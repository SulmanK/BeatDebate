"""
Unified Rate Limiter

Provides configurable rate limiting for all API clients in the BeatDebate system.
Supports both per-second and per-hour rate limiting strategies.
"""

import asyncio
import time
from collections import deque
from typing import Optional, Union
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    calls_per_second: Optional[float] = None
    calls_per_minute: Optional[int] = None
    calls_per_hour: Optional[int] = None
    burst_size: Optional[int] = None  # Maximum burst size for token bucket


class UnifiedRateLimiter:
    """
    Unified rate limiter supporting multiple rate limit strategies.
    
    Supports:
    - Per-second limiting (for LastFM: 3 requests/second)
    - Per-hour limiting (for Spotify: 50 requests/hour)
    - Token bucket algorithm for burst handling
    - Multiple time window tracking
    """
    
    def __init__(
        self,
        calls_per_second: Optional[float] = None,
        calls_per_minute: Optional[int] = None,
        calls_per_hour: Optional[int] = None,
        burst_size: Optional[int] = None,
        service_name: str = "api"
    ):
        """
        Initialize rate limiter with specified limits.
        
        Args:
            calls_per_second: Maximum calls per second (float for sub-second intervals)
            calls_per_minute: Maximum calls per minute
            calls_per_hour: Maximum calls per hour
            burst_size: Maximum burst size (defaults to calls_per_second * 2)
            service_name: Service name for logging
        """
        self.calls_per_second = calls_per_second
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        self.service_name = service_name
        
        # Token bucket for burst handling
        if calls_per_second:
            self.burst_size = burst_size or max(int(calls_per_second * 2), 1)
            self.tokens = self.burst_size
            self.last_refill = time.time()
        else:
            self.burst_size = None
            self.tokens = 0
            self.last_refill = 0
        
        # Request history tracking
        self.request_times: deque = deque()
        self.lock = asyncio.Lock()
        
        self.logger = logger.bind(service=f"RateLimiter-{service_name}")
        
        self.logger.info(
            "Rate limiter initialized",
            calls_per_second=calls_per_second,
            calls_per_minute=calls_per_minute,
            calls_per_hour=calls_per_hour,
            burst_size=self.burst_size
        )
    
    @classmethod
    def for_lastfm(cls, calls_per_second: float = 3.0) -> "UnifiedRateLimiter":
        """
        Create rate limiter configured for Last.fm API.
        
        Args:
            calls_per_second: Calls per second (default: 3.0)
            
        Returns:
            Configured rate limiter for Last.fm
        """
        return cls(
            calls_per_second=calls_per_second,
            service_name="LastFM"
        )
    
    @classmethod
    def for_spotify(cls, calls_per_hour: int = 50) -> "UnifiedRateLimiter":
        """
        Create rate limiter configured for Spotify API.
        
        Args:
            calls_per_hour: Calls per hour (default: 50)
            
        Returns:
            Configured rate limiter for Spotify
        """
        return cls(
            calls_per_hour=calls_per_hour,
            service_name="Spotify"
        )
    
    @classmethod
    def for_gemini(cls, calls_per_minute: int = 15) -> "UnifiedRateLimiter":
        """
        Create rate limiter configured for Gemini LLM API.
        
        Args:
            calls_per_minute: Calls per minute (default: 15)
            
        Returns:
            Configured rate limiter for Gemini
        """
        return cls(
            calls_per_minute=calls_per_minute,
            service_name="Gemini"
        )
    
    async def wait_if_needed(self) -> None:
        """
        Wait if necessary to respect rate limits.
        
        This method should be called before making each API request.
        """
        async with self.lock:
            current_time = time.time()
            
            # Clean up old request times
            self._cleanup_old_requests(current_time)
            
            # Check each rate limit type
            wait_time = 0.0
            
            # Per-second rate limiting with token bucket
            if self.calls_per_second:
                wait_time = max(wait_time, await self._check_per_second_limit(current_time))
            
            # Per-minute rate limiting
            if self.calls_per_minute:
                wait_time = max(wait_time, self._check_per_minute_limit(current_time))
            
            # Per-hour rate limiting
            if self.calls_per_hour:
                wait_time = max(wait_time, self._check_per_hour_limit(current_time))
            
            # Wait if necessary
            if wait_time > 0:
                self.logger.debug(
                    "Rate limit wait required",
                    wait_time=wait_time,
                    current_requests=len(self.request_times)
                )
                await asyncio.sleep(wait_time)
                current_time = time.time()
            
            # Record this request
            self.request_times.append(current_time)
            
            # Consume token if using token bucket
            if self.calls_per_second and self.tokens > 0:
                self.tokens -= 1
    
    async def _check_per_second_limit(self, current_time: float) -> float:
        """
        Check per-second rate limit using token bucket algorithm.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        # Refill tokens based on time elapsed
        time_elapsed = current_time - self.last_refill
        tokens_to_add = time_elapsed * self.calls_per_second
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_refill = current_time
        
        # Check if we have tokens available
        if self.tokens >= 1:
            return 0.0
        
        # Calculate wait time for next token
        wait_time = (1.0 - self.tokens) / self.calls_per_second
        return wait_time
    
    def _check_per_minute_limit(self, current_time: float) -> float:
        """
        Check per-minute rate limit.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        minute_ago = current_time - 60
        recent_requests = [t for t in self.request_times if t > minute_ago]
        
        if len(recent_requests) < self.calls_per_minute:
            return 0.0
        
        # Find the oldest request in the current minute window
        oldest_request = min(recent_requests)
        wait_time = 60 - (current_time - oldest_request)
        return max(0.0, wait_time)
    
    def _check_per_hour_limit(self, current_time: float) -> float:
        """
        Check per-hour rate limit.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        hour_ago = current_time - 3600
        recent_requests = [t for t in self.request_times if t > hour_ago]
        
        if len(recent_requests) < self.calls_per_hour:
            return 0.0
        
        # Find the oldest request in the current hour window
        oldest_request = min(recent_requests)
        wait_time = 3600 - (current_time - oldest_request)
        return max(0.0, wait_time)
    
    def _cleanup_old_requests(self, current_time: float) -> None:
        """
        Remove old request timestamps that are outside all rate limit windows.
        
        Args:
            current_time: Current timestamp
        """
        # Determine the longest time window we need to track
        max_window = 0
        if self.calls_per_second:
            max_window = max(max_window, 60)  # Keep 1 minute for burst analysis
        if self.calls_per_minute:
            max_window = max(max_window, 60)
        if self.calls_per_hour:
            max_window = max(max_window, 3600)
        
        if max_window == 0:
            return
        
        # Remove requests older than the maximum window
        cutoff_time = current_time - max_window
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
    
    def get_current_usage(self) -> dict:
        """
        Get current rate limit usage statistics.
        
        Returns:
            Dictionary with current usage information
        """
        current_time = time.time()
        
        # Count requests in each time window
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        
        minute_requests = len([t for t in self.request_times if t > minute_ago])
        hour_requests = len([t for t in self.request_times if t > hour_ago])
        
        usage = {
            "timestamp": current_time,
            "total_requests_tracked": len(self.request_times),
            "requests_last_minute": minute_requests,
            "requests_last_hour": hour_requests,
        }
        
        if self.calls_per_second:
            usage["tokens_available"] = self.tokens
            usage["burst_capacity"] = self.burst_size
            usage["calls_per_second_limit"] = self.calls_per_second
        
        if self.calls_per_minute:
            usage["calls_per_minute_limit"] = self.calls_per_minute
            usage["minute_usage_percent"] = (minute_requests / self.calls_per_minute) * 100
        
        if self.calls_per_hour:
            usage["calls_per_hour_limit"] = self.calls_per_hour
            usage["hour_usage_percent"] = (hour_requests / self.calls_per_hour) * 100
        
        return usage
    
    def reset(self) -> None:
        """Reset rate limiter state (useful for testing)."""
        self.request_times.clear()
        if self.calls_per_second:
            self.tokens = self.burst_size
            self.last_refill = time.time()
        
        self.logger.info("Rate limiter reset")


# Legacy compatibility - these will be deprecated
class RateLimiter(UnifiedRateLimiter):
    """Legacy rate limiter class for backward compatibility."""
    
    def __init__(self, calls_per_second: float):
        super().__init__(calls_per_second=calls_per_second, service_name="Legacy")


class SpotifyRateLimiter(UnifiedRateLimiter):
    """Legacy Spotify rate limiter class for backward compatibility."""
    
    def __init__(self, calls_per_hour: int):
        super().__init__(calls_per_hour=calls_per_hour, service_name="SpotifyLegacy") 