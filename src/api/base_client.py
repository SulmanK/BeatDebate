"""
Base API Client

Provides unified HTTP request handling, rate limiting, and error handling
for all external API clients in the BeatDebate system.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import json

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class BaseAPIClient(ABC):
    """
    Base HTTP client with unified request handling, rate limiting, and error handling.
    
    All API clients (LastFM, Spotify, etc.) should inherit from this class to ensure
    consistent behavior across the system.
    """
    
    def __init__(
        self, 
        base_url: str, 
        rate_limiter: "UnifiedRateLimiter",
        timeout: int = 10,
        service_name: str = "api"
    ):
        """
        Initialize base API client.
        
        Args:
            base_url: Base URL for the API
            rate_limiter: Rate limiter instance for this client
            timeout: Request timeout in seconds
            service_name: Service name for logging
        """
        self.base_url = base_url.rstrip('/')
        self.rate_limiter = rate_limiter
        self.timeout = timeout
        self.service_name = service_name
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logger.bind(service=service_name)
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        self.logger.debug(f"{self.service_name} client session started")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.logger.debug(f"{self.service_name} client session closed")
            
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Make rate-limited HTTP request with error handling and retries.
        
        Args:
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            method: HTTP method (GET, POST, etc.)
            headers: Additional headers
            retries: Number of retry attempts
            
        Returns:
            Parsed JSON response data
            
        Raises:
            Exception: For unrecoverable errors
        """
        if not self.session:
            raise RuntimeError(f"{self.service_name} client not initialized. Use async context manager.")
            
        # Wait for rate limiter
        await self.rate_limiter.wait_if_needed()
        
        # Build full URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
        
        # Prepare request parameters
        request_params = params or {}
        request_headers = headers or {}
        
        # Add default headers
        request_headers.setdefault('User-Agent', f'BeatDebate-{self.service_name}/1.0')
        
        for attempt in range(retries + 1):
            try:
                self.logger.debug(
                    f"{self.service_name} API request",
                    method=method,
                    url=url,
                    attempt=attempt + 1
                )
                
                async with self.session.request(
                    method=method,
                    url=url,
                    params=request_params if method == "GET" else None,
                    json=request_params if method in ["POST", "PUT", "PATCH"] else None,
                    headers=request_headers
                ) as response:
                    
                    # Handle successful responses
                    if response.status == 200:
                        data = await self._parse_response(response)
                        
                        # Check for API-specific errors in response body
                        error_info = self._extract_api_error(data)
                        if error_info:
                            self.logger.error(
                                f"{self.service_name} API error in response",
                                error=error_info,
                                endpoint=endpoint
                            )
                            raise Exception(f"{self.service_name} API error: {error_info}")
                            
                        self.logger.debug(
                            f"{self.service_name} API request successful",
                            endpoint=endpoint,
                            status=response.status
                        )
                        return data
                        
                    # Handle rate limiting
                    elif response.status == 429:
                        wait_time = await self._calculate_backoff_time(response, attempt)
                        self.logger.warning(
                            f"{self.service_name} rate limited",
                            attempt=attempt + 1,
                            wait_time=wait_time,
                            endpoint=endpoint
                        )
                        await asyncio.sleep(wait_time)
                        continue
                        
                    # Handle other HTTP errors
                    else:
                        await self._handle_http_error(response, endpoint, attempt, retries)
                        if attempt < retries:
                            await self._exponential_backoff(attempt)
                            continue
                        else:
                            raise Exception(
                                f"{self.service_name} request failed after {retries + 1} attempts"
                            )
                            
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"{self.service_name} request timeout",
                    attempt=attempt + 1,
                    endpoint=endpoint
                )
                if attempt == retries:
                    raise Exception(f"{self.service_name} request timed out after {retries + 1} attempts")
                await self._exponential_backoff(attempt)
                
            except aiohttp.ClientError as e:
                self.logger.error(
                    f"{self.service_name} client error",
                    error=str(e),
                    attempt=attempt + 1,
                    endpoint=endpoint
                )
                if attempt == retries:
                    raise Exception(f"{self.service_name} client error: {str(e)}")
                await self._exponential_backoff(attempt)
                
            except Exception as e:
                self.logger.error(
                    f"{self.service_name} unexpected error",
                    error=str(e),
                    attempt=attempt + 1,
                    endpoint=endpoint
                )
                if attempt == retries:
                    raise
                await self._exponential_backoff(attempt)
                
        raise Exception(f"{self.service_name} request failed after {retries + 1} attempts")
    
    async def _parse_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """
        Parse API response. Can be overridden by subclasses for custom parsing.
        
        Args:
            response: HTTP response object
            
        Returns:
            Parsed response data
        """
        try:
            return await response.json()
        except json.JSONDecodeError as e:
            self.logger.error(f"{self.service_name} invalid JSON response", error=str(e))
            raise Exception(f"{self.service_name} returned invalid JSON")
    
    @abstractmethod
    def _extract_api_error(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract API-specific error information from response data.
        Must be implemented by subclasses.
        
        Args:
            data: Parsed response data
            
        Returns:
            Error message if found, None otherwise
        """
        pass
    
    async def _calculate_backoff_time(
        self, 
        response: aiohttp.ClientResponse, 
        attempt: int
    ) -> float:
        """
        Calculate backoff time for rate limiting.
        
        Args:
            response: HTTP response with rate limit information
            attempt: Current attempt number
            
        Returns:
            Wait time in seconds
        """
        # Check for Retry-After header
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
                
        # Default exponential backoff
        return min(2 ** attempt, 60)  # Max 60 seconds
    
    async def _handle_http_error(
        self, 
        response: aiohttp.ClientResponse, 
        endpoint: str, 
        attempt: int, 
        max_retries: int
    ):
        """
        Handle HTTP error responses.
        
        Args:
            response: HTTP response object
            endpoint: Request endpoint
            attempt: Current attempt number
            max_retries: Maximum retry attempts
        """
        self.logger.warning(
            f"{self.service_name} HTTP error",
            status=response.status,
            endpoint=endpoint,
            attempt=attempt + 1,
            max_retries=max_retries + 1
        )
        
        # For 4xx errors (except 429), don't retry
        if 400 <= response.status < 500 and response.status != 429:
            raise Exception(f"{self.service_name} client error: {response.status}")
    
    async def _exponential_backoff(self, attempt: int, base_delay: float = 1.0):
        """
        Implement exponential backoff with jitter.
        
        Args:
            attempt: Current attempt number (0-based)
            base_delay: Base delay in seconds
        """
        import random
        
        # Exponential backoff with jitter
        delay = base_delay * (2 ** attempt)
        jitter = random.uniform(0.1, 0.3) * delay
        total_delay = min(delay + jitter, 60.0)  # Max 60 seconds
        
        self.logger.debug(
            f"{self.service_name} backing off",
            attempt=attempt + 1,
            delay=total_delay
        )
        
        await asyncio.sleep(total_delay) 