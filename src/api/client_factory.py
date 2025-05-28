"""
API Client Factory

Provides standardized creation and configuration of all API clients.
Eliminates duplicate client instantiation patterns across the codebase.
"""

import os
from typing import Optional, Dict, Any

import structlog

from .base_client import BaseAPIClient
from .rate_limiter import UnifiedRateLimiter
from ..models.agent_models import SystemConfig

logger = structlog.get_logger(__name__)


class APIClientFactory:
    """
    Factory for creating configured API clients.
    
    Provides a centralized way to create all API clients with proper
    rate limiting, configuration, and dependency injection.
    """
    
    def __init__(self, system_config: Optional[SystemConfig] = None):
        """
        Initialize client factory.
        
        Args:
            system_config: System configuration (optional)
        """
        self.system_config = system_config
        self.logger = logger.bind(service="APIClientFactory")
        
        # Cache for rate limiters (shared across clients of same type)
        self._rate_limiters: Dict[str, UnifiedRateLimiter] = {}
        
        self.logger.info("API Client Factory initialized")
    
    async def create_lastfm_client(
        self, 
        api_key: Optional[str] = None,
        rate_limit: Optional[float] = None,
        shared_secret: Optional[str] = None
    ) -> "LastFmClient":
        """
        Create configured Last.fm client.
        
        Args:
            api_key: Last.fm API key (defaults to env var or system config)
            rate_limit: Requests per second (defaults to system config)
            shared_secret: Last.fm shared secret (optional)
            
        Returns:
            Configured LastFmClient instance
        """
        # Resolve configuration
        api_key = api_key or self._get_lastfm_api_key()
        rate_limit = rate_limit or self._get_lastfm_rate_limit()
        
        if not api_key:
            raise ValueError("Last.fm API key is required")
        
        # Get or create rate limiter
        rate_limiter_key = f"lastfm_{rate_limit}"
        if rate_limiter_key not in self._rate_limiters:
            self._rate_limiters[rate_limiter_key] = UnifiedRateLimiter.for_lastfm(rate_limit)
        
        # Import here to avoid circular imports
        from .lastfm_client import LastFmClient
        
        client = LastFmClient(
            api_key=api_key,
            shared_secret=shared_secret,
            rate_limiter=self._rate_limiters[rate_limiter_key]
        )
        
        self.logger.info(
            "Last.fm client created",
            rate_limit=rate_limit,
            has_shared_secret=bool(shared_secret)
        )
        
        return client
    
    async def create_spotify_client(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        rate_limit: Optional[int] = None
    ) -> "SpotifyClient":
        """
        Create configured Spotify client.
        
        Args:
            client_id: Spotify client ID (defaults to env var or system config)
            client_secret: Spotify client secret (defaults to env var or system config)
            rate_limit: Requests per hour (defaults to system config)
            
        Returns:
            Configured SpotifyClient instance
        """
        # Resolve configuration
        client_id = client_id or self._get_spotify_client_id()
        client_secret = client_secret or self._get_spotify_client_secret()
        rate_limit = rate_limit or self._get_spotify_rate_limit()
        
        if not client_id or not client_secret:
            raise ValueError("Spotify client ID and secret are required")
        
        # Get or create rate limiter
        rate_limiter_key = f"spotify_{rate_limit}"
        if rate_limiter_key not in self._rate_limiters:
            self._rate_limiters[rate_limiter_key] = UnifiedRateLimiter.for_spotify(rate_limit)
        
        # Import here to avoid circular imports
        from .spotify_client import SpotifyClient
        
        client = SpotifyClient(
            client_id=client_id,
            client_secret=client_secret,
            rate_limiter=self._rate_limiters[rate_limiter_key]
        )
        
        self.logger.info(
            "Spotify client created",
            rate_limit=rate_limit
        )
        
        return client
    
    def create_gemini_rate_limiter(
        self,
        calls_per_minute: Optional[int] = None
    ) -> UnifiedRateLimiter:
        """
        Create rate limiter for Gemini LLM API.
        
        Args:
            calls_per_minute: Calls per minute (defaults to system config)
            
        Returns:
            Configured rate limiter for Gemini
        """
        calls_per_minute = calls_per_minute or self._get_gemini_rate_limit()
        
        rate_limiter_key = f"gemini_{calls_per_minute}"
        if rate_limiter_key not in self._rate_limiters:
            self._rate_limiters[rate_limiter_key] = UnifiedRateLimiter.for_gemini(calls_per_minute)
        
        return self._rate_limiters[rate_limiter_key]
    
    def get_rate_limiter_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all active rate limiters.
        
        Returns:
            Dictionary of rate limiter statistics
        """
        stats = {}
        for key, limiter in self._rate_limiters.items():
            stats[key] = limiter.get_current_usage()
        
        return stats
    
    def reset_rate_limiters(self) -> None:
        """Reset all rate limiters (useful for testing)."""
        for limiter in self._rate_limiters.values():
            limiter.reset()
        
        self.logger.info("All rate limiters reset")
    
    # Configuration resolution methods
    def _get_lastfm_api_key(self) -> Optional[str]:
        """Get Last.fm API key from configuration or environment."""
        if self.system_config and hasattr(self.system_config, 'lastfm_api_key'):
            return self.system_config.lastfm_api_key
        return os.getenv('LASTFM_API_KEY')
    
    def _get_lastfm_rate_limit(self) -> float:
        """Get Last.fm rate limit from configuration."""
        if self.system_config and hasattr(self.system_config, 'lastfm_rate_limit'):
            return self.system_config.lastfm_rate_limit
        return 3.0  # Default
    
    def _get_spotify_client_id(self) -> Optional[str]:
        """Get Spotify client ID from configuration or environment."""
        if self.system_config and hasattr(self.system_config, 'spotify_client_id'):
            return self.system_config.spotify_client_id
        return os.getenv('SPOTIFY_CLIENT_ID')
    
    def _get_spotify_client_secret(self) -> Optional[str]:
        """Get Spotify client secret from configuration or environment."""
        if self.system_config and hasattr(self.system_config, 'spotify_client_secret'):
            return self.system_config.spotify_client_secret
        return os.getenv('SPOTIFY_CLIENT_SECRET')
    
    def _get_spotify_rate_limit(self) -> int:
        """Get Spotify rate limit from configuration."""
        if self.system_config and hasattr(self.system_config, 'spotify_rate_limit'):
            return self.system_config.spotify_rate_limit
        return 50  # Default
    
    def _get_gemini_rate_limit(self) -> int:
        """Get Gemini rate limit from configuration."""
        if self.system_config and hasattr(self.system_config, 'gemini_rate_limit'):
            return self.system_config.gemini_rate_limit
        return 15  # Default


# Global factory instance for convenience
_global_factory: Optional[APIClientFactory] = None


def get_client_factory(system_config: Optional[SystemConfig] = None) -> APIClientFactory:
    """
    Get global client factory instance.
    
    Args:
        system_config: System configuration (optional, used for initialization)
        
    Returns:
        Global APIClientFactory instance
    """
    global _global_factory
    
    if _global_factory is None:
        _global_factory = APIClientFactory(system_config)
    
    return _global_factory


def reset_client_factory():
    """Reset global client factory (useful for testing)."""
    global _global_factory
    _global_factory = None


# Convenience functions for quick client creation
async def create_lastfm_client(
    api_key: Optional[str] = None,
    rate_limit: Optional[float] = None
) -> "LastFmClient":
    """Create Last.fm client using global factory."""
    factory = get_client_factory()
    return await factory.create_lastfm_client(api_key, rate_limit)


async def create_spotify_client(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    rate_limit: Optional[int] = None
) -> "SpotifyClient":
    """Create Spotify client using global factory."""
    factory = get_client_factory()
    return await factory.create_spotify_client(client_id, client_secret, rate_limit) 