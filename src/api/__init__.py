"""
API Module

Unified API layer for all external service clients.
Provides consistent HTTP handling, rate limiting, and error handling.
"""

from .base_client import BaseAPIClient
from .rate_limiter import UnifiedRateLimiter, RateLimitConfig
from .lastfm_client import LastFmClient, TrackMetadata, ArtistMetadata
from .spotify_client import SpotifyClient, SpotifyTrack, AudioFeatures
from .client_factory import (
    APIClientFactory,
    get_client_factory,
    reset_client_factory,
    create_lastfm_client,
    create_spotify_client
)

__all__ = [
    # Base infrastructure
    "BaseAPIClient",
    "UnifiedRateLimiter",
    "RateLimitConfig",
    
    # LastFM client and models
    "LastFmClient",
    "TrackMetadata",
    "ArtistMetadata",
    
    # Spotify client and models
    "SpotifyClient",
    "SpotifyTrack",
    "AudioFeatures",
    
    # Client factory
    "APIClientFactory",
    "get_client_factory",
    "reset_client_factory",
    "create_lastfm_client",
    "create_spotify_client",
]
