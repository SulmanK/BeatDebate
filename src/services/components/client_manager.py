"""
Client Manager Component

Handles API client instantiation and session management for the BeatDebate system.
Extracted from APIService to follow single responsibility principle.
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

import structlog

# Handle imports gracefully
try:
    from ...api import (
        APIClientFactory,
        LastFmClient,
        SpotifyClient
    )
except ImportError:
    # Fallback imports for testing
    import sys
    sys.path.append('src')
    from api import (
        APIClientFactory,
        LastFmClient,
        SpotifyClient
    )

logger = structlog.get_logger(__name__)


class ClientManager:
    """
    Manages API client instances and sessions.
    
    Responsibilities:
    - Client instantiation and caching
    - Session management
    - Credential handling
    - Connection lifecycle management
    """
    
    def __init__(
        self,
        lastfm_api_key: Optional[str] = None,
        spotify_client_id: Optional[str] = None,
        spotify_client_secret: Optional[str] = None
    ):
        """
        Initialize client manager.
        
        Args:
            lastfm_api_key: Last.fm API key (defaults to env var)
            spotify_client_id: Spotify client ID (defaults to env var)
            spotify_client_secret: Spotify client secret (defaults to env var)
        """
        self.logger = logger.bind(component="ClientManager")
        
        # Initialize client factory
        self.client_factory = APIClientFactory()
        
        # Store credentials
        self.lastfm_api_key = lastfm_api_key or os.getenv('LASTFM_API_KEY')
        self.spotify_client_id = spotify_client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.spotify_client_secret = (
            spotify_client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        )
        
        # Client instances (created on demand)
        self._lastfm_client: Optional[LastFmClient] = None
        self._spotify_client: Optional[SpotifyClient] = None
        
        self.logger.info(
            "Client Manager initialized",
            has_lastfm_key=bool(self.lastfm_api_key),
            has_spotify_credentials=bool(
                self.spotify_client_id and self.spotify_client_secret
            )
        )
    
    async def get_lastfm_client(self) -> LastFmClient:
        """
        Get shared Last.fm client instance.
        
        Returns:
            Configured LastFmClient instance
        """
        if self._lastfm_client is None:
            if not self.lastfm_api_key:
                raise ValueError("Last.fm API key is required")
            
            self._lastfm_client = await self.client_factory.create_lastfm_client(
                api_key=self.lastfm_api_key
            )
            
            # Initialize the session immediately
            await self._lastfm_client.__aenter__()
            
            self.logger.info("Last.fm client created and cached")
        
        return self._lastfm_client
    
    async def get_spotify_client(self) -> SpotifyClient:
        """
        Get shared Spotify client instance.
        
        Returns:
            Configured SpotifyClient instance
        """
        if self._spotify_client is None:
            if not self.spotify_client_id or not self.spotify_client_secret:
                self.logger.warning("Spotify credentials not provided")
                return None
            
            self._spotify_client = await self.client_factory.create_spotify_client(
                client_id=self.spotify_client_id,
                client_secret=self.spotify_client_secret
            )
            
            # Initialize the session
            try:
                await self._spotify_client.__aenter__()
                self.logger.info("Spotify client created and cached")
            except Exception as e:
                self.logger.warning("Failed to initialize Spotify client session", error=str(e))
                self._spotify_client = None
                return None
        
        return self._spotify_client
    
    @asynccontextmanager
    async def lastfm_session(self):
        """
        Context manager for Last.fm API operations.
        
        Usage:
            async with client_manager.lastfm_session() as client:
                tracks = await client.search_tracks("indie rock")
        """
        client = await self.get_lastfm_client()
        async with client:
            yield client
    
    @asynccontextmanager
    async def spotify_session(self):
        """
        Context manager for Spotify API operations.
        
        Usage:
            async with client_manager.spotify_session() as client:
                track = await client.search_track("Artist", "Track")
        """
        client = await self.get_spotify_client()
        async with client:
            yield client
    
    async def close(self):
        """Close all API client connections."""
        try:
            if self._lastfm_client:
                await self._lastfm_client.__aexit__(None, None, None)
                self._lastfm_client = None
        except Exception as e:
            self.logger.warning("Error closing Last.fm client", error=str(e))
        
        try:
            if self._spotify_client:
                await self._spotify_client.__aexit__(None, None, None)
                self._spotify_client = None
        except Exception as e:
            self.logger.warning("Error closing Spotify client", error=str(e))
        
        self.logger.info("Client Manager connections closed") 