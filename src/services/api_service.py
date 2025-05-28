"""
API Service

Centralized API client management for the BeatDebate system.
Eliminates duplicate client instantiation patterns across agents and services.
"""

import os
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import structlog

# Handle imports gracefully
try:
    from ..api import (
        APIClientFactory,
        LastFmClient,
        SpotifyClient,
        UnifiedRateLimiter,
        TrackMetadata,
        ArtistMetadata,
        SpotifyTrack,
        AudioFeatures
    )
except ImportError:
    # Fallback imports for testing
    import sys
    sys.path.append('src')
    from api import (
        APIClientFactory,
        LastFmClient,
        SpotifyClient,
        UnifiedRateLimiter,
        TrackMetadata,
        ArtistMetadata,
        SpotifyTrack,
        AudioFeatures
    )

try:
    from ..models.metadata_models import (
        UnifiedTrackMetadata,
        UnifiedArtistMetadata,
        MetadataSource,
        merge_track_metadata
    )
except ImportError:
    # Fallback imports for testing
    from models.metadata_models import (
        UnifiedTrackMetadata,
        UnifiedArtistMetadata,
        MetadataSource,
        merge_track_metadata
    )

logger = structlog.get_logger(__name__)


class APIService:
    """
    Centralized API service for all external API interactions.
    
    Provides:
    - Shared API client instances with proper rate limiting
    - Unified metadata retrieval and merging
    - Consistent error handling and logging
    - Cache integration for API responses
    """
    
    def __init__(
        self,
        lastfm_api_key: Optional[str] = None,
        spotify_client_id: Optional[str] = None,
        spotify_client_secret: Optional[str] = None,
        cache_manager: Optional["CacheManager"] = None
    ):
        """
        Initialize API service.
        
        Args:
            lastfm_api_key: Last.fm API key (defaults to env var)
            spotify_client_id: Spotify client ID (defaults to env var)
            spotify_client_secret: Spotify client secret (defaults to env var)
            cache_manager: Cache manager instance (optional)
        """
        self.logger = logger.bind(service="APIService")
        self.cache_manager = cache_manager
        
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
            "API Service initialized",
            has_lastfm_key=bool(self.lastfm_api_key),
            has_spotify_credentials=bool(
                self.spotify_client_id and self.spotify_client_secret
            ),
            has_cache=bool(cache_manager)
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
                raise ValueError("Spotify client ID and secret are required")
            
            self._spotify_client = await self.client_factory.create_spotify_client(
                client_id=self.spotify_client_id,
                client_secret=self.spotify_client_secret
            )
            
            self.logger.info("Spotify client created and cached")
        
        return self._spotify_client
    
    @asynccontextmanager
    async def lastfm_session(self):
        """
        Context manager for Last.fm API operations.
        
        Usage:
            async with api_service.lastfm_session() as client:
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
            async with api_service.spotify_session() as client:
                track = await client.search_track("Artist", "Track")
        """
        client = await self.get_spotify_client()
        async with client:
            yield client
    
    async def search_unified_tracks(
        self,
        query: str,
        limit: int = 20,
        include_spotify: bool = True
    ) -> List[UnifiedTrackMetadata]:
        """
        Search for tracks across multiple sources and return unified metadata.
        
        Args:
            query: Search query
            limit: Maximum results
            include_spotify: Whether to include Spotify data
            
        Returns:
            List of unified track metadata
        """
        unified_tracks = []
        
        # Search Last.fm
        try:
            async with self.lastfm_session() as lastfm:
                lastfm_tracks = await lastfm.search_tracks(query, limit)
                
                for track in lastfm_tracks:
                    unified_track = UnifiedTrackMetadata(
                        name=track.name,
                        artist=track.artist,
                        source=MetadataSource.LASTFM,
                        lastfm_data=track,
                        tags=track.tags,
                        listeners=track.listeners,
                        playcount=track.playcount
                    )
                    unified_tracks.append(unified_track)
                    
        except Exception as e:
            self.logger.error("Last.fm search failed", error=str(e))
        
        # Enhance with Spotify data if requested
        if include_spotify and unified_tracks:
            try:
                async with self.spotify_session() as spotify:
                    for unified_track in unified_tracks:
                        spotify_track = await spotify.search_track(
                            unified_track.artist,
                            unified_track.name,
                            limit=1
                        )
                        
                        if spotify_track:
                            unified_track.spotify_data = spotify_track
                            unified_track.preview_url = spotify_track.preview_url
                            unified_track.popularity = spotify_track.popularity
                            
            except Exception as e:
                self.logger.error("Spotify enhancement failed", error=str(e))
        
        self.logger.info(
            "Unified track search completed",
            query=query,
            results=len(unified_tracks),
            include_spotify=include_spotify
        )
        
        return unified_tracks
    
    async def get_unified_track_info(
        self,
        artist: str,
        track: str,
        include_audio_features: bool = True
    ) -> Optional[UnifiedTrackMetadata]:
        """
        Get comprehensive track information from multiple sources.
        
        Args:
            artist: Artist name
            track: Track name
            include_audio_features: Whether to fetch Spotify audio features
            
        Returns:
            Unified track metadata or None
        """
        # Check cache first
        cache_key = f"{artist}:{track}"
        if self.cache_manager:
            cached = self.cache_manager.get_track_metadata(artist, track)
            if cached:
                self.logger.debug("Track metadata cache hit", artist=artist, track=track)
                return cached
        
        unified_track = None
        
        # Get Last.fm data
        try:
            async with self.lastfm_session() as lastfm:
                lastfm_track = await lastfm.get_track_info(artist, track)
                
                if lastfm_track:
                    unified_track = UnifiedTrackMetadata(
                        name=lastfm_track.name,
                        artist=lastfm_track.artist,
                        source=MetadataSource.LASTFM,
                        lastfm_data=lastfm_track,
                        tags=lastfm_track.tags,
                        listeners=lastfm_track.listeners,
                        playcount=lastfm_track.playcount,
                        summary=lastfm_track.summary
                    )
                    
        except Exception as e:
            self.logger.error("Last.fm track info failed", error=str(e))
        
        # Enhance with Spotify data
        if unified_track:
            try:
                async with self.spotify_session() as spotify:
                    spotify_track = await spotify.search_track(artist, track)
                    
                    if spotify_track:
                        unified_track.spotify_data = spotify_track
                        unified_track.preview_url = spotify_track.preview_url
                        unified_track.popularity = spotify_track.popularity
                        unified_track.source = MetadataSource.UNIFIED
                        
                        # Get audio features if requested
                        if include_audio_features:
                            audio_features = await spotify.get_audio_features(
                                spotify_track.id
                            )
                            if audio_features:
                                unified_track.audio_features = audio_features
                                
            except Exception as e:
                self.logger.error("Spotify track enhancement failed", error=str(e))
        
        # Cache the result
        if unified_track and self.cache_manager:
            self.cache_manager.cache_track_metadata(artist, track, unified_track)
        
        return unified_track
    
    async def get_similar_tracks(
        self,
        artist: str,
        track: str,
        limit: int = 20,
        include_spotify_features: bool = False
    ) -> List[UnifiedTrackMetadata]:
        """
        Get similar tracks with unified metadata.
        
        Args:
            artist: Seed artist name
            track: Seed track name
            limit: Maximum results
            include_spotify_features: Whether to include audio features
            
        Returns:
            List of similar tracks with unified metadata
        """
        similar_tracks = []
        
        try:
            async with self.lastfm_session() as lastfm:
                lastfm_similar = await lastfm.get_similar_tracks(artist, track, limit)
                
                for similar_track in lastfm_similar:
                    unified_track = UnifiedTrackMetadata(
                        name=similar_track.name,
                        artist=similar_track.artist,
                        source=MetadataSource.LASTFM,
                        lastfm_data=similar_track
                    )
                    similar_tracks.append(unified_track)
                    
        except Exception as e:
            self.logger.error("Similar tracks search failed", error=str(e))
        
        # Enhance with Spotify data if requested
        if include_spotify_features and similar_tracks:
            try:
                async with self.spotify_session() as spotify:
                    for unified_track in similar_tracks:
                        spotify_track = await spotify.search_track(
                            unified_track.artist,
                            unified_track.name
                        )
                        
                        if spotify_track:
                            unified_track.spotify_data = spotify_track
                            unified_track.source = MetadataSource.UNIFIED
                            
                            if include_spotify_features:
                                audio_features = await spotify.get_audio_features(
                                    spotify_track.id
                                )
                                if audio_features:
                                    unified_track.audio_features = audio_features
                                    
            except Exception as e:
                self.logger.error("Spotify similar tracks enhancement failed", error=str(e))
        
        self.logger.info(
            "Similar tracks search completed",
            seed_artist=artist,
            seed_track=track,
            results=len(similar_tracks)
        )
        
        return similar_tracks
    
    async def search_by_tags(
        self,
        tags: List[str],
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """
        Search tracks by tags using Last.fm.
        
        Args:
            tags: List of tags to search
            limit: Maximum results
            
        Returns:
            List of tracks matching tags
        """
        try:
            async with self.lastfm_session() as lastfm:
                lastfm_tracks = await lastfm.search_by_tags(tags, limit)
                
                unified_tracks = []
                for track in lastfm_tracks:
                    unified_track = UnifiedTrackMetadata(
                        name=track.name,
                        artist=track.artist,
                        source=MetadataSource.LASTFM,
                        lastfm_data=track,
                        tags=track.tags
                    )
                    unified_tracks.append(unified_track)
                
                self.logger.info(
                    "Tag search completed",
                    tags=tags,
                    results=len(unified_tracks)
                )
                
                return unified_tracks
                
        except Exception as e:
            self.logger.error("Tag search failed", tags=tags, error=str(e))
            return []
    
    async def get_artist_info(
        self,
        artist: str,
        include_top_tracks: bool = True
    ) -> Optional[UnifiedArtistMetadata]:
        """
        Get comprehensive artist information.
        
        Args:
            artist: Artist name
            include_top_tracks: Whether to include top tracks
            
        Returns:
            Unified artist metadata or None
        """
        try:
            async with self.lastfm_session() as lastfm:
                artist_info = await lastfm.get_artist_info(artist)
                
                if not artist_info:
                    return None
                
                unified_artist = UnifiedArtistMetadata(
                    name=artist_info.name,
                    source=MetadataSource.LASTFM,
                    lastfm_data=artist_info,
                    tags=artist_info.tags,
                    similar_artists=artist_info.similar_artists,
                    listeners=artist_info.listeners,
                    playcount=artist_info.playcount,
                    bio=artist_info.bio
                )
                
                # Get top tracks if requested
                if include_top_tracks:
                    top_tracks = await lastfm.get_artist_top_tracks(artist, limit=10)
                    unified_artist.top_tracks = [
                        track.name for track in top_tracks
                    ]
                
                return unified_artist
                
        except Exception as e:
            self.logger.error("Artist info failed", artist=artist, error=str(e))
            return None
    
    async def close(self):
        """Close all API client connections."""
        if self._lastfm_client:
            await self._lastfm_client.__aexit__(None, None, None)
        
        if self._spotify_client:
            await self._spotify_client.__aexit__(None, None, None)
        
        self.logger.info("API Service connections closed")


# Global API service instance
_global_api_service: Optional[APIService] = None


def get_api_service(
    lastfm_api_key: Optional[str] = None,
    spotify_client_id: Optional[str] = None,
    spotify_client_secret: Optional[str] = None,
    cache_manager: Optional["CacheManager"] = None
) -> APIService:
    """
    Get global API service instance.
    
    Args:
        lastfm_api_key: Last.fm API key (optional)
        spotify_client_id: Spotify client ID (optional)
        spotify_client_secret: Spotify client secret (optional)
        cache_manager: Cache manager instance (optional)
        
    Returns:
        Global APIService instance
    """
    global _global_api_service
    
    if _global_api_service is None:
        _global_api_service = APIService(
            lastfm_api_key=lastfm_api_key,
            spotify_client_id=spotify_client_id,
            spotify_client_secret=spotify_client_secret,
            cache_manager=cache_manager
        )
    
    return _global_api_service


async def close_api_service():
    """Close global API service."""
    global _global_api_service
    
    if _global_api_service:
        await _global_api_service.close()
        _global_api_service = None 