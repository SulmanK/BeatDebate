"""
Refactored API Service

Centralized API client management for the BeatDebate system using modular components.
Follows single responsibility principle with delegated operations.
"""

import os
from typing import Optional, Dict, Any, List

import structlog

# Handle imports gracefully
try:
    from ..models.metadata_models import (
        UnifiedTrackMetadata,
        UnifiedArtistMetadata
    )
except ImportError:
    # Fallback imports for testing
    from models.metadata_models import (
        UnifiedTrackMetadata,
        UnifiedArtistMetadata
    )

from .components import (
    ClientManager,
    TrackOperations,
    ArtistOperations,
    GenreAnalyzer
)

logger = structlog.get_logger(__name__)


class APIService:
    """
    Refactored API service using modular components.
    
    Provides:
    - Shared API client instances with proper rate limiting
    - Unified metadata retrieval and merging
    - Consistent error handling and logging
    - Cache integration for API responses
    - Modular component architecture
    """
    
    def __init__(
        self,
        lastfm_api_key: Optional[str] = None,
        spotify_client_id: Optional[str] = None,
        spotify_client_secret: Optional[str] = None,
        cache_manager: Optional["CacheManager"] = None
    ):
        """
        Initialize API service with modular components.
        
        Args:
            lastfm_api_key: Last.fm API key (defaults to env var)
            spotify_client_id: Spotify client ID (defaults to env var)
            spotify_client_secret: Spotify client secret (defaults to env var)
            cache_manager: Cache manager instance (optional)
        """
        self.logger = logger.bind(service="APIService")
        self.cache_manager = cache_manager
        
        # Initialize components
        self.client_manager = ClientManager(
            lastfm_api_key=lastfm_api_key,
            spotify_client_id=spotify_client_id,
            spotify_client_secret=spotify_client_secret
        )
        
        self.track_operations = TrackOperations(
            client_manager=self.client_manager,
            cache_manager=cache_manager
        )
        
        self.artist_operations = ArtistOperations(
            client_manager=self.client_manager
        )
        
        self.genre_analyzer = GenreAnalyzer(
            client_manager=self.client_manager,
            artist_operations=self.artist_operations
        )
        
        self.logger.info(
            "Refactored API Service initialized",
            has_cache=bool(cache_manager),
            components_loaded=4
        )
    
    # Client Management (delegated to ClientManager)
    
    async def get_lastfm_client(self):
        """Get shared Last.fm client instance."""
        return await self.client_manager.get_lastfm_client()
    
    async def get_spotify_client(self):
        """Get shared Spotify client instance."""
        return await self.client_manager.get_spotify_client()
    
    async def lastfm_session(self):
        """Context manager for Last.fm API operations."""
        return self.client_manager.lastfm_session()
    
    async def spotify_session(self):
        """Context manager for Spotify API operations."""
        return self.client_manager.spotify_session()
    
    # Track Operations (delegated to TrackOperations)
    
    async def search_unified_tracks(
        self,
        query: str,
        limit: int = 20,
        include_spotify: bool = False
    ) -> List[UnifiedTrackMetadata]:
        """Search for tracks across multiple sources and return unified metadata."""
        return await self.track_operations.search_unified_tracks(
            query=query,
            limit=limit,
            include_spotify=include_spotify
        )
    
    async def get_unified_track_info(
        self,
        artist: str,
        track: str,
        include_audio_features: bool = True,
        include_spotify: bool = False
    ) -> Optional[UnifiedTrackMetadata]:
        """Get comprehensive track information from multiple sources."""
        return await self.track_operations.get_unified_track_info(
            artist=artist,
            track=track,
            include_audio_features=include_audio_features,
            include_spotify=include_spotify
        )
    
    async def get_similar_tracks(
        self,
        artist: str,
        track: str,
        limit: int = 20,
        include_spotify_features: bool = False
    ) -> List[UnifiedTrackMetadata]:
        """Get similar tracks with unified metadata."""
        return await self.track_operations.get_similar_tracks(
            artist=artist,
            track=track,
            limit=limit,
            include_spotify_features=include_spotify_features
        )
    
    async def search_by_tags(
        self,
        tags: List[str],
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """Search tracks by tags using Last.fm."""
        return await self.track_operations.search_by_tags(
            tags=tags,
            limit=limit
        )
    
    async def search_tracks_by_tags(
        self,
        tags: List[str],
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """Search tracks by tags using Last.fm (alias for search_by_tags)."""
        return await self.search_by_tags(tags=tags, limit=limit)
    
    async def search_tracks(
        self,
        query: str,
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """Search tracks with a general query."""
        return await self.track_operations.search_tracks(
            query=query,
            limit=limit
        )
    
    async def get_tracks_by_tag(
        self,
        tag: str,
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """Get tracks by a specific tag."""
        return await self.track_operations.get_tracks_by_tag(
            tag=tag,
            limit=limit
        )
    
    async def search_underground_tracks_by_tags(
        self,
        tags: List[str],
        limit: int = 20,
        max_listeners: int = 50000
    ) -> List[UnifiedTrackMetadata]:
        """Search for underground tracks by tags with strict popularity filtering."""
        return await self.track_operations.search_underground_tracks_by_tags(
            tags=tags,
            limit=limit,
            max_listeners=max_listeners
        )
    
    # Artist Operations (delegated to ArtistOperations)
    
    async def get_artist_info(
        self,
        artist: str,
        include_top_tracks: bool = True
    ) -> Optional[UnifiedArtistMetadata]:
        """Get comprehensive artist information."""
        return await self.artist_operations.get_artist_info(
            artist=artist,
            include_top_tracks=include_top_tracks
        )
    
    async def get_similar_artist_tracks(
        self,
        artist: str,
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """Get tracks from artists similar to the given artist."""
        return await self.artist_operations.get_similar_artist_tracks(
            artist=artist,
            limit=limit
        )
    
    async def get_artist_top_tracks(
        self,
        artist: str,
        limit: int = 20,
        page: int = 1
    ) -> List[UnifiedTrackMetadata]:
        """Get top tracks for an artist."""
        return await self.artist_operations.get_artist_top_tracks(
            artist=artist,
            limit=limit,
            page=page
        )
    
    async def get_artist_primary_genres(
        self,
        artist: str,
        max_genres: int = 3
    ) -> List[str]:
        """Get the primary genres for an artist based on their Last.fm tags."""
        return await self.artist_operations.get_artist_primary_genres(
            artist=artist,
            max_genres=max_genres
        )
    
    async def get_similar_artists(self, artist: str, limit: int = 10) -> List[Any]:
        """Get similar artists from Last.fm API."""
        return await self.artist_operations.get_similar_artists(
            artist=artist,
            limit=limit
        )
    
    # Genre Analysis (delegated to GenreAnalyzer)
    
    async def check_artist_genre_match(
        self,
        artist: str,
        target_genre: str,
        include_related_genres: bool = True
    ) -> Dict[str, Any]:
        """Check if an artist matches a specific genre by looking up their metadata."""
        return await self.genre_analyzer.check_artist_genre_match(
            artist=artist,
            target_genre=target_genre,
            include_related_genres=include_related_genres
        )
    
    async def check_track_genre_match(
        self,
        artist: str,
        track: str,
        target_genre: str,
        include_related_genres: bool = True
    ) -> Dict[str, Any]:
        """Check if a track matches a specific genre by looking up track and artist metadata."""
        return await self.genre_analyzer.check_track_genre_match(
            artist=artist,
            track=track,
            target_genre=target_genre,
            include_related_genres=include_related_genres
        )
    
    async def check_genre_relationship_llm(
        self,
        target_genre: str,
        candidate_genre: str,
        llm_client=None
    ) -> Dict[str, Any]:
        """Use LLM to determine if two genres are related dynamically."""
        return await self.genre_analyzer.check_genre_relationship_llm(
            target_genre=target_genre,
            candidate_genre=candidate_genre,
            llm_client=llm_client
        )
    
    async def batch_check_tracks_genre_match(
        self,
        tracks: List[Dict[str, Any]],
        target_genre: str,
        llm_client=None,
        include_related_genres: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Check multiple tracks against a target genre in a single LLM call."""
        return await self.genre_analyzer.batch_check_tracks_genre_match(
            tracks=tracks,
            target_genre=target_genre,
            llm_client=llm_client,
            include_related_genres=include_related_genres
        )
    
    # Lifecycle Management
    
    async def close(self):
        """Close all API client connections."""
        await self.client_manager.close()
        self.logger.info("Refactored API Service closed")


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
