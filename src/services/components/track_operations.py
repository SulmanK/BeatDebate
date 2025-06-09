"""
Track Operations Component

Handles track-related API operations for the BeatDebate system.
Extracted from APIService to follow single responsibility principle.
"""

from typing import List, Optional, Dict, Any

import structlog

# Handle imports gracefully
try:
    from ...models.metadata_models import (
        UnifiedTrackMetadata,
        MetadataSource
    )
except ImportError:
    # Fallback imports for testing
    from models.metadata_models import (
        UnifiedTrackMetadata,
        MetadataSource
    )

from .client_manager import ClientManager

logger = structlog.get_logger(__name__)


class TrackOperations:
    """
    Handles track-related API operations.
    
    Responsibilities:
    - Track search across platforms
    - Track metadata retrieval and unification
    - Similar track discovery
    - Tag-based track search
    """
    
    def __init__(self, client_manager: ClientManager, cache_manager: Optional["CacheManager"] = None):
        """
        Initialize track operations.
        
        Args:
            client_manager: Client manager instance
            cache_manager: Cache manager instance (optional)
        """
        self.client_manager = client_manager
        self.cache_manager = cache_manager
        self.logger = logger.bind(component="TrackOperations")
        
        self.logger.info("Track Operations initialized", has_cache=bool(cache_manager))
    
    async def search_unified_tracks(
        self,
        query: str,
        limit: int = 20,
        include_spotify: bool = False
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
            lastfm_client = await self.client_manager.get_lastfm_client()
            if lastfm_client:
                lastfm_tracks = await lastfm_client.search_tracks(query, limit)
                
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
                spotify_client = await self.client_manager.get_spotify_client()
                if spotify_client:
                    for unified_track in unified_tracks:
                        try:
                            spotify_track = await spotify_client.search_track(
                                unified_track.artist,
                                unified_track.name,
                                limit=1
                            )
                            
                            if spotify_track:
                                unified_track.spotify_data = spotify_track
                                unified_track.preview_url = spotify_track.preview_url
                                unified_track.popularity = spotify_track.popularity
                        except Exception as e:
                            self.logger.debug(
                                "Spotify track search failed", 
                                artist=unified_track.artist,
                                track=unified_track.name,
                                error=str(e)
                            )
                            continue
                        
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
        include_audio_features: bool = True,
        include_spotify: bool = False
    ) -> Optional[UnifiedTrackMetadata]:
        """
        Get comprehensive track information from multiple sources.
        
        Args:
            artist: Artist name
            track: Track name
            include_audio_features: Whether to fetch Spotify audio features
            include_spotify: Whether to include Spotify data
            
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
            lastfm_client = await self.client_manager.get_lastfm_client()
            if lastfm_client:
                lastfm_track = await lastfm_client.get_track_info(artist, track)
                
                if lastfm_track:
                    unified_track = UnifiedTrackMetadata(
                        name=lastfm_track.name,
                        artist=lastfm_track.artist,
                        source=MetadataSource.LASTFM,
                        lastfm_data=lastfm_track,
                        tags=lastfm_track.tags,
                        listeners=lastfm_track.listeners,
                        playcount=lastfm_track.playcount
                    )
                    
        except Exception as e:
            self.logger.error("Last.fm track info failed", error=str(e))
        
        # Enhance with Spotify data if requested
        if unified_track and include_spotify:
            try:
                spotify_client = await self.client_manager.get_spotify_client()
                if spotify_client:
                    spotify_track = await spotify_client.search_track(artist, track)
                    
                    if spotify_track:
                        unified_track.spotify_data = spotify_track
                        unified_track.preview_url = spotify_track.preview_url
                        unified_track.popularity = spotify_track.popularity
                        unified_track.source = MetadataSource.UNIFIED
                        
                        # Get audio features if requested
                        if include_audio_features:
                            audio_features = await spotify_client.get_audio_features(
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
            lastfm_client = await self.client_manager.get_lastfm_client()
            if lastfm_client:
                lastfm_similar = await lastfm_client.get_similar_tracks(artist, track, limit)
                
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
                spotify_client = await self.client_manager.get_spotify_client()
                if spotify_client:
                    for unified_track in similar_tracks:
                        try:
                            spotify_track = await spotify_client.search_track(
                                unified_track.artist,
                                unified_track.name
                            )
                            
                            if spotify_track:
                                unified_track.spotify_data = spotify_track
                                unified_track.source = MetadataSource.UNIFIED
                                
                                if include_spotify_features:
                                    audio_features = await spotify_client.get_audio_features(
                                        spotify_track.id
                                    )
                                    if audio_features:
                                        unified_track.audio_features = audio_features
                        except Exception as e:
                            self.logger.debug(
                                "Spotify similar track enhancement failed",
                                artist=unified_track.artist,
                                track=unified_track.name,
                                error=str(e)
                            )
                            continue
                                    
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
            lastfm_client = await self.client_manager.get_lastfm_client()
            if lastfm_client:
                lastfm_tracks = await lastfm_client.search_by_tags(tags, limit)
                
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
    
    async def search_tracks(
        self,
        query: str,
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """
        Search tracks with a general query.
        Alias for search_unified_tracks for compatibility.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of unified track metadata
        """
        return await self.search_unified_tracks(
            query=query,
            limit=limit,
            include_spotify=False
        )
    
    async def get_tracks_by_tag(
        self,
        tag: str,
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """
        Get tracks by a specific tag.
        Alias for search_by_tags for compatibility.
        
        Args:
            tag: Tag to search for
            limit: Maximum results
            
        Returns:
            List of tracks matching the tag
        """
        return await self.search_by_tags(tags=[tag], limit=limit)
    
    async def search_underground_tracks_by_tags(
        self,
        tags: List[str],
        limit: int = 20,
        max_listeners: int = 50000
    ) -> List[UnifiedTrackMetadata]:
        """
        Search for underground tracks by tags with strict popularity filtering.
        
        Uses track.search instead of tag.getTopTracks to avoid popular tracks.
        
        Args:
            tags: List of tags to search
            limit: Maximum results
            max_listeners: Maximum listener count for underground classification
            
        Returns:
            List of underground tracks matching tags
        """
        try:
            lastfm_client = await self.client_manager.get_lastfm_client()
            if lastfm_client:
                lastfm_tracks = await lastfm_client.search_underground_tracks_by_tags(
                    tags, limit, max_listeners
                )
                
                unified_tracks = []
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
                    # Calculate underground score immediately
                    unified_track.underground_score = unified_track.calculate_underground_score()
                    unified_tracks.append(unified_track)
                
                self.logger.info(
                    "Underground track search completed",
                    tags=tags,
                    results=len(unified_tracks),
                    max_listeners=max_listeners
                )
                
                return unified_tracks
                
        except Exception as e:
            self.logger.error("Underground track search failed", tags=tags, error=str(e))
            return [] 