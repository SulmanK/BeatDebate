"""
Metadata Service for BeatDebate

Unified service for metadata operations across all APIs.
Provides consistent metadata handling and eliminates duplicate metadata logic.
"""

from typing import Dict, List, Any, Optional
import structlog

from ..models.metadata_models import UnifiedTrackMetadata, UnifiedArtistMetadata
from ..api.lastfm_client import LastFmClient

logger = structlog.get_logger(__name__)


class MetadataService:
    """
    Unified service for metadata operations across all APIs.
    
    Provides:
    - Unified track metadata retrieval
    - Unified artist metadata retrieval
    - Cross-service metadata enrichment
    - Consistent metadata formatting
    """
    
    def __init__(self, lastfm_client: LastFmClient, spotify_client=None):
        """
        Initialize metadata service with API clients.
        
        Args:
            lastfm_client: Last.fm API client
            spotify_client: Spotify API client (optional)
        """
        self.lastfm_client = lastfm_client
        self.spotify_client = spotify_client
        self.logger = logger.bind(service="MetadataService")
        
        self.logger.info("MetadataService initialized")
    
    async def get_unified_track_metadata(
        self, 
        artist: str, 
        track: str
    ) -> Optional[UnifiedTrackMetadata]:
        """
        Get unified track metadata from multiple sources.
        
        Args:
            artist: Artist name
            track: Track name
            
        Returns:
            Unified track metadata or None if not found
        """
        try:
            # Get Last.fm metadata
            lastfm_data = await self._get_lastfm_track_metadata(artist, track)
            
            # Get Spotify metadata if available
            spotify_data = None
            if self.spotify_client:
                spotify_data = await self._get_spotify_track_metadata(artist, track)
            
            # Merge metadata
            unified_metadata = self._merge_track_metadata(lastfm_data, spotify_data)
            
            return unified_metadata
            
        except Exception as e:
            self.logger.error("Failed to get unified track metadata", 
                            artist=artist, track=track, error=str(e))
            return None
    
    async def search_tracks_unified(self, query: str, limit: int = 20) -> List[UnifiedTrackMetadata]:
        """
        Search tracks across multiple services with unified results.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of unified track metadata
        """
        try:
            unified_tracks = []
            
            # Search Last.fm
            if self.lastfm_client:
                lastfm_tracks = await self.lastfm_client.search_tracks(query, limit=limit)
                for track_data in lastfm_tracks:
                    unified_track = self._convert_lastfm_to_unified(track_data)
                    if unified_track:
                        unified_tracks.append(unified_track)
            
            # Search Spotify if available
            if self.spotify_client:
                spotify_tracks = await self.spotify_client.search_tracks(query, limit=limit)
                for track_data in spotify_tracks:
                    unified_track = self._convert_spotify_to_unified(track_data)
                    if unified_track:
                        unified_tracks.append(unified_track)
            
            # Remove duplicates and limit results
            unique_tracks = self._deduplicate_tracks(unified_tracks)
            return unique_tracks[:limit]
            
        except Exception as e:
            self.logger.error("Failed to search tracks unified", query=query, error=str(e))
            return []
    
    async def get_unified_artist_metadata(self, artist: str) -> Optional[UnifiedArtistMetadata]:
        """
        Get unified artist metadata from multiple sources.
        
        Args:
            artist: Artist name
            
        Returns:
            Unified artist metadata or None if not found
        """
        try:
            # Get Last.fm metadata
            lastfm_data = await self._get_lastfm_artist_metadata(artist)
            
            # Get Spotify metadata if available
            spotify_data = None
            if self.spotify_client:
                spotify_data = await self._get_spotify_artist_metadata(artist)
            
            # Merge metadata
            unified_metadata = self._merge_artist_metadata(lastfm_data, spotify_data)
            
            return unified_metadata
            
        except Exception as e:
            self.logger.error("Failed to get unified artist metadata", 
                            artist=artist, error=str(e))
            return None
    
    async def _get_lastfm_track_metadata(self, artist: str, track: str) -> Optional[Dict[str, Any]]:
        """Get track metadata from Last.fm."""
        try:
            track_info = await self.lastfm_client.get_track_info(artist, track)
            return track_info
        except Exception as e:
            self.logger.warning("Failed to get Last.fm track metadata", 
                              artist=artist, track=track, error=str(e))
            return None
    
    async def _get_spotify_track_metadata(self, artist: str, track: str) -> Optional[Dict[str, Any]]:
        """Get track metadata from Spotify."""
        try:
            if not self.spotify_client:
                return None
            
            # Search for track on Spotify
            search_results = await self.spotify_client.search_tracks(f"{artist} {track}", limit=1)
            if search_results:
                return search_results[0]
            return None
            
        except Exception as e:
            self.logger.warning("Failed to get Spotify track metadata", 
                              artist=artist, track=track, error=str(e))
            return None
    
    async def _get_lastfm_artist_metadata(self, artist: str) -> Optional[Dict[str, Any]]:
        """Get artist metadata from Last.fm."""
        try:
            artist_info = await self.lastfm_client.get_artist_info(artist)
            return artist_info
        except Exception as e:
            self.logger.warning("Failed to get Last.fm artist metadata", 
                              artist=artist, error=str(e))
            return None
    
    async def _get_spotify_artist_metadata(self, artist: str) -> Optional[Dict[str, Any]]:
        """Get artist metadata from Spotify."""
        try:
            if not self.spotify_client:
                return None
            
            # Search for artist on Spotify
            search_results = await self.spotify_client.search_artists(artist, limit=1)
            if search_results:
                return search_results[0]
            return None
            
        except Exception as e:
            self.logger.warning("Failed to get Spotify artist metadata", 
                              artist=artist, error=str(e))
            return None
    
    def _merge_track_metadata(
        self, 
        lastfm_data: Optional[Dict[str, Any]], 
        spotify_data: Optional[Dict[str, Any]]
    ) -> Optional[UnifiedTrackMetadata]:
        """Merge track metadata from multiple sources."""
        if not lastfm_data and not spotify_data:
            return None
        
        # Use Last.fm as primary source, enrich with Spotify
        primary_data = lastfm_data or spotify_data
        secondary_data = spotify_data if lastfm_data else None
        
        try:
            unified_track = UnifiedTrackMetadata(
                name=primary_data.get('name', ''),
                artist=primary_data.get('artist', ''),
                album=primary_data.get('album', {}).get('title') if isinstance(primary_data.get('album'), dict) else primary_data.get('album'),
                duration_ms=primary_data.get('duration_ms'),
                genres=primary_data.get('genres', []),
                tags=primary_data.get('tags', []),
                popularity=primary_data.get('popularity', 0),
                source_data={
                    'lastfm': lastfm_data,
                    'spotify': spotify_data
                }
            )
            
            # Enrich with secondary source data
            if secondary_data:
                if not unified_track.duration_ms and secondary_data.get('duration_ms'):
                    unified_track.duration_ms = secondary_data['duration_ms']
                
                if not unified_track.genres and secondary_data.get('genres'):
                    unified_track.genres = secondary_data['genres']
            
            return unified_track
            
        except Exception as e:
            self.logger.error("Failed to merge track metadata", error=str(e))
            return None
    
    def _merge_artist_metadata(
        self, 
        lastfm_data: Optional[Dict[str, Any]], 
        spotify_data: Optional[Dict[str, Any]]
    ) -> Optional[UnifiedArtistMetadata]:
        """Merge artist metadata from multiple sources."""
        if not lastfm_data and not spotify_data:
            return None
        
        # Use Last.fm as primary source, enrich with Spotify
        primary_data = lastfm_data or spotify_data
        
        try:
            unified_artist = UnifiedArtistMetadata(
                name=primary_data.get('name', ''),
                genres=primary_data.get('genres', []),
                tags=primary_data.get('tags', []),
                popularity=primary_data.get('popularity', 0),
                source_data={
                    'lastfm': lastfm_data,
                    'spotify': spotify_data
                }
            )
            
            return unified_artist
            
        except Exception as e:
            self.logger.error("Failed to merge artist metadata", error=str(e))
            return None
    
    def _convert_lastfm_to_unified(self, lastfm_data: Dict[str, Any]) -> Optional[UnifiedTrackMetadata]:
        """Convert Last.fm track data to unified format."""
        try:
            return UnifiedTrackMetadata(
                name=lastfm_data.get('name', ''),
                artist=lastfm_data.get('artist', ''),
                album=lastfm_data.get('album', {}).get('title') if isinstance(lastfm_data.get('album'), dict) else lastfm_data.get('album'),
                genres=lastfm_data.get('genres', []),
                tags=lastfm_data.get('tags', []),
                popularity=int(lastfm_data.get('listeners', 0)),
                source_data={'lastfm': lastfm_data}
            )
        except Exception as e:
            self.logger.warning("Failed to convert Last.fm data", error=str(e))
            return None
    
    def _convert_spotify_to_unified(self, spotify_data: Dict[str, Any]) -> Optional[UnifiedTrackMetadata]:
        """Convert Spotify track data to unified format."""
        try:
            return UnifiedTrackMetadata(
                name=spotify_data.get('name', ''),
                artist=spotify_data.get('artists', [{}])[0].get('name', '') if spotify_data.get('artists') else '',
                album=spotify_data.get('album', {}).get('name'),
                duration_ms=spotify_data.get('duration_ms'),
                genres=spotify_data.get('genres', []),
                popularity=spotify_data.get('popularity', 0),
                source_data={'spotify': spotify_data}
            )
        except Exception as e:
            self.logger.warning("Failed to convert Spotify data", error=str(e))
            return None
    
    def _deduplicate_tracks(self, tracks: List[UnifiedTrackMetadata]) -> List[UnifiedTrackMetadata]:
        """Remove duplicate tracks based on artist and name."""
        seen = set()
        unique_tracks = []
        
        for track in tracks:
            track_key = f"{track.artist.lower()}::{track.name.lower()}"
            if track_key not in seen:
                seen.add(track_key)
                unique_tracks.append(track)
        
        return unique_tracks
    
    async def close(self):
        """Close service connections."""
        # Close any persistent connections if needed
        self.logger.info("MetadataService closed")


# Global service instance
_global_metadata_service: Optional[MetadataService] = None


def get_metadata_service(
    lastfm_client: Optional[LastFmClient] = None,
    spotify_client=None
) -> MetadataService:
    """
    Get global metadata service instance.
    
    Args:
        lastfm_client: Last.fm client (optional)
        spotify_client: Spotify client (optional)
        
    Returns:
        Global MetadataService instance
    """
    global _global_metadata_service
    
    if _global_metadata_service is None and lastfm_client:
        _global_metadata_service = MetadataService(
            lastfm_client=lastfm_client,
            spotify_client=spotify_client
        )
    
    return _global_metadata_service


async def close_metadata_service():
    """Close global metadata service."""
    global _global_metadata_service
    
    if _global_metadata_service:
        await _global_metadata_service.close()
        _global_metadata_service = None 