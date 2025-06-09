"""
Artist Operations Component

Handles artist-related API operations for the BeatDebate system.
Extracted from APIService to follow single responsibility principle.
"""

from typing import List, Optional, Any

import structlog

# Handle imports gracefully
try:
    from ...models.metadata_models import (
        UnifiedTrackMetadata,
        UnifiedArtistMetadata,
        MetadataSource
    )
except ImportError:
    # Fallback imports for testing
    from models.metadata_models import (
        UnifiedTrackMetadata,
        UnifiedArtistMetadata,
        MetadataSource
    )

from .client_manager import ClientManager

logger = structlog.get_logger(__name__)


class ArtistOperations:
    """
    Handles artist-related API operations.
    
    Responsibilities:
    - Artist information retrieval
    - Artist top tracks
    - Similar artist discovery
    - Artist genre analysis
    """
    
    def __init__(self, client_manager: ClientManager):
        """
        Initialize artist operations.
        
        Args:
            client_manager: Client manager instance
        """
        self.client_manager = client_manager
        self.logger = logger.bind(component="ArtistOperations")
        
        self.logger.info("Artist Operations initialized")
    
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
            lastfm_client = await self.client_manager.get_lastfm_client()
            if lastfm_client:
                artist_info = await lastfm_client.get_artist_info(artist)
                
                if not artist_info:
                    return None
                
                unified_artist = UnifiedArtistMetadata(
                    name=artist_info.name,
                    source=MetadataSource.LASTFM,
                    tags=artist_info.tags,
                    similar_artists=artist_info.similar_artists,
                    listeners=artist_info.listeners,
                    playcount=artist_info.playcount,
                    bio=artist_info.bio,
                    source_data={"lastfm": artist_info.__dict__}
                )
                
                # Get top tracks if requested
                if include_top_tracks:
                    top_tracks = await lastfm_client.get_artist_top_tracks(artist, limit=10)
                    unified_artist.top_tracks = [
                        track.name for track in top_tracks
                    ]
                
                return unified_artist
                
        except Exception as e:
            self.logger.error(f"Artist info failed for {artist}: {e}")
            return None
    
    async def get_similar_artist_tracks(
        self,
        artist: str,
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """
        Get tracks from artists similar to the given artist.
        
        Args:
            artist: Artist name
            limit: Maximum results
            
        Returns:
            List of tracks from similar artists
        """
        try:
            # Get similar artists first
            lastfm_client = await self.client_manager.get_lastfm_client()
            if lastfm_client:
                artist_info = await lastfm_client.get_artist_info(artist)
                
                if not artist_info or not artist_info.similar_artists:
                    return []
                
                similar_tracks = []
                tracks_per_artist = max(1, limit // len(artist_info.similar_artists))
                
                for similar_artist in artist_info.similar_artists[:5]:  # Limit to top 5
                    try:
                        artist_tracks = await lastfm_client.get_artist_top_tracks(
                            similar_artist, 
                            limit=tracks_per_artist
                        )
                        
                        for track in artist_tracks:
                            unified_track = UnifiedTrackMetadata(
                                name=track.name,
                                artist=track.artist,
                                source=MetadataSource.LASTFM,
                                lastfm_data=track,
                                tags=track.tags if hasattr(track, 'tags') else [],
                                listeners=track.listeners,
                                playcount=track.playcount
                            )
                            similar_tracks.append(unified_track)
                            
                            if len(similar_tracks) >= limit:
                                break
                                
                    except Exception as e:
                        self.logger.warning(
                            "Failed to get tracks for similar artist",
                            similar_artist=similar_artist,
                            error=str(e)
                        )
                        continue
                    
                    if len(similar_tracks) >= limit:
                        break
                
                self.logger.info(
                    "Similar artist tracks search completed",
                    seed_artist=artist,
                    results=len(similar_tracks)
                )
                
                return similar_tracks[:limit]
                
        except Exception as e:
            self.logger.error(
                "Similar artist tracks search failed", 
                artist=artist, 
                error=str(e)
            )
            return []
    
    async def get_artist_top_tracks(
        self,
        artist: str,
        limit: int = 20,
        page: int = 1
    ) -> List[UnifiedTrackMetadata]:
        """
        Get top tracks for an artist.
        
        Args:
            artist: Artist name
            limit: Maximum results
            page: Page number (1-indexed)
            
        Returns:
            List of top tracks with unified metadata
        """
        try:
            lastfm_client = await self.client_manager.get_lastfm_client()
            if lastfm_client:
                lastfm_tracks = await lastfm_client.get_artist_top_tracks(artist, limit, page)
                
                unified_tracks = []
                for track in lastfm_tracks:
                    unified_track = UnifiedTrackMetadata(
                        name=track.name,
                        artist=track.artist,
                        source=MetadataSource.LASTFM,
                        lastfm_data=track,
                        tags=track.tags if hasattr(track, 'tags') else [],
                        listeners=track.listeners,
                        playcount=track.playcount
                    )
                    unified_tracks.append(unified_track)
                
                self.logger.info(
                    "Artist top tracks search completed",
                    artist=artist,
                    results=len(unified_tracks)
                )
                
                return unified_tracks
                
        except Exception as e:
            self.logger.error(
                "Artist top tracks search failed", 
                artist=artist, 
                error=str(e)
            )
            return []
    
    async def get_artist_primary_genres(
        self,
        artist: str,
        max_genres: int = 3
    ) -> List[str]:
        """
        Get the primary genres for an artist based on their Last.fm tags.
        
        Args:
            artist: Artist name
            max_genres: Maximum number of genres to return
            
        Returns:
            List of primary genres for the artist
        """
        try:
            artist_info = await self.get_artist_info(artist, include_top_tracks=False)
            
            if not artist_info or not artist_info.tags:
                return []
            
            # Filter for genre-like tags (exclude non-genre descriptors)
            genre_keywords = [
                'rock', 'pop', 'jazz', 'blues', 'country', 'folk', 'metal', 
                'electronic', 'hip hop', 'hip-hop', 'rap', 'r&b', 'rnb', 
                'soul', 'funk', 'reggae', 'punk', 'indie', 'alternative',
                'classical', 'ambient', 'techno', 'house', 'dance'
            ]
            
            genre_tags = []
            for tag in artist_info.tags[:10]:  # Check top 10 tags
                tag_lower = tag.lower()
                if any(genre_keyword in tag_lower for genre_keyword in genre_keywords):
                    genre_tags.append(tag)
                    
                if len(genre_tags) >= max_genres:
                    break
            
            return genre_tags
            
        except Exception as e:
            self.logger.error(f"Failed to get primary genres for {artist}: {e}")
            return []
    
    async def get_similar_artists(self, artist: str, limit: int = 10) -> List[Any]:
        """Get similar artists from Last.fm API."""
        try:
            lastfm_client = await self.client_manager.get_lastfm_client()
            if lastfm_client:
                artists = await lastfm_client.get_similar_artists(artist, limit=limit)
                return artists[:limit]  # get_similar_artists already returns a list of ArtistMetadata objects
            return []
        except Exception as e:
            self.logger.warning(f"Failed to get similar artists for {artist}: {e}")
            return [] 