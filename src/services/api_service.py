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
            lastfm_client = await self.get_lastfm_client()
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
                spotify_client = await self.get_spotify_client()
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
            lastfm_client = await self.get_lastfm_client()
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
                spotify_client = await self.get_spotify_client()
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
            lastfm_client = await self.get_lastfm_client()
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
                spotify_client = await self.get_spotify_client()
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
            lastfm_client = await self.get_lastfm_client()
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
            lastfm_client = await self.get_lastfm_client()
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
    
    # Additional methods expected by unified candidate generator
    
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
            lastfm_client = await self.get_lastfm_client()
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
    
    async def get_artist_top_tracks(
        self,
        artist: str,
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """
        Get top tracks for an artist.
        
        Args:
            artist: Artist name
            limit: Maximum results
            
        Returns:
            List of top tracks with unified metadata
        """
        try:
            lastfm_client = await self.get_lastfm_client()
            if lastfm_client:
                lastfm_tracks = await lastfm_client.get_artist_top_tracks(artist, limit)
                
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
    
    async def check_artist_genre_match(
        self,
        artist: str,
        target_genre: str,
        include_related_genres: bool = True
    ) -> Dict[str, Any]:
        """
        Check if an artist matches a specific genre by looking up their metadata.
        
        Args:
            artist: Artist name
            target_genre: Genre to check against (e.g., "r&b", "jazz", "rock")
            include_related_genres: Whether to include related/synonym genres
            
        Returns:
            Dict with match info: {
                'matches': bool,
                'confidence': float,
                'matched_tags': List[str],
                'artist_tags': List[str],
                'match_type': str  # 'direct', 'related', 'none'
            }
        """
        try:
            # Get artist information from Last.fm
            artist_info = await self.get_artist_info(artist, include_top_tracks=False)
            
            if not artist_info or not artist_info.tags:
                self.logger.debug(f"No tags found for artist: {artist}")
                return {
                    'matches': False,
                    'confidence': 0.0,
                    'matched_tags': [],
                    'artist_tags': [],
                    'match_type': 'none'
                }
            
            # Normalize target genre
            target_genre_lower = target_genre.lower().strip()
            artist_tags_lower = [tag.lower() for tag in artist_info.tags]
            
            # Create genre mappings for related genres
            genre_mappings = self._get_genre_mappings()
            
            # Check for direct match first
            if target_genre_lower in artist_tags_lower:
                return {
                    'matches': True,
                    'confidence': 1.0,
                    'matched_tags': [target_genre_lower],
                    'artist_tags': artist_info.tags,
                    'match_type': 'direct'
                }
            
            # Check for related genres if enabled
            if include_related_genres and target_genre_lower in genre_mappings:
                related_genres = genre_mappings[target_genre_lower]
                matched_related = []
                
                for related_genre in related_genres:
                    if related_genre in artist_tags_lower:
                        matched_related.append(related_genre)
                
                if matched_related:
                    # Calculate confidence based on how many related genres match
                    confidence = min(0.9, len(matched_related) * 0.3)
                    return {
                        'matches': True,
                        'confidence': confidence,
                        'matched_tags': matched_related,
                        'artist_tags': artist_info.tags,
                        'match_type': 'related'
                    }
            
            # No match found
            return {
                'matches': False,
                'confidence': 0.0,
                'matched_tags': [],
                'artist_tags': artist_info.tags,
                'match_type': 'none'
            }
            
        except Exception as e:
            self.logger.error(f"Artist genre check failed for {artist}: {e}")
            return {
                'matches': False,
                'confidence': 0.0,
                'matched_tags': [],
                'artist_tags': [],
                'match_type': 'error'
            }
    
    async def check_track_genre_match(
        self,
        artist: str,
        track: str,
        target_genre: str,
        include_related_genres: bool = True
    ) -> Dict[str, Any]:
        """
        Check if a track matches a specific genre by looking up track and artist metadata.
        
        Args:
            artist: Artist name
            track: Track name
            target_genre: Genre to check against
            include_related_genres: Whether to include related/synonym genres
            
        Returns:
            Dict with match info: {
                'matches': bool,
                'confidence': float,
                'matched_tags': List[str],
                'track_tags': List[str],
                'artist_match': Dict[str, Any],
                'match_type': str
            }
        """
        try:
            # Check both track-specific tags and artist genre
            track_info = await self.get_unified_track_info(artist, track, include_spotify=False)
            artist_match = await self.check_artist_genre_match(artist, target_genre, include_related_genres)
            
            track_tags = []
            track_match_result = {
                'matches': False,
                'confidence': 0.0,
                'matched_tags': [],
                'track_tags': track_tags,
                'artist_match': artist_match,
                'match_type': 'none'
            }
            
            # Get track tags if available
            if track_info and track_info.tags:
                track_tags = track_info.tags
                track_match_result['track_tags'] = track_tags
                
                # Check track tags for genre match
                target_genre_lower = target_genre.lower().strip()
                track_tags_lower = [tag.lower() for tag in track_tags]
                
                # Direct match in track tags
                if target_genre_lower in track_tags_lower:
                    track_match_result.update({
                        'matches': True,
                        'confidence': 1.0,
                        'matched_tags': [target_genre_lower],
                        'match_type': 'direct_track'
                    })
                    return track_match_result
                
                # Related genre match in track tags
                if include_related_genres:
                    genre_mappings = self._get_genre_mappings()
                    if target_genre_lower in genre_mappings:
                        related_genres = genre_mappings[target_genre_lower]
                        matched_related = []
                        
                        for related_genre in related_genres:
                            if related_genre in track_tags_lower:
                                matched_related.append(related_genre)
                        
                        if matched_related:
                            confidence = min(0.9, len(matched_related) * 0.3)
                            track_match_result.update({
                                'matches': True,
                                'confidence': confidence,
                                'matched_tags': matched_related,
                                'match_type': 'related_track'
                            })
                            return track_match_result
            
            # If track doesn't have genre info, fall back to artist genre
            if artist_match['matches']:
                track_match_result.update({
                    'matches': True,
                    'confidence': artist_match['confidence'] * 0.8,  # Slightly lower confidence
                    'matched_tags': artist_match['matched_tags'],
                    'match_type': f"artist_{artist_match['match_type']}"
                })
                return track_match_result
            
            # No match found
            return track_match_result
            
        except Exception as e:
            self.logger.error(f"Track genre check failed for {artist} - {track}: {e}")
            return {
                'matches': False,
                'confidence': 0.0,
                'matched_tags': [],
                'track_tags': [],
                'artist_match': {'matches': False, 'confidence': 0.0},
                'match_type': 'error'
            }
    
    def _get_genre_mappings(self) -> Dict[str, List[str]]:
        """
        Get genre mappings for related genre detection.
        
        Returns:
            Dict mapping primary genres to related genres
        """
        return {
            'r&b': [
                'rnb', 'rhythm and blues', 'soul', 'neo-soul', 'contemporary r&b',
                'motown', 'funk', 'urban', 'smooth r&b', 'r&b soul'
            ],
            'jazz': [
                'bebop', 'swing', 'cool jazz', 'fusion', 'smooth jazz',
                'jazz fusion', 'contemporary jazz', 'acid jazz'
            ],
            'rock': [
                'classic rock', 'alternative rock', 'indie rock', 'hard rock',
                'soft rock', 'progressive rock', 'art rock', 'garage rock'
            ],
            'electronic': [
                'edm', 'techno', 'house', 'ambient', 'electronica',
                'synth', 'synthwave', 'electronic music', 'dance'
            ],
            'hip hop': [
                'hip-hop', 'rap', 'trap', 'underground hip hop', 'conscious hip hop',
                'old school hip hop', 'east coast hip hop', 'west coast hip hop'
            ],
            'pop': [
                'pop music', 'mainstream pop', 'indie pop', 'electro pop',
                'synth pop', 'dance pop', 'alternative pop'
            ],
            'indie': [
                'indie rock', 'indie pop', 'indie folk', 'alternative',
                'independent', 'lo-fi', 'bedroom pop'
            ],
            'folk': [
                'folk music', 'indie folk', 'contemporary folk', 'acoustic',
                'singer-songwriter', 'americana', 'country folk'
            ],
            'metal': [
                'heavy metal', 'death metal', 'black metal', 'thrash metal',
                'progressive metal', 'power metal', 'doom metal'
            ],
            'country': [
                'country music', 'modern country', 'classic country',
                'country rock', 'americana', 'bluegrass'
            ]
        }
    
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