"""
Last.fm API Client

Provides access to Last.fm's music database with rate limiting and caching.
Focus on indie/underground track metadata and discovery.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import structlog

from .base_client import BaseAPIClient
from .rate_limiter import UnifiedRateLimiter

logger = structlog.get_logger(__name__)


@dataclass
class TrackMetadata:
    """Last.fm track metadata."""
    name: str
    artist: str
    mbid: Optional[str] = None
    url: Optional[str] = None
    tags: List[str] = None
    similar_tracks: List[str] = None
    listeners: Optional[int] = None
    playcount: Optional[int] = None
    summary: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.similar_tracks is None:
            self.similar_tracks = []


@dataclass 
class ArtistMetadata:
    """Last.fm artist metadata."""
    name: str
    mbid: Optional[str] = None
    url: Optional[str] = None
    tags: List[str] = None
    similar_artists: List[str] = None
    listeners: Optional[int] = None
    playcount: Optional[int] = None
    bio: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.similar_artists is None:
            self.similar_artists = []


class LastFmClient(BaseAPIClient):
    """
    Last.fm API client with unified rate limiting and error handling.
    
    Focused on music discovery and metadata extraction for indie/underground tracks.
    Inherits from BaseAPIClient for consistent HTTP handling across all API clients.
    """
    
    BASE_URL = "https://ws.audioscrobbler.com/2.0/"
    
    def __init__(
        self, 
        api_key: str, 
        shared_secret: Optional[str] = None, 
        rate_limiter: Optional[UnifiedRateLimiter] = None
    ):
        """
        Initialize Last.fm client.
        
        Args:
            api_key: Last.fm API key (required)
            shared_secret: Last.fm shared secret (optional, for user auth features)
            rate_limiter: Rate limiter instance (optional, will create default if not provided)
        """
        # Create default rate limiter if not provided
        if rate_limiter is None:
            rate_limiter = UnifiedRateLimiter.for_lastfm()
        
        # Initialize base client
        super().__init__(
            base_url=self.BASE_URL,
            rate_limiter=rate_limiter,
            timeout=10,
            service_name="LastFM"
        )
        
        self.api_key = api_key
        self.shared_secret = shared_secret
        
        self.logger.info(
            "Last.fm client initialized",
            has_shared_secret=bool(shared_secret)
        )
    
    def _extract_api_error(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract Last.fm API error information from response data.
        
        Args:
            data: Parsed response data
            
        Returns:
            Error message if found, None otherwise
        """
        if "error" in data:
            return data.get("message", f"Error {data['error']}")
        return None
    
    async def _make_lastfm_request(
        self, 
        method: str, 
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Make Last.fm API request with automatic parameter injection.
        
        Args:
            method: Last.fm API method
            params: Additional parameters
            retries: Number of retry attempts
            
        Returns:
            API response data
        """
        # Build request parameters with Last.fm specifics
        request_params = {
            "method": method,
            "api_key": self.api_key,
            "format": "json",
            **(params or {})
        }
        
        # Use base client's _make_request with empty endpoint (Last.fm uses query params)
        return await self._make_request(
            endpoint="",  # Last.fm uses base URL with query params
            params=request_params,
            method="GET",
            retries=retries
        )
    
    async def search_tracks(
        self, 
        query: str, 
        limit: int = 20,
        page: int = 1
    ) -> List[TrackMetadata]:
        """
        Search for tracks by query.
        
        Args:
            query: Search query (e.g., "indie rock underground")
            limit: Maximum results per page
            page: Page number
            
        Returns:
            List of track metadata
        """
        try:
            data = await self._make_lastfm_request(
                "track.search",
                {
                    "track": query,
                    "limit": limit,
                    "page": page
                }
            )
            
            tracks = []
            if "results" in data and "trackmatches" in data["results"]:
                track_matches = data["results"]["trackmatches"]
                track_list = track_matches.get("track", [])
                
                # Handle single track result (not in list)
                if isinstance(track_list, dict):
                    track_list = [track_list]
                    
                for track_data in track_list:
                    # Extract listeners and playcount first
                    listeners = int(track_data.get("listeners", 0))
                    playcount = int(track_data.get("playcount", 0))
                    
                    # For discovery, allow tracks with zero popularity but only skip if missing essential data
                    track_name = track_data.get("name", "")
                    artist_name = track_data.get("artist", "")
                    
                    if not track_name or not artist_name:
                        self.logger.debug(
                            "Skipping track with missing name/artist",
                            track=track_name,
                            artist=artist_name,
                            query=query
                        )
                        continue
                    
                    # Log when tracks have zero popularity but still include them
                    if listeners == 0 and playcount == 0:
                        self.logger.debug(
                            "Including track with zero popularity (discovery mode)",
                            track=track_name,
                            artist=artist_name,
                            query=query
                        )
                    
                    track = TrackMetadata(
                        name=track_name,
                        artist=artist_name,
                        mbid=track_data.get("mbid"),
                        url=track_data.get("url"),
                        listeners=listeners,
                        playcount=playcount
                    )
                    tracks.append(track)
                    
            self.logger.info(
                "Track search completed",
                query=query,
                results_count=len(tracks),
                limit=limit
            )
            
            return tracks
            
        except Exception as e:
            self.logger.error(
                "Track search failed",
                query=query,
                error=str(e)
            )
            return []
    
    async def get_track_info(
        self, 
        artist: str, 
        track: str,
        mbid: Optional[str] = None
    ) -> Optional[TrackMetadata]:
        """
        Get detailed track information.
        
        Args:
            artist: Artist name
            track: Track name  
            mbid: MusicBrainz ID (optional)
            
        Returns:
            Detailed track metadata or None if not found
        """
        try:
            params = {"artist": artist, "track": track}
            if mbid:
                params["mbid"] = mbid
                
            data = await self._make_lastfm_request("track.getInfo", params)
            
            if "track" not in data:
                return None
                
            track_data = data["track"]
            
            # Extract tags
            tags = []
            if "toptags" in track_data and "tag" in track_data["toptags"]:
                tag_list = track_data["toptags"]["tag"]
                if isinstance(tag_list, dict):
                    tag_list = [tag_list]
                tags = [tag.get("name", "") for tag in tag_list]
            
            # Extract similar tracks
            similar_tracks = []
            if "similarartists" in track_data and "artist" in track_data["similarartists"]:
                similar_list = track_data["similarartists"]["artist"]
                if isinstance(similar_list, dict):
                    similar_list = [similar_list]
                similar_tracks = [artist.get("name", "") for artist in similar_list]
            
            return TrackMetadata(
                name=track_data.get("name", ""),
                artist=track_data.get("artist", {}).get("name", artist),
                mbid=track_data.get("mbid"),
                url=track_data.get("url"),
                tags=tags,
                similar_tracks=similar_tracks,
                listeners=int(track_data.get("listeners", 0)),
                playcount=int(track_data.get("playcount", 0)),
                summary=track_data.get("wiki", {}).get("summary", "")
            )
            
        except Exception as e:
            self.logger.error(
                "Get track info failed",
                artist=artist,
                track=track,
                error=str(e)
            )
            return None
    
    async def get_similar_tracks(
        self, 
        artist: str, 
        track: str,
        limit: int = 20
    ) -> List[TrackMetadata]:
        """
        Get similar tracks for discovery.
        
        Args:
            artist: Artist name
            track: Track name
            limit: Maximum results
            
        Returns:
            List of similar tracks
        """
        try:
            data = await self._make_lastfm_request(
                "track.getSimilar",
                {
                    "artist": artist,
                    "track": track,
                    "limit": limit
                }
            )
            
            tracks = []
            if "similartracks" in data and "track" in data["similartracks"]:
                track_list = data["similartracks"]["track"]
                if isinstance(track_list, dict):
                    track_list = [track_list]
                    
                for track_data in track_list:
                    # Extract listeners and playcount first
                    listeners = int(track_data.get("listeners", 0))
                    playcount = int(track_data.get("playcount", 0))
                    
                    # For discovery, allow tracks with zero popularity but only skip if missing essential data
                    track_name = track_data.get("name", "")
                    artist_name = track_data.get("artist", {}).get("name", "")
                    
                    if not track_name or not artist_name:
                        self.logger.debug(
                            "Skipping track with missing name/artist",
                            track=track_name,
                            artist=artist_name,
                            seed_artist=artist,
                            seed_track=track
                        )
                        continue
                    
                    # Log when tracks have zero popularity but still include them
                    if listeners == 0 and playcount == 0:
                        self.logger.debug(
                            "Including track with zero popularity (discovery mode)",
                            track=track_name,
                            artist=artist_name,
                            seed_artist=artist,
                            seed_track=track
                        )
                    
                    track = TrackMetadata(
                        name=track_name,
                        artist=artist_name,
                        mbid=track_data.get("mbid"),
                        url=track_data.get("url"),
                        listeners=listeners,
                        playcount=playcount
                    )
                    tracks.append(track)
                    
            self.logger.info(
                "Similar tracks search completed",
                seed_artist=artist,
                seed_track=track,
                results_count=len(tracks)
            )
            
            return tracks
            
        except Exception as e:
            self.logger.error(
                "Similar tracks search failed",
                artist=artist,
                track=track,
                error=str(e)
            )
            return []
    
    async def get_artist_info(self, artist: str) -> Optional[ArtistMetadata]:
        """
        Get detailed artist information.
        
        Args:
            artist: Artist name
            
        Returns:
            Artist metadata or None if not found
        """
        try:
            data = await self._make_lastfm_request(
                "artist.getInfo",
                {"artist": artist}
            )
            
            if "artist" not in data:
                return None
                
            artist_data = data["artist"]
            
            # Extract tags
            tags = []
            if "tags" in artist_data and "tag" in artist_data["tags"]:
                tag_list = artist_data["tags"]["tag"]
                if isinstance(tag_list, dict):
                    tag_list = [tag_list]
                tags = [tag.get("name", "") for tag in tag_list]
            
            # Extract similar artists
            similar_artists = []
            if "similar" in artist_data and "artist" in artist_data["similar"]:
                similar_list = artist_data["similar"]["artist"]
                if isinstance(similar_list, dict):
                    similar_list = [similar_list]
                similar_artists = [a.get("name", "") for a in similar_list]
            
            return ArtistMetadata(
                name=artist_data.get("name", ""),
                mbid=artist_data.get("mbid"),
                url=artist_data.get("url"),
                tags=tags,
                similar_artists=similar_artists,
                listeners=int(artist_data.get("stats", {}).get("listeners", 0)),
                playcount=int(artist_data.get("stats", {}).get("playcount", 0)),
                bio=artist_data.get("bio", {}).get("summary", "")
            )
            
        except Exception as e:
            self.logger.error(
                "Get artist info failed",
                artist=artist,
                error=str(e)
            )
            return None
    
    async def get_similar_artists(
        self, 
        artist: str, 
        limit: int = 10
    ) -> List[ArtistMetadata]:
        """
        Get similar artists for a given artist.
        
        Args:
            artist: Artist name
            limit: Maximum number of similar artists to return
            
        Returns:
            List of similar artist metadata
        """
        try:
            data = await self._make_lastfm_request(
                "artist.getSimilar",
                {
                    "artist": artist,
                    "limit": limit
                }
            )
            
            artists = []
            if "similarartists" in data and "artist" in data["similarartists"]:
                similar_artists = data["similarartists"]["artist"]
                
                # Handle single artist result (not in list)
                if isinstance(similar_artists, dict):
                    similar_artists = [similar_artists]
                    
                for artist_data in similar_artists:
                    artist_meta = ArtistMetadata(
                        name=artist_data.get("name", ""),
                        mbid=artist_data.get("mbid"),
                        url=artist_data.get("url"),
                    )
                    artists.append(artist_meta)
                    
            self.logger.info(
                "Similar artists search completed",
                artist=artist,
                results_count=len(artists),
                limit=limit
            )
            return artists
            
        except Exception as e:
            self.logger.error(
                "Failed to get similar artists",
                artist=artist,
                error=str(e)
            )
            return []
    
    async def search_by_tags(
        self, 
        tags: List[str], 
        limit: int = 20
    ) -> List[TrackMetadata]:
        """
        Search tracks by tags for genre/mood discovery.
        
        Args:
            tags: List of tags (e.g., ["indie", "experimental"])
            limit: Maximum results
            
        Returns:
            List of tracks matching tags
        """
        tracks = []
        
        for tag in tags:
            try:
                data = await self._make_lastfm_request(
                    "tag.getTopTracks",
                    {
                        "tag": tag,
                        "limit": min(limit // len(tags), 200)  # Allow much higher per-tag limit
                    }
                )
                
                if "tracks" in data and "track" in data["tracks"]:
                    track_list = data["tracks"]["track"]
                    if isinstance(track_list, dict):
                        track_list = [track_list]
                        
                    for track_data in track_list:
                        # Extract listeners and playcount first
                        listeners = int(track_data.get("listeners", 0))
                        playcount = int(track_data.get("playcount", 0))
                        
                        # For discovery, allow tracks with zero popularity but flag with low confidence
                        # Only skip tracks with no name/artist (truly invalid data)
                        track_name = track_data.get("name", "")
                        artist_name = track_data.get("artist", {}).get("name", "")
                        
                        if not track_name or not artist_name:
                            self.logger.debug(
                                "Skipping track with missing name/artist",
                                track=track_name,
                                artist=artist_name,
                                tag=tag
                            )
                            continue
                        
                        # Log when tracks have zero popularity but still include them
                        if listeners == 0 and playcount == 0:
                            self.logger.debug(
                                "Including track with zero popularity (discovery mode)",
                                track=track_name,
                                artist=artist_name,
                                tag=tag
                            )
                        
                        track = TrackMetadata(
                            name=track_name,
                            artist=artist_name,
                            mbid=track_data.get("mbid"),
                            url=track_data.get("url"),
                            tags=[tag],  # Add the search tag
                            listeners=listeners,
                            playcount=playcount
                        )
                        tracks.append(track)
                        
            except Exception as e:
                self.logger.warning(
                    "Tag search failed for tag",
                    tag=tag,
                    error=str(e)
                )
                continue
                
        self.logger.info(
            "Tag search completed",
            tags=tags,
            results_count=len(tracks)
        )
        
        return tracks[:limit]  # Limit final results
    
    async def get_artist_top_tracks(
        self, 
        artist: str, 
        limit: int = 20,
        page: int = 1
    ) -> List[TrackMetadata]:
        """
        Get top tracks for an artist.
        
        Args:
            artist: Artist name
            limit: Maximum results
            page: Page number (1-indexed)
            
        Returns:
            List of top tracks for the artist
        """
        try:
            data = await self._make_lastfm_request(
                "artist.getTopTracks",
                {
                    "artist": artist,
                    "limit": limit,
                    "page": page
                }
            )
            
            tracks = []
            if "toptracks" in data and "track" in data["toptracks"]:
                track_list = data["toptracks"]["track"]
                if isinstance(track_list, dict):
                    track_list = [track_list]
                    
                for track_data in track_list:
                    # Extract listeners and playcount first
                    listeners = int(track_data.get("listeners", 0))
                    playcount = int(track_data.get("playcount", 0))
                    
                    # For discovery, allow tracks with zero popularity but only skip if missing essential data
                    track_name = track_data.get("name", "")
                    artist_name = track_data.get("artist", {}).get("name", artist)
                    
                    if not track_name or not artist_name:
                        self.logger.debug(
                            "Skipping track with missing name/artist",
                            track=track_name,
                            artist=artist_name,
                            target_artist=artist
                        )
                        continue
                    
                    # Log when tracks have zero popularity but still include them
                    if listeners == 0 and playcount == 0:
                        self.logger.debug(
                            "Including track with zero popularity (discovery mode)",
                            track=track_name,
                            artist=artist_name,
                            target_artist=artist
                        )
                    
                    track = TrackMetadata(
                        name=track_name,
                        artist=artist_name,
                        mbid=track_data.get("mbid"),
                        url=track_data.get("url"),
                        listeners=listeners,
                        playcount=playcount
                    )
                    tracks.append(track)
                    
            self.logger.info(
                "Artist top tracks search completed",
                artist=artist,
                results_count=len(tracks)
            )
            
            return tracks
            
        except Exception as e:
            self.logger.error(
                "Artist top tracks search failed",
                artist=artist,
                error=str(e)
            )
            return []
    
    async def search_artists(
        self, 
        query: str, 
        limit: int = 20,
        page: int = 1
    ) -> List[ArtistMetadata]:
        """
        Search for artists by query.
        
        Args:
            query: Search query (artist name)
            limit: Maximum results per page
            page: Page number
            
        Returns:
            List of artist metadata
        """
        try:
            data = await self._make_lastfm_request(
                "artist.search",
                {
                    "artist": query,
                    "limit": limit,
                    "page": page
                }
            )
            
            artists = []
            if "results" in data and "artistmatches" in data["results"]:
                artist_matches = data["results"]["artistmatches"]
                artist_list = artist_matches.get("artist", [])
                
                # Handle single artist result (not in list)
                if isinstance(artist_list, dict):
                    artist_list = [artist_list]
                    
                for artist_data in artist_list:
                    artist = ArtistMetadata(
                        name=artist_data.get("name", ""),
                        mbid=artist_data.get("mbid"),
                        url=artist_data.get("url"),
                        listeners=int(artist_data.get("listeners", 0))
                    )
                    artists.append(artist)
                    
            self.logger.info(
                "Artist search completed",
                query=query,
                results_count=len(artists),
                limit=limit
            )
            
            return artists
            
        except Exception as e:
            self.logger.error(
                "Artist search failed",
                query=query,
                error=str(e)
            )
            return []
    
    async def search_underground_tracks_by_tags(
        self, 
        tags: List[str], 
        limit: int = 20,
        max_listeners: int = 50000
    ) -> List[TrackMetadata]:
        """
        Search for underground tracks by tags using track.search instead of tag.getTopTracks.
        
        This method specifically targets underground tracks by:
        1. Using track.search instead of tag.getTopTracks (which returns popular tracks)
        2. Searching with underground-specific query terms
        3. Filtering results by listener thresholds
        
        Args:
            tags: List of tags (e.g., ["minimal techno", "experimental electronic"])
            limit: Maximum results
            max_listeners: Maximum listener count for "underground" classification
            
        Returns:
            List of underground tracks matching tags
        """
        underground_tracks = []
        
        # Create underground-focused search queries
        underground_queries = []
        for tag in tags:
            # Primary tag search
            underground_queries.append(tag)
            # Add underground modifiers
            underground_queries.extend([
                f"{tag} underground",
                f"{tag} minimal",
                f"{tag} experimental",
                f"underground {tag}",
                f"minimal {tag}"
            ])
        
        # Limit queries to avoid rate limiting
        search_queries = underground_queries[:min(len(underground_queries), 8)]
        
        for query in search_queries:
            try:
                data = await self._make_lastfm_request(
                    "track.search",  # Using track.search instead of tag.getTopTracks
                    {
                        "track": query,
                        "limit": max(5, limit // len(search_queries))  # Distribute across queries
                    }
                )
                
                if "results" in data and "trackmatches" in data["results"]:
                    track_matches = data["results"]["trackmatches"]
                    track_list = track_matches.get("track", [])
                    
                    if isinstance(track_list, dict):
                        track_list = [track_list]
                    
                    for track_data in track_list:
                        listeners = int(track_data.get("listeners", 0))
                        playcount = int(track_data.get("playcount", 0))
                        
                        track_name = track_data.get("name", "")
                        artist_name = track_data.get("artist", "")
                        
                        if not track_name or not artist_name:
                            continue
                        
                        # Apply underground filtering: strict listener limits
                        is_underground = (
                            listeners <= max_listeners or  # Low listener count
                            listeners == 0 or             # Unknown (potentially very underground)
                            playcount < max_listeners * 20  # Low playcount relative to listeners
                        )
                        
                        if is_underground:
                            self.logger.debug(
                                "Found underground track",
                                track=track_name,
                                artist=artist_name,
                                listeners=listeners,
                                query=query
                            )
                            
                            track = TrackMetadata(
                                name=track_name,
                                artist=artist_name,
                                mbid=track_data.get("mbid"),
                                url=track_data.get("url"),
                                tags=[query],  # Add the search query as context
                                listeners=listeners,
                                playcount=playcount
                            )
                            underground_tracks.append(track)
                        else:
                            self.logger.debug(
                                "Filtering out popular track",
                                track=track_name,
                                artist=artist_name,
                                listeners=listeners,
                                max_allowed=max_listeners
                            )
                        
            except Exception as e:
                self.logger.warning(
                    "Underground track search failed for query",
                    query=query,
                    error=str(e)
                )
                continue
        
        # Remove duplicates and sort by "undergroundness" (lower listener count = more underground)
        seen_tracks = set()
        unique_tracks = []
        
        for track in underground_tracks:
            track_key = f"{track.artist.lower()}::{track.name.lower()}"
            if track_key not in seen_tracks:
                seen_tracks.add(track_key)
                unique_tracks.append(track)
        
        # Sort by listener count (ascending - fewer listeners = more underground)
        unique_tracks.sort(key=lambda t: t.listeners or 0)
        
        self.logger.info(
            "Underground track search completed",
            original_tags=tags,
            search_queries_used=len(search_queries),
            total_found=len(underground_tracks),
            unique_results=len(unique_tracks),
            max_listeners_threshold=max_listeners
        )
        
        return unique_tracks[:limit]


# Legacy compatibility - maintain the old RateLimiter class for backward compatibility
class RateLimiter:
    """Legacy rate limiter for backward compatibility."""
    
    def __init__(self, calls_per_second: float = 3.0):
        self.calls_per_second = calls_per_second
        self._unified_limiter = UnifiedRateLimiter.for_lastfm(calls_per_second)
        
    async def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        await self._unified_limiter.wait_if_needed() 