"""
Last.fm API Client

Provides access to Last.fm's music database with rate limiting and caching.
Focus on indie/underground track metadata and discovery.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from functools import wraps

import aiohttp
import structlog
from pydantic import BaseModel, Field

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


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float = 3.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        
    async def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        now = time.time()
        time_since_last = now - self.last_call
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
            
        self.last_call = time.time()


class LastFmClient:
    """
    Last.fm API client with rate limiting and error handling.
    
    Focused on music discovery and metadata extraction for indie/underground tracks.
    """
    
    BASE_URL = "https://ws.audioscrobbler.com/2.0/"
    
    def __init__(self, api_key: str, shared_secret: Optional[str] = None, rate_limit: float = 3.0):
        """
        Initialize Last.fm client.
        
        Args:
            api_key: Last.fm API key (required)
            shared_secret: Last.fm shared secret (optional, for user auth features)
            rate_limit: Requests per second (default: 3.0)
        """
        self.api_key = api_key
        self.shared_secret = shared_secret
        self.rate_limiter = RateLimiter(rate_limit)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def _make_request(
        self, 
        method: str, 
        params: Dict[str, Any],
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Make rate-limited request to Last.fm API.
        
        Args:
            method: Last.fm API method
            params: Additional parameters
            retries: Number of retry attempts
            
        Returns:
            API response data
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
            
        await self.rate_limiter.wait_if_needed()
        
        # Build request parameters
        request_params = {
            "method": method,
            "api_key": self.api_key,
            "format": "json",
            **params
        }
        
        for attempt in range(retries + 1):
            try:
                async with self.session.get(
                    self.BASE_URL, 
                    params=request_params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "error" in data:
                            logger.error(
                                "Last.fm API error",
                                error=data["error"],
                                message=data.get("message", "Unknown error"),
                                method=method
                            )
                            raise Exception(f"Last.fm API error: {data.get('message', 'Unknown')}")
                            
                        logger.debug(
                            "Last.fm API request successful",
                            method=method,
                            params=params
                        )
                        return data
                        
                    elif response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(
                            "Rate limited by Last.fm",
                            attempt=attempt,
                            wait_time=wait_time
                        )
                        await asyncio.sleep(wait_time)
                        continue
                        
                    else:
                        logger.warning(
                            "Last.fm API request failed",
                            status=response.status,
                            method=method
                        )
                        response.raise_for_status()
                        
            except asyncio.TimeoutError:
                logger.warning(
                    "Last.fm API timeout",
                    attempt=attempt,
                    method=method
                )
                if attempt == retries:
                    raise
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(
                    "Last.fm API request failed",
                    error=str(e),
                    method=method,
                    attempt=attempt
                )
                if attempt == retries:
                    raise
                await asyncio.sleep(2 ** attempt)
                
        raise Exception(f"Failed to complete Last.fm request after {retries + 1} attempts")
    
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
            data = await self._make_request(
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
                    track = TrackMetadata(
                        name=track_data.get("name", ""),
                        artist=track_data.get("artist", ""),
                        mbid=track_data.get("mbid"),
                        url=track_data.get("url"),
                        listeners=int(track_data.get("listeners", 0)),
                    )
                    tracks.append(track)
                    
            logger.info(
                "Track search completed",
                query=query,
                results_count=len(tracks),
                limit=limit
            )
            
            return tracks
            
        except Exception as e:
            logger.error(
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
                
            data = await self._make_request("track.getInfo", params)
            
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
            logger.error(
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
            data = await self._make_request(
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
                    track = TrackMetadata(
                        name=track_data.get("name", ""),
                        artist=track_data.get("artist", {}).get("name", ""),
                        mbid=track_data.get("mbid"),
                        url=track_data.get("url"),
                    )
                    tracks.append(track)
                    
            logger.info(
                "Similar tracks search completed",
                seed_artist=artist,
                seed_track=track,
                results_count=len(tracks)
            )
            
            return tracks
            
        except Exception as e:
            logger.error(
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
            data = await self._make_request(
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
            logger.error(
                "Get artist info failed",
                artist=artist,
                error=str(e)
            )
            return None
    
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
                data = await self._make_request(
                    "tag.getTopTracks",
                    {
                        "tag": tag,
                        "limit": min(limit // len(tags), 10)  # Distribute across tags
                    }
                )
                
                if "tracks" in data and "track" in data["tracks"]:
                    track_list = data["tracks"]["track"]
                    if isinstance(track_list, dict):
                        track_list = [track_list]
                        
                    for track_data in track_list:
                        track = TrackMetadata(
                            name=track_data.get("name", ""),
                            artist=track_data.get("artist", {}).get("name", ""),
                            mbid=track_data.get("mbid"),
                            url=track_data.get("url"),
                            tags=[tag],  # Add the search tag
                        )
                        tracks.append(track)
                        
            except Exception as e:
                logger.warning(
                    "Tag search failed for tag",
                    tag=tag,
                    error=str(e)
                )
                continue
                
        logger.info(
            "Tag search completed",
            tags=tags,
            results_count=len(tracks)
        )
        
        return tracks[:limit]  # Limit final results
    
    async def get_artist_top_tracks(
        self, 
        artist: str, 
        limit: int = 20
    ) -> List[TrackMetadata]:
        """
        Get top tracks for an artist.
        
        Args:
            artist: Artist name
            limit: Maximum results
            
        Returns:
            List of top tracks for the artist
        """
        try:
            data = await self._make_request(
                "artist.getTopTracks",
                {
                    "artist": artist,
                    "limit": limit
                }
            )
            
            tracks = []
            if "toptracks" in data and "track" in data["toptracks"]:
                track_list = data["toptracks"]["track"]
                if isinstance(track_list, dict):
                    track_list = [track_list]
                    
                for track_data in track_list:
                    track = TrackMetadata(
                        name=track_data.get("name", ""),
                        artist=track_data.get("artist", {}).get("name", artist),
                        mbid=track_data.get("mbid"),
                        url=track_data.get("url"),
                        listeners=int(track_data.get("listeners", 0)),
                        playcount=int(track_data.get("playcount", 0))
                    )
                    tracks.append(track)
                    
            logger.info(
                "Artist top tracks search completed",
                artist=artist,
                results_count=len(tracks)
            )
            
            return tracks
            
        except Exception as e:
            logger.error(
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
            data = await self._make_request(
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
                    
            logger.info(
                "Artist search completed",
                query=query,
                results_count=len(artists),
                limit=limit
            )
            
            return artists
            
        except Exception as e:
            logger.error(
                "Artist search failed",
                query=query,
                error=str(e)
            )
            return [] 