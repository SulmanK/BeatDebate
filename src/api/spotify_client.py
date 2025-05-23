"""
Spotify Web API Client

Provides access to Spotify's database for audio features and preview URLs.
Used as secondary data source for BeatDebate.
"""

import asyncio
import base64
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SpotifyTrack:
    """Spotify track data."""
    id: str
    name: str
    artist: str
    album: str
    preview_url: Optional[str] = None
    external_urls: Optional[Dict[str, str]] = None
    duration_ms: Optional[int] = None
    popularity: Optional[int] = None
    
    
@dataclass
class AudioFeatures:
    """Spotify audio features."""
    track_id: str
    danceability: float
    energy: float
    valence: float
    acousticness: float
    instrumentalness: float
    speechiness: float
    liveness: float
    loudness: float
    tempo: float
    time_signature: int
    key: int
    mode: int


class SpotifyRateLimiter:
    """Rate limiter for Spotify API calls."""
    
    def __init__(self, calls_per_hour: int = 50):
        self.calls_per_hour = calls_per_hour
        self.min_interval = 3600.0 / calls_per_hour  # seconds between calls
        self.last_call = 0.0
        
    async def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        now = time.time()
        time_since_last = now - self.last_call
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
            
        self.last_call = time.time()


class SpotifyClient:
    """
    Spotify Web API client with authentication and rate limiting.
    
    Focuses on audio features and preview URLs for BeatDebate.
    """
    
    BASE_URL = "https://api.spotify.com/v1"
    AUTH_URL = "https://accounts.spotify.com/api/token"
    
    def __init__(
        self, 
        client_id: str, 
        client_secret: str,
        rate_limit: int = 50
    ):
        """
        Initialize Spotify client.
        
        Args:
            client_id: Spotify client ID
            client_secret: Spotify client secret
            rate_limit: Requests per hour (default: 50)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.rate_limiter = SpotifyRateLimiter(rate_limit)
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.token_expires_at: float = 0.0
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        await self._authenticate()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def _authenticate(self) -> None:
        """Authenticate with Spotify API using client credentials flow."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
            
        # Prepare authentication
        auth_str = f"{self.client_id}:{self.client_secret}"
        auth_b64 = base64.b64encode(auth_str.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {"grant_type": "client_credentials"}
        
        try:
            async with self.session.post(
                self.AUTH_URL,
                headers=headers,
                data=data,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data["access_token"]
                    expires_in = token_data.get("expires_in", 3600)
                    self.token_expires_at = time.time() + expires_in - 60  # 1min buffer
                    
                    logger.info(
                        "Spotify authentication successful",
                        expires_in=expires_in
                    )
                else:
                    error_data = await response.json()
                    logger.error(
                        "Spotify authentication failed",
                        status=response.status,
                        error=error_data
                    )
                    raise Exception(f"Spotify auth failed: {error_data}")
                    
        except Exception as e:
            logger.error("Spotify authentication error", error=str(e))
            raise
            
    async def _ensure_valid_token(self) -> None:
        """Ensure we have a valid access token."""
        if not self.access_token or time.time() >= self.token_expires_at:
            await self._authenticate()
            
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Make authenticated request to Spotify API.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            retries: Number of retry attempts
            
        Returns:
            API response data
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
            
        await self._ensure_valid_token()
        await self.rate_limiter.wait_if_needed()
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        for attempt in range(retries + 1):
            try:
                async with self.session.get(
                    url,
                    headers=headers,
                    params=params or {},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(
                            "Spotify API request successful",
                            endpoint=endpoint,
                            params=params
                        )
                        return data
                        
                    elif response.status == 401:  # Unauthorized
                        logger.warning("Spotify token expired, re-authenticating")
                        await self._authenticate()
                        # Don't count this as a retry attempt
                        continue
                        
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 1))
                        logger.warning(
                            "Rate limited by Spotify",
                            retry_after=retry_after,
                            attempt=attempt
                        )
                        await asyncio.sleep(retry_after)
                        continue
                        
                    elif response.status == 404:  # Not found
                        logger.debug(
                            "Spotify resource not found",
                            endpoint=endpoint,
                            params=params
                        )
                        return {}
                        
                    else:
                        logger.warning(
                            "Spotify API request failed",
                            status=response.status,
                            endpoint=endpoint
                        )
                        response.raise_for_status()
                        
            except asyncio.TimeoutError:
                logger.warning(
                    "Spotify API timeout",
                    attempt=attempt,
                    endpoint=endpoint
                )
                if attempt == retries:
                    raise
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(
                    "Spotify API request failed",
                    error=str(e),
                    attempt=attempt,
                    endpoint=endpoint
                )
                if attempt == retries:
                    raise
                await asyncio.sleep(2 ** attempt)
                
        raise Exception(f"Failed to complete Spotify request after {retries + 1} attempts")
    
    async def search_track(
        self,
        artist: str,
        track: str,
        limit: int = 1
    ) -> Optional[SpotifyTrack]:
        """
        Search for a track on Spotify.
        
        Args:
            artist: Artist name
            track: Track name
            limit: Number of results (default: 1)
            
        Returns:
            First matching Spotify track or None
        """
        try:
            query = f"artist:{artist} track:{track}"
            data = await self._make_request(
                "search",
                {
                    "q": query,
                    "type": "track",
                    "limit": limit
                }
            )
            
            if "tracks" in data and data["tracks"]["items"]:
                track_data = data["tracks"]["items"][0]
                
                return SpotifyTrack(
                    id=track_data["id"],
                    name=track_data["name"],
                    artist=track_data["artists"][0]["name"],
                    album=track_data["album"]["name"],
                    preview_url=track_data.get("preview_url"),
                    external_urls=track_data.get("external_urls"),
                    duration_ms=track_data.get("duration_ms"),
                    popularity=track_data.get("popularity")
                )
                
            return None
            
        except Exception as e:
            logger.error(
                "Spotify track search failed",
                artist=artist,
                track=track,
                error=str(e)
            )
            return None
    
    async def get_track(self, track_id: str) -> Optional[SpotifyTrack]:
        """
        Get track details by Spotify ID.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Track details or None
        """
        try:
            data = await self._make_request(f"tracks/{track_id}")
            
            if not data:
                return None
                
            return SpotifyTrack(
                id=data["id"],
                name=data["name"],
                artist=data["artists"][0]["name"],
                album=data["album"]["name"],
                preview_url=data.get("preview_url"),
                external_urls=data.get("external_urls"),
                duration_ms=data.get("duration_ms"),
                popularity=data.get("popularity")
            )
            
        except Exception as e:
            logger.error(
                "Get Spotify track failed",
                track_id=track_id,
                error=str(e)
            )
            return None
    
    async def get_audio_features(self, track_id: str) -> Optional[AudioFeatures]:
        """
        Get audio features for a track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Audio features or None
        """
        try:
            data = await self._make_request(f"audio-features/{track_id}")
            
            if not data or "id" not in data:
                return None
                
            return AudioFeatures(
                track_id=data["id"],
                danceability=data.get("danceability", 0.0),
                energy=data.get("energy", 0.0),
                valence=data.get("valence", 0.0),
                acousticness=data.get("acousticness", 0.0),
                instrumentalness=data.get("instrumentalness", 0.0),
                speechiness=data.get("speechiness", 0.0),
                liveness=data.get("liveness", 0.0),
                loudness=data.get("loudness", 0.0),
                tempo=data.get("tempo", 0.0),
                time_signature=data.get("time_signature", 4),
                key=data.get("key", 0),
                mode=data.get("mode", 0)
            )
            
        except Exception as e:
            logger.error(
                "Get audio features failed",
                track_id=track_id,
                error=str(e)
            )
            return None
    
    async def get_multiple_audio_features(
        self, 
        track_ids: List[str]
    ) -> Dict[str, AudioFeatures]:
        """
        Get audio features for multiple tracks (batch request).
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            Dictionary mapping track IDs to audio features
        """
        features = {}
        
        # Process in batches of 100 (Spotify limit)
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i + 100]
            
            try:
                data = await self._make_request(
                    "audio-features",
                    {"ids": ",".join(batch)}
                )
                
                if "audio_features" in data:
                    for feature_data in data["audio_features"]:
                        if feature_data and "id" in feature_data:
                            features[feature_data["id"]] = AudioFeatures(
                                track_id=feature_data["id"],
                                danceability=feature_data.get("danceability", 0.0),
                                energy=feature_data.get("energy", 0.0),
                                valence=feature_data.get("valence", 0.0),
                                acousticness=feature_data.get("acousticness", 0.0),
                                instrumentalness=feature_data.get("instrumentalness", 0.0),
                                speechiness=feature_data.get("speechiness", 0.0),
                                liveness=feature_data.get("liveness", 0.0),
                                loudness=feature_data.get("loudness", 0.0),
                                tempo=feature_data.get("tempo", 0.0),
                                time_signature=feature_data.get("time_signature", 4),
                                key=feature_data.get("key", 0),
                                mode=feature_data.get("mode", 0)
                            )
                            
            except Exception as e:
                logger.error(
                    "Batch audio features request failed",
                    batch_size=len(batch),
                    error=str(e)
                )
                continue
                
        logger.info(
            "Audio features batch completed",
            requested=len(track_ids),
            retrieved=len(features)
        )
        
        return features
    
    async def search_tracks(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[SpotifyTrack]:
        """
        Search for tracks with a general query.
        
        Args:
            query: Search query
            limit: Number of results
            offset: Result offset
            
        Returns:
            List of matching tracks
        """
        try:
            data = await self._make_request(
                "search",
                {
                    "q": query,
                    "type": "track",
                    "limit": limit,
                    "offset": offset
                }
            )
            
            tracks = []
            if "tracks" in data and "items" in data["tracks"]:
                for track_data in data["tracks"]["items"]:
                    track = SpotifyTrack(
                        id=track_data["id"],
                        name=track_data["name"],
                        artist=track_data["artists"][0]["name"],
                        album=track_data["album"]["name"],
                        preview_url=track_data.get("preview_url"),
                        external_urls=track_data.get("external_urls"),
                        duration_ms=track_data.get("duration_ms"),
                        popularity=track_data.get("popularity")
                    )
                    tracks.append(track)
                    
            logger.info(
                "Spotify search completed",
                query=query,
                results_count=len(tracks)
            )
            
            return tracks
            
        except Exception as e:
            logger.error(
                "Spotify search failed",
                query=query,
                error=str(e)
            )
            return [] 