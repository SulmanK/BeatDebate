"""
Spotify Web API Client

Provides access to Spotify's database for audio features and preview URLs.
Used as secondary data source for BeatDebate.
"""

import base64
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import structlog

from .base_client import BaseAPIClient
from .rate_limiter import UnifiedRateLimiter

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


class SpotifyClient(BaseAPIClient):
    """
    Spotify Web API client with unified authentication and rate limiting.
    
    Focuses on audio features and preview URLs for BeatDebate.
    Inherits from BaseAPIClient for consistent HTTP handling across all API clients.
    """
    
    BASE_URL = "https://api.spotify.com/v1"
    AUTH_URL = "https://accounts.spotify.com/api/token"
    
    def __init__(
        self, 
        client_id: str, 
        client_secret: str,
        rate_limiter: Optional[UnifiedRateLimiter] = None
    ):
        """
        Initialize Spotify client.
        
        Args:
            client_id: Spotify client ID
            client_secret: Spotify client secret
            rate_limiter: Rate limiter instance (optional, will create default if not provided)
        """
        # Create default rate limiter if not provided
        if rate_limiter is None:
            rate_limiter = UnifiedRateLimiter.for_spotify()
        
        # Initialize base client
        super().__init__(
            base_url=self.BASE_URL,
            rate_limiter=rate_limiter,
            timeout=10,
            service_name="Spotify"
        )
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: Optional[str] = None
        self.token_expires_at: float = 0.0
        
        self.logger.info("Spotify client initialized")
    
    def _extract_api_error(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract Spotify API error information from response data.
        
        Args:
            data: Parsed response data
            
        Returns:
            Error message if found, None otherwise
        """
        if "error" in data:
            error_info = data["error"]
            if isinstance(error_info, dict):
                return error_info.get("message", f"Error {error_info.get('status', 'unknown')}")
            return str(error_info)
        return None
    
    async def __aenter__(self):
        """Async context manager entry with authentication."""
        await super().__aenter__()
        await self._authenticate()
        return self
    
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
                data=data
            ) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data["access_token"]
                    expires_in = token_data.get("expires_in", 3600)
                    self.token_expires_at = time.time() + expires_in - 60  # 1min buffer
                    
                    self.logger.info(
                        "Spotify authentication successful",
                        expires_in=expires_in
                    )
                else:
                    error_data = await response.json()
                    self.logger.error(
                        "Spotify authentication failed",
                        status=response.status,
                        error=error_data
                    )
                    raise Exception(f"Spotify auth failed: {error_data}")
                    
        except Exception as e:
            self.logger.error("Spotify authentication error", error=str(e))
            raise
            
    async def _ensure_valid_token(self) -> None:
        """Ensure we have a valid access token."""
        if not self.access_token or time.time() >= self.token_expires_at:
            await self._authenticate()
    
    async def _make_spotify_request(
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
        await self._ensure_valid_token()
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        return await self._make_request(
            endpoint=endpoint,
            params=params,
            headers=headers,
            retries=retries
        )
    
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
            data = await self._make_spotify_request(
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
            self.logger.error(
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
            Spotify track data or None if not found
        """
        try:
            data = await self._make_spotify_request(f"tracks/{track_id}")
            
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
            self.logger.error(
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
            Audio features or None if not found
        """
        try:
            data = await self._make_spotify_request(f"audio-features/{track_id}")
            
            if not data or data.get("id") is None:
                return None
                
            return AudioFeatures(
                track_id=data["id"],
                danceability=data["danceability"],
                energy=data["energy"],
                valence=data["valence"],
                acousticness=data["acousticness"],
                instrumentalness=data["instrumentalness"],
                speechiness=data["speechiness"],
                liveness=data["liveness"],
                loudness=data["loudness"],
                tempo=data["tempo"],
                time_signature=data["time_signature"],
                key=data["key"],
                mode=data["mode"]
            )
            
        except Exception as e:
            self.logger.error(
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
        Get audio features for multiple tracks.
        
        Args:
            track_ids: List of Spotify track IDs (max 100)
            
        Returns:
            Dictionary mapping track IDs to audio features
        """
        if not track_ids:
            return {}
            
        # Spotify API allows max 100 IDs per request
        track_ids = track_ids[:100]
        
        try:
            data = await self._make_spotify_request(
                "audio-features",
                {"ids": ",".join(track_ids)}
            )
            
            features_dict = {}
            if "audio_features" in data:
                for features_data in data["audio_features"]:
                    if features_data:  # Can be None for tracks without features
                        features = AudioFeatures(
                            track_id=features_data["id"],
                            danceability=features_data["danceability"],
                            energy=features_data["energy"],
                            valence=features_data["valence"],
                            acousticness=features_data["acousticness"],
                            instrumentalness=features_data["instrumentalness"],
                            speechiness=features_data["speechiness"],
                            liveness=features_data["liveness"],
                            loudness=features_data["loudness"],
                            tempo=features_data["tempo"],
                            time_signature=features_data["time_signature"],
                            key=features_data["key"],
                            mode=features_data["mode"]
                        )
                        features_dict[features.track_id] = features
                        
            self.logger.info(
                "Multiple audio features retrieved",
                requested=len(track_ids),
                retrieved=len(features_dict)
            )
            
            return features_dict
            
        except Exception as e:
            self.logger.error(
                "Get multiple audio features failed",
                track_count=len(track_ids),
                error=str(e)
            )
            return {}
    
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
            data = await self._make_spotify_request(
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
                    
            self.logger.info(
                "Spotify search completed",
                query=query,
                results_count=len(tracks)
            )
            
            return tracks
            
        except Exception as e:
            self.logger.error(
                "Spotify search failed",
                query=query,
                error=str(e)
            )
            return []


# Legacy compatibility - maintain the old SpotifyRateLimiter class for backward compatibility
class SpotifyRateLimiter:
    """Legacy Spotify rate limiter for backward compatibility."""
    
    def __init__(self, calls_per_hour: int = 50):
        self.calls_per_hour = calls_per_hour
        self._unified_limiter = UnifiedRateLimiter.for_spotify(calls_per_hour)
        
    async def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        await self._unified_limiter.wait_if_needed() 