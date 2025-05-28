"""
Unified Metadata Models

Provides consistent data models for track and artist metadata across all services.
Consolidates LastFM and Spotify metadata into unified structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


class MetadataSource(Enum):
    """Enumeration of metadata sources."""
    LASTFM = "lastfm"
    SPOTIFY = "spotify"
    UNIFIED = "unified"  # For merged data from multiple sources


@dataclass
class UnifiedTrackMetadata:
    """
    Unified track metadata across all services.
    
    Combines data from LastFM and Spotify into a consistent structure
    while preserving service-specific data in source_data.
    """
    # Core identification fields (always present)
    name: str
    artist: str
    
    # Common optional fields
    album: Optional[str] = None
    duration_ms: Optional[int] = None
    
    # URLs and identifiers
    spotify_id: Optional[str] = None
    lastfm_mbid: Optional[str] = None  # MusicBrainz ID from LastFM
    preview_url: Optional[str] = None
    external_urls: Optional[Dict[str, str]] = None
    
    # Popularity and statistics
    popularity: Optional[int] = None  # Spotify popularity (0-100)
    listeners: Optional[int] = None   # LastFM listeners
    playcount: Optional[int] = None   # LastFM playcount
    
    # Discovery and categorization
    tags: List[str] = field(default_factory=list)  # LastFM tags
    genres: List[str] = field(default_factory=list)  # Unified genres
    similar_tracks: List[str] = field(default_factory=list)  # Similar track names
    
    # Metadata about the metadata
    source: MetadataSource = MetadataSource.UNIFIED
    source_data: Dict[str, Any] = field(default_factory=dict)  # Raw source data
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Quality and underground indicators
    underground_score: Optional[float] = None  # 0-1, higher = more underground
    quality_score: Optional[float] = None      # 0-1, higher = better quality
    
    # Recommendation-specific fields (added by recommendation service)
    recommendation_score: Optional[float] = None  # Score from recommendation agent
    recommendation_reason: Optional[str] = None   # Reason for recommendation
    agent_source: Optional[str] = None            # Which agent recommended this track
    
    # Audio features (from Spotify)
    audio_features: Optional[Dict[str, Any]] = None  # Spotify audio features
    
    # Source object references (for backward compatibility)
    spotify_data: Optional[Any] = None  # Original Spotify track object
    lastfm_data: Optional[Any] = None   # Original LastFM track object
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Normalize track and artist names
        self.name = self.name.strip() if self.name else ""
        self.artist = self.artist.strip() if self.artist else ""
        
        # Initialize collections if None
        if self.tags is None:
            self.tags = []
        if self.genres is None:
            self.genres = []
        if self.similar_tracks is None:
            self.similar_tracks = []
        if self.external_urls is None:
            self.external_urls = {}
        if self.source_data is None:
            self.source_data = {}
    
    @classmethod
    def from_lastfm(cls, lastfm_track: "TrackMetadata") -> "UnifiedTrackMetadata":
        """
        Create unified metadata from LastFM TrackMetadata.
        
        Args:
            lastfm_track: LastFM TrackMetadata object
            
        Returns:
            UnifiedTrackMetadata instance
        """
        return cls(
            name=lastfm_track.name,
            artist=lastfm_track.artist,
            lastfm_mbid=lastfm_track.mbid,
            external_urls={"lastfm": lastfm_track.url} if lastfm_track.url else {},
            listeners=lastfm_track.listeners,
            playcount=lastfm_track.playcount,
            tags=lastfm_track.tags or [],
            similar_tracks=lastfm_track.similar_tracks or [],
            source=MetadataSource.LASTFM,
            source_data={"lastfm": lastfm_track.__dict__}
        )
    
    @classmethod
    def from_spotify(cls, spotify_track: "SpotifyTrack") -> "UnifiedTrackMetadata":
        """
        Create unified metadata from Spotify SpotifyTrack.
        
        Args:
            spotify_track: Spotify SpotifyTrack object
            
        Returns:
            UnifiedTrackMetadata instance
        """
        return cls(
            name=spotify_track.name,
            artist=spotify_track.artist,
            album=spotify_track.album,
            spotify_id=spotify_track.id,
            duration_ms=spotify_track.duration_ms,
            preview_url=spotify_track.preview_url,
            external_urls=spotify_track.external_urls or {},
            popularity=spotify_track.popularity,
            source=MetadataSource.SPOTIFY,
            source_data={"spotify": spotify_track.__dict__}
        )
    
    def merge_with(self, other: "UnifiedTrackMetadata") -> "UnifiedTrackMetadata":
        """
        Merge this metadata with another instance.
        
        Args:
            other: Another UnifiedTrackMetadata instance
            
        Returns:
            New merged UnifiedTrackMetadata instance
        """
        # Verify tracks match
        if not self._matches_track(other):
            raise ValueError(f"Cannot merge different tracks: {self.name} vs {other.name}")
        
        # Create merged instance
        merged = UnifiedTrackMetadata(
            name=self.name,  # Use current name
            artist=self.artist,  # Use current artist
            album=self.album or other.album,
            duration_ms=self.duration_ms or other.duration_ms,
            spotify_id=self.spotify_id or other.spotify_id,
            lastfm_mbid=self.lastfm_mbid or other.lastfm_mbid,
            preview_url=self.preview_url or other.preview_url,
            external_urls={**self.external_urls, **other.external_urls},
            popularity=self.popularity or other.popularity,
            listeners=self.listeners or other.listeners,
            playcount=self.playcount or other.playcount,
            tags=list(set(self.tags + other.tags)),  # Merge and deduplicate
            genres=list(set(self.genres + other.genres)),
            similar_tracks=list(set(self.similar_tracks + other.similar_tracks)),
            source=MetadataSource.UNIFIED,
            source_data={**self.source_data, **other.source_data},
            underground_score=self.underground_score or other.underground_score,
            quality_score=self.quality_score or other.quality_score,
            recommendation_score=self.recommendation_score or other.recommendation_score,
            recommendation_reason=self.recommendation_reason or other.recommendation_reason,
            agent_source=self.agent_source or other.agent_source,
            audio_features=self.audio_features or other.audio_features,
            spotify_data=self.spotify_data or other.spotify_data,
            lastfm_data=self.lastfm_data or other.lastfm_data
        )
        
        return merged
    
    def _matches_track(self, other: "UnifiedTrackMetadata") -> bool:
        """
        Check if another track metadata represents the same track.
        
        Args:
            other: Another UnifiedTrackMetadata instance
            
        Returns:
            True if tracks match, False otherwise
        """
        # Normalize names for comparison
        name1 = self.name.lower().strip()
        name2 = other.name.lower().strip()
        artist1 = self.artist.lower().strip()
        artist2 = other.artist.lower().strip()
        
        # Basic name and artist match
        return name1 == name2 and artist1 == artist2
    
    def calculate_underground_score(self) -> float:
        """
        Calculate underground score based on available metrics.
        
        Returns:
            Underground score (0-1, higher = more underground)
        """
        score = 0.0
        factors = 0
        
        # LastFM popularity indicators
        if self.listeners is not None:
            # Lower listeners = more underground
            if self.listeners < 1000:
                score += 0.8
            elif self.listeners < 10000:
                score += 0.6
            elif self.listeners < 100000:
                score += 0.4
            else:
                score += 0.2
            factors += 1
        
        # Spotify popularity
        if self.popularity is not None:
            # Lower popularity = more underground
            underground_factor = (100 - self.popularity) / 100
            score += underground_factor
            factors += 1
        
        # Tag-based indicators
        underground_tags = [
            'experimental', 'underground', 'indie', 'lo-fi', 'avant-garde',
            'noise', 'drone', 'ambient', 'post-rock', 'math rock'
        ]
        tag_score = sum(1 for tag in self.tags if tag.lower() in underground_tags)
        if self.tags:
            score += min(tag_score / len(self.tags), 1.0)
            factors += 1
        
        # Average the factors
        final_score = score / factors if factors > 0 else 0.5
        self.underground_score = min(max(final_score, 0.0), 1.0)
        
        return self.underground_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "artist": self.artist,
            "album": self.album,
            "duration_ms": self.duration_ms,
            "spotify_id": self.spotify_id,
            "lastfm_mbid": self.lastfm_mbid,
            "preview_url": self.preview_url,
            "external_urls": self.external_urls,
            "popularity": self.popularity,
            "listeners": self.listeners,
            "playcount": self.playcount,
            "tags": self.tags,
            "genres": self.genres,
            "similar_tracks": self.similar_tracks,
            "source": self.source.value,
            "underground_score": self.underground_score,
            "quality_score": self.quality_score,
            "recommendation_score": self.recommendation_score,
            "recommendation_reason": self.recommendation_reason,
            "agent_source": self.agent_source,
            "audio_features": self.audio_features,
            "last_updated": self.last_updated.isoformat(),
            "spotify_data": self.spotify_data,
            "lastfm_data": self.lastfm_data
        }


@dataclass
class UnifiedArtistMetadata:
    """
    Unified artist metadata across all services.
    
    Combines data from LastFM and Spotify into a consistent structure.
    """
    # Core identification
    name: str
    
    # Identifiers
    spotify_id: Optional[str] = None
    lastfm_mbid: Optional[str] = None
    
    # URLs
    external_urls: Optional[Dict[str, str]] = None
    
    # Popularity and statistics
    popularity: Optional[int] = None  # Spotify popularity
    followers: Optional[int] = None   # Spotify followers
    listeners: Optional[int] = None   # LastFM listeners
    playcount: Optional[int] = None   # LastFM playcount
    
    # Categorization
    tags: List[str] = field(default_factory=list)
    genres: List[str] = field(default_factory=list)
    similar_artists: List[str] = field(default_factory=list)
    
    # Additional info
    bio: Optional[str] = None
    
    # Metadata
    source: MetadataSource = MetadataSource.UNIFIED
    source_data: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Post-initialization processing."""
        self.name = self.name.strip() if self.name else ""
        
        if self.tags is None:
            self.tags = []
        if self.genres is None:
            self.genres = []
        if self.similar_artists is None:
            self.similar_artists = []
        if self.external_urls is None:
            self.external_urls = {}
        if self.source_data is None:
            self.source_data = {}
    
    @classmethod
    def from_lastfm(cls, lastfm_artist: "ArtistMetadata") -> "UnifiedArtistMetadata":
        """
        Create unified metadata from LastFM ArtistMetadata.
        
        Args:
            lastfm_artist: LastFM ArtistMetadata object
            
        Returns:
            UnifiedArtistMetadata instance
        """
        return cls(
            name=lastfm_artist.name,
            lastfm_mbid=lastfm_artist.mbid,
            external_urls={"lastfm": lastfm_artist.url} if lastfm_artist.url else {},
            listeners=lastfm_artist.listeners,
            playcount=lastfm_artist.playcount,
            tags=lastfm_artist.tags or [],
            similar_artists=lastfm_artist.similar_artists or [],
            bio=lastfm_artist.bio,
            source=MetadataSource.LASTFM,
            source_data={"lastfm": lastfm_artist.__dict__}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "spotify_id": self.spotify_id,
            "lastfm_mbid": self.lastfm_mbid,
            "external_urls": self.external_urls,
            "popularity": self.popularity,
            "followers": self.followers,
            "listeners": self.listeners,
            "playcount": self.playcount,
            "tags": self.tags,
            "genres": self.genres,
            "similar_artists": self.similar_artists,
            "bio": self.bio,
            "source": self.source.value,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class SearchResult:
    """Unified search result containing tracks and artists."""
    tracks: List[UnifiedTrackMetadata] = field(default_factory=list)
    artists: List[UnifiedArtistMetadata] = field(default_factory=list)
    query: str = ""
    source: MetadataSource = MetadataSource.UNIFIED
    total_results: int = 0
    search_time_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tracks": [track.to_dict() for track in self.tracks],
            "artists": [artist.to_dict() for artist in self.artists],
            "query": self.query,
            "source": self.source.value,
            "total_results": self.total_results,
            "search_time_ms": self.search_time_ms
        }


# Utility functions for metadata operations
def merge_track_metadata(
    tracks: List[UnifiedTrackMetadata]
) -> List[UnifiedTrackMetadata]:
    """
    Merge duplicate tracks from different sources.
    
    Args:
        tracks: List of track metadata to merge
        
    Returns:
        List of merged track metadata with duplicates combined
    """
    merged_tracks = {}
    
    for track in tracks:
        # Create a key for matching tracks
        key = f"{track.artist.lower().strip()}||{track.name.lower().strip()}"
        
        if key in merged_tracks:
            # Merge with existing track
            merged_tracks[key] = merged_tracks[key].merge_with(track)
        else:
            merged_tracks[key] = track
    
    return list(merged_tracks.values())


def calculate_quality_scores(
    tracks: List[UnifiedTrackMetadata]
) -> List[UnifiedTrackMetadata]:
    """
    Calculate quality scores for a list of tracks.
    
    Args:
        tracks: List of track metadata
        
    Returns:
        List of tracks with quality scores calculated
    """
    for track in tracks:
        score = 0.0
        factors = 0
        
        # Popularity indicators
        if track.popularity is not None:
            score += track.popularity / 100
            factors += 1
        
        if track.listeners is not None:
            # Normalize listeners to 0-1 scale (logarithmic)
            import math
            normalized = min(math.log10(max(track.listeners, 1)) / 6, 1.0)
            score += normalized
            factors += 1
        
        # Metadata completeness
        completeness = 0
        if track.album:
            completeness += 1
        if track.duration_ms:
            completeness += 1
        if track.tags:
            completeness += 1
        if track.preview_url:
            completeness += 1
        
        score += completeness / 4  # Normalize to 0-1
        factors += 1
        
        # Calculate final score
        track.quality_score = score / factors if factors > 0 else 0.5
    
    return tracks 