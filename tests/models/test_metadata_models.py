"""
Tests for metadata models.

Validates data integrity and logic within data models including:
- Unified track creation from LastFM and Spotify sources
- Track metadata merging
- Underground score calculation
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.models.metadata_models import (
    UnifiedTrackMetadata,
    MetadataSource,
    merge_track_metadata
)
from src.api.lastfm_client import TrackMetadata
from src.api.spotify_client import SpotifyTrack


class TestUnifiedTrackMetadata:
    """Test UnifiedTrackMetadata functionality."""
    
    def test_unified_track_from_lastfm(self):
        """Ensures correct mapping from LastFmTrack to UnifiedTrackMetadata."""
        # Create mock LastFM track
        lastfm_track = TrackMetadata(
            name="Paranoid Android",
            artist="Radiohead",
            mbid="test-mbid-123",
            url="https://last.fm/music/Radiohead/_/Paranoid+Android",
            tags=["alternative rock", "experimental", "progressive rock"],
            similar_tracks=["Exit Music (For a Film)", "Karma Police"],
            listeners=500000,
            playcount=2000000,
            summary="A classic Radiohead track"
        )
        
        # Convert to unified metadata
        unified_track = UnifiedTrackMetadata.from_lastfm(lastfm_track)
        
        # Verify mapping
        assert unified_track.name == "Paranoid Android"
        assert unified_track.artist == "Radiohead"
        assert unified_track.lastfm_mbid == "test-mbid-123"
        assert unified_track.external_urls == {"lastfm": "https://last.fm/music/Radiohead/_/Paranoid+Android"}
        assert unified_track.listeners == 500000
        assert unified_track.playcount == 2000000
        assert unified_track.tags == ["alternative rock", "experimental", "progressive rock"]
        assert unified_track.similar_tracks == ["Exit Music (For a Film)", "Karma Police"]
        assert unified_track.source == MetadataSource.LASTFM
        assert "lastfm" in unified_track.source_data
        assert unified_track.source_data["lastfm"]["name"] == "Paranoid Android"
    
    def test_unified_track_from_spotify(self):
        """Ensures correct mapping from SpotifyTrack to UnifiedTrackMetadata."""
        # Create mock Spotify track
        spotify_track = SpotifyTrack(
            id="spotify-track-123",
            name="Paranoid Android",
            artist="Radiohead",
            album="OK Computer",
            preview_url="https://p.scdn.co/mp3-preview/123",
            external_urls={"spotify": "https://open.spotify.com/track/123"},
            duration_ms=383000,
            popularity=85
        )
        
        # Convert to unified metadata
        unified_track = UnifiedTrackMetadata.from_spotify(spotify_track)
        
        # Verify mapping
        assert unified_track.name == "Paranoid Android"
        assert unified_track.artist == "Radiohead"
        assert unified_track.album == "OK Computer"
        assert unified_track.spotify_id == "spotify-track-123"
        assert unified_track.duration_ms == 383000
        assert unified_track.preview_url == "https://p.scdn.co/mp3-preview/123"
        assert unified_track.external_urls == {"spotify": "https://open.spotify.com/track/123"}
        assert unified_track.popularity == 85
        assert unified_track.source == MetadataSource.SPOTIFY
        assert "spotify" in unified_track.source_data
        assert unified_track.source_data["spotify"]["id"] == "spotify-track-123"
    
    def test_merge_track_metadata(self):
        """Validates that merging two UnifiedTrackMetadata objects correctly combines their data."""
        # Create LastFM track
        lastfm_track = UnifiedTrackMetadata(
            name="Paranoid Android",
            artist="Radiohead",
            lastfm_mbid="test-mbid-123",
            external_urls={"lastfm": "https://last.fm/track/123"},
            listeners=500000,
            playcount=2000000,
            tags=["alternative rock", "experimental"],
            source=MetadataSource.LASTFM,
            source_data={"lastfm": {"name": "Paranoid Android"}}
        )
        
        # Create Spotify track
        spotify_track = UnifiedTrackMetadata(
            name="Paranoid Android",
            artist="Radiohead",
            album="OK Computer",
            spotify_id="spotify-123",
            duration_ms=383000,
            preview_url="https://preview.url",
            external_urls={"spotify": "https://spotify.com/track/123"},
            popularity=85,
            tags=["progressive rock"],
            source=MetadataSource.SPOTIFY,
            source_data={"spotify": {"id": "spotify-123"}}
        )
        
        # Merge tracks
        merged_track = lastfm_track.merge_with(spotify_track)
        
        # Verify merged data
        assert merged_track.name == "Paranoid Android"
        assert merged_track.artist == "Radiohead"
        assert merged_track.album == "OK Computer"  # From Spotify
        assert merged_track.spotify_id == "spotify-123"  # From Spotify
        assert merged_track.lastfm_mbid == "test-mbid-123"  # From LastFM
        assert merged_track.duration_ms == 383000  # From Spotify
        assert merged_track.preview_url == "https://preview.url"  # From Spotify
        assert merged_track.listeners == 500000  # From LastFM
        assert merged_track.playcount == 2000000  # From LastFM
        assert merged_track.popularity == 85  # From Spotify
        
        # Verify merged external URLs
        assert "lastfm" in merged_track.external_urls
        assert "spotify" in merged_track.external_urls
        assert merged_track.external_urls["lastfm"] == "https://last.fm/track/123"
        assert merged_track.external_urls["spotify"] == "https://spotify.com/track/123"
        
        # Verify merged tags (deduplicated)
        expected_tags = {"alternative rock", "experimental", "progressive rock"}
        assert set(merged_track.tags) == expected_tags
        
        # Verify merged source data
        assert "lastfm" in merged_track.source_data
        assert "spotify" in merged_track.source_data
        assert merged_track.source == MetadataSource.UNIFIED
    
    def test_merge_different_tracks_raises_error(self):
        """Confirms that attempting to merge two different tracks raises a ValueError."""
        # Create two different tracks
        track1 = UnifiedTrackMetadata(
            name="Paranoid Android",
            artist="Radiohead"
        )
        
        track2 = UnifiedTrackMetadata(
            name="Creep",
            artist="Radiohead"
        )
        
        # Attempt to merge should raise ValueError
        with pytest.raises(ValueError, match="Cannot merge different tracks"):
            track1.merge_with(track2)
    
    def test_calculate_underground_score(self):
        """Checks the logic for calculating a track's underground score based on its popularity metrics."""
        # Test high underground score (low popularity)
        underground_track = UnifiedTrackMetadata(
            name="Obscure Track",
            artist="Unknown Artist",
            listeners=500,  # Very low listeners
            popularity=10,  # Very low Spotify popularity
            tags=["experimental", "underground", "noise"]  # Underground tags
        )
        
        score = underground_track.calculate_underground_score()
        
        # Should be high underground score (close to 1.0)
        assert score > 0.7
        assert underground_track.underground_score == score
        
        # Test low underground score (high popularity)
        mainstream_track = UnifiedTrackMetadata(
            name="Popular Song",
            artist="Famous Artist",
            listeners=5000000,  # High listeners
            popularity=95,  # High Spotify popularity
            tags=["pop", "mainstream", "radio"]  # Mainstream tags
        )
        
        score = mainstream_track.calculate_underground_score()
        
        # Should be low underground score (close to 0.0)
        assert score < 0.3
        assert mainstream_track.underground_score == score
        
        # Test medium underground score
        medium_track = UnifiedTrackMetadata(
            name="Indie Track",
            artist="Indie Artist",
            listeners=50000,  # Medium listeners
            popularity=50,  # Medium Spotify popularity
            tags=["indie", "alternative"]  # Some underground elements
        )
        
        score = medium_track.calculate_underground_score()
        
        # Should be medium underground score
        assert 0.3 <= score <= 0.7
        assert medium_track.underground_score == score
    
    def test_calculate_underground_score_no_data(self):
        """Test underground score calculation with no popularity data."""
        track = UnifiedTrackMetadata(
            name="Unknown Track",
            artist="Unknown Artist"
            # No listeners, popularity, or tags
        )
        
        score = track.calculate_underground_score()
        
        # Should default to 0.5 when no data available
        assert score == 0.5
        assert track.underground_score == score
    
    def test_track_matching_logic(self):
        """Test the _matches_track method for track comparison."""
        track1 = UnifiedTrackMetadata(
            name="Paranoid Android",
            artist="Radiohead"
        )
        
        # Same track, different case
        track2 = UnifiedTrackMetadata(
            name="paranoid android",
            artist="RADIOHEAD"
        )
        
        # Different track
        track3 = UnifiedTrackMetadata(
            name="Creep",
            artist="Radiohead"
        )
        
        assert track1._matches_track(track2) is True
        assert track1._matches_track(track3) is False
    
    def test_post_init_normalization(self):
        """Test that __post_init__ properly normalizes and initializes fields."""
        track = UnifiedTrackMetadata(
            name="  Paranoid Android  ",  # Extra whitespace
            artist="  Radiohead  ",  # Extra whitespace
            tags=None,  # Should be initialized to empty list
            genres=None,  # Should be initialized to empty list
            similar_tracks=None,  # Should be initialized to empty list
            external_urls=None,  # Should be initialized to empty dict
            source_data=None  # Should be initialized to empty dict
        )
        
        # Verify normalization
        assert track.name == "Paranoid Android"
        assert track.artist == "Radiohead"
        
        # Verify initialization of collections
        assert track.tags == []
        assert track.genres == []
        assert track.similar_tracks == []
        assert track.external_urls == {}
        assert track.source_data == {}


class TestMergeTrackMetadata:
    """Test the merge_track_metadata utility function."""
    
    def test_merge_track_metadata_function(self):
        """Test the standalone merge_track_metadata function."""
        # Create tracks with same name/artist but different data
        track1 = UnifiedTrackMetadata(
            name="Test Track",
            artist="Test Artist",
            listeners=1000,
            tags=["rock"]
        )
        
        track2 = UnifiedTrackMetadata(
            name="Test Track",
            artist="Test Artist",
            popularity=75,
            tags=["alternative"]
        )
        
        track3 = UnifiedTrackMetadata(
            name="Different Track",
            artist="Test Artist",
            popularity=50
        )
        
        tracks = [track1, track2, track3]
        merged_tracks = merge_track_metadata(tracks)
        
        # Should have 2 tracks (track1 and track2 merged, track3 separate)
        assert len(merged_tracks) == 2
        
        # Find the merged track
        merged_track = next(t for t in merged_tracks if t.name == "Test Track")
        different_track = next(t for t in merged_tracks if t.name == "Different Track")
        
        # Verify merged data
        assert merged_track.listeners == 1000  # From track1
        assert merged_track.popularity == 75   # From track2
        assert set(merged_track.tags) == {"rock", "alternative"}  # Merged tags
        assert merged_track.source == MetadataSource.UNIFIED
        
        # Verify separate track unchanged
        assert different_track.popularity == 50
        assert different_track.name == "Different Track" 