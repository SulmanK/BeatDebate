"""
Tests for APIService.

Ensures APIService correctly orchestrates and delegates calls to its modular components.
All components will be mocked to isolate the service logic.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.services.api_service import APIService
from src.models.metadata_models import UnifiedTrackMetadata, UnifiedArtistMetadata, MetadataSource


@pytest.fixture
def mock_cache_manager():
    """Create a mock cache manager."""
    return Mock()


@pytest.fixture
def mock_client_manager():
    """Create a mock client manager."""
    mock_manager = Mock()
    mock_manager.get_lastfm_client = AsyncMock()
    mock_manager.get_spotify_client = AsyncMock()
    mock_manager.lastfm_session = AsyncMock()
    mock_manager.spotify_session = AsyncMock()
    return mock_manager


@pytest.fixture
def mock_track_operations():
    """Create a mock track operations component."""
    mock_ops = Mock()
    mock_ops.search_unified_tracks = AsyncMock()
    mock_ops.get_unified_track_info = AsyncMock()
    mock_ops.get_similar_tracks = AsyncMock()
    mock_ops.search_by_tags = AsyncMock()
    mock_ops.search_tracks = AsyncMock()
    mock_ops.get_tracks_by_tag = AsyncMock()
    return mock_ops


@pytest.fixture
def mock_artist_operations():
    """Create a mock artist operations component."""
    mock_ops = Mock()
    mock_ops.get_artist_info = AsyncMock()
    mock_ops.get_similar_artist_tracks = AsyncMock()
    mock_ops.get_artist_top_tracks = AsyncMock()
    mock_ops.get_artist_primary_genres = AsyncMock()
    mock_ops.get_similar_artists = AsyncMock()
    return mock_ops


@pytest.fixture
def mock_genre_analyzer():
    """Create a mock genre analyzer component."""
    mock_analyzer = Mock()
    mock_analyzer.check_artist_genre_match = AsyncMock()
    mock_analyzer.check_track_genre_match = AsyncMock()
    mock_analyzer.check_genre_relationship_llm = AsyncMock()
    mock_analyzer.batch_check_tracks_genre_match = AsyncMock()
    return mock_analyzer


@pytest.fixture
def api_service(mock_cache_manager):
    """Create an APIService instance with mocked components."""
    with patch('src.services.api_service.ClientManager') as mock_client_manager_class, \
         patch('src.services.api_service.TrackOperations') as mock_track_ops_class, \
         patch('src.services.api_service.ArtistOperations') as mock_artist_ops_class, \
         patch('src.services.api_service.GenreAnalyzer') as mock_genre_analyzer_class:
        
        # Create the service
        service = APIService(
            lastfm_api_key="test_key",
            spotify_client_id="test_id",
            spotify_client_secret="test_secret",
            cache_manager=mock_cache_manager
        )
        
        # Replace the component instances with properly mocked versions
        service.client_manager = Mock()
        service.client_manager.get_lastfm_client = AsyncMock()
        service.client_manager.get_spotify_client = AsyncMock()
        service.client_manager.lastfm_session = AsyncMock()
        service.client_manager.spotify_session = AsyncMock()
        service.client_manager.close = AsyncMock()
        
        service.track_operations = Mock()
        service.track_operations.search_unified_tracks = AsyncMock()
        service.track_operations.get_unified_track_info = AsyncMock()
        service.track_operations.get_similar_tracks = AsyncMock()
        service.track_operations.search_by_tags = AsyncMock()
        service.track_operations.search_tracks = AsyncMock()
        service.track_operations.get_tracks_by_tag = AsyncMock()
        
        service.artist_operations = Mock()
        service.artist_operations.get_artist_info = AsyncMock()
        service.artist_operations.get_similar_artist_tracks = AsyncMock()
        service.artist_operations.get_artist_top_tracks = AsyncMock()
        service.artist_operations.get_artist_primary_genres = AsyncMock()
        service.artist_operations.get_similar_artists = AsyncMock()
        
        service.genre_analyzer = Mock()
        service.genre_analyzer.check_artist_genre_match = AsyncMock()
        service.genre_analyzer.check_track_genre_match = AsyncMock()
        service.genre_analyzer.check_genre_relationship_llm = AsyncMock()
        service.genre_analyzer.batch_check_tracks_genre_match = AsyncMock()
        
        return service


@pytest.fixture
def sample_unified_track():
    """Create a sample unified track for testing."""
    return UnifiedTrackMetadata(
        name="Test Track",
        artist="Test Artist",
        album="Test Album",
        source=MetadataSource.UNIFIED
    )


@pytest.fixture
def sample_unified_artist():
    """Create a sample unified artist for testing."""
    return UnifiedArtistMetadata(
        name="Test Artist",
        source=MetadataSource.UNIFIED
    )


class TestAPIServiceDelegation:
    """Test that APIService correctly delegates to its components."""
    
    @pytest.mark.asyncio
    async def test_search_unified_tracks_delegates_to_track_ops(self, api_service, sample_unified_track):
        """Verifies that a call to api_service.search_unified_tracks correctly calls the track_operations.search_unified_tracks method with the right arguments."""
        # Setup mock return value
        expected_tracks = [sample_unified_track]
        api_service.track_operations.search_unified_tracks.return_value = expected_tracks
        
        # Call the method
        result = await api_service.search_unified_tracks(
            query="test query",
            limit=10,
            include_spotify=True
        )
        
        # Verify delegation
        api_service.track_operations.search_unified_tracks.assert_called_once_with(
            query="test query",
            limit=10,
            include_spotify=True
        )
        
        # Verify result
        assert result == expected_tracks
    
    @pytest.mark.asyncio
    async def test_get_unified_track_info_delegates_to_track_ops(self, api_service, sample_unified_track):
        """Verifies delegation to the TrackOperations component for track info."""
        # Setup mock return value
        api_service.track_operations.get_unified_track_info.return_value = sample_unified_track
        
        # Call the method
        result = await api_service.get_unified_track_info(
            artist="Test Artist",
            track="Test Track",
            include_audio_features=True,
            include_spotify=False
        )
        
        # Verify delegation
        api_service.track_operations.get_unified_track_info.assert_called_once_with(
            artist="Test Artist",
            track="Test Track",
            include_audio_features=True,
            include_spotify=False
        )
        
        # Verify result
        assert result == sample_unified_track
    
    @pytest.mark.asyncio
    async def test_get_similar_tracks_delegates_to_track_ops(self, api_service, sample_unified_track):
        """Verifies delegation to TrackOperations for similar tracks."""
        # Setup mock return value
        expected_tracks = [sample_unified_track]
        api_service.track_operations.get_similar_tracks.return_value = expected_tracks
        
        # Call the method
        result = await api_service.get_similar_tracks(
            artist="Test Artist",
            track="Test Track",
            limit=15,
            include_spotify_features=True
        )
        
        # Verify delegation
        api_service.track_operations.get_similar_tracks.assert_called_once_with(
            artist="Test Artist",
            track="Test Track",
            limit=15,
            include_spotify_features=True
        )
        
        # Verify result
        assert result == expected_tracks
    
    @pytest.mark.asyncio
    async def test_search_by_tags_delegates_to_track_ops(self, api_service, sample_unified_track):
        """Verifies delegation to TrackOperations for tag search."""
        # Setup mock return value
        expected_tracks = [sample_unified_track]
        api_service.track_operations.search_by_tags.return_value = expected_tracks
        
        # Call the method
        result = await api_service.search_by_tags(
            tags=["rock", "alternative"],
            limit=20
        )
        
        # Verify delegation
        api_service.track_operations.search_by_tags.assert_called_once_with(
            tags=["rock", "alternative"],
            limit=20
        )
        
        # Verify result
        assert result == expected_tracks
    
    @pytest.mark.asyncio
    async def test_get_artist_info_delegates_to_artist_ops(self, api_service, sample_unified_artist):
        """Verifies delegation to the ArtistOperations component."""
        # Setup mock return value
        api_service.artist_operations.get_artist_info.return_value = sample_unified_artist
        
        # Call the method
        result = await api_service.get_artist_info(
            artist="Test Artist",
            include_top_tracks=True
        )
        
        # Verify delegation
        api_service.artist_operations.get_artist_info.assert_called_once_with(
            artist="Test Artist",
            include_top_tracks=True
        )
        
        # Verify result
        assert result == sample_unified_artist
    
    @pytest.mark.asyncio
    async def test_get_artist_top_tracks_delegates_to_artist_ops(self, api_service, sample_unified_track):
        """Verifies delegation to ArtistOperations for top tracks."""
        # Setup mock return value
        expected_tracks = [sample_unified_track]
        api_service.artist_operations.get_artist_top_tracks.return_value = expected_tracks
        
        # Call the method
        result = await api_service.get_artist_top_tracks(
            artist="Test Artist",
            limit=10,
            page=1
        )
        
        # Verify delegation
        api_service.artist_operations.get_artist_top_tracks.assert_called_once_with(
            artist="Test Artist",
            limit=10,
            page=1
        )
        
        # Verify result
        assert result == expected_tracks
    
    @pytest.mark.asyncio
    async def test_check_artist_genre_match_delegates_to_genre_analyzer(self, api_service):
        """Verifies delegation to the GenreAnalyzer component."""
        # Setup mock return value
        expected_result = {
            "match": True,
            "confidence": 0.9,
            "reason": "Artist is known for this genre"
        }
        api_service.genre_analyzer.check_artist_genre_match.return_value = expected_result
        
        # Call the method
        result = await api_service.check_artist_genre_match(
            artist="Test Artist",
            target_genre="rock",
            include_related_genres=True
        )
        
        # Verify delegation
        api_service.genre_analyzer.check_artist_genre_match.assert_called_once_with(
            artist="Test Artist",
            target_genre="rock",
            include_related_genres=True
        )
        
        # Verify result
        assert result == expected_result
    
    @pytest.mark.asyncio
    async def test_check_track_genre_match_delegates_to_genre_analyzer(self, api_service):
        """Verifies delegation to GenreAnalyzer for track genre matching."""
        # Setup mock return value
        expected_result = {
            "match": False,
            "confidence": 0.3,
            "reason": "Track doesn't match genre characteristics"
        }
        api_service.genre_analyzer.check_track_genre_match.return_value = expected_result
        
        # Call the method
        result = await api_service.check_track_genre_match(
            artist="Test Artist",
            track="Test Track",
            target_genre="jazz",
            include_related_genres=False
        )
        
        # Verify delegation
        api_service.genre_analyzer.check_track_genre_match.assert_called_once_with(
            artist="Test Artist",
            track="Test Track",
            target_genre="jazz",
            include_related_genres=False
        )
        
        # Verify result
        assert result == expected_result


class TestAPIServiceClientManagement:
    """Test client management delegation."""
    
    @pytest.mark.asyncio
    async def test_get_lastfm_client_delegates_to_client_manager(self, api_service):
        """Confirms that client retrieval is handled by the ClientManager."""
        # Setup mock return value
        mock_client = Mock()
        api_service.client_manager.get_lastfm_client.return_value = mock_client
        
        # Call the method
        result = await api_service.get_lastfm_client()
        
        # Verify delegation
        api_service.client_manager.get_lastfm_client.assert_called_once()
        
        # Verify result
        assert result == mock_client
    
    @pytest.mark.asyncio
    async def test_get_spotify_client_delegates_to_client_manager(self, api_service):
        """Confirms that Spotify client retrieval is handled by the ClientManager."""
        # Setup mock return value
        mock_client = Mock()
        api_service.client_manager.get_spotify_client.return_value = mock_client
        
        # Call the method
        result = await api_service.get_spotify_client()
        
        # Verify delegation
        api_service.client_manager.get_spotify_client.assert_called_once()
        
        # Verify result
        assert result == mock_client
    
    @pytest.mark.asyncio
    async def test_lastfm_session_delegates_to_client_manager(self, api_service):
        """Test that lastfm_session is delegated to ClientManager."""
        # Setup mock return value - lastfm_session() is not awaited in the API service
        # so we need a regular Mock, not AsyncMock for this method
        mock_session = Mock()
        api_service.client_manager.lastfm_session = Mock(return_value=mock_session)
        
        # Call the method
        result = await api_service.lastfm_session()
        
        # Verify delegation
        api_service.client_manager.lastfm_session.assert_called_once()
        
        # Verify result
        assert result == mock_session


class TestAPIServiceInitialization:
    """Test APIService initialization and component setup."""
    
    def test_api_service_initialization_with_cache(self, mock_cache_manager):
        """Test that APIService initializes correctly with cache manager."""
        with patch('src.services.api_service.ClientManager') as mock_client_manager_class, \
             patch('src.services.api_service.TrackOperations') as mock_track_ops_class, \
             patch('src.services.api_service.ArtistOperations') as mock_artist_ops_class, \
             patch('src.services.api_service.GenreAnalyzer') as mock_genre_analyzer_class:
            
            # Create service
            service = APIService(
                lastfm_api_key="test_key",
                spotify_client_id="test_id",
                spotify_client_secret="test_secret",
                cache_manager=mock_cache_manager
            )
            
            # Verify components were initialized
            mock_client_manager_class.assert_called_once_with(
                lastfm_api_key="test_key",
                spotify_client_id="test_id",
                spotify_client_secret="test_secret"
            )
            
            mock_track_ops_class.assert_called_once()
            mock_artist_ops_class.assert_called_once()
            mock_genre_analyzer_class.assert_called_once()
            
            # Verify cache manager is stored
            assert service.cache_manager == mock_cache_manager
    
    def test_api_service_initialization_without_cache(self):
        """Test that APIService initializes correctly without cache manager."""
        with patch('src.services.api_service.ClientManager') as mock_client_manager_class, \
             patch('src.services.api_service.TrackOperations') as mock_track_ops_class, \
             patch('src.services.api_service.ArtistOperations') as mock_artist_ops_class, \
             patch('src.services.api_service.GenreAnalyzer') as mock_genre_analyzer_class:
            
            # Create service without cache
            service = APIService(
                lastfm_api_key="test_key",
                spotify_client_id="test_id",
                spotify_client_secret="test_secret"
            )
            
            # Verify components were initialized
            mock_client_manager_class.assert_called_once()
            mock_track_ops_class.assert_called_once()
            mock_artist_ops_class.assert_called_once()
            mock_genre_analyzer_class.assert_called_once()
            
            # Verify cache manager is None
            assert service.cache_manager is None


class TestAPIServiceAliases:
    """Test alias methods that provide alternative names for the same functionality."""
    
    @pytest.mark.asyncio
    async def test_search_tracks_by_tags_alias(self, api_service, sample_unified_track):
        """Test that search_tracks_by_tags is an alias for search_by_tags."""
        # Setup mock return value
        expected_tracks = [sample_unified_track]
        api_service.track_operations.search_by_tags.return_value = expected_tracks
        
        # Call the alias method
        result = await api_service.search_tracks_by_tags(
            tags=["indie", "rock"],
            limit=15
        )
        
        # Verify it calls search_by_tags
        api_service.track_operations.search_by_tags.assert_called_once_with(
            tags=["indie", "rock"],
            limit=15
        )
        
        # Verify result
        assert result == expected_tracks


class TestAPIServiceErrorHandling:
    """Test error handling in APIService delegation."""
    
    @pytest.mark.asyncio
    async def test_component_error_propagation(self, api_service):
        """Test that errors from components are properly propagated."""
        # Setup mock to raise exception
        api_service.track_operations.search_unified_tracks.side_effect = Exception("Component error")
        
        # Verify exception is propagated
        with pytest.raises(Exception, match="Component error"):
            await api_service.search_unified_tracks("test query")
    
    @pytest.mark.asyncio
    async def test_close_method(self, api_service):
        """Test the close method delegates to client manager."""
        api_service.client_manager.close = AsyncMock()
        
        # Call close
        await api_service.close()
        
        # Verify delegation
        api_service.client_manager.close.assert_called_once() 