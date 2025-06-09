"""
Tests for FastAPI backend endpoints.

Validates the FastAPI endpoints, ensuring correct request handling and response formatting.
The RecommendationService will be mocked to isolate endpoint testing.
"""

import pytest
import time
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

# from src.api.backend import app  # Not used in test setup
from src.models.metadata_models import UnifiedTrackMetadata, MetadataSource
from src.services.recommendation_service import RecommendationResponse


@pytest.fixture(scope="function")
def client():
    """Create a test client for the FastAPI app."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from unittest.mock import Mock
    
    # Create a test app without the problematic middleware
    test_app = FastAPI(
        title="BeatDebate API Test",
        description="Test version of the API",
        version="1.0.0"
    )
    
    # Add only CORS middleware for testing
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Set up a mock logger directly in the backend module
    import src.api.backend
    mock_logger = Mock()
    mock_logger.info = Mock()
    mock_logger.error = Mock()
    mock_logger.warning = Mock()
    mock_logger.debug = Mock()
    
    # Store original logger
    original_logger = src.api.backend.logger
    src.api.backend.logger = mock_logger
    
    try:
        # Import and add the routes from the main app
        from src.api.backend import health_check, get_recommendations, get_planning_strategy, submit_feedback, get_session_history, get_session_context
        
        test_app.get("/health")(health_check)
        test_app.post("/recommendations")(get_recommendations)
        test_app.post("/planning")(get_planning_strategy)
        test_app.post("/feedback")(submit_feedback)
        test_app.get("/sessions/{session_id}/history")(get_session_history)
        test_app.get("/sessions/{session_id}/context")(get_session_context)
        
        yield TestClient(test_app)
    finally:
        # Cleanup: restore original logger
        src.api.backend.logger = original_logger


@pytest.fixture
def mock_recommendation_service():
    """Create a mock recommendation service."""
    mock_service = AsyncMock()
    return mock_service


@pytest.fixture
def sample_unified_track():
    """Create a sample unified track for testing."""
    return UnifiedTrackMetadata(
        name="Paranoid Android",
        artist="Radiohead",
        album="OK Computer",
        spotify_id="test-spotify-id",
        duration_ms=383000,
        popularity=85,
        listeners=500000,
        tags=["alternative rock", "experimental"],
        source=MetadataSource.UNIFIED,
        recommendation_score=0.95,
        recommendation_reason="Matches your taste for experimental rock",
        agent_source="DiscoveryAgent"
    )


@pytest.fixture
def sample_recommendation_result(sample_unified_track):
    """Create a sample recommendation result."""
    return RecommendationResponse(
        recommendations=[sample_unified_track],
        session_id="test-session-123",
        processing_time=1.5,
        reasoning=["Identified as artist similarity query", "Found experimental tracks similar to Radiohead"],
        strategy_used={
            "intent": "artist_similarity",
            "approach": "discovery_focused",
            "target_count": 3
        },
        metadata={
            "agent_explanations": {
                "planner": "Identified as artist similarity query",
                "discovery": "Found experimental tracks similar to Radiohead",
                "judge": "Selected highest quality matches"
            }
        }
    )


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_check(self, client):
        """Verifies the /health endpoint returns a 200 OK status and the expected JSON payload."""
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
        assert "components" in data
        assert isinstance(data["timestamp"], (int, float))
        assert isinstance(data["components"], dict)
        
        # Verify timestamp is recent (within last 10 seconds)
        current_time = time.time()
        assert abs(current_time - data["timestamp"]) < 10


class TestRecommendationsEndpoint:
    """Test the recommendations endpoint."""
    
    @patch('src.api.backend.recommendation_service')
    def test_get_recommendations_success(self, mock_service, client, sample_recommendation_result):
        """Mocks a successful RecommendationService response and asserts the /recommendations endpoint returns 200 OK with correctly formatted track data."""
        # Setup mock
        mock_service.get_recommendations = AsyncMock(return_value=sample_recommendation_result)
        
        # Make request
        request_data = {
            "query": "music like Radiohead",
            "session_id": "test-session-123",
            "max_recommendations": 3,
            "include_previews": True
        }
        
        response = client.post("/recommendations", json=request_data)
        
        # Verify response
        assert response.status_code == 200
        
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert isinstance(data["processing_time"], float)  # Don't assert exact timing as it varies
        assert len(data["recommendations"]) == 1
        
        # Verify track data format (using transform_unified_to_ui_format field names)
        track = data["recommendations"][0]
        assert track["title"] == "Paranoid Android"  # transform function uses "title" not "name"
        assert track["artist"] == "Radiohead"
        assert track["album"] == "OK Computer"
        assert track["confidence"] == 0.95  # recommendation_score becomes confidence
        assert track["explanation"] == "Matches your taste for experimental rock"  # recommendation_reason becomes explanation
        assert track["source"] == "DiscoveryAgent"  # agent_source becomes source
        
        # Verify agent explanations in metadata
        assert "metadata" in data
        assert "agent_explanations" in data["metadata"]
        assert data["metadata"]["agent_explanations"]["planner"] == "Identified as artist similarity query"
        assert data["metadata"]["agent_explanations"]["discovery"] == "Found experimental tracks similar to Radiohead"
        
        # Verify strategy used
        assert "strategy_used" in data
        assert data["strategy_used"]["intent"] == "artist_similarity"
        
        # Verify mock was called correctly
        mock_service.get_recommendations.assert_called_once()
        call_args = mock_service.get_recommendations.call_args[0][0]
        assert call_args.query == "music like Radiohead"
        assert call_args.session_id == "test-session-123"
        assert call_args.max_recommendations == 3
    
    @patch('src.api.backend.recommendation_service')
    def test_get_recommendations_service_failure(self, mock_service, client):
        """Mocks the RecommendationService to raise an exception and asserts the endpoint returns a 500 Internal Server Error with a structured error message."""
        # Setup mock to raise exception
        mock_service.get_recommendations = AsyncMock(side_effect=Exception("Service unavailable"))
        
        # Make request
        request_data = {
            "query": "music like Radiohead",
            "max_recommendations": 3
        }
        
        response = client.post("/recommendations", json=request_data)
        
        # Verify error response
        assert response.status_code == 500
        
        data = response.json()
        assert "error" in data
        assert data["error"] == "Recommendation generation failed"  # Actual error message from backend
        assert "processing_time" in data
    
    def test_get_recommendations_invalid_request(self, client):
        """Sends a request with invalid data (e.g., missing query) and asserts the endpoint returns a 422 Unprocessable Entity error."""
        # Request missing required 'query' field
        request_data = {
            "max_recommendations": 3,
            "include_previews": True
        }
        
        response = client.post("/recommendations", json=request_data)
        
        # Verify validation error
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
        # FastAPI validation error should mention the missing field
        assert any("query" in str(error) for error in data["detail"])
    
    def test_get_recommendations_invalid_max_recommendations(self, client):
        """Test validation of max_recommendations field."""
        # Test with max_recommendations too high
        request_data = {
            "query": "test query",
            "max_recommendations": 15  # Above the limit of 10
        }
        
        response = client.post("/recommendations", json=request_data)
        assert response.status_code == 422
        
        # Test with max_recommendations too low
        request_data = {
            "query": "test query",
            "max_recommendations": 0  # Below the minimum of 1
        }
        
        response = client.post("/recommendations", json=request_data)
        assert response.status_code == 422
    
    @patch('src.api.backend.recommendation_service', None)
    def test_get_recommendations_no_service(self, client):
        """Test behavior when recommendation service is not available."""
        request_data = {
            "query": "music like Radiohead",
            "max_recommendations": 3
        }
        
        response = client.post("/recommendations", json=request_data)
        
        # Should return 503 when service is unavailable (as per the actual implementation)
        assert response.status_code == 503
        
        data = response.json()
        assert "detail" in data  # FastAPI HTTPException returns "detail" not "error"
        assert data["detail"] == "Recommendation service not available"


class TestPlanningEndpoint:
    """Test the planning strategy endpoint."""
    
    @patch('src.api.backend.recommendation_service')
    def test_get_planning_strategy_success(self, mock_service, client):
        """Mocks a successful response from the get_planning_strategy method and asserts the /planning endpoint returns 200 OK with the strategy data."""
        # Setup mock planning strategy
        mock_strategy = {
            "intent": "artist_similarity",
            "approach": "discovery_focused",
            "target_count": 3,
            "confidence": 0.9
        }
        
        mock_service.get_planning_strategy = AsyncMock(return_value=mock_strategy)
        
        # Make request
        request_data = {
            "query": "music like Radiohead",
            "session_id": "test-session-123"
        }
        
        response = client.post("/planning", json=request_data)
        
        # Verify response
        assert response.status_code == 200
        
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert isinstance(data["execution_time"], float)  # Don't assert exact timing as it varies
        assert data["strategy"]["intent"] == "artist_similarity"
        assert data["strategy"]["approach"] == "discovery_focused"
        assert data["strategy"]["confidence"] == 0.9
        
        # Verify mock was called
        mock_service.get_planning_strategy.assert_called_once()
    
    @patch('src.api.backend.recommendation_service')
    def test_get_planning_strategy_service_failure(self, mock_service, client):
        """Test planning endpoint when service fails."""
        mock_service.get_planning_strategy = AsyncMock(side_effect=Exception("Planning failed"))
        
        request_data = {
            "query": "music like Radiohead"
        }
        
        response = client.post("/planning", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data  # FastAPI HTTPException returns "detail" not "error"
        assert "Planning failed" in data["detail"]


class TestFeedbackEndpoint:
    """Test the feedback endpoint."""
    
    def test_feedback_endpoint(self, client):
        """Tests the /feedback endpoint to ensure it accepts valid feedback and returns a 200 OK status."""
        response = client.post(
            "/feedback",
            params={
                "session_id": "test-session-123",
                "recommendation_id": "track-123",
                "feedback": "thumbs_up"
            }
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Feedback submitted successfully"  # Actual message returned by endpoint
    
    def test_feedback_endpoint_invalid_feedback(self, client):
        """Test feedback endpoint with invalid feedback value."""
        response = client.post(
            "/feedback",
            params={
                "session_id": "test-session-123",
                "recommendation_id": "track-123",
                "feedback": "invalid_feedback"
            }
        )
        
        # Should return 400 for invalid feedback values (as per the actual implementation)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert data["detail"] == "Invalid feedback value"


class TestSessionEndpoints:
    """Test session-related endpoints."""
    
    @patch('src.api.backend.recommendation_service')
    def test_get_session_history(self, mock_service, client):
        """Test getting session history."""
        mock_history = [
            {
                "query": "music like Radiohead",
                "timestamp": time.time(),
                "recommendations_count": 3
            }
        ]
        
        mock_service.get_session_history = AsyncMock(return_value=mock_history)
        
        response = client.get("/sessions/test-session-123/history")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert "history" in data
        assert "message" in data
        # The actual endpoint returns a placeholder response
    
    @patch('src.api.backend.recommendation_service')
    def test_get_session_context(self, mock_service, client):
        """Test getting session context."""
        mock_context = {
            "original_query": "music like Radiohead",
            "session_start": time.time(),
            "interaction_count": 2
        }
        
        # Mock the smart_context_manager attribute and its method
        mock_service.smart_context_manager.get_session_context = AsyncMock(return_value=mock_context)
        
        response = client.get("/sessions/test-session-123/context")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert data["context_summary"] == mock_context
        assert "timestamp" in data


class TestErrorHandling:
    """Test error handling and exception responses."""
    
    def test_404_endpoint(self, client):
        """Test accessing non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test using wrong HTTP method."""
        response = client.get("/recommendations")  # Should be POST
        assert response.status_code == 405


class TestTransformFunction:
    """Test the transform_unified_to_ui_format function."""
    
    def test_transform_unified_to_ui_format(self, sample_unified_track):
        """Test the transformation of UnifiedTrackMetadata to UI format."""
        from src.api.backend import transform_unified_to_ui_format
        
        result = transform_unified_to_ui_format(sample_unified_track)
        
        # Check the actual field names used by the transform function
        assert result["title"] == "Paranoid Android"  # transform uses "title" not "name"
        assert result["artist"] == "Radiohead"
        assert result["album"] == "OK Computer"
        assert result["popularity"] == 85
        assert result["listeners"] == 500000
        assert result["moods"] == ["alternative rock", "experimental"]  # tags become moods
        assert result["confidence"] == 0.95  # recommendation_score becomes confidence
        assert result["explanation"] == "Matches your taste for experimental rock"  # recommendation_reason becomes explanation
        assert result["source"] == "DiscoveryAgent"  # agent_source becomes source 