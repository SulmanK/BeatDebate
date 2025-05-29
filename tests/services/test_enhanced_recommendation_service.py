"""
Tests for Enhanced Recommendation Service

Tests to verify the enhanced recommendation service functionality,
agent integration, and unified API access patterns.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.services import (
    EnhancedRecommendationService,
    RecommendationRequest,
    RecommendationResponse,
    get_recommendation_service
)
from src.models.agent_models import SystemConfig, AgentConfig
from src.models.metadata_models import UnifiedTrackMetadata


class TestEnhancedRecommendationService:
    """Test suite for Enhanced Recommendation Service"""
    
    @pytest.fixture
    def system_config(self):
        """Create test system configuration"""
        return SystemConfig(
            lastfm_api_key="test_lastfm_key",
            gemini_api_key="test_gemini_key",
            lastfm_rate_limit=3.0,
            agent_configs={
                "planner": AgentConfig(
                    agent_name="PlannerAgent",
                    agent_type="planner"
                ),
                "genre_mood": AgentConfig(
                    agent_name="GenreMoodAgent", 
                    agent_type="advocate"
                ),
                "discovery": AgentConfig(
                    agent_name="DiscoveryAgent",
                    agent_type="advocate"
                ),
                "judge": AgentConfig(
                    agent_name="JudgeAgent",
                    agent_type="judge"
                )
            }
        )
    
    @pytest.fixture
    def mock_api_service(self):
        """Create mock API service"""
        api_service = Mock()
        api_service.get_lastfm_client = AsyncMock()
        api_service.get_unified_track_info = AsyncMock()
        api_service.search_unified_tracks = AsyncMock()
        api_service.close = AsyncMock()
        return api_service
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager"""
        cache_manager = Mock()
        cache_manager.close = Mock()
        return cache_manager
    
    @pytest.fixture
    def enhanced_service(self, system_config, mock_api_service, mock_cache_manager):
        """Create Enhanced Recommendation Service instance for testing"""
        return EnhancedRecommendationService(
            system_config=system_config,
            api_service=mock_api_service,
            cache_manager=mock_cache_manager
        )
    
    def test_enhanced_service_initialization(self, enhanced_service):
        """Test that Enhanced Recommendation Service initializes correctly"""
        assert enhanced_service.system_config is not None
        assert enhanced_service.api_service is not None
        assert enhanced_service.cache_manager is not None
        assert enhanced_service.context_manager is not None
        assert enhanced_service._agents_initialized is False
        assert enhanced_service.graph is None
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, enhanced_service):
        """Test agent initialization with shared services"""
        # Mock the LastFM client
        mock_lastfm_client = Mock()
        enhanced_service.api_service.get_lastfm_client.return_value = mock_lastfm_client
        
        # Mock agent constructors to avoid import issues
        with patch('src.services.enhanced_recommendation_service.PlannerAgent') as mock_planner, \
             patch('src.services.enhanced_recommendation_service.GenreMoodAgent') as mock_genre_mood, \
             patch('src.services.enhanced_recommendation_service.DiscoveryAgent') as mock_discovery, \
             patch('src.services.enhanced_recommendation_service.JudgeAgent') as mock_judge:
            
            await enhanced_service.initialize_agents()
            
            # Verify agents were created with correct parameters
            assert mock_planner.called
            assert mock_genre_mood.called
            assert mock_discovery.called
            assert mock_judge.called
            
            # Verify initialization completed
            assert enhanced_service._agents_initialized is True
            assert enhanced_service.graph is not None
    
    @pytest.mark.asyncio
    async def test_recommendation_request_response_flow(self, enhanced_service):
        """Test the complete recommendation request/response flow"""
        # Mock agent initialization
        enhanced_service._agents_initialized = True
        enhanced_service.graph = Mock()
        enhanced_service.graph.ainvoke = AsyncMock()
        
        # Mock context manager
        enhanced_service.context_manager.analyze_context_decision = AsyncMock(return_value={
            "decision": "use_context",
            "confidence": 0.8,
            "reasoning": "Test context decision"
        })
        enhanced_service.context_manager.update_context_after_recommendation = AsyncMock()
        
        # Mock workflow result
        mock_final_state = Mock()
        mock_final_state.final_recommendations = [
            {
                "title": "Test Track",
                "artist": "Test Artist",
                "score": 0.9,
                "reason": "Test recommendation"
            }
        ]
        mock_final_state.reasoning_log = ["Test reasoning"]
        mock_final_state.session_id = "test_session"
        enhanced_service.graph.ainvoke.return_value = mock_final_state
        
        # Mock unified metadata conversion
        mock_unified_track = UnifiedTrackMetadata(
            name="Test Track",
            artist="Test Artist",
            source=["test"],
            recommendation_score=0.9,
            recommendation_reason="Test recommendation"
        )
        enhanced_service.api_service.get_unified_track_info.return_value = mock_unified_track
        
        # Create request
        request = RecommendationRequest(
            query="I want some chill indie music",
            session_id="test_session",
            max_recommendations=5
        )
        
        # Execute request
        response = await enhanced_service.get_recommendations(request)
        
        # Verify response
        assert isinstance(response, RecommendationResponse)
        assert len(response.recommendations) > 0
        assert response.session_id == "test_session"
        assert response.processing_time > 0
        assert len(response.reasoning) > 0
        assert "context_decision" in response.metadata
    
    @pytest.mark.asyncio
    async def test_fallback_recommendations(self, enhanced_service):
        """Test fallback recommendations when workflow fails"""
        # Mock agent initialization
        enhanced_service._agents_initialized = True
        enhanced_service.graph = Mock()
        enhanced_service.graph.ainvoke = AsyncMock(side_effect=Exception("Workflow failed"))
        
        # Mock context manager
        enhanced_service.context_manager.analyze_context_decision = AsyncMock(return_value={
            "decision": "reset_context",
            "confidence": 0.5,
            "reasoning": "Test fallback"
        })
        
        # Mock fallback API service call
        mock_fallback_tracks = [
            UnifiedTrackMetadata(
                name="Fallback Track",
                artist="Fallback Artist",
                source=["fallback"]
            )
        ]
        enhanced_service.api_service.search_unified_tracks.return_value = mock_fallback_tracks
        
        # Create request
        request = RecommendationRequest(
            query="test query",
            max_recommendations=3
        )
        
        # Execute request
        response = await enhanced_service.get_recommendations(request)
        
        # Verify fallback response
        assert isinstance(response, RecommendationResponse)
        assert response.strategy_used["type"] == "fallback"
        assert "Error occurred" in response.reasoning[0]
        assert response.metadata["fallback_used"] is True
    
    @pytest.mark.asyncio
    async def test_unified_metadata_conversion(self, enhanced_service):
        """Test conversion of agent recommendations to unified metadata"""
        # Mock API service responses
        mock_unified_track = UnifiedTrackMetadata(
            name="Test Track",
            artist="Test Artist",
            source=["lastfm"],
            recommendation_score=0.8,
            recommendation_reason="Great match"
        )
        enhanced_service.api_service.get_unified_track_info.return_value = mock_unified_track
        
        # Test recommendations
        recommendations = [
            {
                "artist": "Test Artist",
                "track": "Test Track",
                "score": 0.8,
                "reason": "Great match",
                "agent": "GenreMoodAgent"
            }
        ]
        
        # Convert to unified metadata
        unified_tracks = await enhanced_service._convert_to_unified_metadata(
            recommendations, include_audio_features=True
        )
        
        # Verify conversion
        assert len(unified_tracks) == 1
        assert unified_tracks[0].name == "Test Track"
        assert unified_tracks[0].artist == "Test Artist"
        assert unified_tracks[0].recommendation_score == 0.8
        assert unified_tracks[0].agent_source == "GenreMoodAgent"
    
    @pytest.mark.asyncio
    async def test_service_cleanup(self, enhanced_service):
        """Test service cleanup and resource management"""
        await enhanced_service.close()
        
        # Verify cleanup calls
        enhanced_service.api_service.close.assert_called_once()
        enhanced_service.cache_manager.close.assert_called_once()
    
    def test_global_service_factory(self):
        """Test global service factory function"""
        service1 = get_recommendation_service()
        service2 = get_recommendation_service()
        
        # Should return the same instance (singleton pattern)
        assert service1 is service2
        assert isinstance(service1, EnhancedRecommendationService)


class TestRecommendationRequestResponse:
    """Test suite for request/response models"""
    
    def test_recommendation_request_creation(self):
        """Test RecommendationRequest creation and defaults"""
        request = RecommendationRequest(query="test query")
        
        assert request.query == "test query"
        assert request.session_id is None
        assert request.max_recommendations == 10
        assert request.include_audio_features is True
        assert request.context is None
    
    def test_recommendation_request_with_options(self):
        """Test RecommendationRequest with all options"""
        request = RecommendationRequest(
            query="test query",
            session_id="test_session",
            max_recommendations=5,
            include_audio_features=False,
            context={"test": "context"}
        )
        
        assert request.query == "test query"
        assert request.session_id == "test_session"
        assert request.max_recommendations == 5
        assert request.include_audio_features is False
        assert request.context == {"test": "context"}
    
    def test_recommendation_response_creation(self):
        """Test RecommendationResponse creation"""
        mock_track = UnifiedTrackMetadata(
            name="Test Track",
            artist="Test Artist",
            source=["test"]
        )
        
        response = RecommendationResponse(
            recommendations=[mock_track],
            strategy_used={"type": "enhanced"},
            reasoning=["Test reasoning"],
            session_id="test_session",
            processing_time=1.5,
            metadata={"test": "metadata"}
        )
        
        assert len(response.recommendations) == 1
        assert response.recommendations[0].name == "Test Track"
        assert response.strategy_used["type"] == "enhanced"
        assert response.reasoning == ["Test reasoning"]
        assert response.session_id == "test_session"
        assert response.processing_time == 1.5
        assert response.metadata["test"] == "metadata"


if __name__ == "__main__":
    pytest.main([__file__]) 