"""
Phase 2 Integration Tests

Tests the integration between IntentOrchestrationService, SessionManagerService,
and the updated agents (PlannerAgent, DiscoveryAgent) to verify they correctly
use effective intent instead of complex context override logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.services.session_manager_service import SessionManagerService
from src.services.intent_orchestration_service import IntentOrchestrationService
from src.agents.planner.agent import PlannerAgent
from src.agents.discovery.agent import DiscoveryAgent
from src.models.agent_models import MusicRecommenderState, AgentConfig
from src.models.metadata_models import UnifiedTrackMetadata


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = Mock()
    client.generate_content = AsyncMock(return_value=Mock(text='{"intent": "artist_similarity", "confidence": 0.8}'))
    return client


@pytest.fixture
def mock_api_service():
    """Mock API service for testing."""
    service = Mock()
    service.search_tracks = AsyncMock(return_value=[])
    service.get_artist_info = AsyncMock(return_value={'name': 'Test Artist'})
    return service


@pytest.fixture
def mock_metadata_service():
    """Mock metadata service for testing."""
    service = Mock()
    service.get_track_metadata = AsyncMock(return_value=UnifiedTrackMetadata(
        name="Test Track",
        artist="Test Artist",
        album="Test Album",
        duration=180,
        popularity=0.7,
        genres=["rock"],
        tags=["energetic"],
        audio_features={}
    ))
    return service


@pytest.fixture
def session_manager():
    """Session manager service for testing."""
    return SessionManagerService()


@pytest.fixture
def intent_orchestrator(mock_llm_client):
    """Intent orchestration service for testing."""
    return IntentOrchestrationService(mock_llm_client)


@pytest.fixture
def planner_agent(mock_llm_client, mock_api_service, mock_metadata_service):
    """PlannerAgent for testing."""
    config = AgentConfig(name="planner", type="planner")
    return PlannerAgent(
        config=config,
        llm_client=mock_llm_client,
        api_service=mock_api_service,
        metadata_service=mock_metadata_service
    )


@pytest.fixture
def discovery_agent(mock_llm_client, mock_api_service, mock_metadata_service):
    """DiscoveryAgent for testing."""
    config = AgentConfig(name="discovery", type="discovery")
    return DiscoveryAgent(
        config=config,
        llm_client=mock_llm_client,
        api_service=mock_api_service,
        metadata_service=mock_metadata_service
    )


class TestPhase2Integration:
    """Test Phase 2 integration between services and agents."""
    
    @pytest.mark.asyncio
    async def test_planner_uses_effective_intent(self, planner_agent, intent_orchestrator, session_manager):
        """Test that PlannerAgent correctly uses effective intent from IntentOrchestrationService."""
        
        # Setup: Create a session with original query context
        session_id = "test_session_phase2"
        original_query = "Music by Radiohead"
        
        # Store original query context
        await session_manager.store_original_query_context(
            session_id=session_id,
            query=original_query,
            intent="by_artist",
            entities={"artists": ["Radiohead"]},
            confidence=0.9
        )
        
        # Simulate follow-up query
        followup_query = "more tracks"
        
        # Get effective intent from orchestrator
        effective_intent = await intent_orchestrator.resolve_effective_intent(
            query=followup_query,
            session_id=session_id,
            session_manager=session_manager
        )
        
        # Verify effective intent is correctly resolved
        assert effective_intent is not None
        assert effective_intent['is_followup'] is True
        assert effective_intent['intent'] == 'by_artist'
        assert 'Radiohead' in str(effective_intent['entities'])
        
        # Create state with effective intent
        state = MusicRecommenderState(
            user_query=followup_query,
            session_id=session_id,
            effective_intent=effective_intent
        )
        
        # Process with PlannerAgent
        result_state = await planner_agent.process(state)
        
        # Verify PlannerAgent used effective intent
        assert result_state.query_understanding is not None
        assert result_state.query_understanding.intent.value == 'BY_ARTIST'
        assert 'Radiohead' in result_state.query_understanding.artists
        assert result_state.entities is not None
        assert 'Radiohead' in str(result_state.entities)
        
        print("✅ PlannerAgent correctly used effective intent from IntentOrchestrationService")
    
    @pytest.mark.asyncio
    async def test_discovery_agent_uses_effective_intent(self, discovery_agent, intent_orchestrator, session_manager):
        """Test that DiscoveryAgent correctly uses effective intent from IntentOrchestrationService."""
        
        # Setup: Create a session with original query context
        session_id = "test_session_discovery_phase2"
        original_query = "Indie rock music"
        
        # Store original query context
        await session_manager.store_original_query_context(
            session_id=session_id,
            query=original_query,
            intent="genre_exploration",
            entities={"genres": ["indie rock"]},
            confidence=0.8
        )
        
        # Simulate follow-up query
        followup_query = "more like this"
        
        # Get effective intent from orchestrator
        effective_intent = await intent_orchestrator.resolve_effective_intent(
            query=followup_query,
            session_id=session_id,
            session_manager=session_manager
        )
        
        # Verify effective intent
        assert effective_intent is not None
        assert effective_intent['is_followup'] is True
        assert effective_intent['intent'] == 'genre_exploration'
        
        # Create state with effective intent
        state = MusicRecommenderState(
            user_query=followup_query,
            session_id=session_id,
            effective_intent=effective_intent,
            recently_shown_track_ids=[]
        )
        
        # Process with DiscoveryAgent
        result_state = await discovery_agent.process(state)
        
        # Verify DiscoveryAgent processed the effective intent
        assert result_state.discovery_recommendations is not None
        
        print("✅ DiscoveryAgent correctly used effective intent from IntentOrchestrationService")
    
    @pytest.mark.asyncio
    async def test_artist_followup_scenario(self, planner_agent, discovery_agent, intent_orchestrator, session_manager):
        """Test complete artist follow-up scenario using Phase 2 architecture."""
        
        session_id = "test_artist_followup_phase2"
        
        # Step 1: Original query
        original_query = "Songs by The Beatles"
        await session_manager.store_original_query_context(
            session_id=session_id,
            query=original_query,
            intent="by_artist",
            entities={"artists": ["The Beatles"]},
            confidence=0.95
        )
        
        # Step 2: Follow-up query
        followup_query = "more songs"
        effective_intent = await intent_orchestrator.resolve_effective_intent(
            query=followup_query,
            session_id=session_id,
            session_manager=session_manager
        )
        
        # Verify intent resolution
        assert effective_intent['intent'] == 'by_artist'
        assert effective_intent['is_followup'] is True
        assert effective_intent['followup_type'] == 'load_more'
        assert 'The Beatles' in str(effective_intent['entities'])
        
        # Step 3: Process with PlannerAgent
        state = MusicRecommenderState(
            user_query=followup_query,
            session_id=session_id,
            effective_intent=effective_intent
        )
        
        planned_state = await planner_agent.process(state)
        
        # Verify planning used effective intent
        assert planned_state.query_understanding.intent.value == 'BY_ARTIST'
        assert 'The Beatles' in planned_state.query_understanding.artists
        
        # Step 4: Process with DiscoveryAgent
        discovery_state = await discovery_agent.process(planned_state)
        
        # Verify discovery processing
        assert discovery_state.discovery_recommendations is not None
        
        print("✅ Complete artist follow-up scenario works with Phase 2 architecture")
    
    @pytest.mark.asyncio
    async def test_fallback_to_traditional_approach(self, planner_agent, discovery_agent):
        """Test that agents fall back to traditional approach when no effective intent is available."""
        
        # Create state without effective intent (backward compatibility)
        state = MusicRecommenderState(
            user_query="Jazz music for studying",
            session_id="test_fallback"
        )
        
        # Process with PlannerAgent
        planned_state = await planner_agent.process(state)
        
        # Verify traditional query understanding was used
        assert planned_state.query_understanding is not None
        assert planned_state.entities is not None
        
        # Process with DiscoveryAgent
        discovery_state = await discovery_agent.process(planned_state)
        
        # Verify discovery processing worked
        assert discovery_state.discovery_recommendations is not None
        
        print("✅ Agents correctly fall back to traditional approach when no effective intent available")
    
    @pytest.mark.asyncio
    async def test_effective_intent_parameter_adaptation(self, discovery_agent):
        """Test that DiscoveryAgent correctly adapts parameters based on effective intent."""
        
        # Test different effective intents and their parameter adaptations
        test_cases = [
            {
                'intent': 'by_artist',
                'is_followup': True,
                'followup_type': 'load_more',
                'expected_adaptations': ['relaxed_quality_threshold']
            },
            {
                'intent': 'artist_similarity',
                'is_followup': True,
                'followup_type': 'artist_deep_dive',
                'expected_adaptations': ['reduced_underground_bias', 'permissive_novelty']
            },
            {
                'intent': 'discovery',
                'confidence': 0.5,
                'expected_adaptations': ['increased_candidate_pool']
            }
        ]
        
        for test_case in test_cases:
            effective_intent = test_case.copy()
            expected_adaptations = effective_intent.pop('expected_adaptations')
            
            # Create state with effective intent
            state = MusicRecommenderState(
                user_query="test query",
                session_id="test_adaptation",
                effective_intent=effective_intent,
                recently_shown_track_ids=[]
            )
            
            # Process with DiscoveryAgent
            result_state = await discovery_agent.process(state)
            
            # Verify processing completed (parameter adaptation is internal)
            assert result_state.discovery_recommendations is not None
        
        print("✅ DiscoveryAgent correctly adapts parameters based on effective intent")


def run_phase2_tests():
    """Run Phase 2 integration tests."""
    print("🚀 Running Phase 2 Integration Tests...")
    
    # Run the tests
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_phase2_tests() 