"""
Tests for DiscoveryAgent

Tests the discovery agent's workflow orchestration and component integration,
including candidate generation, scoring, filtering, and diversity management.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

from src.agents.discovery.agent import DiscoveryAgent
from src.models.agent_models import MusicRecommenderState, AgentConfig
from src.models.recommendation_models import TrackRecommendation
from src.services.api_service import APIService
from src.services.metadata_service import MetadataService


@pytest.fixture
def mock_config():
    """Create a mock AgentConfig."""
    return AgentConfig(
        agent_name="discovery_agent",
        agent_type="discovery",
        llm_model="gemini-2.0-flash-exp",
        temperature=0.7,
        max_tokens=1000,
        timeout_seconds=30
    )


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    return AsyncMock()


@pytest.fixture
def mock_api_service():
    """Create a mock APIService."""
    return Mock(spec=APIService)


@pytest.fixture
def mock_metadata_service():
    """Create a mock MetadataService."""
    return Mock(spec=MetadataService)


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManagerService."""
    return AsyncMock()


@pytest.fixture
def discovery_agent(mock_config, mock_llm_client, mock_api_service, mock_metadata_service, mock_session_manager):
    """Create a DiscoveryAgent instance with mocked dependencies."""
    with patch('src.agents.discovery.agent.DiscoveryConfig') as mock_discovery_config, \
         patch('src.agents.discovery.agent.DiscoveryScorer') as mock_discovery_scorer, \
         patch('src.agents.discovery.agent.DiscoveryFilter') as mock_discovery_filter, \
         patch('src.agents.discovery.agent.DiscoveryDiversity') as mock_discovery_diversity, \
         patch('src.agents.discovery.agent.UnifiedCandidateGenerator') as mock_candidate_generator, \
         patch('src.agents.discovery.agent.QualityScorer') as mock_quality_scorer, \
         patch('src.agents.components.llm_utils.LLMUtils') as mock_llm_utils:
        
        # Configure mocks
        mock_discovery_config.return_value.base_config = {
            'target_candidates': 200,
            'final_recommendations': 20,
            'quality_threshold': 0.3,
            'novelty_threshold': 0.4
        }
        
        # Mock LLMUtils
        mock_llm_utils_instance = AsyncMock()
        mock_llm_utils.return_value = mock_llm_utils_instance
        
        agent = DiscoveryAgent(
            config=mock_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service,
            session_manager=mock_session_manager
        )
        
        # Store mock references for test access
        agent._mock_discovery_config = mock_discovery_config.return_value
        agent._mock_discovery_scorer = mock_discovery_scorer.return_value
        agent._mock_discovery_filter = mock_discovery_filter.return_value
        agent._mock_discovery_diversity = mock_discovery_diversity.return_value
        agent._mock_candidate_generator = mock_candidate_generator.return_value
        agent._mock_quality_scorer = mock_quality_scorer.return_value
        agent._mock_llm_utils = mock_llm_utils_instance
        
        return agent


@pytest.fixture
def sample_state():
    """Create a sample MusicRecommenderState for testing."""
    # Create a proper mock for query_understanding
    mock_query_understanding = Mock()
    mock_query_understanding.intent = 'discovery'
    
    return MusicRecommenderState(
        user_query="Find me some underground electronic music",
        session_id="test_session_123",
        entities={
            'genres': ['electronic'],
            'artists': [],
            'moods': ['underground']
        },
        intent_analysis={
            'intent': 'discovery',
            'is_followup': False,
            'query_understanding': mock_query_understanding
        },
        planning_strategy={}  # Add empty planning_strategy to avoid None errors
    )


@pytest.fixture
def sample_candidates():
    """Create sample candidate tracks for testing."""
    return [
        {
            'name': 'Underground Track 1',
            'artist': 'Hidden Artist',
            'listeners': 5000,
            'source': 'lastfm',
            'genres': ['electronic', 'ambient'],
            'url': 'http://example.com/track1'
        },
        {
            'name': 'Discovery Track 2',
            'artist': 'Emerging Artist',
            'listeners': 15000,
            'source': 'spotify',
            'genres': ['electronic', 'experimental'],
            'url': 'http://example.com/track2'
        },
        {
            'name': 'Hidden Gem 3',
            'artist': 'Underground Collective',
            'listeners': 2000,
            'source': 'lastfm',
            'genres': ['electronic', 'drone'],
            'url': 'http://example.com/track3'
        }
    ]


@pytest.fixture
def sample_scored_candidates():
    """Create sample scored candidates for testing."""
    return [
        {
            'name': 'Underground Track 1',
            'artist': 'Hidden Artist',
            'listeners': 5000,
            'source': 'lastfm',
            'genres': ['electronic', 'ambient'],
            'url': 'http://example.com/track1',
            'quality_score': 0.8,
            'novelty_score': 0.9,
            'underground_score': 0.85,
            'similarity_score': 0.7,
            'discovery_score': 0.82,
            'combined_score': 0.81
        },
        {
            'name': 'Discovery Track 2',
            'artist': 'Emerging Artist',
            'listeners': 15000,
            'source': 'spotify',
            'genres': ['electronic', 'experimental'],
            'url': 'http://example.com/track2',
            'quality_score': 0.75,
            'novelty_score': 0.6,
            'underground_score': 0.4,
            'similarity_score': 0.8,
            'discovery_score': 0.64,
            'combined_score': 0.70
        }
    ]


class TestDiscoveryAgentInitialization:
    """Test DiscoveryAgent initialization and component setup."""
    
    def test_discovery_agent_initialization(self, discovery_agent):
        """Test that DiscoveryAgent initializes correctly with all components."""
        assert discovery_agent is not None
        assert hasattr(discovery_agent, 'discovery_config')
        assert hasattr(discovery_agent, 'discovery_scorer')
        assert hasattr(discovery_agent, 'discovery_filter')
        assert hasattr(discovery_agent, 'discovery_diversity')
        assert hasattr(discovery_agent, 'candidate_generator')
        assert hasattr(discovery_agent, 'quality_scorer')
        assert hasattr(discovery_agent, 'current_params')
    
    def test_component_initialization(self, discovery_agent):
        """Test that all components are properly initialized."""
        # Check that components are not None
        assert discovery_agent.discovery_config is not None
        assert discovery_agent.discovery_scorer is not None
        assert discovery_agent.discovery_filter is not None
        assert discovery_agent.discovery_diversity is not None
        assert discovery_agent.candidate_generator is not None
        assert discovery_agent.quality_scorer is not None
        
        # Check current_params is set
        assert isinstance(discovery_agent.current_params, dict)
        assert 'target_candidates' in discovery_agent.current_params


class TestDiscoveryAgentProcessWorkflow:
    """Test the main discovery workflow processing."""
    
    @pytest.mark.asyncio
    async def test_discovery_agent_process_workflow(self, discovery_agent, sample_state, sample_candidates, sample_scored_candidates):
        """Test that DiscoveryAgent processes workflow correctly with all components."""
        # Mock candidate generation
        discovery_agent._mock_candidate_generator.generate_candidate_pool = AsyncMock(return_value=sample_candidates)
        
        # Mock scoring
        discovery_agent._mock_discovery_scorer.score_discovery_candidates = AsyncMock(return_value=sample_scored_candidates)
        
        # Mock filtering
        filtered_candidates = sample_scored_candidates[:2]  # Filter to 2 candidates
        discovery_agent._mock_discovery_filter.filter_for_discovery = AsyncMock(return_value=filtered_candidates)
        
        # Mock diversity management
        diverse_candidates = filtered_candidates  # Keep same for simplicity
        discovery_agent._mock_discovery_diversity.ensure_discovery_diversity = Mock(return_value=diverse_candidates)
        
        # Mock LLM reasoning generation
        with patch.object(discovery_agent, '_generate_batch_discovery_reasoning', new_callable=AsyncMock) as mock_reasoning:
            mock_reasoning.return_value = [
                "This underground track offers unique electronic textures",
                "An emerging artist with experimental electronic sounds"
            ]
            
            # Process the state
            result_state = await discovery_agent.process(sample_state)
            
            # Verify workflow execution
            assert result_state is not None
            assert hasattr(result_state, 'discovery_recommendations')
            assert isinstance(result_state.discovery_recommendations, list)
            assert len(result_state.discovery_recommendations) == 2
            
            # Verify component calls
            discovery_agent._mock_candidate_generator.generate_candidate_pool.assert_called_once()
            discovery_agent._mock_discovery_scorer.score_discovery_candidates.assert_called_once()
            discovery_agent._mock_discovery_filter.filter_for_discovery.assert_called_once()
            discovery_agent._mock_discovery_diversity.ensure_discovery_diversity.assert_called_once()
            
            # Verify recommendations structure
            for rec in result_state.discovery_recommendations:
                assert isinstance(rec, TrackRecommendation)
                assert rec.source == 'discovery_agent'
                assert rec.title is not None
                assert rec.artist is not None
    
    @pytest.mark.asyncio
    async def test_discovery_agent_handles_artist_similarity_intent(self, discovery_agent, sample_candidates, sample_scored_candidates):
        """Test that DiscoveryAgent handles artist_similarity intent correctly."""
        # Create a proper mock for query_understanding
        mock_query_understanding = Mock()
        mock_query_understanding.intent = 'artist_similarity'
        
        # Create state with artist_similarity intent
        state = MusicRecommenderState(
            user_query="Find artists similar to Radiohead",
            session_id="test_session_123",
            entities={
                'artists': ['Radiohead'],
                'genres': ['rock', 'alternative'],
                'moods': []
            },
            intent_analysis={
                'intent': 'artist_similarity',
                'is_followup': False,
                'query_understanding': mock_query_understanding
            },
            planning_strategy={}  # Add empty planning_strategy to avoid None errors
        )
        
        # Mock components
        discovery_agent._mock_candidate_generator.generate_candidate_pool = AsyncMock(return_value=sample_candidates)
        discovery_agent._mock_discovery_scorer.score_discovery_candidates = AsyncMock(return_value=sample_scored_candidates)
        discovery_agent._mock_discovery_filter.filter_for_discovery = AsyncMock(return_value=sample_scored_candidates)
        discovery_agent._mock_discovery_diversity.ensure_discovery_diversity = Mock(return_value=sample_scored_candidates)
        
        with patch.object(discovery_agent, '_generate_batch_discovery_reasoning', new_callable=AsyncMock) as mock_reasoning:
            mock_reasoning.return_value = ["Similar to Radiohead's experimental approach"] * len(sample_scored_candidates)
            
            result_state = await discovery_agent.process(state)
            
            # Verify that the intent was processed
            assert result_state.discovery_recommendations is not None
            assert len(result_state.discovery_recommendations) > 0
            
            # Verify candidate generation was called with correct intent
            call_args = discovery_agent._mock_candidate_generator.generate_candidate_pool.call_args
            assert call_args is not None
            # Check that intent_analysis was passed correctly
            assert 'intent_analysis' in call_args.kwargs or len(call_args.args) >= 2
    
    @pytest.mark.asyncio
    async def test_discovery_agent_skips_generation_for_followup(self, discovery_agent):
        """Test that DiscoveryAgent skips generation for follow-up queries."""
        # Create a proper mock for query_understanding
        mock_query_understanding = Mock()
        mock_query_understanding.intent = 'discovery'
        
        # Create state with follow-up flag
        followup_state = MusicRecommenderState(
            user_query="more tracks like these",
            session_id="test_session_123",
            entities={'artists': [], 'genres': [], 'moods': []},
            intent_analysis={
                'intent': 'discovery',
                'is_followup': True,
                'query_understanding': mock_query_understanding
            },
            planning_strategy={}  # Add empty planning_strategy to avoid None errors
        )
        
        result_state = await discovery_agent.process(followup_state)
        
        # Verify that generation was skipped
        assert result_state.discovery_recommendations == []
        
        # Verify that candidate generation was NOT called
        discovery_agent._mock_candidate_generator.generate_candidate_pool.assert_not_called()


class TestDiscoveryAgentCandidateGeneration:
    """Test candidate generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_discovery_candidates_standard(self, discovery_agent, sample_candidates):
        """Test standard candidate generation."""
        entities = {'genres': ['electronic'], 'artists': [], 'moods': ['underground']}
        intent_analysis = {'intent': 'discovery', 'is_followup': False}
        
        discovery_agent._mock_candidate_generator.generate_candidate_pool = AsyncMock(return_value=sample_candidates)
        
        result = await discovery_agent._generate_discovery_candidates(entities, intent_analysis)
        
        assert result == sample_candidates
        discovery_agent._mock_candidate_generator.generate_candidate_pool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_discovery_candidates_with_pool_generation(self, discovery_agent, sample_candidates, sample_state):
        """Test candidate generation with large pool generation."""
        entities = {'genres': ['electronic'], 'artists': [], 'moods': ['underground']}
        intent_analysis = {'intent': 'discovery', 'is_followup': False}
        
        # Mock pool generation
        discovery_agent._mock_candidate_generator.generate_and_persist_large_pool = AsyncMock(return_value="pool_key_123")
        discovery_agent._mock_candidate_generator.generate_candidate_pool = AsyncMock(return_value=sample_candidates)
        
        result = await discovery_agent._generate_discovery_candidates(
            entities, intent_analysis, should_generate_pool=True, state=sample_state
        )
        
        assert result == sample_candidates
        discovery_agent._mock_candidate_generator.generate_and_persist_large_pool.assert_called_once()
        discovery_agent._mock_candidate_generator.generate_candidate_pool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_discovery_candidates_handles_failure(self, discovery_agent):
        """Test that candidate generation handles failures gracefully."""
        entities = {'genres': ['electronic'], 'artists': [], 'moods': ['underground']}
        intent_analysis = {'intent': 'discovery', 'is_followup': False}
        
        discovery_agent._mock_candidate_generator.generate_candidate_pool = AsyncMock(side_effect=Exception("Generation failed"))
        
        result = await discovery_agent._generate_discovery_candidates(entities, intent_analysis)
        
        assert result == []


class TestDiscoveryAgentUtilityMethods:
    """Test utility methods and parameter management."""
    
    def test_get_current_parameters(self, discovery_agent):
        """Test getting current parameters."""
        params = discovery_agent.get_current_parameters()
        
        assert isinstance(params, dict)
        assert 'target_candidates' in params
        assert 'final_recommendations' in params
        assert 'quality_threshold' in params
        assert 'novelty_threshold' in params
    
    def test_update_parameters(self, discovery_agent):
        """Test updating parameters."""
        new_params = {
            'quality_threshold': 0.5,
            'novelty_threshold': 0.6
        }
        
        # Mock the validation method
        discovery_agent._mock_discovery_config.validate_parameters = Mock(return_value=new_params)
        
        discovery_agent.update_parameters(new_params)
        
        # Verify validation was called
        discovery_agent._mock_discovery_config.validate_parameters.assert_called_once_with(new_params)
    
    def test_create_discovery_fallback_reasoning(self, discovery_agent):
        """Test fallback reasoning creation."""
        candidate = {
            'name': 'Test Track',
            'artist': 'Test Artist',
            'novelty_score': 0.8,
            'underground_score': 0.7,
            'quality_score': 0.6,
            'genres': ['electronic']
        }
        entities = {'genres': ['electronic']}
        intent_analysis = {'intent': 'discovery'}
        
        reasoning = discovery_agent._create_discovery_fallback_reasoning(
            candidate, entities, intent_analysis, 1
        )
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert 'Test Track' in reasoning
        assert 'Test Artist' in reasoning


class TestDiscoveryAgentIntegration:
    """Integration tests for complete discovery workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_discovery_workflow_integration(self, discovery_agent, sample_state, sample_candidates):
        """Test complete discovery workflow with realistic data flow."""
        # Setup realistic mock responses
        scored_candidates = [
            {**candidate, 'quality_score': 0.8, 'discovery_score': 0.7, 'combined_score': 0.75}
            for candidate in sample_candidates
        ]
        
        discovery_agent._mock_candidate_generator.generate_candidate_pool = AsyncMock(return_value=sample_candidates)
        discovery_agent._mock_discovery_scorer.score_discovery_candidates = AsyncMock(return_value=scored_candidates)
        discovery_agent._mock_discovery_filter.filter_for_discovery = AsyncMock(return_value=scored_candidates)
        discovery_agent._mock_discovery_diversity.ensure_discovery_diversity = Mock(return_value=scored_candidates)
        
        with patch.object(discovery_agent, '_generate_batch_discovery_reasoning', new_callable=AsyncMock) as mock_reasoning:
            mock_reasoning.return_value = ["Great discovery track"] * len(scored_candidates)
            
            result_state = await discovery_agent.process(sample_state)
            
            # Verify complete workflow
            assert result_state.discovery_recommendations is not None
            assert len(result_state.discovery_recommendations) == len(sample_candidates)
            
            # Verify all components were called in correct order
            discovery_agent._mock_candidate_generator.generate_candidate_pool.assert_called_once()
            discovery_agent._mock_discovery_scorer.score_discovery_candidates.assert_called_once()
            discovery_agent._mock_discovery_filter.filter_for_discovery.assert_called_once()
            discovery_agent._mock_discovery_diversity.ensure_discovery_diversity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_state_preservation_across_processing(self, discovery_agent, sample_state, sample_candidates):
        """Test that existing state attributes are preserved during processing."""
        # Set some existing attributes to state
        sample_state.planning_strategy = {"some": "strategy"}
        sample_state.reasoning_log = ["existing reasoning"]
        
        discovery_agent._mock_candidate_generator.generate_candidate_pool = AsyncMock(return_value=sample_candidates)
        discovery_agent._mock_discovery_scorer.score_discovery_candidates = AsyncMock(return_value=[])
        discovery_agent._mock_discovery_filter.filter_for_discovery = AsyncMock(return_value=[])
        discovery_agent._mock_discovery_diversity.ensure_discovery_diversity = Mock(return_value=[])
        
        result_state = await discovery_agent.process(sample_state)
        
        # Verify existing attributes are preserved
        assert hasattr(result_state, 'planning_strategy')
        assert result_state.planning_strategy == {"some": "strategy"}
        assert hasattr(result_state, 'reasoning_log')
        assert "existing reasoning" in result_state.reasoning_log
        
        # Verify new attribute was added
        assert hasattr(result_state, 'discovery_recommendations')
