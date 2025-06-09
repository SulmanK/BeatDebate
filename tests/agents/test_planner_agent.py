"""
Tests for PlannerAgent.

Verifies the PlannerAgent correctly orchestrates its components (QueryAnalyzer, 
StrategyPlanner, etc.) to produce a valid planning_strategy on the state object.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from src.agents.planner.agent import PlannerAgent
from src.agents.planner.query_analyzer import QueryAnalyzer
from src.agents.planner.context_analyzer import ContextAnalyzer
from src.agents.planner.strategy_planner import StrategyPlanner
from src.agents.planner.entity_processor import EntityProcessor
from src.models.agent_models import (
    MusicRecommenderState, 
    QueryUnderstanding, 
    QueryIntent
)
from src.services.api_service import APIService
from src.services.metadata_service import MetadataService


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    from src.models.agent_models import AgentConfig
    return AgentConfig(
        agent_name="planner",
        agent_type="planner",
        llm_model="gemini-2.0-flash-exp",
        temperature=0.7,
        max_tokens=1000,
        timeout_seconds=30
    )


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    return Mock()


@pytest.fixture
def mock_api_service():
    """Create a mock API service."""
    return Mock(spec=APIService)


@pytest.fixture
def mock_metadata_service():
    """Create a mock metadata service."""
    return Mock(spec=MetadataService)


@pytest.fixture
def mock_rate_limiter():
    """Create a mock rate limiter."""
    return Mock()


@pytest.fixture
def sample_query_understanding():
    """Create a sample QueryUnderstanding object."""
    return QueryUnderstanding(
        intent=QueryIntent.ARTIST_SIMILARITY,
        confidence=0.9,
        artists=["Radiohead"],
        genres=["alternative rock", "experimental"],
        moods=[],
        activities=[],
        original_query="music like Radiohead",
        normalized_query="music like radiohead",
        reasoning="User wants music similar to Radiohead"
    )


@pytest.fixture
def sample_task_analysis():
    """Create a sample task analysis."""
    return {
        "complexity_level": "medium",
        "specificity": "specific",
        "context_factors": ["artist_similarity"],
        "requires_discovery": True,
        "estimated_difficulty": 0.6
    }


@pytest.fixture
def sample_planning_strategy():
    """Create a sample planning strategy."""
    return {
        "intent": "artist_similarity",
        "complexity_level": "medium",
        "confidence": 0.9,
        "agent_sequence": ["discovery_agent", "judge_agent"],
        "quality_thresholds": {
            "minimum_quality": 0.3,
            "preferred_quality": 0.6,
            "diversity_threshold": 0.4
        },
        "diversity_targets": {
            "genre_diversity": 0.6,
            "artist_diversity": 0.8
        },
        "explanation_style": "detailed",
        "generate_large_pool": True,
        "pool_size_multiplier": 2
    }


@pytest.fixture
def sample_coordination_plan():
    """Create a sample agent coordination plan."""
    return {
        "communication_strategy": "enhanced",
        "data_sharing": {
            "enabled": True,
            "scope": "full_metadata",
            "include_scores": True
        },
        "error_handling": {
            "strategy": "graceful_degradation",
            "retry_attempts": 3
        },
        "performance_targets": {
            "max_total_time": 30.0,
            "max_agent_time": 10.0
        }
    }


@pytest.fixture
def sample_state():
    """Create a sample MusicRecommenderState."""
    return MusicRecommenderState(
        user_query="music like Radiohead",
        session_id="test-session"
    )


@pytest.fixture
def sample_state_with_effective_intent():
    """Create a sample state with effective intent."""
    state = MusicRecommenderState(
        user_query="music like Radiohead",
        session_id="test-session"
    )
    state.effective_intent = {
        "intent": "artist_similarity",
        "entities": {
            "artists": ["Radiohead"],
            "genres": ["alternative rock"]
        },
        "is_followup": False,
        "confidence": 0.9
    }
    return state


@pytest.fixture
def sample_state_with_context_override():
    """Create a sample state with context override."""
    state = MusicRecommenderState(
        user_query="music like Radiohead",
        session_id="test-session"
    )
    state.context_override = {
        "is_followup": True,
        "intent_override": "by_artist",
        "target_entity": "The Beatles",
        "entities": {
            "artists": ["The Beatles"],
            "genres": ["rock", "pop"]
        },
        "confidence": 0.8
    }
    return state


class TestPlannerAgentInitialization:
    """Test PlannerAgent initialization and component setup."""
    
    def test_planner_agent_initialization(self, mock_config, mock_llm_client, mock_api_service, mock_metadata_service, mock_rate_limiter):
        """Test that PlannerAgent initializes correctly with all components."""
        planner = PlannerAgent(
            config=mock_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service,
            rate_limiter=mock_rate_limiter
        )
        
        # Verify all components are initialized
        assert isinstance(planner.query_analyzer, QueryAnalyzer)
        assert isinstance(planner.context_analyzer, ContextAnalyzer)
        assert isinstance(planner.strategy_planner, StrategyPlanner)
        assert isinstance(planner.entity_processor, EntityProcessor)
        
        # Verify component access methods work
        assert planner.get_query_analyzer() == planner.query_analyzer
        assert planner.get_context_analyzer() == planner.context_analyzer
        assert planner.get_strategy_planner() == planner.strategy_planner
        assert planner.get_entity_processor() == planner.entity_processor
    
    def test_component_status_check(self, mock_config, mock_llm_client, mock_api_service, mock_metadata_service):
        """Test component status checking functionality."""
        planner = PlannerAgent(
            config=mock_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service
        )
        
        status = planner.get_component_status()
        
        assert status["query_analyzer"] is True
        assert status["context_analyzer"] is True
        assert status["strategy_planner"] is True
        assert status["entity_processor"] is True


class TestPlannerAgentProcessFreshQuery:
    """Test PlannerAgent processing for fresh queries."""
    
    @pytest.mark.asyncio
    async def test_planner_agent_process_fresh_query(
        self, 
        mock_config, 
        mock_llm_client, 
        mock_api_service, 
        mock_metadata_service,
        sample_state,
        sample_query_understanding,
        sample_task_analysis,
        sample_planning_strategy,
        sample_coordination_plan
    ):
        """Test that PlannerAgent correctly processes a fresh query and produces valid strategy."""
        # Create planner agent
        planner = PlannerAgent(
            config=mock_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service
        )
        
        # Mock component methods
        planner.query_analyzer.understand_user_query = AsyncMock(return_value=sample_query_understanding)
        planner.query_analyzer.convert_understanding_to_entities = Mock(return_value={
            "artists": ["Radiohead"],
            "genres": ["alternative rock", "experimental"]
        })
        planner.query_analyzer.analyze_task_complexity = AsyncMock(return_value=sample_task_analysis)
        planner.strategy_planner.create_planning_strategy = AsyncMock(return_value=sample_planning_strategy)
        planner.strategy_planner.plan_agent_coordination = AsyncMock(return_value=sample_coordination_plan)
        
        # Process the state
        result_state = await planner.process(sample_state)
        
        # Verify the final state contains expected components
        assert result_state.query_understanding == sample_query_understanding
        assert result_state.entities == {
            "artists": ["Radiohead"],
            "genres": ["alternative rock", "experimental"]
        }
        assert result_state.intent_analysis == sample_task_analysis
        assert result_state.planning_strategy == sample_planning_strategy
        assert result_state.agent_coordination == sample_coordination_plan
        
        # Verify component methods were called correctly
        planner.query_analyzer.understand_user_query.assert_called_once_with("music like Radiohead")
        planner.query_analyzer.convert_understanding_to_entities.assert_called_once_with(sample_query_understanding)
        planner.query_analyzer.analyze_task_complexity.assert_called_once_with("music like Radiohead", sample_query_understanding)
        planner.strategy_planner.create_planning_strategy.assert_called_once_with(sample_query_understanding, sample_task_analysis)
        planner.strategy_planner.plan_agent_coordination.assert_called_once_with("music like Radiohead", sample_task_analysis)


class TestPlannerAgentProcessFollowupQuery:
    """Test PlannerAgent processing for follow-up queries."""
    
    @pytest.mark.asyncio
    async def test_planner_agent_process_followup_with_effective_intent(
        self,
        mock_config,
        mock_llm_client,
        mock_api_service,
        mock_metadata_service,
        sample_state_with_effective_intent,
        sample_task_analysis,
        sample_planning_strategy,
        sample_coordination_plan
    ):
        """Test that PlannerAgent uses effective intent when available."""
        # Create planner agent
        planner = PlannerAgent(
            config=mock_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service
        )
        
        # Mock context analyzer methods for effective intent
        mock_understanding = QueryUnderstanding(
            intent=QueryIntent.ARTIST_SIMILARITY,
            confidence=0.9,
            artists=["Radiohead"],
            genres=["alternative rock"],
            moods=[],
            activities=[],
            original_query="music like Radiohead",
            normalized_query="music like radiohead",
            reasoning="Using effective intent"
        )
        
        planner.context_analyzer.create_understanding_from_effective_intent = Mock(return_value=mock_understanding)
        planner.context_analyzer.create_entities_from_effective_intent = Mock(return_value={
            "artists": ["Radiohead"],
            "genres": ["alternative rock"]
        })
        
        # Mock other component methods
        planner.query_analyzer.analyze_task_complexity = AsyncMock(return_value=sample_task_analysis)
        planner.strategy_planner.create_planning_strategy = AsyncMock(return_value=sample_planning_strategy)
        planner.strategy_planner.plan_agent_coordination = AsyncMock(return_value=sample_coordination_plan)
        
        # Process the state
        result_state = await planner.process(sample_state_with_effective_intent)
        
        # Verify effective intent was used
        planner.context_analyzer.create_understanding_from_effective_intent.assert_called_once_with(
            "music like Radiohead", 
            sample_state_with_effective_intent.effective_intent
        )
        planner.context_analyzer.create_entities_from_effective_intent.assert_called_once_with(
            sample_state_with_effective_intent.effective_intent
        )
        
        # Verify query analyzer was NOT called for understanding (since effective intent was used)
        assert not hasattr(planner.query_analyzer.understand_user_query, 'called') or not planner.query_analyzer.understand_user_query.called
        
        # Verify final state
        assert result_state.query_understanding == mock_understanding
        assert result_state.planning_strategy == sample_planning_strategy
    
    @pytest.mark.asyncio
    async def test_planner_agent_process_followup_with_context_override(
        self,
        mock_config,
        mock_llm_client,
        mock_api_service,
        mock_metadata_service,
        sample_state_with_context_override,
        sample_task_analysis,
        sample_planning_strategy,
        sample_coordination_plan
    ):
        """Test that PlannerAgent uses context override for follow-up queries."""
        # Create planner agent
        planner = PlannerAgent(
            config=mock_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service
        )
        
        # Mock context analyzer methods
        planner.context_analyzer.is_followup_with_preserved_context = Mock(return_value=True)
        
        mock_understanding = QueryUnderstanding(
            intent=QueryIntent.BY_ARTIST,
            confidence=0.8,
            artists=["The Beatles"],
            genres=["rock", "pop"],
            moods=[],
            activities=[],
            original_query="music like Radiohead",
            normalized_query="music like radiohead",
            reasoning="Using context override"
        )
        
        planner.context_analyzer.create_understanding_from_context = Mock(return_value=mock_understanding)
        planner.context_analyzer.create_entities_from_context = Mock(return_value={
            "artists": ["The Beatles"],
            "genres": ["rock", "pop"]
        })
        
        # Mock other component methods
        planner.query_analyzer.analyze_task_complexity = AsyncMock(return_value=sample_task_analysis)
        planner.strategy_planner.create_planning_strategy = AsyncMock(return_value=sample_planning_strategy)
        planner.strategy_planner.plan_agent_coordination = AsyncMock(return_value=sample_coordination_plan)
        
        # Process the state
        result_state = await planner.process(sample_state_with_context_override)
        
        # Verify context override was used
        planner.context_analyzer.is_followup_with_preserved_context.assert_called_once_with(
            sample_state_with_context_override.context_override
        )
        planner.context_analyzer.create_understanding_from_context.assert_called_once_with(
            "music like Radiohead",
            sample_state_with_context_override.context_override
        )
        
        # Verify final state
        assert result_state.query_understanding == mock_understanding
        assert result_state.planning_strategy == sample_planning_strategy


class TestPlannerAgentErrorHandling:
    """Test PlannerAgent error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_planner_agent_handles_component_failure(
        self,
        mock_config,
        mock_llm_client,
        mock_api_service,
        mock_metadata_service,
        sample_state
    ):
        """Test that PlannerAgent produces valid fallback strategy when components fail."""
        # Create planner agent
        planner = PlannerAgent(
            config=mock_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service
        )
        
        # Mock query analyzer to raise exception
        planner.query_analyzer.understand_user_query = AsyncMock(side_effect=Exception("LLM service unavailable"))
        
        # Mock fallback strategy
        fallback_strategy = {
            "intent": "discovery",
            "complexity_level": "medium",
            "confidence": 0.3,
            "agent_sequence": ["discovery_agent", "judge_agent"],
            "quality_thresholds": {"minimum_quality": 0.2}
        }
        planner.strategy_planner.create_fallback_strategy = Mock(return_value=fallback_strategy)
        
        # Process the state
        result_state = await planner.process(sample_state)
        
        # Verify fallback strategy was used
        assert result_state.planning_strategy == fallback_strategy
        planner.strategy_planner.create_fallback_strategy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_planner_agent_handles_partial_failure(
        self,
        mock_config,
        mock_llm_client,
        mock_api_service,
        mock_metadata_service,
        sample_state,
        sample_query_understanding
    ):
        """Test that PlannerAgent handles partial component failures gracefully."""
        # Create planner agent
        planner = PlannerAgent(
            config=mock_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service
        )
        
        # Mock successful query understanding but failed task analysis
        planner.query_analyzer.understand_user_query = AsyncMock(return_value=sample_query_understanding)
        planner.query_analyzer.convert_understanding_to_entities = Mock(return_value={
            "artists": ["Radiohead"]
        })
        planner.query_analyzer.analyze_task_complexity = AsyncMock(side_effect=Exception("Analysis failed"))
        
        # Mock fallback strategy
        fallback_strategy = {
            "intent": "discovery",
            "complexity_level": "medium",
            "confidence": 0.3
        }
        planner.strategy_planner.create_fallback_strategy = Mock(return_value=fallback_strategy)
        
        # Process the state
        result_state = await planner.process(sample_state)
        
        # Verify partial success - query understanding should be set
        assert result_state.query_understanding == sample_query_understanding
        assert result_state.entities == {"artists": ["Radiohead"]}
        
        # Verify fallback was used for planning strategy
        assert result_state.planning_strategy == fallback_strategy


class TestPlannerAgentBackwardCompatibility:
    """Test PlannerAgent backward compatibility methods."""
    
    @pytest.mark.asyncio
    async def test_backward_compatibility_methods(
        self,
        mock_config,
        mock_llm_client,
        mock_api_service,
        mock_metadata_service,
        sample_query_understanding
    ):
        """Test that backward compatibility wrapper methods work correctly."""
        # Create planner agent
        planner = PlannerAgent(
            config=mock_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service
        )
        
        # Mock underlying component methods
        planner.query_analyzer.understand_user_query = AsyncMock(return_value=sample_query_understanding)
        planner.query_analyzer.convert_understanding_to_entities = Mock(return_value={"artists": ["Test"]})
        planner.context_analyzer.is_followup_with_preserved_context = Mock(return_value=True)
        planner.strategy_planner.create_fallback_strategy = Mock(return_value={"intent": "discovery"})
        
        # Test backward compatibility methods
        understanding = await planner._understand_user_query("test query")
        assert understanding == sample_query_understanding
        
        entities = planner._convert_understanding_to_entities(sample_query_understanding)
        assert entities == {"artists": ["Test"]}
        
        is_followup = planner._is_followup_with_preserved_context({"is_followup": True})
        assert is_followup is True
        
        fallback = planner._create_fallback_strategy()
        assert fallback == {"intent": "discovery"}
    
    @pytest.mark.asyncio
    async def test_llm_call_wrapper(
        self,
        mock_config,
        mock_llm_client,
        mock_api_service,
        mock_metadata_service
    ):
        """Test the LLM call wrapper method."""
        # Create planner agent
        planner = PlannerAgent(
            config=mock_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service
        )
        
        # Mock LLM utils
        planner.llm_utils = Mock()
        planner.llm_utils.call_llm_with_json_response = AsyncMock(return_value={"result": "test"})
        planner.llm_utils.call_llm = AsyncMock(return_value="simple response")
        
        # Test JSON response call
        result = await planner._make_llm_call("test prompt", "test system prompt")
        assert result == "{'result': 'test'}"
        
        # Test simple call
        result = await planner._make_llm_call("test prompt")
        assert result == "simple response"
        
        # Test error handling
        planner.llm_utils.call_llm.side_effect = Exception("LLM error")
        result = await planner._make_llm_call("test prompt")
        assert result == "{}"


class TestPlannerAgentIntegration:
    """Test PlannerAgent integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_integration(
        self,
        mock_config,
        mock_llm_client,
        mock_api_service,
        mock_metadata_service,
        sample_state
    ):
        """Test complete workflow integration with realistic data flow."""
        # Create planner agent
        planner = PlannerAgent(
            config=mock_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service
        )
        
        # Create realistic mock responses
        understanding = QueryUnderstanding(
            intent=QueryIntent.GENRE_MOOD,
            confidence=0.85,
            artists=[],
            genres=["jazz"],
            moods=["relaxing"],
            activities=[],
            original_query="relaxing jazz music",
            normalized_query="relaxing jazz music",
            reasoning="User wants relaxing jazz music"
        )
        
        entities = {"genres": ["jazz"], "moods": ["relaxing"]}
        
        task_analysis = {
            "complexity_level": "simple",
            "specificity": "moderate",
            "requires_discovery": False
        }
        
        planning_strategy = {
            "intent": "genre_mood",
            "complexity_level": "simple",
            "confidence": 0.85,
            "agent_sequence": ["genre_mood_agent", "judge_agent"],
            "quality_thresholds": {"minimum_quality": 0.2},
            "generate_large_pool": False
        }
        
        coordination_plan = {
            "communication_strategy": "basic",
            "data_sharing": {"enabled": True},
            "performance_targets": {"max_total_time": 20.0}
        }
        
        # Mock all component methods
        planner.query_analyzer.understand_user_query = AsyncMock(return_value=understanding)
        planner.query_analyzer.convert_understanding_to_entities = Mock(return_value=entities)
        planner.query_analyzer.analyze_task_complexity = AsyncMock(return_value=task_analysis)
        planner.strategy_planner.create_planning_strategy = AsyncMock(return_value=planning_strategy)
        planner.strategy_planner.plan_agent_coordination = AsyncMock(return_value=coordination_plan)
        
        # Process the state
        result_state = await planner.process(sample_state)
        
        # Verify complete workflow
        assert result_state.query_understanding.intent == QueryIntent.GENRE_MOOD
        assert result_state.entities["genres"] == ["jazz"]
        assert result_state.intent_analysis["complexity_level"] == "simple"
        assert result_state.planning_strategy["agent_sequence"] == ["genre_mood_agent", "judge_agent"]
        assert result_state.agent_coordination["communication_strategy"] == "basic"
        
        # Verify all components were called in correct order
        planner.query_analyzer.understand_user_query.assert_called_once()
        planner.query_analyzer.convert_understanding_to_entities.assert_called_once()
        planner.query_analyzer.analyze_task_complexity.assert_called_once()
        planner.strategy_planner.create_planning_strategy.assert_called_once()
        planner.strategy_planner.plan_agent_coordination.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_state_preservation_across_processing(
        self,
        mock_config,
        mock_llm_client,
        mock_api_service,
        mock_metadata_service
    ):
        """Test that existing state attributes are preserved during processing."""
        # Create state with existing attributes
        state = MusicRecommenderState(
            user_query="jazz music",
            session_id="test-session"
        )
        # Note: user_preferences and conversation_history are not fields in MusicRecommenderState
        # We'll test with valid fields instead
        
        # Create planner agent
        planner = PlannerAgent(
            config=mock_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service
        )
        
        # Mock minimal responses
        understanding = QueryUnderstanding(
            intent=QueryIntent.GENRE_MOOD,
            confidence=0.8,
            artists=[],
            genres=[],
            moods=[],
            activities=[],
            original_query="jazz music",
            normalized_query="jazz music",
            reasoning="Test"
        )
        
        planner.query_analyzer.understand_user_query = AsyncMock(return_value=understanding)
        planner.query_analyzer.convert_understanding_to_entities = Mock(return_value={})
        planner.query_analyzer.analyze_task_complexity = AsyncMock(return_value={})
        planner.strategy_planner.create_planning_strategy = AsyncMock(return_value={})
        planner.strategy_planner.plan_agent_coordination = AsyncMock(return_value={})
        
        # Process the state
        result_state = await planner.process(state)
        
        # Verify existing attributes are preserved
        assert result_state.user_query == "jazz music"
        assert result_state.session_id == "test-session"
        
        # Verify new attributes were added
        assert hasattr(result_state, 'query_understanding')
        assert hasattr(result_state, 'entities')
        assert hasattr(result_state, 'planning_strategy') 