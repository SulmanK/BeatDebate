"""
Tests for PlannerAgent

Basic tests to verify PlannerAgent functionality and strategic planning behavior.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.models.agent_models import MusicRecommenderState, AgentConfig
from src.agents import PlannerAgent


class TestPlannerAgent:
    """Test suite for PlannerAgent"""
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration"""
        return AgentConfig(
            agent_name="PlannerAgent",
            agent_type="planner",
            llm_model="gemini-2.0-flash-exp",
            temperature=0.7,
            timeout_seconds=30
        )
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Create mock Gemini client"""
        client = Mock()
        client.generate_content = AsyncMock()
        return client
    
    @pytest.fixture
    def planner_agent(self, agent_config, mock_gemini_client):
        """Create PlannerAgent instance for testing"""
        return PlannerAgent(agent_config, mock_gemini_client)
    
    @pytest.fixture
    def test_state(self):
        """Create test state with user query"""
        return MusicRecommenderState(
            user_query="I need focus music for coding",
            session_id="test_session_123"
        )
    
    def test_planner_agent_initialization(self, planner_agent):
        """Test that PlannerAgent initializes correctly"""
        assert planner_agent.agent_name == "PlannerAgent"
        assert planner_agent.agent_type == "planner"
        assert planner_agent.llm_client is not None
        assert hasattr(planner_agent, 'query_patterns')
        assert hasattr(planner_agent, 'strategy_templates')
    
    def test_fallback_query_analysis(self, planner_agent):
        """Test fallback query analysis without LLM"""
        # Test simple query
        analysis = planner_agent._fallback_query_analysis("play some music")
        assert analysis['complexity_level'] == 'simple'
        assert analysis['primary_goal'] == 'music_discovery'
        
        # Test complex query
        analysis = planner_agent._fallback_query_analysis("discover underground indie rock for studying")
        assert analysis['complexity_level'] == 'complex'
        assert 'focus' in analysis['mood_indicators']
    
    def test_fallback_coordination_strategy(self, planner_agent):
        """Test fallback coordination strategy creation"""
        task_analysis = {
            'primary_goal': 'focus_music',
            'complexity_level': 'medium',
            'mood_indicators': ['focus', 'chill'],
            'genre_hints': ['indie', 'electronic']
        }
        
        coordination = planner_agent._fallback_coordination_strategy(task_analysis)
        
        assert 'genre_mood_agent' in coordination
        assert 'discovery_agent' in coordination
        assert coordination['genre_mood_agent']['focus_areas'] == ['indie', 'electronic']
        assert coordination['discovery_agent']['novelty_priority'] == 'medium'
    
    def test_fallback_evaluation_framework(self, planner_agent):
        """Test fallback evaluation framework creation"""
        task_analysis = {'complexity_level': 'medium'}
        
        framework = planner_agent._fallback_evaluation_framework(task_analysis)
        
        assert 'primary_weights' in framework
        assert 'diversity_targets' in framework
        assert sum(framework['primary_weights'].values()) == pytest.approx(1.0, rel=1e-2)
    
    def test_enhance_task_analysis(self, planner_agent):
        """Test task analysis enhancement"""
        analysis = {'primary_goal': 'test'}
        query = "I need music for coding and focus"
        
        enhanced = planner_agent._enhance_task_analysis(analysis, query)
        
        assert 'work' in enhanced['context_factors']
        assert enhanced['complexity_level'] == 'medium'  # default
        assert isinstance(enhanced['mood_indicators'], list)
    
    def test_parse_json_response(self, planner_agent):
        """Test JSON response parsing"""
        # Test clean JSON
        response = '{"test": "value", "number": 123}'
        parsed = planner_agent._parse_json_response(response)
        assert parsed['test'] == 'value'
        assert parsed['number'] == 123
        
        # Test JSON with markdown
        response = '```json\n{"test": "value"}\n```'
        parsed = planner_agent._parse_json_response(response)
        assert parsed['test'] == 'value'
    
    @pytest.mark.asyncio
    async def test_process_with_fallback(self, planner_agent, test_state):
        """Test process method using fallback strategies (no LLM calls)"""
        # Mock LLM to raise exception, forcing fallback
        planner_agent.llm_client.generate_content.side_effect = Exception("LLM unavailable")
        
        result_state = await planner_agent.process(test_state)
        
        # Verify strategy was created using fallbacks
        assert result_state.planning_strategy is not None
        assert 'task_analysis' in result_state.planning_strategy
        assert 'coordination_strategy' in result_state.planning_strategy
        assert 'evaluation_framework' in result_state.planning_strategy
        assert 'execution_monitoring' in result_state.planning_strategy
        
        # Verify reasoning log was updated
        assert len(result_state.reasoning_log) > 0
        assert any('PlannerAgent' in log for log in result_state.reasoning_log)
    
    @pytest.mark.asyncio
    async def test_process_with_mock_llm(self, planner_agent, test_state):
        """Test process method with mocked LLM responses"""
        # Mock LLM responses
        mock_responses = [
            '{"primary_goal": "focus_music", "complexity_level": "medium", "context_factors": ["work"], "mood_indicators": ["focus"], "genre_hints": ["instrumental"]}',
            '{"genre_mood_agent": {"focus_areas": ["instrumental"], "energy_level": "medium"}, "discovery_agent": {"novelty_priority": "medium", "underground_bias": 0.6}}',
            '{"primary_weights": {"relevance": 0.4, "novelty": 0.3, "quality": 0.3}, "diversity_targets": {"genre": 2, "artist": 3}}'
        ]
        
        planner_agent.llm_client.generate_content.side_effect = [
            Mock(text=response) for response in mock_responses
        ]
        
        result_state = await planner_agent.process(test_state)
        
        # Verify strategy was created
        assert result_state.planning_strategy is not None
        strategy = result_state.planning_strategy
        
        # Verify task analysis
        assert strategy['task_analysis']['primary_goal'] == 'focus_music'
        assert strategy['task_analysis']['complexity_level'] == 'medium'
        
        # Verify coordination strategy
        assert 'genre_mood_agent' in strategy['coordination_strategy']
        assert 'discovery_agent' in strategy['coordination_strategy']
        
        # Verify evaluation framework
        assert 'primary_weights' in strategy['evaluation_framework']
        assert 'diversity_targets' in strategy['evaluation_framework']
    
    def test_execution_monitoring_setup(self, planner_agent):
        """Test execution monitoring setup"""
        task_analysis = {'complexity_level': 'complex'}
        
        monitoring = asyncio.run(planner_agent._setup_execution_monitoring(task_analysis))
        
        assert 'quality_thresholds' in monitoring
        assert 'fallback_strategies' in monitoring
        assert 'coordination_protocols' in monitoring
        assert 'success_metrics' in monitoring
        
        # Verify complex query gets lower thresholds
        assert monitoring['quality_thresholds']['min_confidence'] == 0.5
        assert monitoring['success_metrics']['target_recommendations'] == 3
    
    def test_strategy_templates_initialization(self, planner_agent):
        """Test that strategy templates are properly initialized"""
        templates = planner_agent.strategy_templates
        
        assert 'work_focus' in templates
        assert 'workout_energy' in templates
        assert 'chill_discovery' in templates
        
        # Verify template structure
        work_template = templates['work_focus']
        assert 'genre_mood_agent' in work_template
        assert 'discovery_agent' in work_template
        assert work_template['genre_mood_agent']['energy_level'] == 'medium-low'
    
    def test_query_patterns_initialization(self, planner_agent):
        """Test that query patterns are properly initialized"""
        patterns = planner_agent.query_patterns
        
        assert 'activity_context' in patterns
        assert 'mood_indicators' in patterns
        assert 'genre_hints' in patterns
        
        # Verify pattern content
        assert 'work' in patterns['activity_context']
        assert 'coding' in patterns['activity_context']['work']
        assert 'happy' in patterns['mood_indicators']
        assert 'rock' in patterns['genre_hints']


if __name__ == "__main__":
    pytest.main([__file__]) 