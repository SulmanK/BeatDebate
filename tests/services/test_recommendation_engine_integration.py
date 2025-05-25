"""
Integration tests for the RecommendationEngine service.

These tests demonstrate how the RecommendationEngine would be used in a 
real application.
"""

import pytest
from unittest.mock import AsyncMock

from src.models.agent_models import (
    SystemConfig, AgentConfig, MusicRecommenderState
)
from src.services.recommendation_engine import RecommendationEngine


class TestRecommendationEngineIntegration:
    """Integration test suite for the RecommendationEngine."""
    
    @pytest.fixture
    def system_config(self):
        """Create a test system configuration."""
        return SystemConfig(
            gemini_api_key="test-gemini-key",
            lastfm_api_key="test-lastfm-key",
            agent_configs={
                "planner": AgentConfig(
                    model="gemini-1.0-pro",
                    temperature=0.7
                ),
                "judge": AgentConfig(
                    model="gemini-1.0-pro",
                    temperature=0.5
                )
            }
        )
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        # Create mock agents
        mock_planner = AsyncMock()
        mock_genre_mood = AsyncMock()
        mock_discovery = AsyncMock()
        mock_judge = AsyncMock()

        # Configure mock behaviors
        async def planner_process(state):
            state.planning_strategy = {
                "evaluation_framework": {
                    "primary_weights": {"relevance": 0.5, "novelty": 0.5}
                }
            }
            return state

        async def genre_mood_process(state):
            state.genre_mood_recommendations = [
                {
                    "title": "Song 1", 
                    "artist": "Artist 1", 
                    "id": "test_track_1",
                    "source": "test",
                    "genres": ["rock"],
                    "reasoning_chain": "Test reasoning", 
                    "confidence_score": 0.8,
                    "relevance": 0.9,
                    "novelty": 0.7,
                    "recommending_agent": "GenreMoodAgent",
                    "strategy_applied": {"focus": "genre"}
                }
            ]
            return state
        
        async def discovery_process(state):
            state.discovery_recommendations = [
                {
                    "title": "Song 2", 
                    "artist": "Artist 2", 
                    "id": "test_track_2",
                    "source": "test",
                    "genres": ["indie"], 
                    "reasoning_chain": "Test reasoning", 
                    "confidence_score": 0.7,
                    "relevance": 0.8,
                    "novelty": 0.9,
                    "recommending_agent": "DiscoveryAgent",
                    "strategy_applied": {"focus": "novelty"}
                }
            ]
            return state
        
        async def judge_evaluate(state):
            # Set final recommendations directly on the Pydantic model
            state.final_recommendations = [
                {
                    "title": "Song 1", 
                    "artist": "Artist 1", 
                    "genres": ["rock"],
                    "reasoning_chain": "Final selection", 
                    "judge_score": 0.85
                }
            ]
            
            # Add to reasoning log
            if (hasattr(state, 'reasoning_log') and 
                isinstance(state.reasoning_log, list)):
                state.reasoning_log.append(
                    "JudgeAgent: Final selection complete."
                )
            
            return state

        mock_planner.process.side_effect = planner_process
        mock_genre_mood.process.side_effect = genre_mood_process
        mock_discovery.process.side_effect = discovery_process
        mock_judge.evaluate_and_select.side_effect = judge_evaluate
        
        return {
            "planner": mock_planner,
            "genre_mood": mock_genre_mood,
            "discovery": mock_discovery,
            "judge": mock_judge
        }
    
    @pytest.mark.asyncio
    async def test_basic_engine_creation_with_mocks(self, mock_agents):
        """Test basic creation of the recommendation engine with mocks."""
        engine = RecommendationEngine(
            planner_agent=mock_agents["planner"],
            genre_mood_agent=mock_agents["genre_mood"],
            discovery_agent=mock_agents["discovery"],
            judge_agent=mock_agents["judge"]
        )
        
        assert isinstance(engine, RecommendationEngine)
        assert engine.planner_agent is not None
        assert engine.judge_agent is not None
        assert engine.genre_mood_agent is not None
        assert engine.discovery_agent is not None
    
    @pytest.mark.asyncio
    async def test_mocked_component_workflow(self, mock_agents):
        """Test the workflow with mocked components."""
        engine = RecommendationEngine(
            planner_agent=mock_agents["planner"],
            genre_mood_agent=mock_agents["genre_mood"],
            discovery_agent=mock_agents["discovery"],
            judge_agent=mock_agents["judge"]
        )

        # Process a test query
        user_query = "Recommend me some chill indie music for studying"
        result = await engine.process_query(user_query)

        # Verify the workflow executed correctly
        assert result.user_query == user_query
        assert result.planning_strategy is not None
        assert len(result.genre_mood_recommendations) > 0
        assert len(result.discovery_recommendations) > 0
        assert len(result.final_recommendations) > 0
        
        # Verify agents were called
        mock_agents["planner"].process.assert_called_once()
        mock_agents["genre_mood"].process.assert_called_once()
        mock_agents["discovery"].process.assert_called_once()
        mock_agents["judge"].evaluate_and_select.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_workflow_usage_example(self, mock_agents):
        """Example of how the workflow would be used in an application."""
        # Create engine with mocked agents
        engine = RecommendationEngine(
            planner_agent=mock_agents["planner"],
            genre_mood_agent=mock_agents["genre_mood"],
            discovery_agent=mock_agents["discovery"],
            judge_agent=mock_agents["judge"]
        )

        # Process a user query
        user_query = "Recommend me some chill indie music for studying"
        result = await engine.process_query(user_query)

        # Verify the result structure
        assert isinstance(result, MusicRecommenderState)
        assert result.user_query == user_query
        assert result.final_recommendations is not None
        assert len(result.reasoning_log) > 0
        
        # Verify workflow completed successfully
        assert result.error_info is None
        assert result.total_processing_time is not None 