"""
Tests for the RecommendationEngine service.

These tests validate the LangGraph workflow orchestration.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.models.agent_models import MusicRecommenderState
from src.services.recommendation_engine import RecommendationEngine


class TestRecommendationEngine:
    """Test suite for the RecommendationEngine class."""
    
    @pytest.fixture
    def mock_agents(self):
        """Fixture to create mocked agents for testing."""
        mock_planner = AsyncMock()
        mock_genre_mood = AsyncMock()
        mock_discovery = AsyncMock()
        mock_judge = AsyncMock()
        
        # Configure agent behavior
        async def planner_process(state):
            state.planning_strategy = {
                "evaluation_framework": {
                    "primary_weights": {"relevance": 0.7, "novelty": 0.3},
                    "diversity_targets": {"genre": 2, "era": 1}
                },
                "task_analysis": {"primary_goal": "Test goal"},
                "coordination_strategy": {
                    "genre_mood_agent": {"focus_areas": ["rock"]},
                    "discovery_agent": {"novelty_priority": "medium"}
                }
            }
            return state
        
        async def genre_mood_process(state):
            state.genre_mood_recommendations = [
                {"title": "Song 1", "artist": "Artist 1", "genres": ["rock"], 
                 "reasoning_chain": "Test reasoning", "confidence_score": 0.8,
                 "relevance_score": 0.9, "recommending_agent": "GenreMoodAgent",
                 "strategy_applied": {"focus": "rock"}}
            ]
            return state
        
        async def discovery_process(state):
            state.discovery_recommendations = [
                {"title": "Song 2", "artist": "Artist 2", "genres": ["indie"], 
                 "reasoning_chain": "Test reasoning", "confidence_score": 0.7,
                 "relevance_score": 0.8, "recommending_agent": "DiscoveryAgent",
                 "strategy_applied": {"focus": "novelty"}}
            ]
            return state
        
        async def judge_evaluate(state):
            state.final_recommendations = [
                {"title": "Song 1", "artist": "Artist 1", "genres": ["rock"],
                 "reasoning_chain": "Final selection", "judge_score": 0.85}
            ]
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
    
    @pytest.fixture
    def engine(self, mock_agents):
        """Fixture to create a RecommendationEngine with mocked agents."""
        return RecommendationEngine(
            planner_agent=mock_agents["planner"],
            genre_mood_agent=mock_agents["genre_mood"],
            discovery_agent=mock_agents["discovery"],
            judge_agent=mock_agents["judge"]
        )
    
    @pytest.mark.asyncio
    async def test_node_functions(self, engine, mock_agents):
        """Test the individual node functions of the RecommendationEngine."""
        # Arrange
        state = MusicRecommenderState(user_query="Recommend me some rock music")
        
        # Act - Test the planner node
        updated_state = await engine._planner_node_func(state)
        
        # Assert - Planner node
        assert updated_state.planning_strategy is not None
        assert "evaluation_framework" in updated_state.planning_strategy
        assert "task_analysis" in updated_state.planning_strategy
        assert len(updated_state.reasoning_log) > 0
        mock_agents["planner"].process.assert_called_once()
        
        # Act - Test the genre_mood_advocate node
        updated_state = await engine._genre_mood_advocate_node_func(updated_state)
        
        # Assert - GenreMoodAgent node
        assert len(updated_state.genre_mood_recommendations) == 1
        assert updated_state.genre_mood_recommendations[0]["title"] == "Song 1"
        mock_agents["genre_mood"].process.assert_called_once()
        
        # Act - Test the discovery_advocate node
        updated_state = await engine._discovery_advocate_node_func(updated_state)
        
        # Assert - DiscoveryAgent node
        assert len(updated_state.discovery_recommendations) == 1
        assert updated_state.discovery_recommendations[0]["title"] == "Song 2"
        mock_agents["discovery"].process.assert_called_once()
        
        # Act - Test the judge node
        updated_state = await engine._judge_node_func(updated_state)
        
        # Assert - JudgeAgent node
        assert len(updated_state.final_recommendations) == 1
        assert updated_state.final_recommendations[0]["title"] == "Song 1"
        mock_agents["judge"].evaluate_and_select.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_planner_node_error_handling(self, engine, mock_agents):
        """Test error handling in the planner node."""
        # Arrange
        state = MusicRecommenderState(user_query="Recommend me some rock music")
        mock_agents["planner"].process.side_effect = Exception("Planner error")
        
        # Act
        updated_state = await engine._planner_node_func(state)
        
        # Assert
        assert updated_state.error_info is not None
        assert updated_state.error_info["agent"] == "PlannerAgent"
        assert "Planner error" in updated_state.error_info["message"]
        assert len(updated_state.reasoning_log) > 0
        assert "Error" in updated_state.reasoning_log[0]
    
    @pytest.mark.asyncio
    async def test_advocate_node_error_handling(self, engine, mock_agents):
        """Test error handling in an advocate node."""
        # Arrange
        state = MusicRecommenderState(user_query="Recommend me some rock music")
        # First run the planner successfully
        state = await engine._planner_node_func(state)
        # Then simulate error in genre_mood_advocate
        mock_agents["genre_mood"].process.side_effect = Exception("GenreMood error")
        
        # Act
        updated_state = await engine._genre_mood_advocate_node_func(state)
        
        # Assert
        assert updated_state.error_info is not None
        assert updated_state.error_info["agent"] == "GenreMoodAgent"
        assert "GenreMood error" in updated_state.error_info["message"]
    
    @pytest.mark.asyncio
    async def test_conditional_routing_logic(self, engine):
        """Test the conditional routing logic."""
        # Arrange - Test with valid planning strategy
        valid_state = MusicRecommenderState(user_query="Valid query")
        valid_state.planning_strategy = {
            "evaluation_framework": {"primary_weights": {}}
        }
        
        # Arrange - Test with empty planning strategy
        empty_state = MusicRecommenderState(user_query="Empty strategy")
        empty_state.planning_strategy = {}
        
        # Arrange - Test with error from planner
        error_state = MusicRecommenderState(user_query="Error state")
        error_state.error_info = {"agent": "PlannerAgent", "message": "Error"}
        
        # Act and Assert
        assert engine._should_proceed_after_planning(valid_state) == "execute_advocates"
        assert engine._should_proceed_after_planning(empty_state) == "end_workflow"
        assert engine._should_proceed_after_planning(error_state) == "end_workflow"
    
    @pytest.mark.asyncio
    async def test_build_graph(self, engine):
        """Test that the graph is correctly built."""
        assert engine.graph is not None 