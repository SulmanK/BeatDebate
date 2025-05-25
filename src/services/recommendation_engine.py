"""
Recommendation Engine Service for BeatDebate Multi-Agent System

This service orchestrates the flow of agents in the music recommendation system using LangGraph.
It manages the workflow from initial user query to final recommendations.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union

import structlog
from langgraph.graph import StateGraph, END

from ..models.agent_models import MusicRecommenderState, AgentConfig, SystemConfig
from ..agents.planner_agent import PlannerAgent
from ..agents.genre_mood_agent import GenreMoodAgent
from ..agents.discovery_agent import DiscoveryAgent
from ..agents.judge_agent import JudgeAgent
from ..api.lastfm_client import LastFmClient

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class RecommendationEngine:
    """
    Orchestrates the multi-agent workflow for music recommendations using LangGraph.
    
    Manages the sequential and parallel execution of:
    - PlannerAgent: Creates the overall strategy
    - GenreMoodAgent: Generates recommendations based on genre and mood
    - DiscoveryAgent: Generates recommendations for discovery and novelty
    - JudgeAgent: Evaluates and selects final recommendations
    """
    
    def __init__(
        self,
        planner_agent: PlannerAgent,
        genre_mood_agent: GenreMoodAgent,
        discovery_agent: DiscoveryAgent,
        judge_agent: JudgeAgent
    ):
        """
        Initialize the recommendation engine with agent instances.
        
        Args:
            planner_agent: Strategy planning agent
            genre_mood_agent: Genre and mood advocate agent
            discovery_agent: Discovery and novelty advocate agent
            judge_agent: Final selection judge agent
        """
        self.planner_agent = planner_agent
        self.genre_mood_agent = genre_mood_agent
        self.discovery_agent = discovery_agent
        self.judge_agent = judge_agent
        self.logger = logger.bind(service="RecommendationEngine")
        self.graph = self._build_graph()
        
        self.logger.info("RecommendationEngine initialized with LangGraph workflow")
    
    async def _planner_node_func(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        LangGraph node function for PlannerAgent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with planning strategy
        """
        # Get user query for logging
        user_query = None
        if hasattr(state, 'user_query'):
            user_query = state.user_query
        elif isinstance(state, dict) and 'user_query' in state:
            user_query = state['user_query']
            
        self.logger.info("Executing Planner Node", user_query=user_query)
        try:
            updated_state = await self.planner_agent.process(state)
            # Add to reasoning log
            if hasattr(updated_state, 'reasoning_log'):
                updated_state.reasoning_log.append("PlannerAgent: Strategy created.")
            elif isinstance(updated_state, dict) and 'reasoning_log' in updated_state:
                updated_state['reasoning_log'].append("PlannerAgent: Strategy created.")
            return updated_state
        except Exception as e:
            self.logger.error("Planner Node Error", error=str(e))
            # Add to reasoning log and error info
            if hasattr(state, 'reasoning_log'):
                state.reasoning_log.append(f"PlannerAgent: Error - {str(e)}")
                state.error_info = {"agent": "PlannerAgent", "message": str(e)}
            elif isinstance(state, dict):
                if 'reasoning_log' not in state:
                    state['reasoning_log'] = []
                state['reasoning_log'].append(f"PlannerAgent: Error - {str(e)}")
                state['error_info'] = {"agent": "PlannerAgent", "message": str(e)}
            return state
    
    async def _genre_mood_advocate_node_func(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        LangGraph node function for GenreMoodAgent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with genre/mood recommendations
        """
        self.logger.info("Executing GenreMood Advocate Node")
        try:
            updated_state = await self.genre_mood_agent.process(state)
            # Add to reasoning log
            if hasattr(updated_state, 'reasoning_log'):
                updated_state.reasoning_log.append("GenreMoodAgent: Recommendations generated.")
            elif isinstance(updated_state, dict) and 'reasoning_log' in updated_state:
                updated_state['reasoning_log'].append("GenreMoodAgent: Recommendations generated.")
            return updated_state
        except Exception as e:
            self.logger.error("GenreMoodAdvocate Node Error", error=str(e))
            # Add to reasoning log and error info
            if hasattr(state, 'reasoning_log'):
                state.reasoning_log.append(f"GenreMoodAgent: Error - {str(e)}")
                state.error_info = {"agent": "GenreMoodAgent", "message": str(e)}
            elif isinstance(state, dict):
                if 'reasoning_log' not in state:
                    state['reasoning_log'] = []
                state['reasoning_log'].append(f"GenreMoodAgent: Error - {str(e)}")
                state['error_info'] = {"agent": "GenreMoodAgent", "message": str(e)}
            return state
    
    async def _discovery_advocate_node_func(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        LangGraph node function for DiscoveryAgent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with discovery recommendations
        """
        self.logger.info("Executing Discovery Advocate Node")
        try:
            updated_state = await self.discovery_agent.process(state)
            # Add to reasoning log
            if hasattr(updated_state, 'reasoning_log'):
                updated_state.reasoning_log.append("DiscoveryAgent: Recommendations generated.")
            elif isinstance(updated_state, dict) and 'reasoning_log' in updated_state:
                updated_state['reasoning_log'].append("DiscoveryAgent: Recommendations generated.")
            return updated_state
        except Exception as e:
            self.logger.error("DiscoveryAdvocate Node Error", error=str(e))
            # Add to reasoning log and error info
            if hasattr(state, 'reasoning_log'):
                state.reasoning_log.append(f"DiscoveryAgent: Error - {str(e)}")
                state.error_info = {"agent": "DiscoveryAgent", "message": str(e)}
            elif isinstance(state, dict):
                if 'reasoning_log' not in state:
                    state['reasoning_log'] = []
                state['reasoning_log'].append(f"DiscoveryAgent: Error - {str(e)}")
                state['error_info'] = {"agent": "DiscoveryAgent", "message": str(e)}
            return state
    
    async def _judge_node_func(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        LangGraph node function for JudgeAgent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with final recommendations
        """
        self.logger.info("Executing Judge Node")
        try:
            updated_state = await self.judge_agent.evaluate_and_select(state)
            # Add to reasoning log
            if hasattr(updated_state, 'reasoning_log'):
                updated_state.reasoning_log.append("JudgeAgent: Final selection complete.")
            elif isinstance(updated_state, dict) and 'reasoning_log' in updated_state:
                updated_state['reasoning_log'].append("JudgeAgent: Final selection complete.")
            return updated_state
        except Exception as e:
            self.logger.error("Judge Node Error", error=str(e))
            # Add to reasoning log and error info
            if hasattr(state, 'reasoning_log'):
                state.reasoning_log.append(f"JudgeAgent: Error - {str(e)}")
                state.error_info = {"agent": "JudgeAgent", "message": str(e)}
            elif isinstance(state, dict):
                if 'reasoning_log' not in state:
                    state['reasoning_log'] = []
                state['reasoning_log'].append(f"JudgeAgent: Error - {str(e)}")
                state['error_info'] = {"agent": "JudgeAgent", "message": str(e)}
            return state
    
    def _should_proceed_after_planning(self, state: MusicRecommenderState) -> str:
        """
        Conditional edge function to determine next step after planning.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next step identifier: "execute_advocates" or "end_workflow"
        """
        # Handle both dict and object access
        error_info = None
        if hasattr(state, 'error_info'):
            error_info = state.error_info
        elif isinstance(state, dict) and 'error_info' in state:
            error_info = state['error_info']
            
        planning_strategy = None
        if hasattr(state, 'planning_strategy'):
            planning_strategy = state.planning_strategy
        elif isinstance(state, dict) and 'planning_strategy' in state:
            planning_strategy = state['planning_strategy']
        
        self.logger.debug(
            "Conditional Edge: Checking planner output", 
            error_info=error_info
        )
        
        # Check for planner errors
        if error_info and error_info.get("agent") == "PlannerAgent":
            self.logger.warning("Planner agent reported an error. Ending workflow.")
            # Add to reasoning log
            if hasattr(state, 'reasoning_log'):
                state.reasoning_log.append("Workflow: Planner encountered an error, ending.")
            elif isinstance(state, dict) and 'reasoning_log' in state:
                state['reasoning_log'].append("Workflow: Planner encountered an error, ending.")
            return "end_workflow"
        
        # Check for valid planning strategy
        if not planning_strategy:
            self.logger.warning("Planner did not produce any strategy. Ending workflow.")
            # Add to reasoning log
            if hasattr(state, 'reasoning_log'):
                state.reasoning_log.append("Workflow: Planner failed to produce strategy, ending.")
            elif isinstance(state, dict) and 'reasoning_log' in state:
                state['reasoning_log'].append("Workflow: Planner failed to produce strategy, ending.")
            return "end_workflow"
        
        # Check for evaluation framework (critical for judge)
        if not planning_strategy.get("evaluation_framework"):
            self.logger.warning("Planner strategy missing evaluation framework. Ending workflow.")
            # Add to reasoning log
            if hasattr(state, 'reasoning_log'):
                state.reasoning_log.append("Workflow: Planning strategy is missing evaluation framework, ending.")
            elif isinstance(state, dict) and 'reasoning_log' in state:
                state['reasoning_log'].append("Workflow: Planning strategy is missing evaluation framework, ending.")
            return "end_workflow"
        
        self.logger.info("Planning successful, proceeding to advocate agents.")
        return "execute_advocates"
    
    def _build_graph(self) -> Callable[[MusicRecommenderState], Union[MusicRecommenderState, asyncio.Future]]:
        """
        Build and compile the LangGraph workflow.
        
        Returns:
            Compiled graph function
        """
        workflow = StateGraph(MusicRecommenderState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node_func)
        workflow.add_node("genre_mood_advocate", self._genre_mood_advocate_node_func)
        workflow.add_node("discovery_advocate", self._discovery_advocate_node_func) 
        workflow.add_node("judge", self._judge_node_func)
        
        # Add a dummy node to handle fan-out after conditional
        workflow.add_node("fan_out_advocates", lambda x: x)  # Identity function
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add conditional edge from planner
        workflow.add_conditional_edges(
            "planner",
            self._should_proceed_after_planning,
            {
                "execute_advocates": "fan_out_advocates",
                "end_workflow": END
            }
        )
        
        # Fan out from the dummy node to both advocate agents
        workflow.add_edge("fan_out_advocates", "genre_mood_advocate")
        workflow.add_edge("fan_out_advocates", "discovery_advocate")
        
        # Both advocates converge to judge
        workflow.add_edge("genre_mood_advocate", "judge")
        workflow.add_edge("discovery_advocate", "judge")
        
        # Judge to end
        workflow.add_edge("judge", END)
        
        self.logger.info("LangGraph workflow built and compiled")
        return workflow.compile()
    
    async def process_query(self, user_query: str) -> MusicRecommenderState:
        """
        Process a user's music query through the entire agent workflow.
        
        Args:
            user_query: User's music recommendation request
            
        Returns:
            Final state with recommendations and reasoning
        """
        self.logger.info("Processing new user query", query=user_query)
        
        # Initialize state with processing start time
        start_time = time.time()
        initial_state = MusicRecommenderState(
            user_query=user_query,
            processing_start_time=start_time
        )
        
        # Execute graph
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            # Calculate processing time - handle both dict and object access
            processing_start = None
            if hasattr(final_state, 'processing_start_time'):
                processing_start = final_state.processing_start_time
            elif isinstance(final_state, dict) and 'processing_start_time' in final_state:
                processing_start = final_state['processing_start_time']
            
            if processing_start:
                total_time = time.time() - processing_start
                # Update total processing time
                if hasattr(final_state, 'total_processing_time'):
                    final_state.total_processing_time = total_time
                elif isinstance(final_state, dict):
                    final_state['total_processing_time'] = total_time
            
            # Get recommendation count for logging
            recommendations = []
            if hasattr(final_state, 'final_recommendations'):
                recommendations = final_state.final_recommendations
            elif isinstance(final_state, dict) and 'final_recommendations' in final_state:
                recommendations = final_state['final_recommendations']
            
            self.logger.info(
                "Query processing completed",
                total_time=total_time if processing_start else None,
                recommendation_count=len(recommendations) if recommendations else 0
            )
            
            # Convert back to MusicRecommenderState if needed
            if isinstance(final_state, dict):
                return MusicRecommenderState(**final_state)
            return final_state
            
        except Exception as e:
            self.logger.error("Graph execution failed", error=str(e))
            # Create error state
            error_state = MusicRecommenderState(user_query=user_query)
            error_state.error_info = {"agent": "RecommendationEngine", "message": str(e)}
            error_state.reasoning_log.append(f"Workflow ERROR: Graph execution failed - {str(e)}")
            return error_state

async def create_recommendation_engine(
    system_config: SystemConfig
) -> RecommendationEngine:
    """
    Factory function to create a fully configured RecommendationEngine with all agents.
    
    Args:
        system_config: System configuration containing API keys and agent configs
        
    Returns:
        Initialized RecommendationEngine with all agent dependencies
    """
    logger.info("Creating RecommendationEngine with agent dependencies")
    
    # Create API clients
    lastfm_client = LastFmClient(
        api_key=system_config.lastfm_api_key,
        rate_limit=system_config.lastfm_rate_limit
    )
    
    # Initialize agents with their configurations
    planner_agent = PlannerAgent(
        config=system_config.agent_configs.get("planner", AgentConfig(
            agent_name="PlannerAgent",
            agent_type="planner"
        ))
    )
    
    genre_mood_agent = GenreMoodAgent(
        config=system_config.agent_configs.get("genre_mood", AgentConfig(
            agent_name="GenreMoodAgent",
            agent_type="advocate"
        )),
        lastfm_client=lastfm_client
    )
    
    discovery_agent = DiscoveryAgent(
        config=system_config.agent_configs.get("discovery", AgentConfig(
            agent_name="DiscoveryAgent",
            agent_type="advocate"
        )),
        lastfm_client=lastfm_client
    )
    
    judge_agent = JudgeAgent()  # JudgeAgent has simpler initialization
    
    # Create and return the engine
    engine = RecommendationEngine(
        planner_agent=planner_agent,
        genre_mood_agent=genre_mood_agent,
        discovery_agent=discovery_agent,
        judge_agent=judge_agent
    )
    
    logger.info("RecommendationEngine created successfully")
    return engine 