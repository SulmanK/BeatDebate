"""
Workflow Orchestrator for Enhanced Recommendation Service

Manages LangGraph workflow creation, routing, and execution.
Extracted from EnhancedRecommendationService to improve modularity and maintainability.
"""

from typing import Dict, Any, Optional
import structlog
from langgraph.graph import StateGraph, END

# Handle imports gracefully
try:
    from ...models.agent_models import MusicRecommenderState
    from .agent_coordinator import AgentCoordinator
except ImportError:
    # Fallback imports for testing
    import sys
    sys.path.append('src')
    from models.agent_models import MusicRecommenderState
    from services.components.agent_coordinator import AgentCoordinator

logger = structlog.get_logger(__name__)


class WorkflowOrchestrator:
    """
    Orchestrates the LangGraph workflow for music recommendation.
    
    Responsibilities:
    - Building the workflow graph
    - Routing between agents
    - Executing workflow nodes
    - Managing workflow state transitions
    """
    
    def __init__(self, agent_coordinator: AgentCoordinator):
        self.agent_coordinator = agent_coordinator
        self.logger = structlog.get_logger(__name__)
        self.graph: Optional[StateGraph] = None
    
    def build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow with conditional routing."""
        workflow = StateGraph(MusicRecommenderState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("genre_mood_advocate", self._genre_mood_node)
        workflow.add_node("discovery_advocate", self._discovery_node)
        workflow.add_node("judge", self._judge_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Intent-aware routing: Add conditional edges based on planner's agent sequence
        workflow.add_conditional_edges(
            "planner",
            self._route_agents,  # Router function that respects intent-aware sequences
            {
                "discovery_only": "discovery_advocate",
                "genre_mood_only": "genre_mood_advocate", 
                "both_agents": "discovery_advocate",  # Start with discovery, then genre_mood
                "judge_only": "judge"  # For edge cases
            }
        )
        
        # Add conditional edges from discovery to either genre_mood or judge
        workflow.add_conditional_edges(
            "discovery_advocate",
            self._route_after_discovery,
            {
                "to_genre_mood": "genre_mood_advocate",
                "to_judge": "judge"
            }
        )
        
        # Add edges from agents to judge
        workflow.add_edge("genre_mood_advocate", "judge")
        workflow.add_edge("judge", END)
        
        self.graph = workflow.compile()
        self.logger.info("Workflow graph built successfully")
        return self.graph
    
    def _route_agents(self, state: MusicRecommenderState) -> str:
        """
        Route to the appropriate agents based on planner's strategy.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node to execute
        """
        planning_strategy = getattr(state, 'planning_strategy', {})
        agent_sequence = planning_strategy.get('agent_sequence', ['discovery', 'genre_mood'])
        
        self.logger.debug(f"Routing agents based on sequence: {agent_sequence}")
        
        if not agent_sequence:
            self.logger.warning("No agent sequence found, defaulting to judge_only")
            return "judge_only"
        
        # Determine routing based on agent sequence
        if len(agent_sequence) == 1:
            if agent_sequence[0] == 'discovery_agent':
                return "discovery_only"
            elif agent_sequence[0] == 'genre_mood_agent':
                return "genre_mood_only"
            else:
                return "judge_only"
        elif len(agent_sequence) >= 2:
            # Check if it's a genre_mood + judge sequence (no discovery)
            if agent_sequence == ['genre_mood_agent', 'judge_agent']:
                return "genre_mood_only"
            # Otherwise, use both agents
            return "both_agents"
        else:
            return "judge_only"
    
    def _route_after_discovery(self, state: MusicRecommenderState) -> str:
        """
        Route after discovery agent execution.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node to execute
        """
        planning_strategy = getattr(state, 'planning_strategy', {})
        agent_sequence = planning_strategy.get('agent_sequence', ['discovery', 'genre_mood'])
        
        # If genre_mood_agent is in the sequence and we're coming from discovery, go to genre_mood
        if 'genre_mood_agent' in agent_sequence and len(agent_sequence) > 1:
            self.logger.debug("Routing from discovery to genre_mood")
            return "to_genre_mood"
        else:
            self.logger.debug("Routing from discovery directly to judge")
            return "to_judge"
    
    async def _planner_node(self, state: MusicRecommenderState) -> Dict[str, Any]:
        """
        Execute the planner agent node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state dictionary
        """
        try:
            planner_agent = self.agent_coordinator.get_planner_agent()
            if not planner_agent:
                raise ValueError("Planner agent not initialized")
            
            self.logger.info("Executing planner node")
            
            # Process with planner agent
            updated_state = await planner_agent.process(state)
            
            # Log planner results
            planning_strategy = getattr(updated_state, 'planning_strategy', {})
            self.logger.info(
                "Planner node completed",
                agent_sequence=planning_strategy.get('agent_sequence', []),
                intent=planning_strategy.get('intent', 'unknown')
            )
            
            return updated_state.__dict__ if hasattr(updated_state, '__dict__') else updated_state
            
        except Exception as e:
            self.logger.error(f"Error in planner node: {e}")
            # Return state with error information
            state.reasoning_log = getattr(state, 'reasoning_log', [])
            state.reasoning_log.append(f"Planner error: {str(e)}")
            return state.__dict__ if hasattr(state, '__dict__') else state
    
    async def _genre_mood_node(self, state: MusicRecommenderState) -> Dict[str, Any]:
        """
        Execute the genre mood agent node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state dictionary
        """
        try:
            genre_mood_agent = self.agent_coordinator.get_genre_mood_agent()
            if not genre_mood_agent:
                raise ValueError("Genre mood agent not initialized")
            
            self.logger.info("Executing genre mood node")
            
            # Process with genre mood agent
            updated_state = await genre_mood_agent.process(state)
            
            # Log results
            genre_mood_recs = getattr(updated_state, 'genre_mood_recommendations', [])
            self.logger.info(
                "Genre mood node completed",
                recommendations_count=len(genre_mood_recs)
            )
            
            return updated_state.__dict__ if hasattr(updated_state, '__dict__') else updated_state
            
        except Exception as e:
            self.logger.error(f"Error in genre mood node: {e}")
            # Return state with error information
            state.reasoning_log = getattr(state, 'reasoning_log', [])
            state.reasoning_log.append(f"Genre mood error: {str(e)}")
            return state.__dict__ if hasattr(state, '__dict__') else state
    
    async def _discovery_node(self, state: MusicRecommenderState) -> Dict[str, Any]:
        """
        Execute the discovery agent node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state dictionary
        """
        try:
            discovery_agent = self.agent_coordinator.get_discovery_agent()
            if not discovery_agent:
                raise ValueError("Discovery agent not initialized")
            
            self.logger.info("Executing discovery node")
            
            # Process with discovery agent
            updated_state = await discovery_agent.process(state)
            
            # Log results
            discovery_recs = getattr(updated_state, 'discovery_recommendations', [])
            self.logger.info(
                "Discovery node completed",
                recommendations_count=len(discovery_recs)
            )
            
            return updated_state.__dict__ if hasattr(updated_state, '__dict__') else updated_state
            
        except Exception as e:
            self.logger.error(f"Error in discovery node: {e}")
            # Return state with error information
            state.reasoning_log = getattr(state, 'reasoning_log', [])
            state.reasoning_log.append(f"Discovery error: {str(e)}")
            return state.__dict__ if hasattr(state, '__dict__') else state
    
    async def _judge_node(self, state: MusicRecommenderState) -> Dict[str, Any]:
        """
        Execute the judge agent node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state dictionary
        """
        try:
            judge_agent = self.agent_coordinator.get_judge_agent()
            if not judge_agent:
                raise ValueError("Judge agent not initialized")
            
            self.logger.info("Executing judge node")
            
            # Process with judge agent
            updated_state = await judge_agent.process(state)
            
            # Log results
            final_recs = getattr(updated_state, 'final_recommendations', [])
            self.logger.info(
                "Judge node completed",
                final_recommendations_count=len(final_recs)
            )
            
            return updated_state.__dict__ if hasattr(updated_state, '__dict__') else updated_state
            
        except Exception as e:
            self.logger.error(f"Error in judge node: {e}")
            # Return state with error information
            state.reasoning_log = getattr(state, 'reasoning_log', [])
            state.reasoning_log.append(f"Judge error: {str(e)}")
            return state.__dict__ if hasattr(state, '__dict__') else state
    
    async def execute_workflow(self, initial_state: MusicRecommenderState) -> Dict[str, Any]:
        """
        Execute the complete workflow.
        
        Args:
            initial_state: Initial workflow state
            
        Returns:
            Final workflow state
        """
        if not self.graph:
            raise ValueError("Workflow graph not built. Call build_workflow_graph() first.")
        
        self.logger.info("Starting workflow execution")
        
        try:
            # Execute workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            self.logger.info("Workflow execution completed successfully")
            return final_state
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise
    
    def get_workflow_graph(self) -> Optional[StateGraph]:
        """Get the compiled workflow graph."""
        return self.graph 