"""
Recommendation Engine Service for BeatDebate Multi-Agent System

This service orchestrates the flow of agents in the music recommendation system using LangGraph.
It manages the workflow from initial user query to final recommendations.
"""

import asyncio
import time
import os
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union

import structlog
from ..utils.logging_config import log_performance
from langgraph.graph import StateGraph, END
import google.generativeai as genai

from ..models.agent_models import MusicRecommenderState, AgentConfig, SystemConfig
from ..agents.planner_agent import PlannerAgent
from ..agents.genre_mood_agent import GenreMoodAgent
from ..agents.discovery_agent import DiscoveryAgent
from ..agents.judge_agent import JudgeAgent
from ..api.lastfm_client import LastFmClient
from .smart_context_manager import SmartContextManager

logger = structlog.get_logger(__name__)

T = TypeVar('T')


def create_gemini_client(api_key: str):
    """
    Create and configure a Gemini client.
    
    Args:
        api_key: Gemini API key
        
    Returns:
        Configured Gemini client
    """
    try:
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Create the model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        return model
        
    except ImportError:
        logger.error("google-generativeai package not installed")
        return None
    except Exception as e:
        logger.error("Failed to create Gemini client", error=str(e))
        return None


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
        judge_agent: JudgeAgent,
        lastfm_client: "LastFmClient"
    ):
        """
        Initialize the recommendation engine with agent instances.
        
        Args:
            planner_agent: Strategy planning agent
            genre_mood_agent: Genre and mood advocate agent
            discovery_agent: Discovery and novelty advocate agent
            judge_agent: Final selection judge agent
            lastfm_client: Last.fm API client
        """
        self.planner_agent = planner_agent
        self.genre_mood_agent = genre_mood_agent
        self.discovery_agent = discovery_agent
        self.judge_agent = judge_agent
        self.logger = logger.bind(service="RecommendationEngine")
        self.graph = self._build_graph()
        
        self.logger.info("RecommendationEngine initialized with LangGraph workflow")
        
        # Initialize LLM clients
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            self.logger.warning("GEMINI_API_KEY not found. LLM features will be limited.")
            
        try:
            genai.configure(api_key=gemini_api_key)
            self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
            self.logger.info("Gemini LLM client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            self.gemini_client = None
            
        # Initialize data clients
        self.lastfm_client = lastfm_client
        
        # Initialize smart context manager
        self.smart_context_manager = SmartContextManager()
    
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
            
            # Log performance metrics
            if processing_start:
                log_performance(
                    operation="full_recommendation_pipeline",
                    duration=total_time,
                    recommendation_count=len(recommendations) if recommendations else 0,
                    query_length=len(user_query),
                    has_recommendations=bool(recommendations)
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

    async def get_planning_strategy(
        self, 
        query: str, 
        session_id: Optional[str] = None
    ) -> "PlanningStrategy":
        """
        Get the planning strategy from PlannerAgent without executing full workflow.
        
        This method is useful for demonstrating the PlannerAgent's strategic
        thinking process in the UI.
        
        Args:
            query: User's music preference query
            session_id: Optional session identifier
            
        Returns:
            Planning strategy from PlannerAgent
        """
        self.logger.info("Generating planning strategy", query=query, session_id=session_id)
        
        # Create initial state
        initial_state = MusicRecommenderState(
            user_query=query,
            session_id=session_id or f"planning_{int(time.time())}"
        )
        
        try:
            # Execute only the planner agent
            updated_state = await self.planner_agent.process(initial_state)
            
            # Extract planning strategy
            planning_strategy = None
            if hasattr(updated_state, 'planning_strategy'):
                planning_strategy = updated_state.planning_strategy
            elif isinstance(updated_state, dict) and 'planning_strategy' in updated_state:
                planning_strategy = updated_state['planning_strategy']
            
            if not planning_strategy:
                raise ValueError("PlannerAgent failed to generate planning strategy")
            
            self.logger.info("Planning strategy generated successfully")
            return planning_strategy
            
        except Exception as e:
            self.logger.error("Planning strategy generation failed", error=str(e))
            raise

    async def get_recommendations(
        self,
        query: str,
        session_id: Optional[str] = None,
        max_recommendations: int = 10,
        chat_context: Optional[Dict] = None
    ) -> "RecommendationResponse":
        """
        Get music recommendations using the complete 4-agent workflow with smart context management.
        
        Args:
            query: User's music preference query
            session_id: Optional session identifier
            max_recommendations: Maximum number of recommendations to return
            chat_context: Previous chat context for continuity (legacy support)
            
        Returns:
            Complete recommendation response with tracks and explanations
        """
        from ..models.recommendation_models import RecommendationResponse, TrackRecommendation
        
        self.logger.info(
            "Getting recommendations with smart context", 
            query=query, 
            session_id=session_id,
            max_recommendations=max_recommendations
        )
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{int(time.time())}"
        
        try:
            # Get LLM understanding of the query if available
            llm_understanding = None
            if self.gemini_client:
                try:
                    llm_understanding = await self._get_llm_query_understanding(query)
                except Exception as e:
                    self.logger.warning(f"LLM understanding failed: {e}")
            
            # Analyze context decision using smart context manager
            context_decision = await self.smart_context_manager.analyze_context_decision(
                current_query=query,
                session_id=session_id,
                llm_understanding=llm_understanding
            )
            
            self.logger.info(
                f"Context decision: {context_decision['decision']} "
                f"(confidence: {context_decision['confidence']:.2f}) - "
                f"{context_decision['reasoning']}"
            )
            
            # Prepare enhanced query based on context decision
            enhanced_query = query
            conversation_context = context_decision.get("context_to_use")
            
            if conversation_context and context_decision["action"] != "reset_context":
                # Add context information to query processing
                previous_queries = []
                previous_recs = []
                
                interaction_history = conversation_context.get("interaction_history", [])
                if interaction_history:
                    previous_queries = [h.get("query", "") for h in interaction_history[-2:]]
                    previous_recs = [h.get("recommendations", [])[:1] for h in interaction_history[-2:]]
                
                # Legacy format for backward compatibility
                chat_context = {
                    "previous_queries": previous_queries,
                    "previous_recommendations": previous_recs,
                    "context_decision": context_decision,
                    "smart_context": True
                }
                
                if previous_queries:
                    context_info = f"Previous requests: {', '.join(previous_queries)}"
                    enhanced_query = f"{query} (Context: {context_info})"
                    self.logger.info(f"Enhanced query with smart context: {enhanced_query}")
            
            # Process query through full workflow
            final_state = await self.process_query(
                enhanced_query, 
                session_id, 
                llm_understanding,
                conversation_context
            )
            
            # Extract recommendations from final state
            recommendations = []
            if hasattr(final_state, 'final_recommendations'):
                recommendations = final_state.final_recommendations
            elif isinstance(final_state, dict) and 'final_recommendations' in final_state:
                recommendations = final_state['final_recommendations']
            
            # Limit to max_recommendations
            recommendations = recommendations[:max_recommendations]
            
            # Convert to TrackRecommendation objects if needed
            track_recommendations = []
            for rec in recommendations:
                if isinstance(rec, dict):
                    track_recommendations.append(TrackRecommendation(**rec))
                else:
                    track_recommendations.append(rec)
            
            # Update context after providing recommendations
            await self.smart_context_manager.update_context_after_recommendation(
                session_id=session_id,
                query=query,
                llm_understanding=llm_understanding,
                recommendations=[track.model_dump() for track in track_recommendations],
                context_decision=context_decision
            )
            
            # Get reasoning log
            reasoning_log = []
            if hasattr(final_state, 'reasoning_log'):
                reasoning_log = final_state.reasoning_log
            elif isinstance(final_state, dict) and 'reasoning_log' in final_state:
                reasoning_log = final_state['reasoning_log']
            
            # Add context reasoning to log
            reasoning_log.insert(0, f"Smart Context: {context_decision['reasoning']}")
            
            # Get processing time
            processing_time = None
            if hasattr(final_state, 'total_processing_time'):
                processing_time = final_state.total_processing_time
            elif isinstance(final_state, dict) and 'total_processing_time' in final_state:
                processing_time = final_state['total_processing_time']
            
            # Create response
            response = RecommendationResponse(
                recommendations=track_recommendations,
                reasoning_log=reasoning_log,
                session_id=session_id,
                response_time=processing_time or 0.0,
                agent_coordination_log=[
                    f"SmartContext: {context_decision['action']} (confidence: {context_decision['confidence']:.2f})",
                    "PlannerAgent: Strategic planning completed",
                    "GenreMoodAgent: Genre/mood recommendations generated", 
                    "DiscoveryAgent: Discovery recommendations generated",
                    "JudgeAgent: Final selection and ranking completed"
                ]
            )
            
            self.logger.info(
                "Recommendations generated successfully with smart context",
                recommendation_count=len(track_recommendations),
                processing_time=processing_time,
                context_decision=context_decision["decision"]
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Smart context recommendation failed: {e}")
            # Fallback to basic recommendations without context
            return await self._fallback_recommendations(query, session_id, max_recommendations)
    
    async def _get_llm_query_understanding(self, query: str) -> Optional[Dict]:
        """Get LLM understanding of the query for context analysis."""
        if not self.gemini_client:
            return None
        
        try:
            prompt = f"""
Analyze this music query and extract key information:

Query: "{query}"

Return JSON with:
{{
    "intent": {{"value": "artist_similarity|genre_exploration|mood_matching|activity_context|discovery"}},
    "artists": ["artist names mentioned"],
    "genres": ["genres mentioned"],
    "moods": ["moods/emotions mentioned"],
    "activities": ["activities mentioned"],
    "confidence": 0.0-1.0
}}
"""
            
            response = await self.gemini_client.generate_content_async(prompt)
            
            # Parse JSON response
            import json
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.split('```json')[1].split('```')[0]
            elif response_text.startswith('```'):
                response_text = response_text.split('```')[1].split('```')[0]
            
            return json.loads(response_text)
            
        except Exception as e:
            self.logger.warning(f"LLM query understanding failed: {e}")
            return None
    
    async def process_query(
        self, 
        query: str, 
        session_id: Optional[str] = None, 
        llm_understanding: Optional[Dict] = None,
        conversation_context: Optional[Dict] = None
    ) -> MusicRecommenderState:
        """
        Process query through the agent workflow with enhanced context.
        
        Args:
            query: User query
            session_id: Session identifier
            llm_understanding: LLM understanding of the query
            conversation_context: Conversation context from smart context manager
        """
        # Create initial state with enhanced context
        initial_state = MusicRecommenderState(
            user_query=query,
            session_id=session_id or f"session_{int(time.time())}",
            query_understanding=llm_understanding,
            conversation_context=conversation_context
        )
        
        # Execute workflow
        try:
            final_state = await self.graph.ainvoke(initial_state)
            return final_state
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def _fallback_recommendations(
        self, 
        query: str, 
        session_id: str, 
        max_recommendations: int
    ) -> "RecommendationResponse":
        """Fallback recommendations when smart context fails."""
        from ..models.recommendation_models import RecommendationResponse, TrackRecommendation
        
        self.logger.warning("Using fallback recommendations")
        
        # Simple fallback recommendations
        demo_tracks = [
            TrackRecommendation(
                title="Fallback Track 1",
                artist="Demo Artist",
                id="fallback_1",
                source="fallback",
                explanation="Fallback recommendation due to context processing error",
                confidence=0.5
            )
        ]
        
        return RecommendationResponse(
            recommendations=demo_tracks,
            reasoning_log=[
                "Smart context processing failed, using fallback",
                "Demo: Basic recommendations provided"
            ],
            agent_coordination_log=[
                "Fallback: Limited functionality due to error"
            ],
            session_id=session_id,
            response_time=0.1
        )

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
    
    # Create Gemini client
    gemini_client = create_gemini_client(system_config.gemini_api_key)
    if not gemini_client:
        logger.warning("Gemini client creation failed, agents will use fallback strategies")
    
    # Initialize agents with their configurations and clients
    planner_agent = PlannerAgent(
        config=system_config.agent_configs.get("planner", AgentConfig(
            agent_name="PlannerAgent",
            agent_type="planner"
        )),
        gemini_client=gemini_client
    )
    
    genre_mood_agent = GenreMoodAgent(
        config=system_config.agent_configs.get("genre_mood", AgentConfig(
            agent_name="GenreMoodAgent",
            agent_type="advocate"
        )),
        lastfm_client=lastfm_client,
        gemini_client=gemini_client
    )
    
    discovery_agent = DiscoveryAgent(
        config=system_config.agent_configs.get("discovery", AgentConfig(
            agent_name="DiscoveryAgent",
            agent_type="advocate"
        )),
        lastfm_client=lastfm_client,
        gemini_client=gemini_client
    )
    
    judge_agent = JudgeAgent(llm_client=gemini_client)  # Pass Gemini client to JudgeAgent too
    
    # Create and return the engine
    engine = RecommendationEngine(
        planner_agent=planner_agent,
        genre_mood_agent=genre_mood_agent,
        discovery_agent=discovery_agent,
        judge_agent=judge_agent,
        lastfm_client=lastfm_client
    )
    
    logger.info("RecommendationEngine created successfully")
    return engine 