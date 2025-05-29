"""
Enhanced Recommendation Service

Streamlined recommendation service that uses the unified API service
and eliminates duplicate business logic patterns.
"""

import asyncio
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

import structlog
from langgraph.graph import StateGraph, END

# Handle imports gracefully
try:
    from ..models.agent_models import MusicRecommenderState, AgentConfig, SystemConfig
    from ..models.metadata_models import UnifiedTrackMetadata, MetadataSource
    from ..agents import PlannerAgent, GenreMoodAgent, DiscoveryAgent, JudgeAgent
    from .api_service import APIService, get_api_service
    from .smart_context_manager import SmartContextManager
    from .cache_manager import CacheManager, get_cache_manager
except ImportError:
    # Fallback imports for testing
    import sys
    sys.path.append('src')
    from models.agent_models import MusicRecommenderState, AgentConfig, SystemConfig
    from models.metadata_models import UnifiedTrackMetadata, MetadataSource
    from agents import PlannerAgent, GenreMoodAgent, DiscoveryAgent, JudgeAgent
    from services.api_service import APIService, get_api_service
    from services.smart_context_manager import SmartContextManager
    from services.cache_manager import CacheManager, get_cache_manager

logger = structlog.get_logger(__name__)


@dataclass
class RecommendationRequest:
    """Request for music recommendations."""
    query: str
    session_id: Optional[str] = None
    max_recommendations: int = 10
    include_audio_features: bool = True
    context: Optional[Dict[str, Any]] = None


@dataclass
class RecommendationResponse:
    """Response containing music recommendations."""
    recommendations: List[UnifiedTrackMetadata]
    strategy_used: Dict[str, Any]
    reasoning: List[str]
    session_id: str
    processing_time: float
    metadata: Dict[str, Any]


class EnhancedRecommendationService:
    """
    Enhanced recommendation service with unified API access.
    
    Key improvements:
    - Uses centralized APIService for all external API calls
    - Eliminates duplicate client instantiation across agents
    - Unified metadata handling with UnifiedTrackMetadata
    - Streamlined workflow with better error handling
    - Integrated caching and context management
    """
    
    def __init__(
        self,
        system_config: Optional[SystemConfig] = None,
        api_service: Optional[APIService] = None,
        cache_manager: Optional[CacheManager] = None,
        context_manager: Optional[SmartContextManager] = None
    ):
        """
        Initialize enhanced recommendation service.
        
        Args:
            system_config: System configuration
            api_service: API service instance (optional, will create if not provided)
            cache_manager: Cache manager instance (optional)
            context_manager: Context manager instance (optional)
        """
        self.logger = logger.bind(service="EnhancedRecommendationService")
        self.system_config = system_config
        
        # Initialize services
        self.cache_manager = cache_manager or get_cache_manager()
        self.api_service = api_service or get_api_service(
            cache_manager=self.cache_manager
        )
        self.context_manager = context_manager or SmartContextManager()
        
        # Initialize agents (will be created with shared API service)
        self._agents_initialized = False
        self.planner_agent: Optional[PlannerAgent] = None
        self.genre_mood_agent: Optional[GenreMoodAgent] = None
        self.discovery_agent: Optional[DiscoveryAgent] = None
        self.judge_agent: Optional[JudgeAgent] = None
        
        # Workflow graph
        self.graph: Optional[StateGraph] = None
        
        self.logger.info("Enhanced Recommendation Service initialized")
    
    async def initialize_agents(self):
        """Initialize agents with shared API service and rate limiting."""
        if self._agents_initialized:
            return
        
        try:
            # Create agent configurations
            agent_config = AgentConfig(
                agent_name="default",
                agent_type="enhanced",
                llm_model="gemini-2.0-flash-exp",
                temperature=0.7,
                max_tokens=1000
            )
            
            # Create Gemini client for LLM interactions
            gemini_api_key = os.getenv('GEMINI_API_KEY', 'demo_gemini_key')
            gemini_client = create_gemini_client(gemini_api_key)
            
            if not gemini_client:
                self.logger.warning("Failed to create Gemini client, agents will have limited functionality")
            
            # Create rate limiter for Gemini API (free tier: 10 requests per minute)
            try:
                from ..api.rate_limiter import UnifiedRateLimiter
                gemini_rate_limiter = UnifiedRateLimiter.for_gemini(calls_per_minute=8)  # Conservative limit
                self.logger.info("Gemini rate limiter created", calls_per_minute=8)
            except ImportError:
                from api.rate_limiter import UnifiedRateLimiter
                gemini_rate_limiter = UnifiedRateLimiter.for_gemini(calls_per_minute=8)
                self.logger.info("Gemini rate limiter created", calls_per_minute=8)
            
            # Get shared clients from API service
            lastfm_client = await self.api_service.get_lastfm_client()
            
            # Create metadata service with shared client
            from .metadata_service import MetadataService
            metadata_service = MetadataService(lastfm_client=lastfm_client)
            
            # Initialize agents with Gemini client and rate limiter
            self.planner_agent = PlannerAgent(
                config=agent_config,
                llm_client=gemini_client,
                api_service=self.api_service,
                metadata_service=metadata_service,
                rate_limiter=gemini_rate_limiter
            )
            
            self.genre_mood_agent = GenreMoodAgent(
                config=agent_config,
                llm_client=gemini_client,
                api_service=self.api_service,
                metadata_service=metadata_service,
                rate_limiter=gemini_rate_limiter
            )
            
            self.discovery_agent = DiscoveryAgent(
                config=agent_config,
                llm_client=gemini_client,
                api_service=self.api_service,
                metadata_service=metadata_service,
                rate_limiter=gemini_rate_limiter
            )
            
            self.judge_agent = JudgeAgent(
                config=agent_config,
                llm_client=gemini_client,
                api_service=self.api_service,
                metadata_service=metadata_service,
                rate_limiter=gemini_rate_limiter
            )
            
            # Build workflow graph
            self.graph = self._build_workflow_graph()
            
            self._agents_initialized = True
            self.logger.info("Agents initialized with shared API service and rate limiting")
            
        except Exception as e:
            self.logger.error("Failed to initialize agents", error=str(e))
            raise
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(MusicRecommenderState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("genre_mood_advocate", self._genre_mood_node)
        workflow.add_node("discovery_advocate", self._discovery_node)
        workflow.add_node("judge", self._judge_node)
        
        # Add edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "genre_mood_advocate")
        workflow.add_edge("planner", "discovery_advocate")
        workflow.add_edge("genre_mood_advocate", "judge")
        workflow.add_edge("discovery_advocate", "judge")
        workflow.add_edge("judge", END)
        
        return workflow.compile()
    
    async def _planner_node(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """Execute planner agent."""
        try:
            self.logger.info("Executing planner node")
            updated_state = await self.planner_agent.process(state)
            
            if hasattr(updated_state, 'reasoning_log'):
                updated_state.reasoning_log.append("Planner: Strategy created")
            
            return updated_state
            
        except Exception as e:
            self.logger.error("Planner node failed", error=str(e))
            if hasattr(state, 'reasoning_log'):
                state.reasoning_log.append(f"Planner: Error - {str(e)}")
            return state
    
    async def _genre_mood_node(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """Execute genre/mood advocate agent."""
        try:
            self.logger.info("Executing genre/mood advocate node")
            updated_state = await self.genre_mood_agent.process(state)
            
            if hasattr(updated_state, 'reasoning_log'):
                updated_state.reasoning_log.append("GenreMood: Recommendations generated")
            
            return updated_state
            
        except Exception as e:
            self.logger.error("Genre/mood node failed", error=str(e))
            if hasattr(state, 'reasoning_log'):
                state.reasoning_log.append(f"GenreMood: Error - {str(e)}")
            return state
    
    async def _discovery_node(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """Execute discovery advocate agent."""
        try:
            self.logger.info("Executing discovery advocate node")
            updated_state = await self.discovery_agent.process(state)
            
            if hasattr(updated_state, 'reasoning_log'):
                updated_state.reasoning_log.append("Discovery: Recommendations generated")
            
            return updated_state
            
        except Exception as e:
            self.logger.error("Discovery node failed", error=str(e))
            if hasattr(state, 'reasoning_log'):
                state.reasoning_log.append(f"Discovery: Error - {str(e)}")
            return state
    
    async def _judge_node(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """Execute judge agent."""
        try:
            self.logger.info("Executing judge node")
            updated_state = await self.judge_agent.evaluate_and_select(state)
            
            if hasattr(updated_state, 'reasoning_log'):
                updated_state.reasoning_log.append("Judge: Final recommendations selected")
            
            return updated_state
            
        except Exception as e:
            self.logger.error("Judge node failed", error=str(e))
            if hasattr(state, 'reasoning_log'):
                state.reasoning_log.append(f"Judge: Error - {str(e)}")
            return state
    
    async def get_recommendations(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """
        Get music recommendations using the enhanced workflow.
        
        Args:
            request: Recommendation request
            
        Returns:
            Recommendation response with unified metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        # Ensure agents are initialized
        await self.initialize_agents()
        
        # Analyze context decision
        context_decision = await self.context_manager.analyze_context_decision(
            current_query=request.query,
            session_id=request.session_id or "default"
        )
        
        # Create initial state
        initial_state = MusicRecommenderState(
            user_query=request.query,
            session_id=request.session_id or "default",
            max_recommendations=request.max_recommendations,
            reasoning_log=[],
            context_decision=context_decision
        )
        
        try:
            # Execute workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Debug: Check what's actually in the final state
            self.logger.debug(f"Final state type: {type(final_state)}")
            self.logger.debug(f"Final state attributes: {dir(final_state)}")
            self.logger.debug(f"Final state dict: {final_state.__dict__ if hasattr(final_state, '__dict__') else 'No __dict__'}")
            
            # Try multiple ways to access final_recommendations
            final_recommendations = None
            if hasattr(final_state, 'final_recommendations'):
                final_recommendations = final_state.final_recommendations
                self.logger.debug(f"Found final_recommendations via hasattr: {len(final_recommendations) if final_recommendations else 'None'}")
            else:
                self.logger.warning("final_recommendations attribute not found in final_state")
            
            if final_recommendations is None:
                final_recommendations = getattr(final_state, 'final_recommendations', [])
                self.logger.debug(f"Found final_recommendations via getattr: {len(final_recommendations)}")
            
            if not final_recommendations:
                # Fallback: check if recommendations are in other fields
                all_possible_recs = []
                for attr_name in ['final_recommendations', 'recommendations', 'genre_mood_recommendations', 'discovery_recommendations']:
                    attr_value = getattr(final_state, attr_name, None)
                    if attr_value:
                        self.logger.debug(f"Found {len(attr_value)} items in {attr_name}")
                        all_possible_recs.extend(attr_value)
                
                if all_possible_recs:
                    self.logger.warning(f"Using fallback recommendations from other fields: {len(all_possible_recs)} items")
                    final_recommendations = all_possible_recs
            
            # Convert recommendations to unified metadata
            unified_recommendations = await self._convert_to_unified_metadata(
                final_recommendations,
                include_audio_features=request.include_audio_features
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            response = RecommendationResponse(
                recommendations=unified_recommendations,
                strategy_used=getattr(final_state, 'strategy', {}),
                reasoning=getattr(final_state, 'reasoning_log', []),
                session_id=getattr(final_state, 'session_id', request.session_id or "default"),
                processing_time=processing_time,
                metadata={
                    "context_decision": context_decision,
                    "agents_used": ["planner", "genre_mood", "discovery", "judge"],
                    "total_candidates": len(getattr(final_state, 'all_recommendations', [])),
                    "final_count": len(unified_recommendations)
                }
            )
            
            # Update context after recommendation
            await self.context_manager.update_context_after_recommendation(
                session_id=response.session_id,
                query=request.query,
                llm_understanding=getattr(final_state, 'query_understanding', None),
                recommendations=[rec.to_dict() for rec in unified_recommendations],
                context_decision=context_decision
            )
            
            self.logger.info(
                "Recommendations generated successfully",
                query=request.query,
                count=len(unified_recommendations),
                processing_time=processing_time
            )
            
            return response
            
        except Exception as e:
            self.logger.error("Recommendation generation failed", error=str(e))
            
            # Return fallback recommendations
            fallback_recommendations = await self._get_fallback_recommendations(
                request.query,
                request.max_recommendations
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return RecommendationResponse(
                recommendations=fallback_recommendations,
                strategy_used={"type": "fallback", "reason": str(e)},
                reasoning=[f"Error occurred: {str(e)}", "Using fallback recommendations"],
                session_id=request.session_id or "default",
                processing_time=processing_time,
                metadata={"error": str(e), "fallback_used": True}
            )
    
    async def _convert_to_unified_metadata(
        self,
        recommendations: List[Union[Dict[str, Any], "TrackRecommendation"]],
        include_audio_features: bool = True
    ) -> List[UnifiedTrackMetadata]:
        """
        Convert agent recommendations to unified metadata format.
        
        Args:
            recommendations: Raw recommendations from agents (can be dicts or TrackRecommendation objects)
            include_audio_features: Whether to include Spotify audio features
            
        Returns:
            List of unified track metadata
        """
        unified_tracks = []
        
        for rec in recommendations:
            try:
                # Handle both TrackRecommendation objects and dictionaries
                if hasattr(rec, 'title'):  # TrackRecommendation object
                    artist = rec.artist
                    track = rec.title
                    confidence = rec.confidence
                    explanation = rec.explanation
                    source = rec.source
                elif isinstance(rec, dict):  # Dictionary format
                    artist = rec.get('artist', '')
                    track = rec.get('title', '') or rec.get('name', '')
                    confidence = rec.get('confidence', 0.0)
                    explanation = rec.get('explanation', '')
                    source = rec.get('source', 'unknown')
                else:
                    self.logger.warning(f"Unknown recommendation format: {type(rec)}")
                    continue
                
                if not artist or not track:
                    self.logger.debug(f"Skipping recommendation missing artist/title: artist='{artist}', track='{track}'")
                    continue
                
                # Get comprehensive track info using API service
                unified_track = await self.api_service.get_unified_track_info(
                    artist=artist,
                    track=track,
                    include_audio_features=include_audio_features
                )
                
                if unified_track:
                    # Add recommendation-specific metadata
                    unified_track.recommendation_score = confidence
                    unified_track.recommendation_reason = explanation
                    unified_track.agent_source = source
                    
                    unified_tracks.append(unified_track)
                    self.logger.debug(f"Successfully converted: {artist} - {track}")
                else:
                    self.logger.debug(f"Failed to get unified track info for {artist} - {track}")
                    
            except Exception as e:
                artist_name = getattr(rec, 'artist', rec.get('artist', 'unknown')) if hasattr(rec, 'artist') or isinstance(rec, dict) else 'unknown'
                track_name = getattr(rec, 'title', rec.get('title', rec.get('name', 'unknown'))) if hasattr(rec, 'title') or isinstance(rec, dict) else 'unknown'
                self.logger.warning(
                    "Failed to convert recommendation to unified metadata",
                    artist=artist_name,
                    title=track_name,
                    error=str(e)
                )
                continue
        
        self.logger.info(f"Converted {len(unified_tracks)} out of {len(recommendations)} recommendations to unified metadata")
        return unified_tracks
    
    async def _get_fallback_recommendations(
        self,
        query: str,
        max_recommendations: int
    ) -> List[UnifiedTrackMetadata]:
        """
        Get fallback recommendations using direct API service calls.
        
        Args:
            query: User query
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of fallback recommendations
        """
        try:
            # Use API service for direct search
            fallback_tracks = await self.api_service.search_unified_tracks(
                query=query,
                limit=max_recommendations,
                include_spotify=True
            )
            
            self.logger.info(
                "Fallback recommendations generated",
                query=query,
                count=len(fallback_tracks)
            )
            
            return fallback_tracks
            
        except Exception as e:
            self.logger.error("Fallback recommendations failed", error=str(e))
            return []
    
    async def get_similar_tracks(
        self,
        artist: str,
        track: str,
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """
        Get similar tracks using the API service.
        
        Args:
            artist: Artist name
            track: Track name
            limit: Maximum results
            
        Returns:
            List of similar tracks with unified metadata
        """
        return await self.api_service.get_similar_tracks(
            artist=artist,
            track=track,
            limit=limit,
            include_spotify_features=True
        )
    
    async def search_by_tags(
        self,
        tags: List[str],
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """
        Search tracks by tags using the API service.
        
        Args:
            tags: List of tags
            limit: Maximum results
            
        Returns:
            List of tracks matching tags
        """
        return await self.api_service.search_by_tags(tags=tags, limit=limit)
    
    async def get_planning_strategy(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get planning strategy from PlannerAgent without executing full workflow.
        
        Args:
            query: User query
            session_id: Session identifier
            
        Returns:
            Planning strategy dictionary
        """
        # Ensure agents are initialized
        await self.initialize_agents()
        
        # Create minimal state for planning
        initial_state = MusicRecommenderState(
            user_query=query,
            session_id=session_id or "default"
        )
        
        try:
            # Execute only the planner node
            planning_state = await self._planner_node(initial_state)
            
            return {
                "planning_strategy": getattr(planning_state, 'planning_strategy', {}),
                "query_understanding": getattr(planning_state, 'query_understanding', {}),
                "agent_coordination": getattr(planning_state, 'agent_coordination', {}),
                "session_id": planning_state.session_id
            }
            
        except Exception as e:
            self.logger.error("Planning strategy generation failed", error=str(e))
            return {
                "planning_strategy": {"type": "fallback", "reason": str(e)},
                "query_understanding": {"intent": "discovery", "confidence": 0.3},
                "agent_coordination": {"strategy": "simple"},
                "session_id": session_id or "default"
            }
    
    @property
    def smart_context_manager(self) -> SmartContextManager:
        """
        Access to smart context manager for backward compatibility.
        
        Returns:
            SmartContextManager instance
        """
        return self.context_manager
    
    async def close(self):
        """Close all service connections."""
        await self.api_service.close()
        if self.cache_manager:
            self.cache_manager.close()
        
        self.logger.info("Enhanced Recommendation Service closed")


# Global service instance
_global_recommendation_service: Optional[EnhancedRecommendationService] = None


def get_recommendation_service(
    system_config: Optional[SystemConfig] = None,
    api_service: Optional[APIService] = None,
    cache_manager: Optional[CacheManager] = None
) -> EnhancedRecommendationService:
    """
    Get global recommendation service instance.
    
    Args:
        system_config: System configuration (optional)
        api_service: API service instance (optional)
        cache_manager: Cache manager instance (optional)
        
    Returns:
        Global EnhancedRecommendationService instance
    """
    global _global_recommendation_service
    
    if _global_recommendation_service is None:
        _global_recommendation_service = EnhancedRecommendationService(
            system_config=system_config,
            api_service=api_service,
            cache_manager=cache_manager
        )
    
    return _global_recommendation_service


async def close_recommendation_service():
    """Close global recommendation service."""
    global _global_recommendation_service
    
    if _global_recommendation_service:
        await _global_recommendation_service.close()
        _global_recommendation_service = None


def create_gemini_client(api_key: str):
    """
    Create a Gemini client for LLM interactions.
    
    Args:
        api_key: Gemini API key
        
    Returns:
        Configured Gemini client
    """
    try:
        import google.generativeai as genai
        
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Create and return the model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        logger.info("Gemini client created successfully")
        return model
        
    except ImportError:
        logger.error("google-generativeai library not installed")
        return None
    except Exception as e:
        logger.error("Failed to create Gemini client", error=str(e))
        return None 