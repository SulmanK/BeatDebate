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
    from ..agents.planner_agent import PlannerAgent
    from ..agents.genre_mood_agent import GenreMoodAgent
    from ..agents.discovery_agent import DiscoveryAgent
    from ..agents.judge_agent import JudgeAgent
    from .api_service import APIService, get_api_service
    from .smart_context_manager import SmartContextManager
    from .cache_manager import CacheManager, get_cache_manager
except ImportError:
    # Fallback imports for testing
    import sys
    sys.path.append('src')
    from models.agent_models import MusicRecommenderState, AgentConfig, SystemConfig
    from models.metadata_models import UnifiedTrackMetadata, MetadataSource
    from agents.planner_agent import PlannerAgent
    from agents.genre_mood_agent import GenreMoodAgent
    from agents.discovery_agent import DiscoveryAgent
    from agents.judge_agent import JudgeAgent
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
        """Initialize agents with shared API service."""
        if self._agents_initialized:
            return
        
        try:
            # Create agent configurations
            agent_config = AgentConfig(
                agent_name="default",
                agent_type="enhanced",
                model_name="gemini-1.5-flash",
                temperature=0.7,
                max_tokens=1000
            )
            
            # Get shared LastFM client
            lastfm_client = await self.api_service.get_lastfm_client()
            
            # Initialize agents with correct constructors
            self.planner_agent = PlannerAgent(
                config=agent_config,
                gemini_client=None  # TODO: Add Gemini client support
            )
            
            self.genre_mood_agent = GenreMoodAgent(
                config=agent_config,
                lastfm_client=lastfm_client
            )
            
            self.discovery_agent = DiscoveryAgent(
                config=agent_config,
                lastfm_client=lastfm_client
            )
            
            self.judge_agent = JudgeAgent(
                llm_client=None  # TODO: Add LLM client support
            )
            
            # Build workflow graph
            self.graph = self._build_workflow_graph()
            
            self._agents_initialized = True
            self.logger.info("Agents initialized with shared API service")
            
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
            
            # Convert recommendations to unified metadata
            unified_recommendations = await self._convert_to_unified_metadata(
                final_state.final_recommendations,
                include_audio_features=request.include_audio_features
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            response = RecommendationResponse(
                recommendations=unified_recommendations,
                strategy_used=getattr(final_state, 'strategy', {}),
                reasoning=getattr(final_state, 'reasoning_log', []),
                session_id=final_state.session_id,
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
        recommendations: List[Dict[str, Any]],
        include_audio_features: bool = True
    ) -> List[UnifiedTrackMetadata]:
        """
        Convert agent recommendations to unified metadata format.
        
        Args:
            recommendations: Raw recommendations from agents
            include_audio_features: Whether to include Spotify audio features
            
        Returns:
            List of unified track metadata
        """
        unified_tracks = []
        
        for rec in recommendations:
            try:
                # Extract basic track info
                artist = rec.get('artist', '')
                track = rec.get('track', '')
                
                if not artist or not track:
                    continue
                
                # Get comprehensive track info using API service
                unified_track = await self.api_service.get_unified_track_info(
                    artist=artist,
                    track=track,
                    include_audio_features=include_audio_features
                )
                
                if unified_track:
                    # Add recommendation-specific metadata
                    unified_track.recommendation_score = rec.get('score', 0.0)
                    unified_track.recommendation_reason = rec.get('reason', '')
                    unified_track.agent_source = rec.get('agent', 'unknown')
                    
                    unified_tracks.append(unified_track)
                    
            except Exception as e:
                self.logger.warning(
                    "Failed to convert recommendation to unified metadata",
                    recommendation=rec,
                    error=str(e)
                )
                continue
        
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