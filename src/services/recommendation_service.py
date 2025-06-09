"""
Recommendation Service

Streamlined recommendation service that delegates to specialized components.
This is the production version with improved modularity and maintainability.

Target: 72KB → ~30KB (60% reduction)
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re  # Add this import for regex pattern matching

import structlog
import uuid

# Handle imports gracefully
try:
    from ..models.agent_models import MusicRecommenderState, SystemConfig
    from ..models.metadata_models import UnifiedTrackMetadata
    from ..models.recommendation_models import TrackRecommendation
    from .api_service import APIService, get_api_service
    # SmartContextManager functionality moved to SessionManagerService
    from .session_manager_service import SessionManagerService
    from .intent_orchestration_service import IntentOrchestrationService
    from .cache_manager import CacheManager, get_cache_manager
    from .components.context_handler import ContextHandler
    from .components.agent_coordinator import AgentCoordinator
    from .components.workflow_orchestrator import WorkflowOrchestrator
    from .components.state_manager import StateManager
except ImportError:
    # Fallback imports for testing
    import sys
    sys.path.append('src')
    from models.agent_models import MusicRecommenderState, SystemConfig
    from models.metadata_models import UnifiedTrackMetadata
    from models.recommendation_models import TrackRecommendation
    from services.api_service import APIService, get_api_service
    # SmartContextManager functionality moved to SessionManagerService
    from services.session_manager_service import SessionManagerService
    from services.intent_orchestration_service import IntentOrchestrationService
    from services.cache_manager import CacheManager, get_cache_manager
    from services.components.context_handler import ContextHandler
    from services.components.agent_coordinator import AgentCoordinator
    from services.components.workflow_orchestrator import WorkflowOrchestrator
    from services.components.state_manager import StateManager

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


class RecommendationService:
    """
    Production recommendation service with modular architecture.
    
    Key improvements:
    - Delegates specialized tasks to focused components
    - Clear separation of concerns
    - Improved testability and maintainability
    - Reduced complexity through composition
    """
    
    def __init__(
        self,
        system_config: Optional[SystemConfig] = None,
        api_service: Optional[APIService] = None,
        cache_manager: Optional[CacheManager] = None,
        # context_manager parameter removed - functionality moved to session_manager
        session_manager: Optional[SessionManagerService] = None,
        intent_orchestrator: Optional[IntentOrchestrationService] = None
    ):
        """
        Initialize recommendation service with component architecture.
        
        Args:
            system_config: System configuration
            api_service: API service instance (optional, will create if not provided)
            cache_manager: Cache manager instance (optional)
            # context_manager: Functionality moved to session_manager
            session_manager: Session manager instance (optional)
            intent_orchestrator: Intent orchestrator instance (optional)
        """
        self.logger = logger.bind(service="RecommendationService")
        self.system_config = system_config
        
        # Initialize core services
        self.cache_manager = cache_manager or get_cache_manager()
        self.api_service = api_service or get_api_service(cache_manager=self.cache_manager)
        self.session_manager = session_manager or SessionManagerService(cache_manager=self.cache_manager)
        
        # Initialize intent orchestrator
        self._intent_orchestrator = intent_orchestrator
        
        # Initialize specialized components
        self.context_handler = ContextHandler(
            session_manager=self.session_manager,
            intent_orchestrator=self.session_manager  # Will be updated after agents init
        )
        
        self.agent_coordinator = AgentCoordinator(
            api_service=self.api_service,
            session_manager=self.session_manager
        )
        
        self.workflow_orchestrator = WorkflowOrchestrator(
            agent_coordinator=self.agent_coordinator
        )
        
        self.state_manager = StateManager(
            session_manager=self.session_manager
        )
        
        # Initialization status
        self._agents_initialized = False
        
        self.logger.info("Recommendation Service initialized")
    
    async def initialize_agents(self):
        """Initialize agents and complete component setup."""
        if self._agents_initialized:
            return
        
        try:
            # Initialize agents through coordinator
            success = await self.agent_coordinator.initialize_agents()
            if not success:
                raise RuntimeError("Failed to initialize agents")
            
            # Initialize context analyzer with LLM client
            gemini_client = self.agent_coordinator.get_gemini_client()
            rate_limiter = self.agent_coordinator.get_rate_limiter()
            
            if gemini_client:
                self.context_handler.initialize_context_analyzer(gemini_client, rate_limiter)
            
            # Initialize intent orchestrator if not provided
            if not self._intent_orchestrator and gemini_client:
                from ..agents.components.llm_utils import LLMUtils
                llm_utils = LLMUtils(gemini_client, rate_limiter)
                self._intent_orchestrator = IntentOrchestrationService(
                    session_manager=self.session_manager,
                    llm_utils=llm_utils
                )
            
            # Update context handler with intent orchestrator
            if self._intent_orchestrator:
                self.context_handler.intent_orchestrator = self._intent_orchestrator
            
            # Build workflow graph
            self.workflow_orchestrator.build_workflow_graph()
            
            self._agents_initialized = True
            self.logger.info("Agents and components initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize agents and components", error=str(e))
            raise
    
    async def get_recommendations(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """
        Get music recommendations using the enhanced modular workflow.
        
        Args:
            request: Recommendation request
            
        Returns:
            Recommendation response with unified metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        # Ensure agents are initialized
        await self.initialize_agents()
        
        try:
            # Step 0: Handle session ID generation and intent change detection
            session_id = await self._handle_session_management(request)
            
            # Step 1: Process conversation history
            conversation_history = await self.context_handler.process_conversation_history(request)
            
            # Step 2: Get session context
            session_context = await self.context_handler.get_session_context(
                session_id
            )
            
            # Step 3: Analyze context for follow-up detection
            context_override = await self.context_handler.analyze_context(
                request.query, 
                conversation_history, 
                session_id
            )
            
            self.logger.info(
                "Context analysis complete",
                followup_detected=context_override['is_followup'],
                target_entity=context_override['target_entity'],
                confidence=context_override['confidence']
            )
            
            # 🔧 NEW STEP 3.5: Use IntentOrchestrationService for proper LLM understanding
            # For fresh queries, we need LLM understanding to extract artists, genres, etc.
            # For follow-ups, we pass the context_override to prioritize follow-up context
            if self._intent_orchestrator:
                effective_intent = await self._intent_orchestrator.resolve_effective_intent(
                    current_query=request.query,
                    session_id=session_id,
                    llm_understanding=None,  # Will be resolved internally by orchestrator
                    context_override=context_override if context_override.get('is_followup') else None
                )
                
                # Only override context_override if it's NOT a followup (fresh queries need better extraction)
                # For followups, preserve the original context_override from context handler
                if effective_intent and not context_override.get('is_followup'):
                    # Fresh query: enhance with IntentOrchestrationService results
                    context_override.update({
                        'intent_override': effective_intent.get('intent'),
                        'entities': effective_intent.get('entities', {}),
                        'confidence': max(context_override.get('confidence', 0.0), effective_intent.get('confidence', 0.0)),
                        'is_followup': effective_intent.get('is_followup', context_override.get('is_followup', False)),
                        'followup_type': effective_intent.get('followup_type'),
                        'target_entity': effective_intent.get('target_entity') or context_override.get('target_entity')
                    })
                    
                    self.logger.info(
                        "🎯 Enhanced context with IntentOrchestrationService",
                        intent=effective_intent.get('intent'),
                        entities_count=len(effective_intent.get('entities', {})),
                        confidence=effective_intent.get('confidence'),
                        is_followup=effective_intent.get('is_followup')
                    )
                elif context_override.get('is_followup'):
                    # Follow-up query: preserve context_override, just log what orchestrator would have done
                    self.logger.info(
                        "🎯 Follow-up detected: preserving context_override from ContextHandler",
                        target_entity=context_override.get('target_entity'),
                        intent_override=context_override.get('intent_override'),
                        followup_type=context_override.get('followup_type'),
                        confidence=context_override.get('confidence')
                    )
            else:
                self.logger.warning("IntentOrchestrationService not available, using basic context analysis only")
            
            # Step 4: Extract recently shown tracks for follow-ups
            recently_shown_track_ids = self.context_handler.extract_recently_shown_tracks(
                conversation_history,
                context_override,
                None  # Will be populated after state creation if needed
            )
            
            if recently_shown_track_ids:
                self.logger.info(
                    f"Prepared {len(recently_shown_track_ids)} recently shown tracks to avoid duplicates"
                )
            
            # Step 5: Create workflow state
            workflow_state = self.state_manager.create_workflow_state(
                query=request.query,
                session_id=session_id,
                max_recommendations=request.max_recommendations or 10,
                context_override=context_override,
                session_context=session_context,
                recently_shown_track_ids=recently_shown_track_ids
            )
            
            # Step 6: Validate state
            self.state_manager.validate_state_for_workflow(workflow_state)
            
            # Step 7: Execute workflow
            final_state = await self.workflow_orchestrator.execute_workflow(workflow_state)
            
            # Step 8: Extract final recommendations
            final_recommendations = self.state_manager.extract_final_recommendations(final_state)
            
            # Step 9: Convert to unified metadata
            unified_recommendations = await self.state_manager.convert_to_unified_metadata(
                final_recommendations,
                include_audio_features=request.include_audio_features
            )
            
            # Step 10: Extract state fields for response
            state_fields = self.state_manager.extract_state_fields(final_state, session_id)
            
            # Step 11: Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Step 12: Create response
            response = RecommendationResponse(
                recommendations=unified_recommendations,
                strategy_used=state_fields['strategy_used'],
                reasoning=state_fields['reasoning_log'],
                session_id=state_fields['session_id'],
                processing_time=processing_time,
                metadata={
                    "context_decision": {
                        'is_followup': context_override['is_followup'],
                        'intent_override': context_override['intent_override'],
                        'target_entity': context_override['target_entity'],
                        'confidence': context_override['confidence'],
                    },
                    "workflow_execution": {
                        'conversation_history_length': len(conversation_history),
                        'recently_shown_tracks_count': len(recently_shown_track_ids),
                        'final_recommendations_count': len(unified_recommendations),
                        'processing_time_seconds': processing_time
                    },
                    "query_understanding": self._serialize_query_understanding(state_fields['query_understanding'])
                }
            )
            
            # Step 13: Save recommendations to history (optional, don't fail if it errors)
            try:
                # Convert unified metadata back to TrackRecommendation format for history
                track_recommendations = []
                for i, track in enumerate(unified_recommendations):
                    track_rec = TrackRecommendation(
                        title=track.name,  # FIXED: Use 'title' field name
                        artist=track.artist,  # FIXED: Use 'artist' field name  
                        id=f"{track.artist}_{track.name}_{i}".replace(' ', '_').lower(),  # FIXED: Add required 'id' field
                        source=getattr(track, 'agent_source', 'recommendation_service'),  # FIXED: Add required 'source' field
                        confidence=getattr(track, 'recommendation_score', 0.0),
                        explanation=getattr(track, 'recommendation_reason', '')
                    )
                    track_recommendations.append(track_rec)
                
                await self.state_manager.save_recommendations_to_history(
                    state_fields['session_id'],
                    track_recommendations
                )
                
                # 🔧 STEP 13.5: Update session with interaction data for follow-up detection
                # This is crucial for follow-up queries to work properly
                if self.session_manager:
                    await self.session_manager.create_or_update_session(
                        session_id=state_fields['session_id'],
                        query=request.query,
                        intent=context_override.get('intent_override', 'discovery'),
                        entities=context_override.get('entities', {}),
                        recommendations=unified_recommendations,
                        is_original_query=not context_override.get('is_followup', False)
                    )
                    self.logger.info(
                        "🎯 Session updated with interaction data",
                        session_id=state_fields['session_id'],
                        is_followup=context_override.get('is_followup', False),
                        intent=context_override.get('intent_override', 'discovery'),
                        entities_count=len(context_override.get('entities', {})),
                        recommendations_count=len(unified_recommendations)
                    )
                else:
                    self.logger.warning("Session manager not available for session update")
                
            except Exception as e:
                self.logger.warning(f"Failed to save recommendations to history: {e}")
            
            self.logger.info(
                "Recommendation request completed successfully",
                session_id=state_fields['session_id'],
                recommendations_count=len(unified_recommendations),
                processing_time=processing_time
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing recommendation request: {e}")
            
            # Return fallback recommendations
            try:
                fallback_recommendations = await self._get_fallback_recommendations(
                    request.query,
                    request.max_recommendations or 10
                )
                
                processing_time = asyncio.get_event_loop().time() - start_time
                
                return RecommendationResponse(
                    recommendations=fallback_recommendations,
                    strategy_used={"fallback": True, "error": str(e)},
                    reasoning=[f"Primary recommendation failed: {str(e)}", "Using fallback recommendations"],
                    session_id=session_id,
                    processing_time=processing_time,
                    metadata={"error": str(e), "fallback_used": True}
                )
                
            except Exception as fallback_error:
                self.logger.error(f"Fallback recommendations also failed: {fallback_error}")
                raise
    
    async def _handle_session_management(self, request: RecommendationRequest) -> str:
        """
        Handle session ID generation and intent change detection.
        
        Args:
            request: Recommendation request
            
        Returns:
            Session ID to use for this request
        """
        # If no session ID provided, generate a new one
        if not request.session_id:
            session_id = str(uuid.uuid4())
            self.logger.info("Generated new session ID", session_id=session_id)
            return session_id
        
        # If session ID provided, check if we should start a new session due to intent change
        try:
            # Get existing session context
            session_context = await self.session_manager.get_session_context(request.session_id)
            
            if not session_context:
                # Session doesn't exist, use provided session ID
                self.logger.info("Session not found, using provided session ID", session_id=request.session_id)
                return request.session_id
            
            # Check if this is an intent change that should trigger a new session
            if await self._should_start_new_session(request, session_context):
                new_session_id = str(uuid.uuid4())
                self.logger.info(
                    "Intent change detected, starting new session",
                    old_session_id=request.session_id,
                    new_session_id=new_session_id
                )
                return new_session_id
            
            # Continue with existing session
            self.logger.info("Continuing existing session", session_id=request.session_id)
            return request.session_id
            
        except Exception as e:
            self.logger.warning(f"Error in session management: {e}, using provided session ID")
            return request.session_id or str(uuid.uuid4())

    async def _should_start_new_session(self, request: RecommendationRequest, session_context: Dict[str, Any]) -> bool:
        """
        Determine if we should start a new session due to intent change.
        
        Args:
            request: Current request
            session_context: Existing session context
            
        Returns:
            True if a new session should be started
        """
        try:
            # Get the last interaction from session history
            interaction_history = session_context.get('interaction_history', [])
            if not interaction_history:
                return False

            last_interaction = interaction_history[-1]
            last_intent = last_interaction.get('intent', '')
            
            # 🔧 CRITICAL FIX: Check if this is a follow-up query first
            # Follow-up queries should NEVER trigger new sessions, even with intent changes
            followup_indicators = [
                'more tracks', 'more songs', 'more music', 'load more', 'show more',
                'next', 'continue', 'similar', 'like this', 'more like', 'keep going'
            ]
            
            query_lower = request.query.lower().strip()
            is_likely_followup = any(indicator in query_lower for indicator in followup_indicators)
            
            # Additional check: avoid detecting new artist queries as follow-ups
            # If query explicitly mentions a different artist than previous, it's NOT a follow-up
            if not is_likely_followup:
                # Extract entities from current query to detect new artist mentions
                current_artists, _ = self._extract_entities_from_query(request.query)
                last_entities = last_interaction.get('entities', {})
                last_artists = set()
                
                # Get previous artists from last interaction
                if 'target_artists' in last_entities:
                    last_artists.update([a.lower().strip() for a in last_entities['target_artists']])
                if 'target_artist' in last_entities:
                    last_artists.add(last_entities['target_artist'].lower().strip())
                
                # If current query mentions different artists, it's definitely NOT a follow-up
                if current_artists and last_artists:
                    artist_overlap = last_artists.intersection(current_artists)
                    if not artist_overlap:
                        self.logger.info(
                            "🎯 Different artist detected - treating as new primary query",
                            last_artists=list(last_artists),
                            current_artists=list(current_artists),
                            query=request.query
                        )
                        # Force new session for different artist queries
                        return True
            
            if is_likely_followup:
                self.logger.info(
                    "🔄 Follow-up query detected - preserving session",
                    query=request.query,
                    last_intent=last_intent,
                    session_id=request.session_id
                )
                return False  # Never create new session for follow-ups

            # Use intent orchestrator to analyze current query intent
            if self._intent_orchestrator:
                current_analysis = await self._intent_orchestrator.resolve_effective_intent(
                    current_query=request.query,
                    session_id=request.session_id,
                    llm_understanding=None
                )
                current_intent = current_analysis.get('intent', '') if current_analysis else ''
            else:
                # Fallback: assume no intent change
                return False
            
            # Check for significant intent changes (only for non-follow-up queries)
            intent_change_triggers = [
                # Artist to genre/mood change
                (last_intent in ['by_artist', 'by_artist_underground'] and current_intent in ['genre_mood', 'discovery']),
                # Genre/mood to artist change  
                (last_intent in ['genre_mood', 'discovery'] and current_intent in ['by_artist', 'by_artist_underground']),
                # Any change to/from contextual queries
                (last_intent == 'contextual' and current_intent != 'contextual'),
                (last_intent != 'contextual' and current_intent == 'contextual'),
                # Hybrid/similarity to artist-specific changes
                (last_intent in ['hybrid_similarity_genre', 'artist_similarity'] and current_intent in ['by_artist', 'by_artist_underground', 'artist_genre']),
                # Artist-specific to hybrid/similarity changes
                (last_intent in ['by_artist', 'by_artist_underground', 'artist_genre'] and current_intent in ['hybrid_similarity_genre', 'artist_similarity']),
                # Between hybrid and artist_genre (different focus)
                (last_intent == 'hybrid_similarity_genre' and current_intent == 'artist_genre'),
                (last_intent == 'artist_genre' and current_intent == 'hybrid_similarity_genre')
            ]
            
            should_start_new_intent = any(intent_change_triggers)
            
            # 🔧 NEW: Check for entity changes within the same intent type
            should_start_new_entities = await self._check_entity_changes(
                last_interaction, request.query, last_intent, current_intent
            )
            
            should_start_new = should_start_new_intent or should_start_new_entities
            
            if should_start_new:
                reason = "Intent change" if should_start_new_intent else "Entity change"
                self.logger.info(
                    f"{reason} detected - starting new session",
                    last_intent=last_intent,
                    current_intent=current_intent,
                    query=request.query[:50] + "..." if len(request.query) > 50 else request.query
                )
            
            return should_start_new
            
        except Exception as e:
            self.logger.warning(f"Error checking intent change: {e}")
            return False
    
    async def _check_entity_changes(
        self, 
        last_interaction: Dict[str, Any], 
        current_query: str, 
        last_intent: str, 
        current_intent: str
    ) -> bool:
        """
        Check if entities (artists, genres) have changed between queries.
        
        Args:
            last_interaction: Previous interaction data
            current_query: Current user query
            last_intent: Previous intent
            current_intent: Current intent
            
        Returns:
            True if entities have changed significantly
        """
        try:
            # Only check entity changes for same intent types that focus on specific entities
            entity_focused_intents = [
                'by_artist', 'by_artist_underground', 'artist_genre', 
                'hybrid_similarity_genre', 'artist_similarity'
            ]
            
            if last_intent not in entity_focused_intents or current_intent not in entity_focused_intents:
                return False
            
            # Extract entities from last interaction
            last_entities = last_interaction.get('entities', {})
            last_artists = set()
            last_genres = set()
            
            if 'target_artists' in last_entities:
                last_artists.update([a.lower().strip() for a in last_entities['target_artists']])
            if 'target_artist' in last_entities:
                last_artists.add(last_entities['target_artist'].lower().strip())
            if 'target_genres' in last_entities:
                last_genres.update([g.lower().strip() for g in last_entities['target_genres']])
            if 'target_genre' in last_entities:
                last_genres.add(last_entities['target_genre'].lower().strip())
            
            # Extract entities from current query using simple pattern matching
            current_artists, current_genres = self._extract_entities_from_query(current_query)
            
            # Check for artist changes
            if last_artists and current_artists:
                artist_overlap = last_artists.intersection(current_artists)
                if not artist_overlap:  # No common artists
                    self.logger.debug(
                        f"Artist change detected: {last_artists} -> {current_artists}"
                    )
                    return True
            
            # Check for genre changes (only if same intent and same artist)
            if (last_intent == current_intent and 
                last_artists and current_artists and 
                last_artists.intersection(current_artists) and  # Same artist
                last_genres and current_genres):
                
                genre_overlap = last_genres.intersection(current_genres)
                if not genre_overlap:  # No common genres
                    self.logger.debug(
                        f"Genre change detected: {last_genres} -> {current_genres}"
                    )
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking entity changes: {e}")
            return False
    
    def _extract_entities_from_query(self, query: str) -> tuple[set[str], set[str]]:
        """
        Extract artists and genres from a query using simple pattern matching.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (artists_set, genres_set)
        """
        query_lower = query.lower()
        artists = set()
        genres = set()
        
        # Artist patterns
        artist_patterns = [
            r'songs by ([^,\n]+?)(?:\s+that|\s+which|\s*$)',
            r'music (?:by|from) ([^,\n]+?)(?:\s+that|\s+which|\s*$)',
            r'tracks by ([^,\n]+?)(?:\s+that|\s+which|\s*$)',
            r'(?:music )?like ([^,\n]+?)(?:\s+but|\s+that|\s*$)',
            r'similar to ([^,\n]+?)(?:\s+but|\s+that|\s*$)',
        ]
        
        for pattern in artist_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                artist = match.strip()
                if artist and len(artist) > 1:  # Avoid single characters
                    artists.add(artist)
        
        # Genre patterns
        genre_patterns = [
            r'that (?:are|is) ([^,\n]+?)(?:\s*$)',
            r'which (?:are|is) ([^,\n]+?)(?:\s*$)',
            r'but ([^,\n]+?)(?:\s*$)',
        ]
        
        for pattern in genre_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                genre = match.strip()
                if genre and len(genre) > 1:  # Avoid single characters
                    genres.add(genre)
        
        return artists, genres
    
    async def _get_fallback_recommendations(
        self,
        query: str,
        max_recommendations: int
    ) -> List[UnifiedTrackMetadata]:
        """
        Get fallback recommendations when the main workflow fails.
        
        Args:
            query: User query
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of fallback recommendations
        """
        try:
            # Use LLM fallback service if available
            from ..llm_fallback_service import LLMFallbackService
            
            fallback_service = LLMFallbackService()
            fallback_response = await fallback_service.get_recommendations(query, max_recommendations)
            
            # Convert fallback response to UnifiedTrackMetadata
            unified_recommendations = []
            for track in fallback_response.get('recommendations', []):
                unified_track = UnifiedTrackMetadata(
                    name=track.get('title', 'Unknown Track'),  # FIXED: Use 'name' field for UnifiedTrackMetadata
                    artist=track.get('artist', 'Unknown Artist'),  # FIXED: Use 'artist' field for UnifiedTrackMetadata
                    album=track.get('album', ''),
                    recommendation_score=0.5,  # Lower confidence for fallback
                    recommendation_reason='Fallback recommendation',
                    agent_source='llm_fallback'
                )
                unified_recommendations.append(unified_track)
            
            self.logger.info(f"Generated {len(unified_recommendations)} fallback recommendations")
            return unified_recommendations
            
        except Exception as e:
            self.logger.error(f"Fallback recommendations failed: {e}")
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
            limit: Maximum number of tracks
            
        Returns:
            List of similar tracks
        """
        try:
            tracks = await self.api_service.get_similar_tracks(artist, track, limit)
            return tracks
        except Exception as e:
            self.logger.error(f"Error getting similar tracks: {e}")
            return []
    
    async def search_by_tags(
        self,
        tags: List[str],
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """
        Search tracks by tags using the API service.
        
        Args:
            tags: List of tags to search
            limit: Maximum number of tracks
            
        Returns:
            List of tracks matching tags
        """
        try:
            tracks = await self.api_service.search_by_tags(tags, limit)
            return tracks
        except Exception as e:
            self.logger.error(f"Error searching by tags: {e}")
            return []
    
    async def get_planning_strategy(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get planning strategy for a query without full execution.
        
        Args:
            query: User query
            session_id: Session identifier
            
        Returns:
            Planning strategy information
        """
        await self.initialize_agents()
        
        try:
            # Create minimal state for planning
            temp_state = MusicRecommenderState(
                user_query=query,
                max_recommendations=10,
                entities={},
                session_id=session_id or "default"
            )
            
            # Get planner agent and process
            planner_agent = self.agent_coordinator.get_planner_agent()
            if not planner_agent:
                raise ValueError("Planner agent not available")
            
            processed_state = await planner_agent.process(temp_state)
            strategy = getattr(processed_state, 'planning_strategy', {})
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error getting planning strategy: {e}")
            return {"error": str(e)}
    
    @property
    def smart_context_manager(self) -> SessionManagerService:
        """Access to the session manager (replaces smart context manager)."""
        return self.session_manager
    
    @property
    def intent_orchestrator(self) -> IntentOrchestrationService:
        """Access to the intent orchestrator."""
        if not self._intent_orchestrator:
            raise ValueError("Intent orchestrator not initialized. Call initialize_agents() first.")
        return self._intent_orchestrator
    
    async def close(self):
        """Clean up resources."""
        try:
            await self.agent_coordinator.close()
            if self.api_service and hasattr(self.api_service, 'close'):
                await self.api_service.close()
            self.logger.info("Recommendation Service closed")
        except Exception as e:
            self.logger.error(f"Error closing service: {e}")

    def _serialize_query_understanding(self, query_understanding):
        """Convert QueryUnderstanding object to JSON-serializable format."""
        if query_understanding is None:
            return None
        
        if isinstance(query_understanding, dict):
            return query_understanding
        
        # Handle QueryUnderstanding dataclass
        if hasattr(query_understanding, '__dict__'):
            result = {}
            for key, value in query_understanding.__dict__.items():
                if hasattr(value, 'value'):  # Handle enum values
                    result[key] = value.value
                elif isinstance(value, (list, dict, str, int, float, bool)):
                    result[key] = value
                else:
                    result[key] = str(value)
            return result
        
        # Fallback to string representation
        return str(query_understanding)


# Factory functions for backward compatibility

def get_recommendation_service(
    system_config: Optional[SystemConfig] = None,
    api_service: Optional[APIService] = None,
    cache_manager: Optional[CacheManager] = None
) -> RecommendationService:
    """
    Get a recommendation service instance.
    
    Args:
        system_config: System configuration
        api_service: API service instance
        cache_manager: Cache manager instance
        
    Returns:
        Recommendation service instance
    """
    return RecommendationService(
        system_config=system_config,
        api_service=api_service,
        cache_manager=cache_manager
    )


_service_instance: Optional[RecommendationService] = None


async def close_recommendation_service():
    """Close the global recommendation service instance."""
    global _service_instance
    if _service_instance:
        await _service_instance.close()
        _service_instance = None 