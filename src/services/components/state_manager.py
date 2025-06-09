"""
State Manager for Enhanced Recommendation Service

Manages MusicRecommenderState creation, validation, and transformation.
Extracted from EnhancedRecommendationService to improve modularity and maintainability.
"""

from typing import Dict, List, Any, Optional, Union
import structlog

# Handle imports gracefully
try:
    from ...models.agent_models import MusicRecommenderState
    from ...models.metadata_models import UnifiedTrackMetadata
    from ...models.recommendation_models import TrackRecommendation
    from ..session_manager_service import SessionManagerService
except ImportError:
    # Fallback imports for testing
    import sys
    sys.path.append('src')
    from models.agent_models import MusicRecommenderState
    from models.metadata_models import UnifiedTrackMetadata
    from models.recommendation_models import TrackRecommendation
    from services.session_manager_service import SessionManagerService

logger = structlog.get_logger(__name__)


class StateManager:
    """
    Manages workflow state creation, validation, and transformation.
    
    Responsibilities:
    - Creating MusicRecommenderState from requests
    - Validating state for workflow execution
    - Converting final state to response format
    - Managing state transitions and updates
    """
    
    def __init__(self, session_manager: SessionManagerService):
        self.session_manager = session_manager
        self.logger = structlog.get_logger(__name__)
    
    def create_workflow_state(
        self,
        query: str,
        session_id: str,
        max_recommendations: int,
        context_override: Dict[str, Any],
        session_context: Dict[str, Any],
        recently_shown_track_ids: List[str]
    ) -> MusicRecommenderState:
        """
        Create a MusicRecommenderState from request parameters.
        
        Args:
            query: User query
            session_id: Session identifier
            max_recommendations: Maximum number of recommendations
            context_override: Context analysis results
            session_context: Session context data
            recently_shown_track_ids: Track IDs to avoid
            
        Returns:
            Initialized MusicRecommenderState
        """
        # Extract entities from context override
        entities = context_override.get('entities', {}) if context_override else {}
        
        # 🔧 NEW: Create effective_intent for judge agent
        # The judge agent's candidate_selector expects state.effective_intent with followup_type
        effective_intent = None
        if context_override:
            effective_intent = {
                'intent': context_override.get('intent_override'),
                'is_followup': context_override.get('is_followup', False),
                'entities': entities,
                'confidence': context_override.get('confidence', 0.8)
            }
            # 🎯 CRITICAL: Include followup_type for pool retrieval
            if context_override.get('is_followup'):
                # Map the context handler's followup types to what candidate_selector expects
                followup_mapping = {
                    'artist_deep_dive': 'artist_deep_dive',
                    'load_more': 'load_more',
                    'style_continuation': 'artist_deep_dive',  # Treat as artist deep dive for pool retrieval
                    'artist_style_refinement': 'artist_deep_dive'  # Treat as artist deep dive for pool retrieval
                }
                original_followup_type = context_override.get('followup_type', 'artist_deep_dive')
                mapped_followup_type = followup_mapping.get(original_followup_type, 'artist_deep_dive')
                effective_intent['followup_type'] = mapped_followup_type
                
                self.logger.info(
                    "🔄 Created effective_intent for follow-up query",
                    original_followup_type=original_followup_type,
                    mapped_followup_type=mapped_followup_type,
                    intent=effective_intent['intent']
                )
        
        # Create workflow state
        workflow_state = MusicRecommenderState(
            user_query=query,
            max_recommendations=max_recommendations,
            entities=entities,
            conversation_context=session_context,
            context_override=context_override,
            session_id=session_id,
            recently_shown_track_ids=recently_shown_track_ids
        )
        
        # 🔧 NEW: Add effective_intent to state for judge agent
        if effective_intent:
            workflow_state.effective_intent = effective_intent
        
        self.logger.info(
            "Workflow state created",
            session_id=session_id,
            query_length=len(query),
            entities_count=len(entities),
            recently_shown_count=len(recently_shown_track_ids)
        )
        
        return workflow_state
    
    def validate_state_for_workflow(self, state: MusicRecommenderState) -> None:
        """
        Validate state before workflow execution.
        
        Args:
            state: Workflow state to validate
            
        Raises:
            ValueError: If state is invalid
        """
        if not state:
            raise ValueError("Workflow state is None")
        
        if not hasattr(state, 'user_query') or not state.user_query:
            raise ValueError("Workflow state missing user_query")
        
        if not hasattr(state, 'session_id') or not state.session_id:
            raise ValueError("Workflow state missing session_id")
        
        if not hasattr(state, 'max_recommendations'):
            raise ValueError("Workflow state missing max_recommendations")
        
        # Ensure required attributes exist with defaults
        if not hasattr(state, 'reasoning_log'):
            state.reasoning_log = []
        
        if not hasattr(state, 'recently_shown_track_ids'):
            state.recently_shown_track_ids = []
        
        if not hasattr(state, 'entities'):
            state.entities = {}
        
        self.logger.debug("State validation passed")
    
    def extract_final_recommendations(self, final_state) -> List:
        """
        Extract final recommendations from workflow state.
        
        Args:
            final_state: Final workflow state (dict or object)
            
        Returns:
            List of final recommendations
        """
        final_recommendations = []
        
        # Try multiple ways to access final_recommendations
        if isinstance(final_state, dict):
            final_recommendations = final_state.get('final_recommendations', [])
            self.logger.debug(f"Found final_recommendations in dict: {len(final_recommendations)} items")
        elif hasattr(final_state, 'final_recommendations'):
            final_recommendations = final_state.final_recommendations
            self.logger.debug(f"Found final_recommendations via hasattr: {len(final_recommendations) if final_recommendations else 'None'}")
        else:
            self.logger.warning("final_recommendations not found in final_state")
        
        # Fallback: check if recommendations are in other fields
        if not final_recommendations:
            all_possible_recs = []
            for attr_name in ['final_recommendations', 'recommendations', 'genre_mood_recommendations', 'discovery_recommendations']:
                if isinstance(final_state, dict):
                    attr_value = final_state.get(attr_name)
                else:
                    attr_value = getattr(final_state, attr_name, None)
                
                if attr_value:
                    self.logger.debug(f"Found {len(attr_value)} items in {attr_name}")
                    all_possible_recs.extend(attr_value)
            
            if all_possible_recs:
                # Limit fallback to configured maximum (typically 20)
                max_recommendations = getattr(final_state, 'max_recommendations', 20)
                limited_recs = all_possible_recs[:max_recommendations]
                self.logger.warning(f"Using fallback recommendations from other fields: {len(limited_recs)}/{len(all_possible_recs)} items (limited to {max_recommendations})")
                final_recommendations = limited_recs
        
        return final_recommendations or []
    
    def extract_state_fields(self, final_state, request_session_id: str) -> Dict[str, Any]:
        """
        Extract relevant fields from final workflow state.
        
        Args:
            final_state: Final workflow state (dict or object)
            request_session_id: Original request session ID
            
        Returns:
            Dictionary with extracted state fields
        """
        # Extract fields from final_state (handle both dict and object)
        if isinstance(final_state, dict):
            strategy_used = final_state.get('planning_strategy', {})
            reasoning_log = final_state.get('reasoning_log', [])
            session_id = final_state.get('session_id', request_session_id)
            query_understanding = final_state.get('query_understanding', None)
        else:
            strategy_used = getattr(final_state, 'planning_strategy', {})
            reasoning_log = getattr(final_state, 'reasoning_log', [])
            session_id = getattr(final_state, 'session_id', request_session_id)
            query_understanding = getattr(final_state, 'query_understanding', None)
        
        return {
            'strategy_used': strategy_used,
            'reasoning_log': reasoning_log,
            'session_id': session_id,
            'query_understanding': query_understanding
        }
    
    async def convert_to_unified_metadata(
        self,
        recommendations: List[Union[Dict[str, Any], TrackRecommendation]],
        include_audio_features: bool = True
    ) -> List[UnifiedTrackMetadata]:
        """
        Convert recommendations to unified metadata format.
        
        Args:
            recommendations: List of recommendations to convert
            include_audio_features: Whether to include audio features
            
        Returns:
            List of UnifiedTrackMetadata objects
        """
        if not recommendations:
            self.logger.warning("No recommendations to convert")
            return []
        
        unified_recommendations = []
        
        for rec in recommendations:
            try:
                if isinstance(rec, dict):
                    # Convert dict to UnifiedTrackMetadata
                    unified_track = self._dict_to_unified_metadata(rec, include_audio_features)
                elif hasattr(rec, '__dict__'):
                    # Convert object to UnifiedTrackMetadata
                    unified_track = self._object_to_unified_metadata(rec, include_audio_features)
                else:
                    self.logger.warning(f"Unknown recommendation type: {type(rec)}")
                    continue
                
                if unified_track:
                    unified_recommendations.append(unified_track)
                    
            except Exception as e:
                self.logger.error(f"Error converting recommendation: {e}")
                continue
        
        self.logger.info(f"Converted {len(unified_recommendations)} recommendations to unified metadata")
        return unified_recommendations
    
    def _dict_to_unified_metadata(self, rec_dict: Dict[str, Any], include_audio_features: bool) -> Optional[UnifiedTrackMetadata]:
        """Convert a dictionary recommendation to UnifiedTrackMetadata."""
        try:
            # Extract basic track information
            track_name = rec_dict.get('track', rec_dict.get('track_name', rec_dict.get('name', '')))
            artist_name = rec_dict.get('artist', rec_dict.get('artist_name', ''))
            album_name = rec_dict.get('album', rec_dict.get('album_name', ''))
            
            if not track_name or not artist_name:
                self.logger.warning(f"Missing required fields in recommendation: {rec_dict}")
                return None
            
            # Create UnifiedTrackMetadata
            unified_track = UnifiedTrackMetadata(
                name=track_name,
                artist=artist_name,
                album=album_name,
                duration_ms=rec_dict.get('duration', 0),
                external_urls={
                    'lastfm': rec_dict.get('lastfm_url', rec_dict.get('url', '')),
                    'spotify': rec_dict.get('spotify_url', '')
                },
                preview_url=rec_dict.get('preview_url', ''),
                genres=rec_dict.get('genres', []),
                tags=rec_dict.get('tags', []),
                popularity=rec_dict.get('popularity', 0),
                recommendation_score=rec_dict.get('confidence', 0.0),
                recommendation_reason=rec_dict.get('reasoning', ''),
                agent_source=rec_dict.get('metadata_source', 'unknown')
            )
            
            # Add audio features if requested and available
            if include_audio_features:
                audio_features = rec_dict.get('audio_features', {})
                if audio_features and isinstance(audio_features, dict):
                    unified_track.audio_features = audio_features
            
            return unified_track
            
        except Exception as e:
            self.logger.error(f"Error converting dict to unified metadata: {e}")
            return None
    
    def _object_to_unified_metadata(self, rec_obj, include_audio_features: bool) -> Optional[UnifiedTrackMetadata]:
        """Convert an object recommendation to UnifiedTrackMetadata."""
        try:
            # If it's already UnifiedTrackMetadata, return as-is
            if isinstance(rec_obj, UnifiedTrackMetadata):
                return rec_obj
            
            # If it's TrackRecommendation, convert it
            if isinstance(rec_obj, TrackRecommendation):
                return UnifiedTrackMetadata(
                    name=rec_obj.title,
                    artist=rec_obj.artist,
                    album=getattr(rec_obj, 'album', ''),
                    duration_ms=getattr(rec_obj, 'duration', 0),
                    external_urls={
                        'lastfm': getattr(rec_obj, 'url', ''),
                        'spotify': getattr(rec_obj, 'spotify_url', '')
                    },
                    preview_url=getattr(rec_obj, 'preview_url', ''),
                    genres=getattr(rec_obj, 'genres', []),
                    tags=getattr(rec_obj, 'tags', []),
                    popularity=getattr(rec_obj, 'popularity', 0),
                    recommendation_score=getattr(rec_obj, 'confidence', 0.0),
                    recommendation_reason=getattr(rec_obj, 'explanation', ''),
                    agent_source=getattr(rec_obj, 'source', 'unknown'),
                    audio_features=getattr(rec_obj, 'audio_features', {}) if include_audio_features else None
                )
            
            # For other objects, extract attributes
            track_name = getattr(rec_obj, 'title', getattr(rec_obj, 'name', getattr(rec_obj, 'track', getattr(rec_obj, 'track_name', ''))))
            artist_name = getattr(rec_obj, 'artist', getattr(rec_obj, 'artist_name', ''))
            
            if not track_name or not artist_name:
                self.logger.warning(f"Missing required fields in recommendation object: {rec_obj}")
                return None
            
            unified_track = UnifiedTrackMetadata(
                name=track_name,
                artist=artist_name,
                album=getattr(rec_obj, 'album_name', getattr(rec_obj, 'album', '')),
                duration_ms=getattr(rec_obj, 'duration', 0),
                external_urls={
                    'lastfm': getattr(rec_obj, 'lastfm_url', getattr(rec_obj, 'url', '')),
                    'spotify': getattr(rec_obj, 'spotify_url', '')
                },
                preview_url=getattr(rec_obj, 'preview_url', ''),
                genres=getattr(rec_obj, 'genres', []),
                tags=getattr(rec_obj, 'tags', []),
                popularity=getattr(rec_obj, 'popularity', 0),
                recommendation_score=getattr(rec_obj, 'confidence', 0.0),
                recommendation_reason=getattr(rec_obj, 'reasoning', ''),
                agent_source=getattr(rec_obj, 'metadata_source', 'unknown')
            )
            
            # Add audio features if requested and available
            if include_audio_features:
                audio_features = getattr(rec_obj, 'audio_features', {})
                if audio_features:
                    unified_track.audio_features = audio_features
            
            return unified_track
            
        except Exception as e:
            self.logger.error(f"Error converting object to unified metadata: {e}")
            return None
    
    async def save_recommendations_to_history(
        self, 
        session_id: str, 
        recommendations: List[TrackRecommendation]
    ) -> None:
        """
        Save recommendations to session history.
        
        Args:
            session_id: Session identifier
            recommendations: List of recommendations to save
        """
        try:
            # Convert recommendations to serializable format
            rec_data = []
            for rec in recommendations:
                if isinstance(rec, dict):
                    rec_data.append(rec)
                elif hasattr(rec, '__dict__'):
                    rec_data.append(rec.__dict__)
                else:
                    # Try to convert to dict
                    try:
                        # Fix: Use correct field names for both TrackRecommendation and UnifiedTrackMetadata
                        # For UnifiedTrackMetadata: use 'name' field, for TrackRecommendation: use 'title' field
                        if hasattr(rec, 'name'):  # UnifiedTrackMetadata
                            track_name = getattr(rec, 'name', '')
                        else:  # TrackRecommendation or other objects
                            track_name = getattr(rec, 'title', getattr(rec, 'track_name', ''))
                        
                        artist_name = getattr(rec, 'artist', getattr(rec, 'artist_name', ''))
                        rec_dict = {
                            'track_name': track_name,
                            'artist_name': artist_name,
                            'confidence': getattr(rec, 'confidence', 0.0)
                        }
                        rec_data.append(rec_dict)
                    except Exception as e:
                        self.logger.warning(f"Could not serialize recommendation: {e}")
                        continue
            
            # Save to session manager
            await self.session_manager.save_recommendations(session_id, rec_data)
            self.logger.info(f"Saved {len(rec_data)} recommendations to session history")
            
        except Exception as e:
            self.logger.error(f"Error saving recommendations to history: {e}")
            # Don't fail the whole request if history saving fails 