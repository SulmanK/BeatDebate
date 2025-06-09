"""
Candidate Selector for Judge Agent

Handles candidate collection, filtering, and initial processing
for the judge agent's recommendation evaluation.

Extracted from judge/agent.py for better modularity.
"""

from typing import Dict, List, Any, Tuple, Optional
import structlog

from src.models.recommendation_models import TrackRecommendation

logger = structlog.get_logger(__name__)


class CandidateSelector:
    """
    Candidate selector for judge agent.
    
    Responsibilities:
    - Collect candidates from all agent sources
    - Filter out recently shown tracks
    - Retrieve candidates from persisted pools (follow-ups)
    - Apply source priority filtering
    - Prepare candidates for scoring and ranking
    """
    
    def __init__(self, session_manager=None):
        """
        Initialize candidate selector.
        
        Args:
            session_manager: SessionManagerService for candidate pool retrieval
        """
        self.session_manager = session_manager
        self.logger = structlog.get_logger(__name__)
    
    async def collect_and_filter_candidates(
        self,
        state,
        max_candidates: int = 50
    ) -> List[TrackRecommendation]:
        """
        Collect and filter candidates from all sources.
        
        Args:
            state: Current workflow state
            max_candidates: Maximum number of candidates for follow-ups
            
        Returns:
            List of filtered candidates ready for scoring
        """
        try:
            # First, try to get candidates from persisted pool for follow-ups
            persisted_candidates = await self._get_candidates_from_persisted_pool(
                state, max_candidates
            )
            
            if persisted_candidates:
                self.logger.info(f"Using {len(persisted_candidates)} candidates from persisted pool")
                # Still apply recently shown filtering
                return self._filter_out_recently_shown(persisted_candidates, state)
            
            # Otherwise, collect candidates from agent recommendations
            all_candidates = self._collect_all_candidates(state)
            
            # Filter out recently shown tracks
            filtered_candidates = self._filter_out_recently_shown(all_candidates, state)
            
            self.logger.info(
                f"Collected {len(all_candidates)} candidates, {len(filtered_candidates)} after filtering"
            )
            
            return filtered_candidates
            
        except Exception as e:
            self.logger.error(f"Candidate collection failed: {e}")
            # Return basic collection as fallback
            return self._collect_all_candidates(state)
    
    async def _get_candidates_from_persisted_pool(
        self,
        state,
        max_candidates: int = 50
    ) -> List[TrackRecommendation]:
        """
        Retrieve candidates from persisted candidate pool for follow-up queries.
        
        Phase 3: This method enables efficient "load more" functionality by
        retrieving candidates from the stored pool instead of regenerating.
        
        Args:
            state: Current workflow state
            max_candidates: Maximum number of candidates to retrieve
            
        Returns:
            List of TrackRecommendation from persisted pool
        """
        if not self.session_manager:
            self.logger.debug("No session manager available for candidate pool retrieval")
            return []
        
        # Check if this is a follow-up query that can use persisted pools
        if not hasattr(state, 'effective_intent') or not state.effective_intent:
            return []
        
        effective_intent = state.effective_intent
        if not effective_intent.get('is_followup'):
            return []
        
        followup_type = effective_intent.get('followup_type')
        # Accept both 'load_more' and 'artist_deep_dive' as pool-retrieval follow-ups
        if followup_type not in ['load_more', 'artist_deep_dive']:
            return []
        
        # Get the intent and entities for pool retrieval
        intent = effective_intent.get('intent')
        entities = effective_intent.get('entities', {})
        session_id = getattr(state, 'session_id', None)
        
        if not session_id:
            self.logger.warning("No session ID available for candidate pool retrieval")
            return []
        
        if self.session_manager and effective_intent.get('is_followup'):
            try:
                original_context = await self.session_manager.get_original_query_context(session_id)
                if original_context:
                    intent = original_context.intent
                    self.logger.debug(
                        f"Using original intent for pool retrieval: {intent} (instead of {effective_intent.get('intent')})"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to get original context, using current intent: {e}")
        
        self.logger.info(
            "Attempting to retrieve candidates from persisted pool",
            session_id=session_id,
            intent=intent,
            followup_type=followup_type
        )
        
        try:
            # Retrieve candidate pool
            candidate_pool = await self.session_manager.get_candidate_pool(
                session_id=session_id,
                intent=intent,
                entities=entities
            )
            
            if not candidate_pool:
                self.logger.info("No compatible candidate pool found")
                return []
            
            # Convert UnifiedTrackMetadata to TrackRecommendation
            track_recommendations = []
            for i, track_metadata in enumerate(candidate_pool.candidates[:max_candidates]):
                try:
                    # Create TrackRecommendation from UnifiedTrackMetadata
                    # FIXED: Use recommendation_score instead of confidence, with proper fallback
                    # Handle the case where recommendation_score might be 0.0 (which is falsy)
                    recommendation_score = getattr(track_metadata, 'recommendation_score', None)
                    quality_score = getattr(track_metadata, 'quality_score', None)
                    
                    if recommendation_score is not None:
                        confidence_score = recommendation_score
                    elif quality_score is not None:
                        confidence_score = quality_score
                    else:
                        confidence_score = 0.7  # Default reasonable confidence for persisted pool tracks
                    
                    # Debug logging for scoring issues
                    self.logger.debug(
                        f"Pool candidate scoring: {track_metadata.artist} - {track_metadata.name}",
                        recommendation_score=recommendation_score,
                        quality_score=quality_score,
                        final_confidence=confidence_score
                    )
                    
                    track_rec = TrackRecommendation(
                        id=getattr(track_metadata, 'id', f"pool_{i}"),
                        title=track_metadata.name,
                        artist=track_metadata.artist,
                        source="persisted_pool",
                        album_title=track_metadata.album,
                        genres=track_metadata.genres,
                        novelty_score=getattr(track_metadata, 'underground_score', None),
                        quality_score=getattr(track_metadata, 'quality_score', None),
                        confidence=confidence_score,
                        additional_scores={
                            "duration_ms": track_metadata.duration_ms,
                            "popularity": track_metadata.popularity,
                            "tags": track_metadata.tags,
                            "audio_features": track_metadata.audio_features,
                            "recommendation_score": getattr(track_metadata, 'recommendation_score', None),
                            "retrieval_reason": "Retrieved from persisted candidate pool for efficient follow-up"
                        }
                    )
                    track_recommendations.append(track_rec)
                except Exception as e:
                    self.logger.warning(f"Failed to convert pool candidate {i}: {e}")
                    continue
            
            self.logger.info(
                "Successfully retrieved candidates from persisted pool",
                session_id=session_id,
                pool_size=len(candidate_pool.candidates),
                retrieved_count=len(track_recommendations),
                usage_count=candidate_pool.used_count
            )
            
            return track_recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve candidates from persisted pool: {e}")
            return []
    
    def _collect_all_candidates(self, state) -> List[TrackRecommendation]:
        """
        Collect all candidate recommendations from agent outputs.
        
        Args:
            state: Current workflow state with agent outputs
            
        Returns:
            Combined list of all candidate recommendations
        """
        try:
            all_candidates = []
            
            # Collect from genre_mood_agent
            if hasattr(state, 'genre_mood_recommendations') and state.genre_mood_recommendations:
                all_candidates.extend(state.genre_mood_recommendations)
                self.logger.debug(f"Added {len(state.genre_mood_recommendations)} genre/mood candidates")
            
            # Collect from discovery_agent
            if hasattr(state, 'discovery_recommendations') and state.discovery_recommendations:
                all_candidates.extend(state.discovery_recommendations)
                self.logger.debug(f"Added {len(state.discovery_recommendations)} discovery candidates")
            
            # Collect from planner_agent (if any direct recommendations)
            if hasattr(state, 'planner_recommendations') and state.planner_recommendations:
                all_candidates.extend(state.planner_recommendations)
                self.logger.debug(f"Added {len(state.planner_recommendations)} planner candidates")
            
            # Remove duplicates based on track name and artist
            unique_candidates = []
            seen_tracks = set()
            
            for candidate in all_candidates:
                track_key = f"{candidate.artist.lower()}||{getattr(candidate, 'title', '').lower()}"
                if track_key not in seen_tracks:
                    unique_candidates.append(candidate)
                    seen_tracks.add(track_key)
                else:
                    self.logger.debug(f"Filtered duplicate: {candidate.artist} - {getattr(candidate, 'title', '')}")
            
            self.logger.info(f"Collected {len(all_candidates)} total, {len(unique_candidates)} unique candidates")
            return unique_candidates
            
        except Exception as e:
            self.logger.error(f"Failed to collect candidates: {e}")
            return []
    
    def _filter_out_recently_shown(
        self, 
        candidates: List[TrackRecommendation], 
        state
    ) -> List[TrackRecommendation]:
        """
        Filter out tracks that were recently shown to avoid repetition.
        
        Args:
            candidates: List of candidate recommendations
            state: Current workflow state with recently shown track IDs
            
        Returns:
            Filtered list excluding recently shown tracks
        """
        try:
            # FIXED: Use the prepared recently_shown_track_ids from state
            recently_shown_track_ids = getattr(state, 'recently_shown_track_ids', [])
            
            if not recently_shown_track_ids:
                # Fallback: try to extract from conversation_history if no prepared IDs
                return self._fallback_conversation_history_extraction(candidates, state)
            
            # Convert recently shown track IDs to the format we use for matching
            recently_shown = set()
            for track_id in recently_shown_track_ids:
                # Handle different formats: "artist::title", "artist:title", etc.
                if "::" in track_id:
                    recently_shown.add(track_id.lower())
                elif ":" in track_id:
                    # Convert "artist:title" to "artist||title" format
                    parts = track_id.split(":", 1)
                    if len(parts) == 2:
                        recently_shown.add(f"{parts[0].lower()}||{parts[1].lower()}")
                else:
                    # Add as-is for any other format
                    recently_shown.add(track_id.lower())
            
            if not recently_shown:
                return candidates
            
            # Filter out recently shown tracks
            filtered_candidates = []
            for candidate in candidates:
                candidate_key = f"{candidate.artist.lower()}||{getattr(candidate, 'title', '').lower()}"
                if candidate_key not in recently_shown:
                    filtered_candidates.append(candidate)
                else:
                    self.logger.debug(f"Filtered recently shown: {candidate.artist} - {getattr(candidate, 'title', '')}")
            
            self.logger.info(f"Filtered {len(candidates) - len(filtered_candidates)} recently shown tracks")
            return filtered_candidates
            
        except Exception as e:
            self.logger.warning(f"Recently shown filtering failed: {e}")
            return candidates
    
    def _fallback_conversation_history_extraction(
        self, 
        candidates: List[TrackRecommendation], 
        state
    ) -> List[TrackRecommendation]:
        """
        Fallback method to extract recently shown tracks from conversation history.
        Used when recently_shown_track_ids is not available in state.
        """
        try:
            if not hasattr(state, 'conversation_history') or not state.conversation_history:
                return candidates
            
            # Extract recently shown tracks from conversation history
            recently_shown = set()
            
            for entry in state.conversation_history[-5:]:  # Check last 5 interactions
                if 'recommendations' in entry:
                    for rec in entry['recommendations']:
                        if isinstance(rec, dict):
                            artist = rec.get('artist', '').lower()
                            name = rec.get('name', rec.get('title', '')).lower()
                        else:
                            artist = getattr(rec, 'artist', '').lower()
                            name = getattr(rec, 'title', getattr(rec, 'name', '')).lower()
                        
                        if artist and name:
                            recently_shown.add(f"{artist}||{name}")
            
            if not recently_shown:
                return candidates
            
            # Filter out recently shown tracks
            filtered_candidates = []
            for candidate in candidates:
                candidate_key = f"{candidate.artist.lower()}||{getattr(candidate, 'title', '').lower()}"
                if candidate_key not in recently_shown:
                    filtered_candidates.append(candidate)
                else:
                    self.logger.debug(f"Filtered recently shown (fallback): {candidate.artist} - {getattr(candidate, 'title', '')}")
            
            self.logger.info(f"Filtered {len(candidates) - len(filtered_candidates)} recently shown tracks (fallback)")
            return filtered_candidates
            
        except Exception as e:
            self.logger.warning(f"Fallback conversation history filtering failed: {e}")
            return candidates
    
    def apply_source_priority_filtering(
        self,
        candidates: List[TrackRecommendation],
        priority_config: Dict[str, Any] = None
    ) -> List[TrackRecommendation]:
        """
        Apply source-based priority filtering to candidates.
        
        Args:
            candidates: List of candidate recommendations
            priority_config: Configuration for source priorities
            
        Returns:
            Filtered and prioritized list of candidates
        """
        try:
            if not priority_config:
                priority_config = {
                    'preferred_sources': ['discovery_agent', 'genre_mood_agent'],
                    'max_per_source': 20,
                    'min_total': 15
                }
            
            # Group candidates by source
            by_source = {}
            for candidate in candidates:
                source = getattr(candidate, 'source', 'unknown')
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append(candidate)
            
            # Apply source limits and preferences
            prioritized_candidates = []
            max_per_source = priority_config.get('max_per_source', 20)
            preferred_sources = priority_config.get('preferred_sources', [])
            
            # First, add from preferred sources
            for source in preferred_sources:
                if source in by_source:
                    source_candidates = by_source[source][:max_per_source]
                    prioritized_candidates.extend(source_candidates)
                    self.logger.debug(f"Added {len(source_candidates)} candidates from preferred source: {source}")
            
            # Then add from other sources if needed
            min_total = priority_config.get('min_total', 15)
            if len(prioritized_candidates) < min_total:
                for source, source_candidates in by_source.items():
                    if source not in preferred_sources:
                        needed = min_total - len(prioritized_candidates)
                        if needed <= 0:
                            break
                        
                        additional = source_candidates[:min(needed, max_per_source)]
                        prioritized_candidates.extend(additional)
                        self.logger.debug(f"Added {len(additional)} candidates from source: {source}")
            
            self.logger.info(f"Source priority filtering: {len(candidates)} -> {len(prioritized_candidates)}")
            return prioritized_candidates
            
        except Exception as e:
            self.logger.warning(f"Source priority filtering failed: {e}")
            return candidates
    
    def validate_candidates(
        self,
        candidates: List[TrackRecommendation]
    ) -> Tuple[List[TrackRecommendation], List[str]]:
        """
        Validate candidates and return valid ones with error list.
        
        Args:
            candidates: List of candidate recommendations
            
        Returns:
            Tuple of (valid_candidates, validation_errors)
        """
        valid_candidates = []
        validation_errors = []
        
        for i, candidate in enumerate(candidates):
            try:
                # Check required fields
                if not hasattr(candidate, 'artist') or not candidate.artist:
                    validation_errors.append(f"Candidate {i}: Missing artist")
                    continue
                
                name = getattr(candidate, 'name', None) or getattr(candidate, 'title', None)
                if not name:
                    validation_errors.append(f"Candidate {i}: Missing track name/title")
                    continue
                
                # Check for reasonable data
                if len(candidate.artist.strip()) < 1:
                    validation_errors.append(f"Candidate {i}: Empty artist name")
                    continue
                
                if len(name.strip()) < 1:
                    validation_errors.append(f"Candidate {i}: Empty track name")
                    continue
                
                # Candidate is valid
                valid_candidates.append(candidate)
                
            except Exception as e:
                validation_errors.append(f"Candidate {i}: Validation error - {e}")
        
        if validation_errors:
            self.logger.warning(f"Candidate validation found {len(validation_errors)} issues")
            for error in validation_errors[:5]:  # Log first 5 errors
                self.logger.debug(error)
        
        return valid_candidates, validation_errors
    
    def get_candidate_statistics(
        self,
        candidates: List[TrackRecommendation]
    ) -> Dict[str, Any]:
        """
        Get statistics about the candidate collection.
        
        Args:
            candidates: List of candidate recommendations
            
        Returns:
            Dictionary with candidate statistics
        """
        try:
            stats = {
                'total_count': len(candidates),
                'sources': {},
                'genres': {},
                'artists': {},
                'confidence_distribution': {
                    'high': 0,    # > 0.7
                    'medium': 0,  # 0.4 - 0.7
                    'low': 0      # < 0.4
                }
            }
            
            for candidate in candidates:
                # Source distribution
                source = getattr(candidate, 'source', 'unknown')
                stats['sources'][source] = stats['sources'].get(source, 0) + 1
                
                # Artist distribution
                artist = candidate.artist
                stats['artists'][artist] = stats['artists'].get(artist, 0) + 1
                
                # Genre distribution
                genres = getattr(candidate, 'genres', [])
                for genre in genres[:3]:  # Top 3 genres per track
                    stats['genres'][genre] = stats['genres'].get(genre, 0) + 1
                
                # Confidence distribution
                confidence = getattr(candidate, 'confidence', 0.5)
                if confidence > 0.7:
                    stats['confidence_distribution']['high'] += 1
                elif confidence >= 0.4:
                    stats['confidence_distribution']['medium'] += 1
                else:
                    stats['confidence_distribution']['low'] += 1
            
            # Sort by count
            stats['sources'] = dict(sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True))
            stats['genres'] = dict(sorted(stats['genres'].items(), key=lambda x: x[1], reverse=True))
            stats['artists'] = dict(sorted(stats['artists'].items(), key=lambda x: x[1], reverse=True))
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Statistics calculation failed: {e}")
            return {'total_count': len(candidates), 'error': str(e)} 