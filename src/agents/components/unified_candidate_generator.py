"""
Refactored Unified Candidate Generation Framework for BeatDebate

A lean orchestrator that delegates to modular strategy components,
fixing the 'str' vs enum bug and improving maintainability.
"""

from typing import Dict, List, Any, Optional, Union
import structlog
from datetime import datetime

from ...services.api_service import APIService
from ...models.metadata_models import UnifiedTrackMetadata
from ...models.agent_models import QueryIntent

from .generation_strategies.factory import StrategyFactory
from .candidate_processor import CandidateProcessor

logger = structlog.get_logger(__name__)


class UnifiedCandidateGenerator:
    """
    Refactored unified candidate generator using the Strategy Pattern.
    
    This lean orchestrator delegates to specialized strategy components
    and enforces type safety to prevent the 'str' vs enum bug.
    
    Key improvements:
    - Modular strategy components
    - Type-safe intent handling
    - Centralized post-processing
    - Better error handling
    - Improved maintainability
    """
    
    def __init__(self, api_service: APIService, session_manager=None, llm_client=None):
        """
        Initialize the refactored candidate generator.
        
        Args:
            api_service: Unified API service for music data access
            session_manager: SessionManagerService for candidate pool persistence
            llm_client: LLM client for strategies that need AI classification
        """
        self.api_service = api_service
        self.session_manager = session_manager
        self.llm_client = llm_client
        self.logger = logger.bind(component="UnifiedCandidateGenerator")
        
        # Initialize modular components
        self.strategy_factory = StrategyFactory(api_service, llm_client=llm_client)
        self.candidate_processor = CandidateProcessor()
        
        # Configuration
        self.target_candidates = 200
        self.final_recommendations = 20
        
        # Pool persistence settings
        self.enable_pool_persistence = session_manager is not None
        self.large_pool_multiplier = 1  # Reduced from 3 to keep pool at 200 candidates
        self.pool_persistence_intents = [
            QueryIntent.BY_ARTIST, 
            QueryIntent.BY_ARTIST_UNDERGROUND,
            QueryIntent.ARTIST_SIMILARITY, 
            QueryIntent.ARTIST_GENRE,  # ✅ NEW: Enable pool persistence for artist_genre
            QueryIntent.HYBRID_SIMILARITY_GENRE,  # ✅ NEW: Enable pool persistence for hybrid_similarity_genre
            QueryIntent.DISCOVERY,
            QueryIntent.DISCOVERING_SERENDIPITY,  # Enable pool persistence for serendipitous discovery
            QueryIntent.GENRE_MOOD,
            QueryIntent.CONTEXTUAL  # Enable pool persistence for contextual queries
        ]
        
        self.logger.info(
            "Refactored Unified Candidate Generator initialized",
            pool_persistence_enabled=self.enable_pool_persistence,
            available_strategies=len(self.strategy_factory.list_available_strategies())
        )
    
    async def generate_and_persist_large_pool(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        session_id: str,
        agent_type: str = "discovery",
        detected_intent: Union[QueryIntent, str, None] = None
    ) -> Optional[str]:
        """
        Generate a large candidate pool and persist it for follow-up queries.
        
        Args:
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent  
            session_id: Session ID for pool storage
            agent_type: "genre_mood" or "discovery" for strategy selection
            detected_intent: QueryIntent enum or string
            
        Returns:
            Pool key for retrieving the stored candidates, or None if failed
        """
        if not self.enable_pool_persistence:
            self.logger.warning("Pool persistence not enabled")
            return None
        
        # CRITICAL FIX: Type-safe intent handling
        try:
            if isinstance(detected_intent, str):
                intent_enum = QueryIntent(detected_intent)
            elif isinstance(detected_intent, QueryIntent):
                intent_enum = detected_intent
            else:
                self.logger.warning(f"Invalid intent type: {type(detected_intent)}")
                return None
        except ValueError as e:
            self.logger.warning(f"Invalid intent value: {detected_intent}, error: {e}")
            return None
        
        # Check if this intent benefits from pool persistence
        if intent_enum not in self.pool_persistence_intents:
            self.logger.info(f"Intent '{intent_enum}' doesn't benefit from pool persistence")
            return None
        
        # Generate large candidate pool
        original_target = self.target_candidates
        self.target_candidates = original_target * self.large_pool_multiplier
        
        self.logger.info(
            "Generating large candidate pool for persistence",
            session_id=session_id,
            intent=intent_enum.value,
            target_candidates=self.target_candidates
        )
        
        try:
            # Generate the large pool using our modular system
            large_pool = await self.generate_candidate_pool(
                entities=entities,
                intent_analysis=intent_analysis,
                agent_type=agent_type,
                target_candidates=self.target_candidates,
                detected_intent=intent_enum,
                recently_shown_track_ids=[]
            )
            
            # CRITICAL FIX: Apply discovery scoring to candidates before storage
            # This ensures that stored candidates have proper combined_score values
            if agent_type == "discovery" and large_pool:
                from ..discovery.discovery_scorer import DiscoveryScorer
                from . import QualityScorer
                
                discovery_scorer = DiscoveryScorer()
                quality_scorer = QualityScorer()
                
                self.logger.info(f"Applying discovery scoring to {len(large_pool)} candidates before storage")
                large_pool = await discovery_scorer.score_discovery_candidates(
                    large_pool, entities, intent_analysis, quality_scorer
                )
                
                # Log scoring results for debugging
                if large_pool:
                    sample_scored = large_pool[:3]
                    for i, candidate in enumerate(sample_scored):
                        self.logger.debug(
                            f"Scored candidate {i+1}: {candidate.get('artist')} - {candidate.get('name')}",
                            combined_score=candidate.get('combined_score'),
                            quality_score=candidate.get('quality_score'),
                            novelty_score=candidate.get('novelty_score')
                        )
            
            # Convert to UnifiedTrackMetadata for storage
            # FIXED: Preserve all scoring information when storing candidates
            metadata_pool = []
            for candidate in large_pool:
                try:
                    metadata = UnifiedTrackMetadata(
                        name=candidate.get('name', 'Unknown'),
                        artist=candidate.get('artist', 'Unknown'),
                        album=candidate.get('album', 'Unknown'),
                        duration_ms=candidate.get('duration', 0),
                        popularity=candidate.get('popularity', 0.0),
                        listeners=candidate.get('listeners', None),
                        genres=candidate.get('genres', []),
                        tags=candidate.get('tags', []),
                        audio_features=candidate.get('audio_features', {}),
                        source=candidate.get('source', 'unified_generator'),
                        # CRITICAL: Preserve scoring information
                        recommendation_score=candidate.get('combined_score') or candidate.get('recommendation_score'),
                        quality_score=candidate.get('quality_score'),
                        underground_score=candidate.get('underground_score') or candidate.get('novelty_score'),
                        agent_source=candidate.get('agent_source', 'unified_generator'),
                        recommendation_reason=candidate.get('recommendation_reason', 'Generated by unified candidate generator')
                    )
                    
                    # Debug logging for score preservation
                    self.logger.debug(
                        f"Storing candidate with scores: {candidate.get('artist')} - {candidate.get('name')}",
                        combined_score=candidate.get('combined_score'),
                        recommendation_score=candidate.get('recommendation_score'),
                        quality_score=candidate.get('quality_score'),
                        final_recommendation_score=metadata.recommendation_score
                    )
                    metadata_pool.append(metadata)
                except Exception as e:
                    self.logger.warning(f"Failed to convert candidate to metadata: {e}")
                    continue
            
            # Store the pool in SessionManagerService - FIXED: Use enum.value
            pool_key = await self.session_manager.store_candidate_pool(
                session_id=session_id,
                candidates=metadata_pool,
                intent=intent_enum.value,  # FIX: Always use .value for storage
                entities=entities
            )
            
            self.logger.info(
                "Large candidate pool stored successfully",
                session_id=session_id,
                pool_key=pool_key,
                pool_size=len(metadata_pool),
                intent=intent_enum.value
            )
            
            return pool_key
            
        except Exception as e:
            self.logger.error(f"Failed to generate/store large candidate pool: {e}")
            return None
            
        finally:
            # Restore original target
            self.target_candidates = original_target
    
    async def generate_candidate_pool(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        agent_type: str = "genre_mood",
        target_candidates: Optional[int] = None,
        detected_intent: Union[QueryIntent, str, None] = None,
        recently_shown_track_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate pool using the modular strategy system.
        
        Args:
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent
            agent_type: "genre_mood" or "discovery" for strategy selection
            target_candidates: Override default target candidate count
            detected_intent: QueryIntent enum or string
            recently_shown_track_ids: List of recently shown track IDs to avoid
            
        Returns:
            List of candidate tracks with source metadata
        """
        if target_candidates:
            self.target_candidates = target_candidates
        
        if recently_shown_track_ids is None:
            recently_shown_track_ids = []
        
        # CRITICAL FIX: Type-safe intent handling
        try:
            if isinstance(detected_intent, str):
                intent_enum = QueryIntent(detected_intent)
            elif isinstance(detected_intent, QueryIntent):
                intent_enum = detected_intent
            elif detected_intent is None:
                intent_enum = QueryIntent.DISCOVERY  # Default fallback
            else:
                self.logger.warning(f"Invalid intent type: {type(detected_intent)}")
                intent_enum = QueryIntent.DISCOVERY
        except ValueError as e:
            self.logger.warning(f"Invalid intent value: {detected_intent}, error: {e}")
            intent_enum = QueryIntent.DISCOVERY
        
        self.logger.info(
            "Starting modular candidate generation",
            agent_type=agent_type,
            intent=intent_enum.value,
            target_candidates=self.target_candidates,
            excluded_tracks=len(recently_shown_track_ids)
        )
        
        try:
            # Get appropriate strategies for the intent
            strategies = self.strategy_factory.get_strategies_for_intent(intent_enum)
            
            if not strategies:
                self.logger.warning(f"No strategies found for intent {intent_enum}")
                return []
            
            # Generate candidates using selected strategies
            all_candidates = []
            candidates_per_strategy = max(10, self.target_candidates // len(strategies))
            
            for strategy in strategies:
                try:
                    strategy_candidates = await strategy.generate(
                        entities=entities,
                        intent_analysis=intent_analysis,
                        limit=candidates_per_strategy
                    )
                    all_candidates.extend(strategy_candidates)
                    
                    self.logger.debug(
                        f"Strategy {strategy.__class__.__name__} generated {len(strategy_candidates)} candidates"
                    )
                    
                except Exception as e:
                    self.logger.warning(
                        f"Strategy {strategy.__class__.__name__} failed: {e}"
                    )
                    continue
            
            # Process candidates (deduplicate, filter, diversify)
            processed_candidates = self.candidate_processor.process_candidates(
                candidates=all_candidates,
                enforce_diversity=True,
                min_confidence=0.1,
                diversity_threshold=0.7
            )
            
            # Filter out recently shown tracks
            if recently_shown_track_ids:
                processed_candidates = self._filter_recently_shown_tracks(
                    processed_candidates, recently_shown_track_ids
                )
            
            # Limit to target count
            final_candidates = processed_candidates[:self.target_candidates]
            
            self.logger.info(
                "Modular candidate generation completed",
                intent=intent_enum.value,
                strategies_used=len(strategies),
                raw_candidates=len(all_candidates),
                processed_candidates=len(processed_candidates),
                final_candidates=len(final_candidates)
            )
            
            return final_candidates
            
        except Exception as e:
            self.logger.error(f"Candidate generation failed: {e}")
            return []
    
    def _filter_recently_shown_tracks(
        self, 
        candidates: List[Dict[str, Any]], 
        recently_shown_track_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter out recently shown tracks."""
        if not recently_shown_track_ids:
            return candidates
        
        filtered_candidates = []
        recently_shown_set = set(recently_shown_track_ids)
        
        for candidate in candidates:
            # FIXED: Create track ID from artist and name fields in the same format as context handler
            artist = candidate.get('artist', '').strip()
            name = candidate.get('name', '').strip()
            
            if artist and name:
                # Use the same format as the context handler: "artist||title"
                candidate_track_id = f"{artist.lower()}||{name.lower()}"
            else:
                # Fallback to explicit track_id or id fields if artist/name not available
                candidate_track_id = candidate.get('track_id') or candidate.get('id') or ""
            
            if candidate_track_id not in recently_shown_set:
                filtered_candidates.append(candidate)
            else:
                self.logger.debug(f"Filtered recently shown track: {artist} - {name}")
        
        self.logger.debug(
            f"Filtered {len(candidates) - len(filtered_candidates)} recently shown tracks"
        )
        
        return filtered_candidates
    
    # Compatibility methods for existing code
    def _extract_seed_artists(self, entities: Dict[str, Any]) -> List[str]:
        """Extract seed artists from entities (compatibility method)."""
        artists = []
        
        # Extract from primary artists
        primary_artists = entities.get('artists', {}).get('primary', [])
        for artist in primary_artists:
            if isinstance(artist, dict):
                artists.append(artist.get('name', ''))
            else:
                artists.append(str(artist))
        
        # Extract from similar_to artists
        similar_artists = entities.get('artists', {}).get('similar_to', [])
        for artist in similar_artists:
            if isinstance(artist, dict):
                artists.append(artist.get('name', ''))
            else:
                artists.append(str(artist))
        
        return [a for a in artists if a]
    
    def _extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target genres from entities (compatibility method)."""
        genres = []
        
        # Extract from primary genres
        primary_genres = entities.get('genres', {}).get('primary', [])
        genres.extend([str(g) for g in primary_genres])
        
        # Extract from secondary genres
        secondary_genres = entities.get('genres', {}).get('secondary', [])
        genres.extend([str(g) for g in secondary_genres])
        
        return [g for g in genres if g]
    
    def _extract_moods(self, entities: Dict[str, Any]) -> List[str]:
        """Extract moods from entities (compatibility method)."""
        moods = []
        
        # Extract from primary moods
        primary_moods = entities.get('moods', {}).get('primary', [])
        moods.extend([str(m) for m in primary_moods])
        
        return [m for m in moods if m]
    
    def _generate_candidate_id(self, candidate: Dict[str, Any]) -> str:
        """Generate a unique ID for a candidate (compatibility method)."""
        artist = candidate.get('artist', 'Unknown')
        title = candidate.get('title', 'Unknown')
        timestamp = datetime.now().isoformat()
        return f"{artist}_{title}_{timestamp}".replace(' ', '_')
    
    def _get_timestamp(self) -> str:
        """Get current timestamp (compatibility method)."""
        return datetime.now().isoformat()
    
    # Legacy method support for gradual migration
    async def _generate_intent_aware_candidates(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        intent: Union[QueryIntent, str],
        agent_type: str
    ) -> List[Dict[str, Any]]:
        """
        Legacy method that routes to the new modular system.
        Maintained for backward compatibility during migration.
        """
        return await self.generate_candidate_pool(
            entities=entities,
            intent_analysis=intent_analysis,
            agent_type=agent_type,
            detected_intent=intent
        ) 