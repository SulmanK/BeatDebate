"""
Simplified Judge Agent

Refactored to use dependency injection and shared components, eliminating:
- LLM calling duplication
- Ranking logic duplication
- Explanation generation duplication
- Quality scoring duplication
"""

from typing import Dict, List, Any, Tuple
import structlog

from ...models.agent_models import MusicRecommenderState, AgentConfig
from ...models.recommendation_models import TrackRecommendation
from ...services.api_service import APIService
from ...services.metadata_service import MetadataService
from ..base_agent import BaseAgent
from ..components import QualityScorer
from .ranking_logic import RankingLogic
from ..components.query_analysis_utils import QueryAnalysisUtils

logger = structlog.get_logger(__name__)


class JudgeAgent(BaseAgent):
    """
    Simplified Judge Agent with dependency injection.
    
    Responsibilities:
    - Evaluate and rank candidate recommendations from all agents
    - Apply contextual relevance scoring
    - Ensure diversity in final selections
    - Generate conversational explanations
    
    Uses shared components to eliminate duplication:
    - QualityScorer for quality assessment
    - LLMUtils for LLM interactions and explanation generation
    - APIService for unified API access
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client,
        api_service: APIService,
        metadata_service: MetadataService,
        rate_limiter=None,
        session_manager=None
    ):
        """
        Initialize simplified judge agent with injected dependencies.
        
        Args:
            config: Agent configuration
            llm_client: LLM client for explanations
            api_service: Unified API service
            metadata_service: Unified metadata service
            rate_limiter: Rate limiter for LLM API calls
            session_manager: SessionManagerService for candidate pool retrieval (Phase 3)
        """
        super().__init__(
            config=config, 
            llm_client=llm_client, 
            api_service=api_service,
            metadata_service=metadata_service,
            rate_limiter=rate_limiter
        )
        
        # Shared components initialized in parent with rate limiter
        self.quality_scorer = QualityScorer()
        
        # Phase 3: Store session manager for candidate pool retrieval
        self.session_manager = session_manager
        
        # Configuration
        self.final_recommendations = 25
        self.diversity_targets = {
            'max_per_artist': 2,
            'min_genres': 3,
            'source_distribution': {
                'genre_mood_agent': 0.4, 
                'discovery_agent': 0.4, 
                'planner_agent': 0.2
            }
        }
        
        # Initialize shared components
        self.ranking_logic = RankingLogic()
        
        self.logger.info("Simplified JudgeAgent initialized with dependency injection")
    
    async def _get_candidates_from_persisted_pool(
        self,
        state: MusicRecommenderState,
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
            self.logger.warning("No session manager available for candidate pool retrieval")
            return []
        
        # Check if this is a follow-up query that can use persisted pools
        if not hasattr(state, 'effective_intent') or not state.effective_intent:
            return []
        
        effective_intent = state.effective_intent
        if not effective_intent.get('is_followup'):
            return []
        
        followup_type = effective_intent.get('followup_type')
        if followup_type != 'load_more':
            return []
        
        # Get the intent and entities for pool retrieval
        intent = effective_intent.get('intent')
        entities = effective_intent.get('entities', {})
        session_id = state.session_id
        
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
                    track_rec = TrackRecommendation(
                        track_id=getattr(track_metadata, 'id', f"pool_{i}"),
                        name=track_metadata.name,
                        artist=track_metadata.artist,
                        album=track_metadata.album,
                        duration=track_metadata.duration,
                        popularity=track_metadata.popularity,
                        genres=track_metadata.genres,
                        tags=track_metadata.tags,
                        audio_features=track_metadata.audio_features,
                        source="persisted_pool",
                        confidence=track_metadata.confidence,
                        reasoning="Retrieved from persisted candidate pool for efficient follow-up"
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
    
    async def process(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Evaluate and select final recommendations from all agent candidates.
        
        Args:
            state: Current workflow state with candidate recommendations
            
        Returns:
            Updated state with final ranked recommendations
        """
        try:
            self.logger.info("Starting judge agent processing")
            
            # Phase 3: Check if we can use persisted candidate pool for follow-up queries
            pool_candidates = await self._get_candidates_from_persisted_pool(state)
            
            if pool_candidates:
                self.logger.info(f"🎯 Phase 3: Using {len(pool_candidates)} candidates from persisted pool")
                all_candidates = pool_candidates
            else:
                # Collect all candidate recommendations from agents
                all_candidates = self._collect_all_candidates(state)
            
            self.logger.debug(f"Collected {len(all_candidates)} total candidates")
            
            # 🔧 DEBUG: Check if recently_shown_track_ids is populated
            self.logger.info(
                f"🔍 JUDGE DEBUG: recently_shown_track_ids = {getattr(state, 'recently_shown_track_ids', 'NOT_SET')}"
            )
            self.logger.info(
                f"🔍 JUDGE DEBUG: recently_shown_track_ids type = {type(getattr(state, 'recently_shown_track_ids', None))}"
            )
            
            if not all_candidates:
                self.logger.warning("No candidates found for evaluation")
                state.final_recommendations = []
                return state
            
            # 🔧 NEW: Filter out recently shown tracks for follow-up queries
            if state.recently_shown_track_ids:
                original_count = len(all_candidates)
                all_candidates = self._filter_out_recently_shown(all_candidates, state)
                filtered_count = len(all_candidates)
                
                self.logger.info(
                    f"🚫 DUPLICATE FILTER: Filtered out {original_count - filtered_count} "
                    f"previously shown tracks (kept {filtered_count})"
                )
                
                if not all_candidates:
                    self.logger.warning("All candidates were filtered as duplicates")
                    state.final_recommendations = []
                    return state
            else:
                self.logger.info("🔍 JUDGE DEBUG: No recently shown tracks to filter, skipping duplicate filtering")
            
            # Phase 1: Score all candidates with contextual relevance
            scored_candidates = await self._score_all_candidates(all_candidates, state)
            
            # Phase 2: Apply ranking based on user intent and context
            ranked_candidates = await self._rank_candidates(scored_candidates, state)
            
            # Phase 3: Select final recommendations with diversity
            final_selections = self._select_with_diversity(ranked_candidates, state)
            
            # Phase 4: Generate enhanced explanations
            final_recommendations = await self._generate_explanations(final_selections, state)
            
            # CRITICAL: Ensure proper state update for LangGraph
            # Convert to dict format for state storage
            final_recommendations_dicts = [rec.model_dump() for rec in final_recommendations]
            
            # Create a new state object to ensure proper propagation
            updated_state = MusicRecommenderState(
                # Copy all existing fields
                user_query=state.user_query,
                session_id=state.session_id,
                max_recommendations=state.max_recommendations,
                planning_strategy=state.planning_strategy,
                execution_plan=state.execution_plan,
                coordination_strategy=state.coordination_strategy,
                agent_coordination=state.agent_coordination,
                entities=state.entities,
                intent_analysis=state.intent_analysis,
                query_understanding=state.query_understanding,
                conversation_context=state.conversation_context,
                entity_reasoning=state.entity_reasoning,
                context_decision=state.context_decision,
                genre_mood_recommendations=state.genre_mood_recommendations,
                discovery_recommendations=state.discovery_recommendations,
                reasoning_log=state.reasoning_log,
                agent_deliberations=state.agent_deliberations,
                error_info=state.error_info,
                processing_start_time=state.processing_start_time,
                total_processing_time=state.total_processing_time,
                confidence=state.confidence,
                # Set the critical final_recommendations field
                final_recommendations=final_recommendations_dicts
            )
            
            self.logger.info(
                "Judge agent processing completed",
                total_candidates=len(all_candidates),
                final_recommendations=len(final_recommendations)
            )
            
            # Verify the field is set
            self.logger.debug(f"Returning state with final_recommendations: {len(updated_state.final_recommendations)}")
            
            return updated_state
            
        except Exception as e:
            self.logger.error("Judge agent processing failed", error=str(e))
            state.final_recommendations = []
            return state
    
    def _collect_all_candidates(self, state: MusicRecommenderState) -> List[TrackRecommendation]:
        """Collect all candidate recommendations from different agents."""
        all_candidates = []
        
        # Collect from genre/mood agent
        if state.genre_mood_recommendations:
            for rec_dict in state.genre_mood_recommendations:
                try:
                    rec = TrackRecommendation(**rec_dict)
                    all_candidates.append(rec)
                except Exception as e:
                    self.logger.warning(f"Failed to parse genre/mood recommendation: {e}")
        
        # Collect from discovery agent
        if state.discovery_recommendations:
            for rec_dict in state.discovery_recommendations:
                try:
                    rec = TrackRecommendation(**rec_dict)
                    all_candidates.append(rec)
                except Exception as e:
                    self.logger.warning(f"Failed to parse discovery recommendation: {e}")
        
        # Remove duplicates based on artist + name
        unique_candidates = []
        seen_tracks = set()
        
        for candidate in all_candidates:
            track_key = f"{candidate.artist.lower()}::{candidate.title.lower()}"
            if track_key not in seen_tracks:
                seen_tracks.add(track_key)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    async def _score_all_candidates(
        self,
        candidates: List[TrackRecommendation],
        state: MusicRecommenderState
    ) -> List[Tuple[TrackRecommendation, Dict[str, float]]]:
        """Score all candidates with contextual relevance."""
        scored_candidates = []
        
        # Collect context from state
        entities = state.entities or {}
        intent_analysis = state.intent_analysis or {}
        
        # 🔧 FIX: Get intent from state and detect hybrid sub-types
        intent = 'balanced'  # default
        if state and state.query_understanding and hasattr(state.query_understanding, 'intent'):
            # Get intent from QueryUnderstanding object
            intent_value = state.query_understanding.intent
            if hasattr(intent_value, 'value'):
                intent = intent_value.value
            else:
                intent = str(intent_value)
            self.logger.debug(f"🔧 Intent from query_understanding: {intent}")
            
            # 🔧 NEW: Detect hybrid sub-types for dynamic scoring
            if intent == 'hybrid':
                hybrid_subtype = self._detect_hybrid_subtype(state)
                intent = hybrid_subtype
                self.logger.info(f"🔧 Using hybrid sub-type for scoring: {intent}")
        
        elif state and state.intent_analysis and 'intent' in state.intent_analysis:
            intent = state.intent_analysis['intent']
            self.logger.debug(f"🔧 Intent from intent_analysis: {intent}")
        else:
            self.logger.warning("🔧 No intent found, using default: balanced")
        
        for candidate in candidates:
            try:
                # Calculate multiple scoring dimensions
                scores = {
                    'quality_score': await self._calculate_quality_score(candidate, entities, intent_analysis),
                    'contextual_relevance': self._calculate_contextual_relevance(candidate, entities, intent_analysis),
                    'intent_alignment': self._calculate_intent_alignment(candidate, intent_analysis, entities, intent),
                    'source_priority': self._calculate_source_priority(candidate),
                    'diversity_value': self._calculate_diversity_value(candidate, entities)
                }
                
                # Calculate combined score
                scores['combined_score'] = (
                    scores['quality_score'] * 0.25 +
                    scores['contextual_relevance'] * 0.25 +
                    scores['intent_alignment'] * 0.25 +
                    scores['source_priority'] * 0.15 +
                    scores['diversity_value'] * 0.10
                )
                
                scored_candidates.append((candidate, scores))
                
            except Exception as e:
                self.logger.warning(f"Failed to score candidate {candidate.title}: {e}")
                continue
        
        return scored_candidates
    
    async def _calculate_quality_score(
        self,
        candidate: TrackRecommendation,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> float:
        """Calculate quality score using shared quality scorer."""
        try:
            # Convert TrackRecommendation to dict for quality scorer
            candidate_dict = {
                'name': candidate.title,
                'artist': candidate.artist,
                'album': candidate.album_title,
                'tags': candidate.moods,
                'url': candidate.track_url,
                'listeners': getattr(candidate, 'listeners', 0),
                'playcount': getattr(candidate, 'playcount', 0)
            }
            
            return await self.quality_scorer.calculate_quality_score(
                candidate_dict, entities, intent_analysis
            )
            
        except Exception as e:
            self.logger.debug(f"Quality scoring failed for {candidate.title}: {e}")
            return 0.5  # Default score
    
    def _calculate_contextual_relevance(
        self,
        candidate: TrackRecommendation,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> float:
        """Calculate contextual relevance score."""
        score = 0.0
        
        # Genre relevance
        target_genres = self._extract_target_genres(entities)
        candidate_genres = [genre.lower() for genre in candidate.genres]
        
        for target_genre in target_genres:
            if any(target_genre.lower() in genre for genre in candidate_genres):
                score += 0.3
        
        # Mood relevance
        target_moods = self._extract_target_moods(entities, intent_analysis)
        candidate_tags = [tag.lower() for tag in candidate.moods]
        
        for target_mood in target_moods:
            if any(target_mood.lower() in tag for tag in candidate_tags):
                score += 0.2
        
        # Activity context relevance
        context_factors = entities.get('context_factors', [])
        for context in context_factors:
            if any(context.lower() in tag for tag in candidate_tags):
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_intent_alignment(
        self,
        candidate: TrackRecommendation,
        intent_analysis: Dict[str, Any],
        entities: Dict[str, Any],
        intent: str
    ) -> float:
        """Calculate alignment with user intent."""
        # 🔧 DEBUG: Log intent detection for troubleshooting
        if candidate.artist == 'Mk.gee':
            self.logger.info(f"🔍 DEBUG Judge: Candidate '{candidate.title}' by {candidate.artist}")
            self.logger.info(f"🔍 DEBUG Judge: Primary intent detected: '{intent}'")
            self.logger.info(f"🔍 DEBUG Judge: Intent analysis keys: {list(intent_analysis.keys())}")
        
        # Intent-specific scoring
        if intent == 'discovery':
            # Favor tracks with novelty indicators
            if candidate.source == 'discovery_agent':
                return 0.8
            elif 'underground' in candidate.moods or 'hidden_gem' in candidate.moods:
                return 0.7
            else:
                return 0.5
        
        elif intent == 'genre_mood':
            # Favor tracks from genre/mood agent
            if candidate.source == 'genre_mood_agent':
                return 0.8
            else:
                return 0.6
        
        elif intent in ['similarity', 'artist_similarity']:
            # ✅ FIXED! Handle both similarity and artist_similarity intents
            # For artist similarity queries, prioritize tracks from the target artist
            if intent == 'artist_similarity':
                # Get target artists from entities (not intent_analysis)
                target_artists = self._extract_target_artists_from_entities(entities)
                
                # 🔧 DEBUG: Log target artist extraction for troubleshooting
                if candidate.artist == 'Mk.gee':
                    self.logger.info(f"🔍 DEBUG Judge: Target artists extracted: {target_artists}")
                    self.logger.info(f"🔍 DEBUG Judge: Candidate artist '{candidate.artist}' in target_artists: {candidate.artist in target_artists}")
                
                if target_artists and candidate.artist in target_artists:
                    if candidate.artist == 'Mk.gee':
                        self.logger.info(f"🎯 DEBUG Judge: Giving 0.95 intent alignment to Mk.gee track: {candidate.title}!")
                    return 0.95  # Very high score for target artist tracks
                elif (hasattr(candidate, 'explanation') and candidate.explanation 
                      and 'similar' in candidate.explanation.lower()):
                    return 0.8   # High score for similar tracks
                else:
                    return 0.6   # Medium score for other tracks
            else:
                # General similarity
                if (hasattr(candidate, 'explanation') and candidate.explanation 
                    and 'similar' in candidate.explanation.lower()):
                    return 0.8
                else:
                    return 0.6
        
        return 0.5  # Default alignment
    
    def _calculate_source_priority(self, candidate: TrackRecommendation) -> float:
        """Calculate source priority score."""
        source_priorities = {
            'genre_mood_agent': 0.8,
            'discovery_agent': 0.7,
            'planner_agent': 0.6,
            'unified_candidate_generator': 0.5
        }
        
        return source_priorities.get(candidate.source, 0.5)
    
    def _calculate_diversity_value(
        self,
        candidate: TrackRecommendation,
        entities: Dict[str, Any]
    ) -> float:
        """Calculate diversity value for the candidate."""
        score = 0.5  # Base diversity score
        
        # Unique genre bonus
        candidate_genres = set(genre.lower() for genre in candidate.genres)
        target_genres = set(genre.lower() for genre in self._extract_target_genres(entities))
        
        if candidate_genres - target_genres:  # Has genres not in target
            score += 0.3
        
        # Unique tags bonus
        unique_tags = ['experimental', 'underground', 'rare', 'hidden_gem', 'cult']
        if any(tag in candidate.moods for tag in unique_tags):
            score += 0.2
        
        return min(score, 1.0)
    
    async def _rank_candidates(
        self,
        scored_candidates: List[Tuple[TrackRecommendation, Dict[str, float]]],
        state: MusicRecommenderState
    ) -> List[Tuple[TrackRecommendation, Dict[str, float]]]:
        """Rank candidates using intent-aware scoring."""
        # Get user intent for scoring weights
        intent = state.query_understanding.intent.value if state.query_understanding else 'balanced'
        
        # 🔧 NEW: Detect hybrid sub-types for dynamic scoring
        if intent == 'hybrid':
            hybrid_subtype = self._detect_hybrid_subtype(state)
            intent = hybrid_subtype
            self.logger.info(f"🔧 Using hybrid sub-type for ranking: {intent}")
        
        # 🔧 Use RankingLogic with intent-specific parameters
        ranking_logic = RankingLogic()
        
        # Get intent-specific scoring weights
        scoring_weights = ranking_logic.get_intent_weights(intent)
        self.logger.info(f"🔧 Using scoring weights for intent '{intent}': {scoring_weights}")
        
        # Rank candidates using intent-aware logic with dynamic novelty threshold detection
        ranked_candidates = ranking_logic.rank_recommendations(
            candidates=scored_candidates, 
            intent=intent,
            entities=state.entities,
            intent_analysis=state.intent_analysis,
            scoring_weights=scoring_weights
        )
        
        self.logger.debug(f"Ranked {len(ranked_candidates)} candidates using intent: {intent}")
        return ranked_candidates
    
    def _select_with_diversity(
        self,
        ranked_candidates: List[Tuple[TrackRecommendation, Dict[str, float]]],
        state: MusicRecommenderState = None
    ) -> List[TrackRecommendation]:
        """Select final recommendations ensuring diversity."""
        selected = []
        artist_counts = {}
        genre_counts = {}
        source_counts = {}
        
        # 🔧 FIX: Get intent-specific diversity limits instead of hardcoded ones
        intent = 'balanced'  # default
        if state and state.query_understanding and hasattr(state.query_understanding, 'intent'):
            # Get intent from QueryUnderstanding object
            intent_value = state.query_understanding.intent
            if hasattr(intent_value, 'value'):
                intent = intent_value.value
            else:
                intent = str(intent_value)
            self.logger.debug(f"🔧 Intent from query_understanding: {intent}")
        elif state and state.intent_analysis and 'intent' in state.intent_analysis:
            intent = state.intent_analysis['intent']
            self.logger.debug(f"🔧 Intent from intent_analysis: {intent}")
        
        # 🚀 NEW: Apply context override constraints for followup intents
        max_per_artist = 2  # default
        context_override_applied = False
        
        if state and hasattr(state, 'context_override') and state.context_override:
            context_override = state.context_override
            
            self.logger.info(f"🔧 DEBUG: Judge Agent received context_override: {type(context_override)}")
            
            # Handle both dictionary and object formats
            is_followup = False
            constraint_overrides = None
            intent_override = None
            target_entity = None
            
            if isinstance(context_override, dict):
                is_followup = context_override.get('is_followup', False)
                constraint_overrides = context_override.get('constraint_overrides')
                intent_override = context_override.get('intent_override')
                target_entity = context_override.get('target_entity')
            elif hasattr(context_override, 'is_followup'):
                is_followup = context_override.is_followup
                constraint_overrides = getattr(context_override, 'constraint_overrides', None)
                intent_override = getattr(context_override, 'intent_override', None)
                target_entity = getattr(context_override, 'target_entity', None)
            
            if is_followup and constraint_overrides:
                # Apply context-specific constraints
                target_artist = constraint_overrides.get('target_artist') if isinstance(constraint_overrides, dict) else None
                
                self.logger.info(
                    "🚀 Context override detected",
                    intent_override=intent_override,
                    target_entity=target_entity,
                    target_artist=target_artist
                )
                
                if intent_override in ['artist_deep_dive', 'artist_similarity'] and target_entity:
                    # For artist-focused followup queries, allow many more tracks from target artist
                    max_per_artist = constraint_overrides.get('max_per_artist', 10) if isinstance(constraint_overrides, dict) else 10
                    context_override_applied = True
                    
                    self.logger.info(
                        f"🎯 Artist followup query: allowing up to {max_per_artist} tracks from {target_entity}"
                    )
                elif intent_override == 'artist_style_refinement' and target_entity:
                    # 🔧 NEW: Artist-style refinement handling
                    # Allow more tracks from target artist, but less than pure artist similarity
                    max_per_artist = 8  # Slightly less than pure artist similarity
                    context_override_applied = True
                    
                    # Get style modifier for logging
                    style_modifier = None
                    if isinstance(context_override, dict):
                        style_modifier = context_override.get('style_modifier')
                    elif hasattr(context_override, 'style_modifier'):
                        style_modifier = getattr(context_override, 'style_modifier', None)
                    
                    self.logger.info(
                        f"🎵 Artist-style refinement: allowing up to {max_per_artist} tracks "
                        f"from {target_entity} with style '{style_modifier}'"
                    )
                elif intent_override == 'style_continuation':
                    # For style continuation, moderate increase
                    max_per_artist = constraint_overrides.get('max_per_artist', 3) if isinstance(constraint_overrides, dict) else 3
                    context_override_applied = True
                    
                    self.logger.info(
                        f"🎵 Style continuation: allowing up to {max_per_artist} tracks per artist"
                    )
            else:
                self.logger.info(f"🔧 DEBUG: No context override applied - is_followup={is_followup}, has_constraints={bool(constraint_overrides)}")
        
        # Apply appropriate diversity limits
        if not context_override_applied:
            diversity_limits = self.ranking_logic.get_diversity_limits(intent)
            max_per_artist = diversity_limits.get('max_per_artist', 2)
        
        # 🎯 NEW: Check if this is a genre-specific query requiring different logic
        is_genre_specific_query = False
        if state and state.entities:
            entities = state.entities
            intent_analysis = state.intent_analysis or {}
            
            # Check if this is a hybrid query with specific genre requirements
            musical_entities = entities.get('musical_entities', {})
            genres_primary = musical_entities.get('genres', {}).get('primary')
            is_hybrid = (intent_analysis.get('intent') == 'hybrid' or 
                        intent_analysis.get('primary_intent') == 'hybrid')
            
            is_genre_specific_query = bool(genres_primary and is_hybrid)
            
            if is_genre_specific_query:
                self.logger.info(f"🎯 Genre-specific query detected: prioritizing genre_mood_agent tracks for genres {genres_primary}")
        
        self.logger.debug(f"🔧 DEBUG: Intent: {intent}, max tracks per artist: {max_per_artist}")
        self.logger.debug(f"Starting diversity selection with {len(ranked_candidates)} candidates")
        
        for i, (candidate, scores) in enumerate(ranked_candidates):
            self.logger.debug(
                f"Evaluating candidate {i+1}: {candidate.title} by {candidate.artist}, "
                f"source={candidate.source}, score={scores.get('final_score', 0):.3f}"
            )
            
            # Check artist diversity using context-aware limits
            if artist_counts.get(candidate.artist, 0) >= max_per_artist:
                self.logger.debug(f"Skipping {candidate.title}: too many tracks from {candidate.artist}")
                continue
            
            # 🎯 MODIFIED: Genre-aware source distribution
            source_count = source_counts.get(candidate.source, 0)
            
            if is_genre_specific_query:
                # For genre-specific queries, relax source constraints for genre_mood_agent
                if candidate.source == 'genre_mood_agent':
                    # Allow more genre_mood_agent tracks (up to 80% of recommendations)
                    max_per_source = int(self.final_recommendations * 0.8)
                else:
                    # Restrict other sources more (discovery, planner) 
                    max_per_source = int(self.final_recommendations * 0.2)
            else:
                # Standard source distribution for non-genre queries
                max_per_source = int(self.final_recommendations * 
                                     self.diversity_targets['source_distribution'].get(candidate.source, 0.3))
            
            self.logger.debug(
                f"Source check: {candidate.source} count={source_count}, max={max_per_source}"
            )
            
            if source_count >= max_per_source:
                self.logger.debug(f"Skipping {candidate.title}: source quota exceeded for {candidate.source}")
                continue
            
            # Add to selection
            selected.append(candidate)
            artist_counts[candidate.artist] = artist_counts.get(candidate.artist, 0) + 1
            source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
            
            # Update genre counts
            for genre in candidate.genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            self.logger.debug(f"Selected {candidate.title} (total selected: {len(selected)})")
            
            # Stop when we have enough
            if len(selected) >= self.final_recommendations:
                break
        
        self.logger.info(
            f"🔧 DEBUG: Diversity filtering: {len(ranked_candidates)} -> {len(selected)} candidates"
        )
        
        if context_override_applied:
            self.logger.info(
                f"🚀 Context override applied - final artist distribution: {dict(artist_counts)}"
            )
        
        self.logger.info(
            f"Selected {len(selected)} recommendations with diversity",
            artist_distribution=dict(artist_counts),
            source_distribution=dict(source_counts),
            diversity_targets=self.diversity_targets
        )
        
        return selected
    
    async def _generate_explanations(
        self,
        selections: List[TrackRecommendation],
        state: MusicRecommenderState
    ) -> List[TrackRecommendation]:
        """Generate enhanced explanations for final selections."""
        enhanced_selections = []
        
        # For now, skip individual LLM calls to avoid rate limits
        # Use existing explanations or create simple ones
        for i, recommendation in enumerate(selections):
            try:
                # Use existing explanation or create a simple one
                if not recommendation.explanation or len(recommendation.explanation.strip()) < 10:
                    recommendation.explanation = self._create_fallback_reasoning(
                        recommendation, 
                        state.entities or {}, 
                        state.intent_analysis or {}, 
                        i + 1
                    )
                
                recommendation.rank = i + 1
                enhanced_selections.append(recommendation)
                
            except Exception as e:
                self.logger.warning(f"Failed to enhance reasoning for {recommendation.title}: {e}")
                # Use original recommendation with updated rank
                recommendation.rank = i + 1
                enhanced_selections.append(recommendation)
        
        return enhanced_selections
    
    def _create_fallback_reasoning(
        self,
        recommendation: TrackRecommendation,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        rank: int
    ) -> str:
        """Create fallback reasoning when LLM is unavailable."""
        reasoning_parts = [f"#{rank}: {recommendation.title} by {recommendation.artist}"]
        
        # Add genre information
        if recommendation.genres:
            reasoning_parts.append(f"A great {'/'.join(recommendation.genres[:2])} track")
        
        # Add source information
        if recommendation.source == 'discovery_agent':
            reasoning_parts.append("Perfect for discovery")
        elif recommendation.source == 'genre_mood_agent':
            reasoning_parts.append("Matches your genre and mood preferences")
        
        # Add confidence information
        if recommendation.confidence > 0.8:
            reasoning_parts.append("High confidence match")
        elif recommendation.confidence > 0.6:
            reasoning_parts.append("Good match for your request")
        
        return ". ".join(reasoning_parts) + "."
    
    def _extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target genres from entities."""
        musical_entities = entities.get('musical_entities', {})
        genres = musical_entities.get('genres', {})
        
        target_genres = []
        target_genres.extend(genres.get('primary', []))
        target_genres.extend(genres.get('secondary', []))
        
        return list(set(target_genres))  # Remove duplicates
    
    def _extract_target_moods(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[str]:
        """Extract target moods from entities and intent analysis."""
        moods = []
        
        # From entities
        contextual_entities = entities.get('contextual_entities', {})
        mood_entities = contextual_entities.get('moods', {})
        moods.extend(mood_entities.get('energy', []))
        moods.extend(mood_entities.get('emotion', []))
        
        # From intent analysis
        moods.extend(intent_analysis.get('mood_indicators', []))
        
        return list(set(moods))  # Remove duplicates
    
    def _extract_target_artists_from_entities(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target artists from entities for artist similarity queries."""
        musical_entities = entities.get('musical_entities', {})
        artists = musical_entities.get('artists', {})
        
        target_artists = []
        target_artists.extend(artists.get('primary', []))
        target_artists.extend(artists.get('similar_to', []))
        
        return list(set(target_artists))  # Remove duplicates
    
    async def evaluate_and_select(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Evaluate and select final recommendations (alias for process method).
        
        This method provides backward compatibility with the enhanced recommendation service.
        
        Args:
            state: Current workflow state with candidate recommendations
            
        Returns:
            Updated state with final ranked recommendations
        """
        return await self.process(state) 
    
    def _detect_hybrid_subtype(self, state: MusicRecommenderState) -> str:
        """
        Detect hybrid sub-type from state information.
        
        Args:
            state: Current recommendation state
            
        Returns:
            Hybrid sub-type or original intent if not hybrid
        """
        try:
            # First check if we stored the sub-type in reasoning
            if state.query_understanding and hasattr(state.query_understanding, 'reasoning'):
                reasoning = state.query_understanding.reasoning
                if 'Hybrid sub-type:' in reasoning:
                    subtype = reasoning.split('Hybrid sub-type:')[1].strip()
                    self.logger.info(f"🔧 Found stored hybrid sub-type: {subtype}")
                    return f"hybrid_{subtype}"
            
            # 🔧 BETTER DETECTION: Analyze the query directly for similarity-primary patterns
            if state.query_understanding and hasattr(state.query_understanding, 'original_query'):
                query = state.query_understanding.original_query.lower()
                
                # Similarity-primary indicators: artist + "like" + modifier
                similarity_phrases = ['like', 'similar to', 'sounds like', 'reminds me of']
                has_similarity = any(phrase in query for phrase in similarity_phrases)
                
                # Style modifiers that indicate this is similarity + genre hybrid
                style_modifiers = ['but', 'with', 'and', 'plus', 'jazzy', 'chill', 'upbeat', 'dark', 'electronic']
                has_style_modifier = any(modifier in query for modifier in style_modifiers)
                
                # Check for artist names in entities
                has_artists = False
                if state.entities and state.entities.get('musical_entities', {}).get('artists', {}).get('primary'):
                    has_artists = True
                
                # 🔧 KEY FIX: "Music like [Artist] but [style]" = similarity_primary 
                if has_similarity and has_artists and has_style_modifier:
                    self.logger.info(f"🔧 DETECTED SIMILARITY-PRIMARY: Query '{query}' has artist + similarity phrase + style modifier")
                    return 'hybrid_similarity_primary'
                
                # Discovery-primary indicators
                discovery_terms = ['underground', 'new', 'hidden', 'unknown', 'discover', 'find']
                has_discovery = any(term in query for term in discovery_terms)
                
                if has_discovery:
                    self.logger.info(f"🔧 DETECTED DISCOVERY-PRIMARY: Query '{query}' has discovery indicators")
                    return 'hybrid_discovery_primary'
                
                # Genre-primary fallback
                self.logger.info(f"🔧 DETECTED GENRE-PRIMARY: Default for style-focused hybrid query '{query}'")
                return 'hybrid_genre_primary'
            
            # Fallback: detect from query and entities using query utils
            if state.query_understanding and state.entities:
                query_utils = QueryAnalysisUtils()
                
                subtype = query_utils.detect_hybrid_subtype(
                    state.query_understanding.original_query,
                    state.entities
                )
                self.logger.info(f"🔧 Detected hybrid sub-type via query utils: {subtype}")
                return f"hybrid_{subtype}"
                
        except Exception as e:
            self.logger.warning(f"Failed to detect hybrid sub-type: {e}")
        
        # Default fallback
        return 'hybrid' 

    async def run(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Main judge agent workflow.
        """
        self.logger.info("🏛️ JUDGE AGENT: Starting final candidate evaluation and ranking")
        
        # Collect all candidates from various agents
        candidates = await self._collect_all_candidates(state)
        
        if not candidates:
            self.logger.warning("No candidates received from advocate agents")
            state.final_recommendations = []
            return state
        
        self.logger.info(f"🔍 JUDGE: Collected {len(candidates)} total candidates for evaluation")
        
        # 🔧 NEW: Filter out recently shown tracks for follow-up queries
        if state.recently_shown_track_ids:
            original_count = len(candidates)
            candidates = self._filter_out_recently_shown(candidates, state)
            filtered_count = len(candidates)
            
            self.logger.info(
                f"🚫 DUPLICATE FILTER: Filtered out {original_count - filtered_count} "
                f"previously shown tracks (kept {filtered_count})"
            )
            
            if not candidates:
                self.logger.warning("All candidates were filtered as duplicates")
                state.final_recommendations = []
                return state
        else:
            self.logger.info("🔍 JUDGE DEBUG: No recently shown tracks to filter, skipping duplicate filtering")
        
        # Score all candidates
        scored_candidates = await self._score_all_candidates(candidates, state)
        
        # Apply ranking and diversity constraints
        final_recommendations = await self._rank_and_select(scored_candidates, state)
        
        state.final_recommendations = final_recommendations
        
        self.logger.info(f"✅ JUDGE: Selected {len(final_recommendations)} final recommendations")
        
        return state 

    def _filter_out_recently_shown(
        self, 
        candidates: List[TrackRecommendation], 
        state: MusicRecommenderState
    ) -> List[TrackRecommendation]:
        """Filter out tracks that were recently shown to avoid duplicates in follow-up queries."""
        if not state.recently_shown_track_ids:
            return candidates
        
        recently_shown_set = set(state.recently_shown_track_ids)
        filtered_candidates = []
        
        # 🔧 DEBUG: Log the recently shown IDs
        self.logger.info(
            f"🚫 FILTERING DUPLICATES: Checking {len(candidates)} candidates against "
            f"{len(recently_shown_set)} recently shown tracks"
        )
        self.logger.info(f"🚫 Recently shown track IDs: {list(recently_shown_set)}")
        
        for candidate in candidates:
            # Create the same track ID format as used in EnhancedRecommendationService
            candidate_track_id = f"{candidate.artist.lower().strip()}::{candidate.title.lower().strip()}"
            
            # 🔧 DEBUG: Log each candidate check
            is_duplicate = candidate_track_id in recently_shown_set
            self.logger.info(
                f"🔍 Checking: '{candidate_track_id}' -> {'DUPLICATE' if is_duplicate else 'NEW'}"
            )
            
            if not is_duplicate:
                filtered_candidates.append(candidate)
            else:
                self.logger.info(
                    f"🚫 Filtering duplicate: {candidate.title} by {candidate.artist} "
                    f"(ID: {candidate_track_id})"
                )
        
        return filtered_candidates 