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
from ..components.llm_utils import LLMUtils
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
        rate_limiter=None
    ):
        """
        Initialize simplified judge agent with injected dependencies.
        
        Args:
            config: Agent configuration
            llm_client: LLM client for explanations
            api_service: Unified API service
            metadata_service: Unified metadata service
            rate_limiter: Rate limiter for LLM API calls
        """
        super().__init__(
            config=config, 
            llm_client=llm_client, 
            api_service=api_service,
            metadata_service=metadata_service,
            rate_limiter=rate_limiter
        )
        
        # Shared components (LLMUtils now initialized in parent with rate limiter)
        self.quality_scorer = QualityScorer()
        
        # Configuration
        self.final_recommendations = 25
        self.diversity_targets = {
            'max_per_artist': 2,
            'min_genres': 3,
            'source_distribution': {'genre_mood_agent': 0.4, 'discovery_agent': 0.4, 'planner_agent': 0.2}
        }
        
        # Initialize shared components
        self.ranking_logic = RankingLogic()
        
        self.logger.info("Simplified JudgeAgent initialized with dependency injection")
    
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
            
            # Collect all candidate recommendations
            all_candidates = self._collect_all_candidates(state)
            
            if not all_candidates:
                self.logger.warning("No candidates found for evaluation")
                state.final_recommendations = []
                return state
            
            self.logger.debug(f"Collected {len(all_candidates)} total candidates")
            
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
        
        # Extract context from state
        entities = state.entities or {}
        intent_analysis = state.intent_analysis or {}
        query_understanding = state.query_understanding
        
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
                        self.logger.info(f"🎯 DEBUG Judge: Giving 0.95 intent alignment to Mk.gee track!")
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
        
        # Get intent-specific novelty threshold
        novelty_threshold = ranking_logic.get_novelty_threshold(intent)
        self.logger.info(f"🔧 Using novelty threshold for intent '{intent}': {novelty_threshold}")
        
        # Rank candidates using intent-aware logic
        ranked_candidates = ranking_logic.rank_recommendations(
            candidates=scored_candidates, 
            intent=intent,
            entities=state.entities,
            intent_analysis=state.intent_analysis,
            novelty_threshold=novelty_threshold,
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
        
        # Use intent-specific diversity limits
        diversity_limits = self.ranking_logic.get_diversity_limits(intent)
        max_per_artist = diversity_limits.get('max_per_artist', 2)
        
        self.logger.debug(f"🔧 DEBUG: Intent: {intent}, max tracks per artist: {max_per_artist}")
        
        self.logger.debug(f"Starting diversity selection with {len(ranked_candidates)} candidates")
        
        for i, (candidate, scores) in enumerate(ranked_candidates):
            self.logger.debug(
                f"Evaluating candidate {i+1}: {candidate.title} by {candidate.artist}, "
                f"source={candidate.source}, score={scores.get('final_score', 0):.3f}"
            )
            
            # Check artist diversity using intent-specific limits
            if artist_counts.get(candidate.artist, 0) >= max_per_artist:
                self.logger.debug(f"Skipping {candidate.title}: too many tracks from {candidate.artist}")
                continue
            
            # Check source distribution
            source_count = source_counts.get(candidate.source, 0)
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
                    self.logger.info(f"🔧 DETECTED SIMILARITY-PRIMARY: Query has artist + similarity phrase + style modifier")
                    return 'hybrid_similarity_primary'
                
                # Discovery-primary indicators
                discovery_terms = ['underground', 'new', 'hidden', 'unknown', 'discover', 'find']
                has_discovery = any(term in query for term in discovery_terms)
                
                if has_discovery:
                    self.logger.info(f"🔧 DETECTED DISCOVERY-PRIMARY: Query has discovery indicators")
                    return 'hybrid_discovery_primary'
                
                # Genre-primary fallback
                self.logger.info(f"🔧 DETECTED GENRE-PRIMARY: Default for style-focused hybrid")
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