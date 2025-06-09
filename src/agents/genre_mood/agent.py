"""
Refactored Genre Mood Agent

Modular GenreMoodAgent using extracted components for better maintainability.
Follows single responsibility principle with clear separation of concerns.
"""

from typing import Dict, List, Any
import structlog

from ...models.agent_models import MusicRecommenderState, AgentConfig
from ...models.recommendation_models import TrackRecommendation
from ...services.api_service import APIService
from ...services.metadata_service import MetadataService
from ..base_agent import BaseAgent
from ..components.unified_candidate_generator import UnifiedCandidateGenerator
from ..components import QualityScorer
from ..components.llm_utils import LLMUtils

# Import modular components
from .components import (
    GenreMoodConfig,
    MoodAnalyzer,
    GenreProcessor,
    TagGenerator
)

logger = structlog.get_logger(__name__)


class GenreMoodAgent(BaseAgent):
    """
    Refactored Genre Mood Agent with modular components.
    
    Responsibilities:
    - Orchestrate genre and mood-based music discovery
    - Coordinate between specialized components
    - Manage workflow and state transitions
    - Handle context overrides and intent adaptation
    
    Uses modular components:
    - GenreMoodConfig: Intent parameter management
    - MoodAnalyzer: Mood detection and analysis
    - GenreProcessor: Genre matching and filtering
    - TagGenerator: Tag generation and enhancement
    - UnifiedCandidateGenerator: Candidate generation
    - QualityScorer: Quality assessment
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
        Initialize refactored genre mood agent with modular components.
        
        Args:
            config: Agent configuration
            llm_client: LLM client for reasoning
            api_service: Unified API service
            metadata_service: Unified metadata service
            rate_limiter: Rate limiter for LLM API calls
            session_manager: SessionManagerService for candidate pool persistence
        """
        super().__init__(
            config=config, 
            llm_client=llm_client, 
            api_service=api_service,
            metadata_service=metadata_service,
            rate_limiter=rate_limiter
        )
        
        # Initialize modular components
        self.config_manager = GenreMoodConfig()
        self.mood_analyzer = MoodAnalyzer()
        self.genre_processor = GenreProcessor(api_service)
        self.tag_generator = TagGenerator()
        
        # Shared components (initialized in parent with rate limiter)
        self.candidate_generator = UnifiedCandidateGenerator(api_service, session_manager, llm_client=llm_client)
        self.quality_scorer = QualityScorer()
        
        self.logger.info("Refactored GenreMoodAgent initialized with modular components")
    
    async def process(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Generate genre and mood-based recommendations using modular components.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with genre/mood recommendations
        """
        try:
            self.logger.info("Starting refactored genre/mood agent processing")
            
            # Extract entities and intent from planner
            entities = state.entities or {}
            intent_analysis = state.intent_analysis or {}
            
            # 🔧 FIX: Check if this is a follow-up query - if so, skip generation
            is_followup = intent_analysis.get('is_followup', False)
            if is_followup:
                self.logger.info("🔄 Follow-up query detected - skipping genre/mood generation, letting judge agent use persisted pool")
                state.genre_mood_recommendations = []
                return state
            
            # Handle context override and intent detection
            context_override_applied, target_artist_from_override = self._handle_context_override(
                state, entities, intent_analysis
            )
            
            # Detect and adapt to intent
            detected_intent = self._detect_intent(state, intent_analysis, context_override_applied)
            self.config_manager.adapt_to_intent(detected_intent)
            
            # 🔧 Phase 3: Check if planner recommends large pool generation
            should_generate_pool = False
            if hasattr(state, 'planning_strategy') and state.planning_strategy:
                should_generate_pool = state.planning_strategy.get('generate_large_pool', False)
                self.logger.info(f"🎯 Planner recommends pool generation: {should_generate_pool}")
            
            # Get session_id from state if available (needed for pool generation)
            session_id = getattr(state, 'session_id', None) or 'default_session'
            
            # Phase 3: Generate large pool if recommended by PlannerAgent
            if should_generate_pool:
                self.logger.info("Phase 3: Generating large candidate pool for future follow-ups")
                pool_key = await self.candidate_generator.generate_and_persist_large_pool(
                    entities=entities,
                    intent_analysis=intent_analysis,
                    session_id=session_id,
                    agent_type="genre_mood",
                    detected_intent=detected_intent
                )
                if pool_key:
                    self.logger.info(f"Large pool generated with key: {pool_key}")
            
            # Generate candidates using shared generator
            candidates = await self.candidate_generator.generate_candidate_pool(
                entities=entities,
                intent_analysis=intent_analysis,
                agent_type="genre_mood",
                target_candidates=self.config_manager.target_candidates,
                detected_intent=detected_intent
            )
            
            self.logger.debug(f"Generated {len(candidates)} candidates")
            
            # Score candidates using modular scoring
            scored_candidates = await self._score_candidates(candidates, entities, intent_analysis)
            
            # Apply context override boosting if needed
            if context_override_applied and target_artist_from_override:
                scored_candidates = self._apply_context_boosting(
                    scored_candidates, target_artist_from_override
                )
            
            # Filter by genre requirements
            filtered_candidates = await self.genre_processor.filter_by_genre_requirements(
                scored_candidates, entities, self.llm_client
            )
            
            # Apply diversity filtering
            diverse_candidates = self._ensure_diversity(
                filtered_candidates, 
                entities, 
                intent_analysis, 
                context_override_applied,
                target_artist_from_override
            )
            
            # Create final recommendations
            recommendations = await self._create_recommendations(
                diverse_candidates[:self.config_manager.final_recommendations],
                entities,
                intent_analysis
            )
            
            # Update state
            state.genre_mood_recommendations = recommendations
            
            self.logger.info(
                "Refactored genre/mood agent processing completed",
                candidates=len(candidates),
                filtered=len(filtered_candidates),
                diverse=len(diverse_candidates),
                recommendations=len(recommendations),
                context_override_applied=context_override_applied,
                pool_generated=should_generate_pool
            )
            
            return state
            
        except Exception as e:
            self.logger.error("Refactored genre/mood agent processing failed", error=str(e))
            state.genre_mood_recommendations = []
            return state
    
    def _handle_context_override(
        self, 
        state: MusicRecommenderState, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Handle context override processing.
        
        Returns:
            Tuple of (context_override_applied, target_artist_from_override)
        """
        context_override_applied = False
        target_artist_from_override = None
        
        if hasattr(state, 'context_override') and state.context_override:
            context_override = state.context_override
            
            self.logger.info(f"🔧 DEBUG: GenreMood Agent received context_override: {type(context_override)}")
            
            # Handle both dictionary and object formats
            is_followup = False
            target_entity = None
            intent_override = None
            
            if isinstance(context_override, dict):
                is_followup = context_override.get('is_followup', False)
                target_entity = context_override.get('target_entity')
                intent_override = context_override.get('intent_override')
            elif hasattr(context_override, 'is_followup'):
                is_followup = context_override.is_followup
                target_entity = getattr(context_override, 'target_entity', None)
                intent_override = getattr(context_override, 'intent_override', None)
            
            if is_followup and target_entity:
                target_artist_from_override = target_entity
                context_override_applied = True
                
                self.logger.info(
                    "🚀 GenreMood Agent: Context override detected",
                    intent_override=intent_override,
                    target_entity=target_entity,
                    is_followup=is_followup
                )
                
                # Override detected intent for followup queries
                if intent_override == 'artist_deep_dive':
                    intent_analysis['intent'] = 'artist_similarity'
                    self.logger.info(f"🎯 Overriding intent to 'artist_similarity' for artist deep dive: {target_artist_from_override}")
                
                # Inject target artist into entities
                self._inject_target_artist_into_entities(entities, target_artist_from_override)
        
        return context_override_applied, target_artist_from_override
    
    def _inject_target_artist_into_entities(self, entities: Dict[str, Any], target_artist: str) -> None:
        """Inject target artist into entities for candidate generation."""
        musical_entities = entities.get('musical_entities', {})
        artists = musical_entities.get('artists', {})
        if 'primary' not in artists:
            artists['primary'] = []
        
        # Ensure target artist is in primary artists list
        if target_artist not in artists['primary']:
            artists['primary'].insert(0, target_artist)  # Put at front
            self.logger.info(f"🎯 Injected target artist '{target_artist}' into entities for candidate generation")
        
        musical_entities['artists'] = artists
        entities['musical_entities'] = musical_entities
    
    def _detect_intent(
        self, 
        state: MusicRecommenderState, 
        intent_analysis: Dict[str, Any], 
        context_override_applied: bool
    ) -> str:
        """Detect the effective intent for this processing round."""
        query_understanding = state.query_understanding
        detected_intent = 'genre_mood'  # Default for genre/mood agent
        
        if query_understanding and hasattr(query_understanding, 'intent'):
            # QueryUnderstanding is an object, get intent value
            intent_from_query = query_understanding.intent.value if hasattr(query_understanding.intent, 'value') else str(query_understanding.intent)
            detected_intent = intent_from_query
            
            # Add intent to intent_analysis if missing
            if not intent_analysis.get('intent') and intent_from_query:
                intent_analysis['intent'] = intent_from_query
        else:
            self.logger.warning("No query_understanding or intent found in state, using default genre/mood parameters")
        
        # If context override applied, use the override intent
        if context_override_applied and hasattr(state, 'context_override'):
            context_override = state.context_override
            if hasattr(context_override, 'intent_override') and context_override.intent_override == 'artist_deep_dive':
                detected_intent = 'artist_similarity'
        
        self.logger.info(f"Detected intent: {detected_intent}")
        return detected_intent
    
    async def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score candidates using modular components."""
        scored_candidates = []
        
        for candidate in candidates:
            try:
                # Use shared quality scorer
                quality_score = await self.quality_scorer.calculate_quality_score(
                    candidate, entities, intent_analysis
                )
                
                # Calculate component-specific scores
                genre_score = await self.genre_processor.calculate_genre_score(candidate, entities)
                mood_score = self.mood_analyzer.calculate_mood_score(candidate, entities, intent_analysis)
                tag_score = self.tag_generator.calculate_tag_based_score(candidate, entities, intent_analysis)
                
                # Combined genre/mood score
                config = self.config_manager.get_current_config()
                genre_mood_score = (
                    genre_score * config['genre_weight'] +
                    mood_score * config['mood_weight'] +
                    tag_score * 0.2  # Tag score weight
                )
                
                # Final combined score
                candidate['quality_score'] = quality_score
                candidate['genre_score'] = genre_score
                candidate['mood_score'] = mood_score
                candidate['tag_score'] = tag_score
                candidate['genre_mood_score'] = genre_mood_score
                candidate['combined_score'] = (quality_score * 0.6) + (genre_mood_score * 0.4)
                
                scored_candidates.append(candidate)
                
            except Exception as e:
                self.logger.warning(f"Failed to score candidate: {e}")
                continue
        
        # Sort by combined score
        scored_candidates.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        return scored_candidates
    
    def _apply_context_boosting(
        self, 
        candidates: List[Dict[str, Any]], 
        target_artist: str
    ) -> List[Dict[str, Any]]:
        """Apply context-based boosting to target artist tracks."""
        self.logger.info(f"🎵 GenreMood Agent: Boosting tracks by target artist: {target_artist}")
        
        for candidate in candidates:
            candidate_artist = candidate.get('artist', '')
            if candidate_artist.lower() == target_artist.lower():
                original_score = candidate.get('combined_score', 0.0)
                candidate['combined_score'] = min(original_score + 0.3, 1.0)
                self.logger.info(f"🚀 BOOSTED genre/mood track: {candidate.get('name')} by {candidate_artist} from {original_score:.3f} to {candidate['combined_score']:.3f}")
        
        # Re-sort after boosting
        candidates.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        return candidates
    
    def _ensure_diversity(
        self,
        candidates: List[Dict[str, Any]], 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        context_override_applied: bool,
        target_artist_from_override: str = None
    ) -> List[Dict[str, Any]]:
        """Ensure diversity in artists and sources using configuration."""
        seen_artists = {}
        diverse_candidates = []
        
        # Get diversity parameters from configuration
        config = self.config_manager.get_current_config()
        max_tracks_per_artist = 1  # Default strict diversity
        
        # Check for by_artist intent
        intent = intent_analysis.get('intent', '').lower()
        if intent == 'by_artist':
            max_tracks_per_artist = 10  # Allow many tracks from target artist
            self.logger.info(f"🎯 BY_ARTIST intent detected: allowing {max_tracks_per_artist} tracks per artist")
        
        # Apply context override constraints
        if context_override_applied and target_artist_from_override:
            max_tracks_per_artist = 10  # Increased for better artist deep dives
            self.logger.info(f"🚀 GenreMood Context override: allowing up to {max_tracks_per_artist} tracks from target artist '{target_artist_from_override}'")
        
        # Check for genre-specific query
        is_genre_specific_query = (
            self.genre_processor.has_genre_requirements(entities) and 
                            (intent_analysis.get('intent') == 'hybrid_similarity_genre' or intent_analysis.get('primary_intent') == 'hybrid_similarity_genre')
        )
        
        if is_genre_specific_query:
            max_tracks_per_artist = max(max_tracks_per_artist, 5)
            self.logger.info(f"🎯 Genre-specific query detected: relaxed diversity - {max_tracks_per_artist} per artist")
        
        # Apply diversity filtering
        total_limit = config['final_recommendations'] * 2
        if intent == 'by_artist' or is_genre_specific_query:
            total_limit = min(config['final_recommendations'] * 4, 50)
        
        for candidate in candidates:
            artist = candidate.get('artist', '').lower()
            artist_count = seen_artists.get(artist, 0)
            
            # Apply different limits for target vs non-target artists
            is_target_artist = target_artist_from_override and artist == target_artist_from_override.lower()
            artist_limit = max_tracks_per_artist if is_target_artist else (1 if target_artist_from_override else max_tracks_per_artist)
            
            if artist_count >= artist_limit:
                continue
            
            # Accept the candidate
            seen_artists[artist] = artist_count + 1
            diverse_candidates.append(candidate)
            
            # Limit to prevent over-representation
            if len(diverse_candidates) >= total_limit:
                break
        
        self.logger.info(f"🔧 GenreMood diversity filtering: {len(candidates)} -> {len(diverse_candidates)} candidates")
        if target_artist_from_override:
            target_count = sum(1 for c in diverse_candidates if c.get('artist', '').lower() == target_artist_from_override.lower())
            self.logger.info(f"🚀 GenreMood target artist '{target_artist_from_override}' tracks in final candidates: {target_count}")
        
        return diverse_candidates
    
    async def _create_recommendations(
        self,
        candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[TrackRecommendation]:
        """Create final track recommendations using modular components."""
        recommendations = []
        
        for i, candidate in enumerate(candidates):
            try:
                # Generate reasoning using shared LLM utils if available
                reasoning = await self._generate_reasoning(candidate, entities, intent_analysis, i + 1)
                
                # Extract enhanced tags using TagGenerator
                enhanced_tags = self.tag_generator.enhance_recommendation_tags(
                    candidate, entities, intent_analysis, self.mood_analyzer
                )
                
                # Extract genres using GenreProcessor
                genres = self.genre_processor.extract_genres_for_recommendation(candidate, entities)
                
                recommendation = TrackRecommendation(
                    title=candidate.get('name', 'Unknown'),
                    artist=candidate.get('artist', 'Unknown'),
                    id=f"{candidate.get('artist', 'Unknown')}_{candidate.get('name', 'Unknown')}".replace(' ', '_').lower(),
                    source='genre_mood_agent',
                    track_url=candidate.get('url', ''),
                    album_title=candidate.get('album', ''),
                    genres=genres,
                    moods=enhanced_tags,
                    confidence=candidate.get('combined_score', 0.5),
                    explanation=reasoning,
                    quality_score=candidate.get('quality_score', 0.0),
                    advocate_source_agent='genre_mood_agent',
                    raw_source_data={
                        'playcount': candidate.get('playcount', 0),
                        'listeners': candidate.get('listeners', 0),
                        'popularity': candidate.get('popularity', 0),
                        'tags': candidate.get('tags', []),
                        'quality_score': candidate.get('quality_score', 0.0)
                    },
                    additional_scores={
                        'combined_score': candidate.get('combined_score', 0.5),
                        'quality_score': candidate.get('quality_score', 0.0),
                        'genre_mood_score': candidate.get('genre_mood_score', 0.0),
                        'genre_score': candidate.get('genre_score', 0.0),
                        'mood_score': candidate.get('mood_score', 0.0),
                        'tag_score': candidate.get('tag_score', 0.0)
                    }
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                self.logger.warning(f"Failed to create recommendation: {e}")
                continue
        
        return recommendations
    
    async def _generate_reasoning(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        rank: int
    ) -> str:
        """Generate reasoning for genre/mood recommendation."""
        try:
            # For now, skip LLM calls to avoid rate limits - use fallback reasoning
            return self._create_fallback_reasoning(candidate, entities, intent_analysis, rank)
            
        except Exception as e:
            self.logger.debug(f"LLM reasoning failed, using fallback: {e}")
            return self._create_fallback_reasoning(candidate, entities, intent_analysis, rank)
    
    def _create_fallback_reasoning(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        rank: int
    ) -> str:
        """Create fallback reasoning when LLM is unavailable."""
        name = candidate.get('name', 'This track')
        artist = candidate.get('artist', 'the artist')
        tags = candidate.get('tags', [])[:3]
        
        reasoning_parts = [f"#{rank}: {name} by {artist}"]
        
        if tags:
            reasoning_parts.append(f"Tagged as {', '.join(tags)}")
        
        quality_score = candidate.get('quality_score', 0)
        if quality_score > 0.8:
            reasoning_parts.append("High quality match")
        elif quality_score > 0.6:
            reasoning_parts.append("Good quality match")
        
        return ". ".join(reasoning_parts) + "." 