"""
Simplified Genre Mood Agent

Refactored to use dependency injection and shared components, eliminating:
- Client instantiation duplication
- Candidate generation duplication
- LLM calling duplication
- Quality scoring duplication
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

logger = structlog.get_logger(__name__)


class GenreMoodAgent(BaseAgent):
    """
    Simplified Genre Mood Agent with dependency injection.
    
    Responsibilities:
    - Genre and mood-based music discovery
    - Tag-based search strategies
    - Energy level matching
    - Quality-filtered recommendations
    
    Uses shared components to eliminate duplication:
    - UnifiedCandidateGenerator for candidate generation
    - QualityScorer for quality assessment
    - LLMUtils for LLM interactions
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
        Initialize simplified genre mood agent with injected dependencies.
        
        Args:
            config: Agent configuration
            llm_client: LLM client for reasoning
            api_service: Unified API service
            metadata_service: Unified metadata service
            rate_limiter: Rate limiter for LLM API calls
            session_manager: SessionManagerService for candidate pool persistence (Phase 3)
        """
        super().__init__(
            config=config, 
            llm_client=llm_client, 
            api_service=api_service,
            metadata_service=metadata_service,
            rate_limiter=rate_limiter
        )
        
        # Shared components (LLMUtils now initialized in parent with rate limiter)
        # Phase 3: Pass session_manager to UnifiedCandidateGenerator for candidate pool persistence
        self.candidate_generator = UnifiedCandidateGenerator(api_service, session_manager)
        self.quality_scorer = QualityScorer()
        
        # Base configuration - will be adapted based on intent
        self.target_candidates = 100
        self.final_recommendations = 20
        self.quality_threshold = 0.4
        
        # Mood and genre mappings
        self.mood_mappings = self._initialize_mood_mappings()
        self.energy_mappings = self._initialize_energy_mappings()
        
        # Intent-specific parameter configurations from design document
        self.intent_parameters = {
            'genre_mood': {
                'quality_threshold': 0.5,     # Higher quality for genre/mood focus
                'genre_weight': 0.6,          # High genre matching weight
                'mood_weight': 0.7,           # High mood matching weight
                'audio_feature_weight': 0.8,  # Strong audio feature focus
                'max_per_genre': 8,           # Allow more per genre
                'candidate_focus': 'style_precision'
            },
            'artist_similarity': {
                'quality_threshold': 0.45,    # Moderate quality
                'genre_weight': 0.4,          # Moderate genre matching
                'mood_weight': 0.3,           # Less mood focus
                'audio_feature_weight': 0.5,  # Moderate audio features
                'max_per_genre': 5,           # Balanced genre diversity
                'candidate_focus': 'similar_style'
            },
            'contextual': {
                'quality_threshold': 0.6,     # High quality for functional use
                'genre_weight': 0.3,          # Less genre restriction
                'mood_weight': 0.8,           # Very high mood importance
                'audio_feature_weight': 0.9,  # Critical audio features (BPM, energy)
                'max_per_genre': 6,           # Moderate genre diversity
                'candidate_focus': 'functional_audio'
            },
            'discovery': {
                'quality_threshold': 0.3,     # Lower for discovery
                'genre_weight': 0.5,          # Moderate genre matching
                'mood_weight': 0.4,           # Moderate mood matching
                'audio_feature_weight': 0.4,  # Moderate audio features
                'max_per_genre': 3,           # High genre diversity
                'candidate_focus': 'genre_exploration'
            },
            'hybrid': {
                'quality_threshold': 0.4,     # Balanced quality
                'genre_weight': 0.5,          # Balanced genre matching
                'mood_weight': 0.6,           # Good mood matching
                'audio_feature_weight': 0.6,  # Good audio feature focus
                'max_per_genre': 4,           # Balanced diversity
                'candidate_focus': 'balanced_style'
            }
        }
        
        self.logger.info("Simplified GenreMoodAgent initialized with intent-aware parameters")
    
    def _adapt_to_intent(self, intent: str) -> None:
        """Adapt agent parameters based on detected intent."""
        if intent in self.intent_parameters:
            params = self.intent_parameters[intent]
            
            self.quality_threshold = params['quality_threshold']
            # Store other parameters for use in processing
            self.current_genre_weight = params['genre_weight']
            self.current_mood_weight = params['mood_weight']
            self.current_audio_feature_weight = params['audio_feature_weight']
            self.current_max_per_genre = params['max_per_genre']
            self.current_candidate_focus = params['candidate_focus']
            
            self.logger.info(
                f"Adapted genre/mood parameters for intent: {intent}",
                quality_threshold=self.quality_threshold,
                genre_weight=self.current_genre_weight,
                mood_weight=self.current_mood_weight,
                audio_feature_weight=self.current_audio_feature_weight
            )
        else:
            # Set defaults
            self.current_genre_weight = 0.5
            self.current_mood_weight = 0.5
            self.current_audio_feature_weight = 0.5
            self.current_max_per_genre = 4
            self.current_candidate_focus = 'balanced_style'
            self.logger.warning(f"Unknown intent: {intent}, using default parameters")
    
    async def process(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Generate genre and mood-based recommendations using shared components.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with genre/mood recommendations
        """
        try:
            self.logger.info("Starting genre/mood agent processing")
            
            # Extract entities and intent from planner
            entities = state.entities or {}
            intent_analysis = state.intent_analysis or {}
            
            # 🚀 CHECK FOR CONTEXT OVERRIDE FIRST
            context_override_applied = False
            target_artist_from_override = None
            
            if hasattr(state, 'context_override') and state.context_override:
                context_override = state.context_override
                
                self.logger.info(f"🔧 DEBUG: GenreMood Agent received context_override: {type(context_override)}")
                self.logger.info(f"🔧 DEBUG: context_override data: {context_override}")
                
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
                else:
                    self.logger.info(f"🔧 DEBUG: No followup detected - is_followup={is_followup}, target_entity={target_entity}")
            else:
                self.logger.info("🔧 DEBUG: No context_override found in state")
            
            # 🔧 Get intent and adapt parameters accordingly
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
            
            # 🔧 If context override applied, use the override intent
            if context_override_applied and hasattr(context_override, 'intent_override'):
                if context_override.intent_override == 'artist_deep_dive':
                    detected_intent = 'artist_similarity'
            
            # 🚀 If we have a target artist from context override, inject it into entities
            if target_artist_from_override:
                musical_entities = entities.get('musical_entities', {})
                artists = musical_entities.get('artists', {})
                if 'primary' not in artists:
                    artists['primary'] = []
                
                # Ensure target artist is in primary artists list
                if target_artist_from_override not in artists['primary']:
                    artists['primary'].insert(0, target_artist_from_override)  # Put at front
                    self.logger.info(f"🎯 Injected target artist '{target_artist_from_override}' into entities for candidate generation")
                
                musical_entities['artists'] = artists
                entities['musical_entities'] = musical_entities
            
            # 🚀 PHASE 2: Adapt agent parameters based on detected intent
            self._adapt_to_intent(detected_intent)
            
            # Phase 1: Generate candidates using shared generator
            candidates = await self.candidate_generator.generate_candidate_pool(
                entities=entities,
                intent_analysis=intent_analysis,
                agent_type="genre_mood",
                target_candidates=self.target_candidates,
                detected_intent=detected_intent
            )
            
            self.logger.debug(f"Generated {len(candidates)} candidates")
            
            # Phase 2: Score candidates using shared quality scorer
            scored_candidates = await self._score_candidates(candidates, entities, intent_analysis)
            
            # 🚀 BOOST target artist tracks if context override applied
            if context_override_applied and target_artist_from_override:
                self.logger.info(f"🎵 GenreMood Agent: Boosting tracks by target artist: {target_artist_from_override}")
                for candidate in scored_candidates:
                    candidate_artist = candidate.get('artist', '')
                    if candidate_artist.lower() == target_artist_from_override.lower():
                        original_score = candidate.get('combined_score', 0.0)
                        candidate['combined_score'] = min(original_score + 0.3, 1.0)
                        self.logger.info(f"🚀 BOOSTED genre/mood track: {candidate.get('name')} by {candidate_artist} from {original_score:.3f} to {candidate['combined_score']:.3f}")
                
                # Re-sort after boosting
                scored_candidates.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            
            # Phase 3: Filter and rank by genre/mood relevance
            filtered_candidates = await self._filter_by_genre_requirements(scored_candidates, entities, self.llm_client)
            
            # Apply diversity filtering
            # 🚀 Pass context override to diversity filtering
            filtered_candidates = self._ensure_diversity(
                filtered_candidates, 
                entities, 
                intent_analysis, 
                context_override if context_override_applied else None
            )
            
            # Phase 4: Create final recommendations
            recommendations = await self._create_recommendations(
                filtered_candidates[:self.final_recommendations],
                entities,
                intent_analysis
            )
            
            # Update state
            state.genre_mood_recommendations = [rec.model_dump() for rec in recommendations]
            
            self.logger.info(
                "Genre/mood agent processing completed",
                candidates=len(candidates),
                filtered=len(filtered_candidates),
                recommendations=len(recommendations),
                context_override_applied=context_override_applied
            )
            
            return state
            
        except Exception as e:
            self.logger.error("Genre/mood agent processing failed", error=str(e))
            state.genre_mood_recommendations = []
            return state
    
    async def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score candidates using shared quality scorer."""
        scored_candidates = []
        
        for candidate in candidates:
            try:
                # Use shared quality scorer
                quality_score = await self.quality_scorer.calculate_quality_score(
                    candidate, entities, intent_analysis
                )
                
                # Add genre/mood specific scoring
                genre_mood_score = await self._calculate_genre_mood_score(candidate, entities, intent_analysis)
                
                # Combined score
                candidate['quality_score'] = quality_score
                candidate['genre_mood_score'] = genre_mood_score
                candidate['combined_score'] = (quality_score * 0.6) + (genre_mood_score * 0.4)
                
                scored_candidates.append(candidate)
                
            except Exception as e:
                self.logger.warning(f"Failed to score candidate: {e}")
                continue
        
        # Sort by combined score
        scored_candidates.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        return scored_candidates
    
    async def _calculate_genre_mood_score(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> float:
        """Calculate genre/mood specific relevance score using enhanced API-based matching."""
        score = 0.1  # Start with lower base score
        
        # Extract candidate information
        candidate_tags = candidate.get('tags', [])
        candidate_name = candidate.get('name', '').lower()
        candidate_artist = candidate.get('artist', '').lower()
        
        # Score based on genre matching using new API service
        target_genres = self._extract_target_genres(entities)
        
        # 🎯 NEW: Use API service for accurate genre matching and scoring
        for genre in target_genres:
            try:
                # Check if candidate matches this genre using our new API method
                match_result = await self._check_genre_match(candidate, genre)
                if match_result.get('matches', False):
                    # Give high score for API-confirmed genre matches
                    score += 0.6  # Much higher than old 0.3
                    self.logger.debug(f"🎵 API-confirmed genre match: {candidate_artist} - {candidate_name} matches {genre}")
                    
            except Exception as e:
                self.logger.debug(f"API genre matching failed, falling back to tag matching for {genre}: {e}")
                # Fallback to simple tag matching if API fails
            if any(genre.lower() in tag.lower() for tag in candidate_tags):
                score += 0.3
            if genre.lower() in candidate_name or genre.lower() in candidate_artist:
                score += 0.2
        
        # Score based on mood matching (keep existing logic)
        target_moods = self._extract_target_moods(entities, intent_analysis)
        for mood in target_moods:
            mood_tags = self.mood_mappings.get(mood, [])
            for mood_tag in mood_tags:
                if any(mood_tag.lower() in tag.lower() for tag in candidate_tags):
                    score += 0.2
        
        # Score based on energy level (keep existing logic)
        energy_level = self._extract_energy_level(entities, intent_analysis)
        energy_tags = self.energy_mappings.get(energy_level, [])
        for energy_tag in energy_tags:
            if any(energy_tag.lower() in tag.lower() for tag in candidate_tags):
                score += 0.3
        
        # Bonus for having any relevant tags (very liberal matching)
        common_music_tags = ['rock', 'indie', 'electronic', 'pop', 'alternative', 'experimental', 'ambient']
        for tag in candidate_tags:
            if any(music_tag in tag.lower() for music_tag in common_music_tags):
                score += 0.1
                break  # Only add bonus once
        
        return min(score, 1.0)
    
    async def _filter_by_genre_requirements(
        self,
        candidates: List[Dict[str, Any]], 
        entities: Dict[str, Any],
        llm_client=None
    ) -> List[Dict[str, Any]]:
        """Filter candidates based on genre requirements using enhanced batch processing."""
        # Extract required genres
        musical_entities = entities.get('musical_entities', {})
        genres = musical_entities.get('genres', {})
        required_genres = genres.get('primary', [])
        
        if not required_genres:
            self.logger.debug("No genre requirements found, keeping all candidates")
            return candidates
        
        self.logger.info(f"🎯 Filtering {len(candidates)} candidates for genres: {required_genres}")
        
        # For genre filtering, we require ALL primary genres to be matched
        filtered_candidates = []
        
        for genre in required_genres:
            genre_matches = []
            
            # Prepare track data for batch processing
            track_data = []
            for candidate in candidates:
                track_info = {
                    'artist': candidate.get('artist', ''),
                    'name': candidate.get('name', ''),
                    'tags': candidate.get('tags', [])
                }
                track_data.append(track_info)
            
            # Use batch processing for genre matching
            from ...services.api_service import get_api_service
            api_service = get_api_service()
            
            batch_results = await api_service.batch_check_tracks_genre_match(
                tracks=track_data,
                target_genre=genre,
                llm_client=llm_client,  # Pass the LLM client for batch processing
                include_related_genres=True
            )
            
            # Apply results to candidates
            for i, candidate in enumerate(candidates):
                track_key = f"{candidate.get('artist', 'Unknown')} - {candidate.get('name', 'Unknown')}"
                match_result = batch_results.get(track_key, {'matches': False})
                
                if match_result['matches']:
                    # Add match information to candidate
                    candidate['genre_match'] = {
                        'genre': genre,
                        'confidence': match_result['confidence'],
                        'matched_tags': match_result['matched_tags'],
                        'match_type': match_result['match_type'],
                        'explanation': match_result['explanation']
                    }
                    genre_matches.append(candidate)
                    
                    self.logger.debug(
                        f"✅ Genre match: {candidate.get('artist')} - {candidate.get('name')} matches {genre}",
                        match_type=match_result['match_type'],
                        confidence=match_result['confidence'],
                        matched_tags=match_result['matched_tags']
                    )
                else:
                    self.logger.debug(
                        f"❌ No genre match: {candidate.get('artist')} - {candidate.get('name')} doesn't match {genre}",
                        track_tags=candidate.get('tags', [])
                    )
            
            self.logger.info(f"🎯 Genre filtering for '{genre}': {len(candidates)} → {len(genre_matches)} candidates")
            
            # Update candidates for next genre (intersection)
            candidates = genre_matches
            
            if not candidates:
                break  # No candidates left
        
        self.logger.info(f"🎯 Final genre filtering: {len(filtered_candidates)} → {len(candidates)} candidates")
        return candidates
    
    def _ensure_diversity(self, candidates: List[Dict[str, Any]], entities: Dict[str, Any], intent_analysis: Dict[str, Any], context_override: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Ensure diversity in artists and sources."""
        seen_artists = {}  # Changed to dict to track count per artist
        diverse_candidates = []
        
        # 🚀 Apply context override constraints for followup intents
        max_tracks_per_artist = 1  # Default strict diversity
        target_artist_from_override = None
        
        # 🎯 NEW: Check for by_artist intent first
        intent = intent_analysis.get('intent', '').lower()
        if intent == 'by_artist':
            # For by_artist queries like "Music by Kendrick Lamar", allow many tracks from target artist
            max_tracks_per_artist = 10  # Allow up to 10 tracks from the target artist
            total_limit = min(self.final_recommendations * 2, 30)  # Allow more total tracks
            self.logger.info(f"🎯 BY_ARTIST intent detected: allowing {max_tracks_per_artist} tracks per artist")
        
        if context_override:
            # Handle both dictionary and object formats
            is_followup = False
            target_entity = None
            
            if isinstance(context_override, dict):
                is_followup = context_override.get('is_followup', False)
                target_entity = context_override.get('target_entity')
            elif hasattr(context_override, 'is_followup'):
                is_followup = context_override.is_followup
                target_entity = getattr(context_override, 'target_entity', None)
            
            if is_followup and target_entity:
                target_artist_from_override = target_entity
                # For followup intents, allow more tracks from target artist
                max_tracks_per_artist = 10  # Increased from 5 to 10 for better artist deep dives
                self.logger.info(f"🚀 GenreMood Context override: allowing up to {max_tracks_per_artist} tracks from target artist '{target_artist_from_override}'")
        
        # 🎯 Check if this is a genre-specific query that should have relaxed diversity
        is_genre_specific_query = (
            self._has_genre_requirements(entities) and 
            (intent_analysis.get('intent') == 'hybrid' or intent_analysis.get('primary_intent') == 'hybrid')
        )
        
        if is_genre_specific_query:
            # For genre-specific queries, allow more tracks per artist to focus on genre quality
            max_tracks_per_artist = max(max_tracks_per_artist, 5)  # Use max to not override by_artist
            total_limit = min(self.final_recommendations * 4, 50)  # Allow more total tracks
            self.logger.info(f"🎯 Genre-specific query detected: relaxed diversity - {max_tracks_per_artist} per artist, max {total_limit} total")
        elif intent != 'by_artist':
            # Standard diversity for non-genre, non-by_artist queries
            total_limit = self.final_recommendations * 2
        
        self.logger.info(f"🔧 DEBUG: GenreMood diversity filtering with {len(candidates)} candidates, max per artist: {max_tracks_per_artist}")
        if target_artist_from_override:
            self.logger.info(f"🔧 DEBUG: Target artist from context override: {target_artist_from_override}")
        
        for candidate in candidates:
            artist = candidate.get('artist', '').lower()
            candidate_name = candidate.get('name', 'Unknown')
            
            # 🔧 DEBUG: Log target artist tracks specifically
            is_target_artist = target_artist_from_override and artist == target_artist_from_override.lower()
            if is_target_artist or 'mk.gee' in artist:
                self.logger.info(f"🔧 DEBUG: GenreMood processing {candidate_name} by {artist}")
                self.logger.info(f"🔧 DEBUG: Is target artist: {is_target_artist}")
                self.logger.info(f"🔧 DEBUG: Artist track count: {seen_artists.get(artist, 0)}")
            
            # Check artist limit
            artist_count = seen_artists.get(artist, 0)
            
            # 🚀 Apply different limits for target vs non-target artists
            if is_target_artist:
                # Target artist from context override gets high limit
                artist_limit = max_tracks_per_artist
            else:
                # Other artists get standard diversity limit
                artist_limit = 1 if target_artist_from_override else max_tracks_per_artist
            
            if artist_count >= artist_limit:
                if is_target_artist or 'mk.gee' in artist:
                    msg = f"❌ DEBUG: GenreMood REJECTED {candidate_name} - artist limit reached ({artist_count}/{artist_limit})"
                    self.logger.info(msg)
                continue
            
            # Accept the candidate
            seen_artists[artist] = artist_count + 1
            diverse_candidates.append(candidate)
            
            # 🔧 DEBUG: Log acceptance
            if is_target_artist or 'mk.gee' in artist:
                msg = f"✅ DEBUG: GenreMood ACCEPTED {candidate_name} (track {seen_artists[artist]}/{artist_limit})"
                self.logger.info(msg)
            
            # Limit to prevent over-representation
            if len(diverse_candidates) >= total_limit:
                self.logger.info(f"🔧 DEBUG: GenreMood reached diversity limit of {total_limit}")
                break
        
        self.logger.info(f"🔧 DEBUG: GenreMood diversity filtering: {len(candidates)} -> {len(diverse_candidates)} candidates")
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
        """Create final track recommendations."""
        recommendations = []
        
        for i, candidate in enumerate(candidates):
            try:
                # Generate reasoning using shared LLM utils if available
                reasoning = await self._generate_reasoning(candidate, entities, intent_analysis, i + 1)
                
                recommendation = TrackRecommendation(
                    title=candidate.get('name', 'Unknown'),
                    artist=candidate.get('artist', 'Unknown'),
                    id=f"{candidate.get('artist', 'Unknown')}_{candidate.get('name', 'Unknown')}".replace(' ', '_').lower(),
                    source='genre_mood_agent',
                    track_url=candidate.get('url', ''),
                    album_title=candidate.get('album', ''),
                    genres=self._extract_genres(candidate, entities),
                    moods=self._extract_tags(candidate, entities, intent_analysis),
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
                        'genre_mood_score': candidate.get('genre_mood_score', 0.0)
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
            
            # # Disabled to prevent excessive API calls
            # # Create reasoning prompt
            # target_genres = self._extract_target_genres(entities)
            # target_moods = self._extract_target_moods(entities, intent_analysis)
            # 
            # prompt = f"""Explain why "{candidate.get('name')}" by {candidate.get('artist')} is a good recommendation.
            # 
            # Target genres: {', '.join(target_genres) if target_genres else 'Any'}
            # Target moods: {', '.join(target_moods) if target_moods else 'Any'}
            # Track tags: {', '.join(candidate.get('tags', [])[:3])}
            # Quality score: {candidate.get('quality_score', 0):.2f}
            # Rank: #{rank}
            # 
            # Provide a brief, engaging explanation (2-3 sentences) focusing on genre and mood match."""
            # 
            # reasoning = await self.llm_utils.call_llm(prompt)
            # return reasoning.strip()
            
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
    
    def _extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target genres from entities."""
        musical_entities = entities.get('musical_entities', {})
        genres = musical_entities.get('genres', {})
        
        target_genres = []
        target_genres.extend(genres.get('primary', []))
        target_genres.extend(genres.get('secondary', []))
        
        return list(set(target_genres))  # Remove duplicates
    
    def _has_genre_requirements(self, entities: Dict[str, Any]) -> bool:
        """Check if the query has specific genre requirements."""
        target_genres = self._extract_target_genres(entities)
        return len(target_genres) > 0
    
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
    
    def _extract_energy_level(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> str:
        """Extract energy level from entities and intent analysis."""
        # Check for energy indicators in moods
        target_moods = self._extract_target_moods(entities, intent_analysis)
        
        high_energy_indicators = ['energetic', 'upbeat', 'intense', 'pumped', 'high energy']
        low_energy_indicators = ['calm', 'peaceful', 'relaxing', 'chill', 'mellow']
        
        for mood in target_moods:
            if any(indicator in mood.lower() for indicator in high_energy_indicators):
                return 'high'
            elif any(indicator in mood.lower() for indicator in low_energy_indicators):
                return 'low'
        
        return 'medium'  # Default
    
    def _extract_genres(self, candidate: Dict[str, Any], entities: Dict[str, Any]) -> List[str]:
        """Extract genres for recommendation."""
        # Use candidate tags as genres
        tags = candidate.get('tags', [])
        
        # Filter for genre-like tags
        genre_tags = []
        for tag in tags[:5]:  # Limit to first 5 tags
            if len(tag) > 2 and not tag.isdigit():
                genre_tags.append(tag)
        
        return genre_tags
    
    def _extract_tags(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract tags for recommendation."""
        tags = candidate.get('tags', [])
        
        # Add mood and energy tags
        target_moods = self._extract_target_moods(entities, intent_analysis)
        energy_level = self._extract_energy_level(entities, intent_analysis)
        
        enhanced_tags = tags[:3]  # Start with top 3 candidate tags
        enhanced_tags.extend(target_moods[:2])  # Add top 2 target moods
        enhanced_tags.append(f"{energy_level}_energy")  # Add energy level
        
        return list(set(enhanced_tags))  # Remove duplicates
    
    def _initialize_mood_mappings(self) -> Dict[str, List[str]]:
        """Initialize mood to tag mappings."""
        return {
            'energetic': ['energetic', 'upbeat', 'high energy', 'pumped', 'intense'],
            'calm': ['calm', 'peaceful', 'relaxing', 'chill', 'mellow'],
            'happy': ['happy', 'joyful', 'uplifting', 'cheerful', 'positive'],
            'melancholic': ['sad', 'melancholic', 'depressing', 'somber', 'moody'],
            'aggressive': ['aggressive', 'angry', 'intense', 'heavy', 'brutal'],
            'romantic': ['romantic', 'love', 'intimate', 'sensual'],
            'nostalgic': ['nostalgic', 'vintage', 'retro', 'classic']
        }
    
    def _initialize_energy_mappings(self) -> Dict[str, List[str]]:
        """Initialize energy level to tag mappings."""
        return {
            'high': ['energetic', 'upbeat', 'intense', 'pumped', 'high energy', 'fast'],
            'medium': ['moderate', 'balanced', 'steady', 'medium energy'],
            'low': ['calm', 'peaceful', 'relaxing', 'chill', 'mellow', 'slow']
        }

    async def _check_genre_match(self, candidate: Dict[str, Any], genre: str) -> Dict[str, Any]:
        """
        Check if a candidate matches the target genre using API service.
        
        Args:
            candidate: Track candidate to check
            genre: Target genre to match against
            
        Returns:
            Dict with match information
        """
        try:
            # Use API service to check genre match
            from ...services.api_service import get_api_service
            api_service = get_api_service()
            
            # Check track-level genre match first
            track_match = await api_service.check_track_genre_match(
                artist=candidate.get('artist', ''),
                track=candidate.get('name', ''),
                target_genre=genre,
                include_related_genres=True
            )
            
            if track_match['matches']:
                self.logger.debug(
                    f"Genre match found for {candidate.get('artist')} - {candidate.get('name')}",
                    genre=genre,
                    match_type=track_match['match_type'],
                    confidence=track_match['confidence'],
                    matched_tags=track_match['matched_tags']
                )
                return track_match
            
            # Fall back to artist-level check if no track match
            artist_match = await api_service.check_artist_genre_match(
                artist=candidate.get('artist', ''),
                target_genre=genre,
                include_related_genres=True
            )
            
            if artist_match['matches']:
                self.logger.debug(
                    f"Artist genre match found for {candidate.get('artist')}",
                    genre=genre,
                    match_type=artist_match['match_type'],
                    confidence=artist_match['confidence'],
                    matched_tags=artist_match['matched_tags']
                )
                
                # Convert artist match to track match format
                return {
                    'matches': True,
                    'confidence': artist_match['confidence'] * 0.8,  # Slightly lower confidence
                    'matched_tags': artist_match['matched_tags'],
                    'track_tags': [],
                    'artist_match': artist_match,
                    'match_type': f"artist_{artist_match['match_type']}"
                }
            
            # No match found
            self.logger.debug(
                f"No genre match for {candidate.get('artist')} - {candidate.get('name')}",
                genre=genre,
                artist_tags=artist_match.get('artist_tags', [])
            )
            
            return {
                'matches': False,
                'confidence': 0.0,
                'matched_tags': [],
                'track_tags': track_match.get('track_tags', []),
                'artist_match': artist_match,
                'match_type': 'none'
            }
            
        except Exception as e:
            self.logger.error(f"Genre matching failed: {e}")
            return {
                'matches': False,
                'confidence': 0.0,
                'matched_tags': [],
                'track_tags': [],
                'artist_match': {'matches': False},
                'match_type': 'error'
        } 