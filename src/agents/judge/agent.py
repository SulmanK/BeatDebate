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
        metadata_service: MetadataService
    ):
        """
        Initialize simplified judge agent with injected dependencies.
        
        Args:
            config: Agent configuration
            llm_client: LLM client for explanations
            api_service: Unified API service
            metadata_service: Unified metadata service
        """
        super().__init__(
            config=config, 
            llm_client=llm_client, 
            agent_name="JudgeAgent",
            api_service=api_service,
            metadata_service=metadata_service
        )
        
        # Shared components
        self.quality_scorer = QualityScorer()
        self.llm_utils = LLMUtils(llm_client)
        
        # Configuration
        self.final_recommendations = 20
        self.diversity_targets = {
            'max_per_artist': 2,
            'min_genres': 3,
            'source_distribution': {'genre_mood': 0.4, 'discovery': 0.4, 'planner': 0.2}
        }
        
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
            final_selections = self._select_with_diversity(ranked_candidates)
            
            # Phase 4: Generate enhanced explanations
            final_recommendations = await self._generate_explanations(final_selections, state)
            
            # Update state
            state.final_recommendations = [rec.model_dump() for rec in final_recommendations]
            
            self.logger.info(
                "Judge agent processing completed",
                total_candidates=len(all_candidates),
                final_recommendations=len(final_recommendations)
            )
            
            return state
            
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
            track_key = f"{candidate.artist.lower()}::{candidate.name.lower()}"
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
        
        for candidate in candidates:
            try:
                # Calculate multiple scoring dimensions
                scores = {
                    'quality_score': await self._calculate_quality_score(candidate, entities, intent_analysis),
                    'contextual_relevance': self._calculate_contextual_relevance(candidate, entities, intent_analysis),
                    'intent_alignment': self._calculate_intent_alignment(candidate, intent_analysis),
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
                self.logger.warning(f"Failed to score candidate {candidate.name}: {e}")
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
                'name': candidate.name,
                'artist': candidate.artist,
                'album': candidate.album,
                'tags': candidate.tags,
                'url': candidate.url,
                'listeners': getattr(candidate, 'listeners', 0),
                'playcount': getattr(candidate, 'playcount', 0)
            }
            
            return await self.quality_scorer.calculate_quality_score(
                candidate_dict, entities, intent_analysis
            )
            
        except Exception as e:
            self.logger.debug(f"Quality scoring failed for {candidate.name}: {e}")
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
        candidate_tags = [tag.lower() for tag in candidate.tags]
        
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
        intent_analysis: Dict[str, Any]
    ) -> float:
        """Calculate alignment with user intent."""
        primary_intent = intent_analysis.get('primary_intent', 'discovery')
        
        # Intent-specific scoring
        if primary_intent == 'discovery':
            # Favor tracks with novelty indicators
            if candidate.source == 'discovery_agent':
                return 0.8
            elif 'underground' in candidate.tags or 'hidden_gem' in candidate.tags:
                return 0.7
            else:
                return 0.5
        
        elif primary_intent == 'genre_mood':
            # Favor tracks from genre/mood agent
            if candidate.source == 'genre_mood_agent':
                return 0.8
            else:
                return 0.6
        
        elif primary_intent == 'similarity':
            # Favor tracks with similarity indicators
            if 'similar' in candidate.reasoning.lower():
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
        if any(tag in candidate.tags for tag in unique_tags):
            score += 0.2
        
        return min(score, 1.0)
    
    async def _rank_candidates(
        self,
        scored_candidates: List[Tuple[TrackRecommendation, Dict[str, float]]],
        state: MusicRecommenderState
    ) -> List[Tuple[TrackRecommendation, Dict[str, float]]]:
        """Rank candidates by combined score."""
        # Sort by combined score
        ranked = sorted(
            scored_candidates,
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        self.logger.debug(f"Ranked {len(ranked)} candidates")
        return ranked
    
    def _select_with_diversity(
        self,
        ranked_candidates: List[Tuple[TrackRecommendation, Dict[str, float]]]
    ) -> List[TrackRecommendation]:
        """Select final recommendations ensuring diversity."""
        selected = []
        artist_counts = {}
        genre_counts = {}
        source_counts = {}
        
        for candidate, scores in ranked_candidates:
            # Check artist diversity
            if artist_counts.get(candidate.artist, 0) >= self.diversity_targets['max_per_artist']:
                continue
            
            # Check source distribution
            source_count = source_counts.get(candidate.source, 0)
            max_per_source = int(self.final_recommendations * 
                               self.diversity_targets['source_distribution'].get(candidate.source, 0.3))
            if source_count >= max_per_source:
                continue
            
            # Add to selection
            selected.append(candidate)
            artist_counts[candidate.artist] = artist_counts.get(candidate.artist, 0) + 1
            source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
            
            # Update genre counts
            for genre in candidate.genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            # Stop when we have enough
            if len(selected) >= self.final_recommendations:
                break
        
        self.logger.debug(
            f"Selected {len(selected)} recommendations with diversity",
            artist_distribution=dict(artist_counts),
            source_distribution=dict(source_counts)
        )
        
        return selected
    
    async def _generate_explanations(
        self,
        selections: List[TrackRecommendation],
        state: MusicRecommenderState
    ) -> List[TrackRecommendation]:
        """Generate enhanced explanations for final selections."""
        enhanced_selections = []
        
        entities = state.entities or {}
        intent_analysis = state.intent_analysis or {}
        
        for i, recommendation in enumerate(selections):
            try:
                # Generate enhanced reasoning using shared LLM utils
                enhanced_reasoning = await self._generate_enhanced_reasoning(
                    recommendation, entities, intent_analysis, i + 1
                )
                
                # Update recommendation with enhanced reasoning
                enhanced_rec = TrackRecommendation(
                    name=recommendation.name,
                    artist=recommendation.artist,
                    album=recommendation.album,
                    url=recommendation.url,
                    genres=recommendation.genres,
                    tags=recommendation.tags,
                    confidence=recommendation.confidence,
                    reasoning=enhanced_reasoning,
                    source=recommendation.source,
                    rank=i + 1
                )
                
                enhanced_selections.append(enhanced_rec)
                
            except Exception as e:
                self.logger.warning(f"Failed to enhance reasoning for {recommendation.name}: {e}")
                # Use original recommendation with updated rank
                recommendation.rank = i + 1
                enhanced_selections.append(recommendation)
        
        return enhanced_selections
    
    async def _generate_enhanced_reasoning(
        self,
        recommendation: TrackRecommendation,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        rank: int
    ) -> str:
        """Generate enhanced reasoning using shared LLM utils."""
        try:
            # Create comprehensive reasoning prompt
            target_genres = self._extract_target_genres(entities)
            target_moods = self._extract_target_moods(entities, intent_analysis)
            primary_intent = intent_analysis.get('primary_intent', 'discovery')
            
            prompt = f"""Create an engaging explanation for why "{recommendation.name}" by {recommendation.artist} is ranked #{rank}.

User Intent: {primary_intent}
Target Genres: {', '.join(target_genres) if target_genres else 'Open to any'}
Target Moods: {', '.join(target_moods) if target_moods else 'Any mood'}
Track Genres: {', '.join(recommendation.genres)}
Track Tags: {', '.join(recommendation.tags)}
Source Agent: {recommendation.source}
Original Reasoning: {recommendation.reasoning}

Create a conversational, engaging explanation (2-3 sentences) that:
1. Explains why this track fits the user's request
2. Highlights what makes it special or interesting
3. Uses natural, enthusiastic language

Focus on the musical qualities and why the user would enjoy this recommendation."""
            
            enhanced_reasoning = await self.llm_utils.call_llm(prompt)
            return enhanced_reasoning.strip()
            
        except Exception as e:
            self.logger.debug(f"LLM reasoning failed, using fallback: {e}")
            return self._create_fallback_reasoning(recommendation, entities, intent_analysis, rank)
    
    def _create_fallback_reasoning(
        self,
        recommendation: TrackRecommendation,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        rank: int
    ) -> str:
        """Create fallback reasoning when LLM is unavailable."""
        reasoning_parts = [f"#{rank}: {recommendation.name} by {recommendation.artist}"]
        
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