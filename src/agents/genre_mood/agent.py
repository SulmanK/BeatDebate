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
        metadata_service: MetadataService
    ):
        """
        Initialize simplified genre mood agent with injected dependencies.
        
        Args:
            config: Agent configuration
            llm_client: LLM client for reasoning
            api_service: Unified API service
            metadata_service: Unified metadata service
        """
        super().__init__(
            config=config, 
            llm_client=llm_client, 
            agent_name="GenreMoodAgent",
            api_service=api_service,
            metadata_service=metadata_service
        )
        
        # Shared components
        self.candidate_generator = UnifiedCandidateGenerator(api_service)
        self.quality_scorer = QualityScorer()
        self.llm_utils = LLMUtils(llm_client)
        
        # Configuration
        self.target_candidates = 100
        self.final_recommendations = 20
        self.quality_threshold = 0.6
        
        # Mood and genre mappings
        self.mood_mappings = self._initialize_mood_mappings()
        self.energy_mappings = self._initialize_energy_mappings()
        
        self.logger.info("Simplified GenreMoodAgent initialized with dependency injection")
    
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
            
            # Phase 1: Generate candidates using shared generator
            candidates = await self.candidate_generator.generate_candidate_pool(
                entities=entities,
                intent_analysis=intent_analysis,
                agent_type="genre_mood",
                target_candidates=self.target_candidates
            )
            
            self.logger.debug(f"Generated {len(candidates)} candidates")
            
            # Phase 2: Score candidates using shared quality scorer
            scored_candidates = await self._score_candidates(candidates, entities, intent_analysis)
            
            # Phase 3: Filter and rank by genre/mood relevance
            filtered_candidates = await self._filter_by_genre_mood_relevance(
                scored_candidates, entities, intent_analysis
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
                recommendations=len(recommendations)
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
                genre_mood_score = self._calculate_genre_mood_score(candidate, entities, intent_analysis)
                
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
    
    def _calculate_genre_mood_score(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> float:
        """Calculate genre/mood specific relevance score."""
        score = 0.0
        
        # Extract candidate information
        candidate_tags = candidate.get('tags', [])
        candidate_name = candidate.get('name', '').lower()
        candidate_artist = candidate.get('artist', '').lower()
        
        # Score based on genre matching
        target_genres = self._extract_target_genres(entities)
        for genre in target_genres:
            if any(genre.lower() in tag.lower() for tag in candidate_tags):
                score += 0.3
            if genre.lower() in candidate_name or genre.lower() in candidate_artist:
                score += 0.2
        
        # Score based on mood matching
        target_moods = self._extract_target_moods(entities, intent_analysis)
        for mood in target_moods:
            mood_tags = self.mood_mappings.get(mood, [])
            for mood_tag in mood_tags:
                if any(mood_tag.lower() in tag.lower() for tag in candidate_tags):
                    score += 0.2
        
        # Score based on energy level
        energy_level = self._extract_energy_level(entities, intent_analysis)
        energy_tags = self.energy_mappings.get(energy_level, [])
        for energy_tag in energy_tags:
            if any(energy_tag.lower() in tag.lower() for tag in candidate_tags):
                score += 0.3
        
        return min(score, 1.0)
    
    async def _filter_by_genre_mood_relevance(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter candidates by genre/mood relevance and quality threshold."""
        filtered = []
        
        for candidate in scored_candidates:
            # Quality threshold check
            if candidate.get('quality_score', 0) < self.quality_threshold:
                continue
            
            # Genre/mood relevance check
            if candidate.get('genre_mood_score', 0) < 0.3:
                continue
            
            filtered.append(candidate)
        
        # Ensure diversity in sources and artists
        filtered = self._ensure_diversity(filtered)
        
        return filtered
    
    def _ensure_diversity(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure diversity in artists and sources."""
        seen_artists = set()
        diverse_candidates = []
        
        for candidate in candidates:
            artist = candidate.get('artist', '').lower()
            
            # Skip if we already have too many tracks from this artist
            if artist in seen_artists:
                continue
            
            seen_artists.add(artist)
            diverse_candidates.append(candidate)
            
            # Limit to prevent over-representation
            if len(diverse_candidates) >= self.final_recommendations * 2:
                break
        
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
                    advocate_source_agent='genre_mood_agent'
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
        """Generate reasoning for recommendation using shared LLM utils."""
        try:
            # Create reasoning prompt
            target_genres = self._extract_target_genres(entities)
            target_moods = self._extract_target_moods(entities, intent_analysis)
            
            prompt = f"""Explain why "{candidate.get('name')}" by {candidate.get('artist')} is a good recommendation.

Target genres: {', '.join(target_genres) if target_genres else 'Any'}
Target moods: {', '.join(target_moods) if target_moods else 'Any'}
Track tags: {', '.join(candidate.get('tags', [])[:5])}
Quality score: {candidate.get('quality_score', 0):.2f}
Genre/mood score: {candidate.get('genre_mood_score', 0):.2f}
Rank: #{rank}

Provide a brief, engaging explanation (2-3 sentences) focusing on genre and mood match."""
            
            reasoning = await self.llm_utils.call_llm(prompt)
            return reasoning.strip()
            
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