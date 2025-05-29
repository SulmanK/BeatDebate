"""
Simplified Discovery Agent

Refactored to use dependency injection and shared components, eliminating:
- Client instantiation duplication
- Candidate generation duplication
- LLM calling duplication
- Underground detection duplication
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


class DiscoveryAgent(BaseAgent):
    """
    Simplified Discovery Agent with dependency injection.
    
    Responsibilities:
    - Multi-hop similarity exploration
    - Underground and hidden gem detection
    - Serendipitous discovery beyond mainstream music
    - Novelty-optimized recommendations
    
    Uses shared components to eliminate duplication:
    - UnifiedCandidateGenerator for discovery-focused candidate generation
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
        Initialize simplified discovery agent with injected dependencies.
        
        Args:
            config: Agent configuration
            llm_client: LLM client for reasoning
            api_service: Unified API service
            metadata_service: Unified metadata service
        """
        super().__init__(
            config=config, 
            llm_client=llm_client, 
            agent_name="DiscoveryAgent",
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
        self.quality_threshold = 0.5  # Lower threshold for discovery
        self.novelty_threshold = 0.6
        
        # Discovery parameters
        self.underground_bias = 0.7
        self.similarity_depth = 2  # Multi-hop depth
        
        self.logger.info("Simplified DiscoveryAgent initialized with dependency injection")
    
    async def process(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Generate discovery recommendations using shared components.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with discovery recommendations
        """
        try:
            self.logger.info("Starting discovery agent processing")
            
            # Extract entities and intent from planner
            entities = state.entities or {}
            intent_analysis = state.intent_analysis or {}
            
            # Phase 1: Generate candidates using shared generator with discovery strategy
            candidates = await self.candidate_generator.generate_candidate_pool(
                entities=entities,
                intent_analysis=intent_analysis,
                agent_type="discovery",
                target_candidates=self.target_candidates
            )
            
            self.logger.debug(f"Generated {len(candidates)} discovery candidates")
            
            # Phase 2: Score candidates with discovery-specific metrics
            scored_candidates = await self._score_discovery_candidates(candidates, entities, intent_analysis)
            
            # Phase 3: Filter for novelty and underground appeal
            filtered_candidates = await self._filter_for_discovery(
                scored_candidates, entities, intent_analysis
            )
            
            # Phase 4: Create final recommendations with discovery reasoning
            recommendations = await self._create_discovery_recommendations(
                filtered_candidates[:self.final_recommendations],
                entities,
                intent_analysis
            )
            
            # Update state
            state.discovery_recommendations = [rec.model_dump() for rec in recommendations]
            
            self.logger.info(
                "Discovery agent processing completed",
                candidates=len(candidates),
                filtered=len(filtered_candidates),
                recommendations=len(recommendations)
            )
            
            return state
            
        except Exception as e:
            self.logger.error("Discovery agent processing failed", error=str(e))
            state.discovery_recommendations = []
            return state
    
    async def _score_discovery_candidates(
        self,
        candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score candidates with discovery-specific metrics."""
        scored_candidates = []
        
        for candidate in candidates:
            try:
                # Use shared quality scorer
                quality_score = await self.quality_scorer.calculate_quality_score(
                    candidate, entities, intent_analysis
                )
                
                # Ensure quality_score is a number
                if quality_score is None:
                    quality_score = 0.0
                elif not isinstance(quality_score, (int, float)):
                    quality_score = 0.0
                
                # Add discovery-specific scoring
                novelty_score = self._calculate_novelty_score(candidate, entities, intent_analysis)
                underground_score = self._calculate_underground_score(candidate)
                similarity_score = self._calculate_similarity_score(candidate, entities)
                
                # Combined discovery score
                discovery_score = (
                    novelty_score * 0.4 +
                    underground_score * 0.3 +
                    similarity_score * 0.3
                )
                
                candidate['quality_score'] = quality_score
                candidate['novelty_score'] = novelty_score
                candidate['underground_score'] = underground_score
                candidate['similarity_score'] = similarity_score
                candidate['discovery_score'] = discovery_score
                candidate['combined_score'] = (quality_score * 0.4) + (discovery_score * 0.6)
                
                scored_candidates.append(candidate)
                
            except Exception as e:
                self.logger.warning(f"Failed to score discovery candidate: {e}")
                continue
        
        # Sort by combined score
        scored_candidates.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        return scored_candidates
    
    def _calculate_novelty_score(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> float:
        """Calculate novelty score for discovery."""
        score = 0.0
        
        # Lower listener count = higher novelty
        listeners = candidate.get('listeners', 0)
        # Handle None values and ensure it's a number
        if listeners is None:
            listeners = 0
        elif not isinstance(listeners, (int, float)):
            try:
                listeners = int(listeners)
            except (ValueError, TypeError):
                listeners = 0
        
        if listeners == 0:
            score += 0.5
        elif listeners < 10000:
            score += 0.4
        elif listeners < 100000:
            score += 0.3
        elif listeners < 1000000:
            score += 0.2
        else:
            score += 0.1
        
        # Uncommon tags indicate novelty
        tags = candidate.get('tags', [])
        if tags is None:
            tags = []
        
        uncommon_tags = ['experimental', 'underground', 'indie', 'alternative', 'obscure', 'rare']
        for tag in tags:
            if tag and any(uncommon in str(tag).lower() for uncommon in uncommon_tags):
                score += 0.2
        
        # Source diversity (non-mainstream sources)
        source = candidate.get('source', '')
        if source and ('underground' in source or 'serendipitous' in source):
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_underground_score(self, candidate: Dict[str, Any]) -> float:
        """Calculate underground appeal score."""
        score = 0.0
        
        # Low play count indicates underground status
        playcount = candidate.get('playcount', 0)
        listeners = candidate.get('listeners', 0)
        
        # Handle None values and ensure they're numbers
        if playcount is None:
            playcount = 0
        elif not isinstance(playcount, (int, float)):
            try:
                playcount = int(playcount)
            except (ValueError, TypeError):
                playcount = 0
        
        if listeners is None:
            listeners = 0
        elif not isinstance(listeners, (int, float)):
            try:
                listeners = int(listeners)
            except (ValueError, TypeError):
                listeners = 0
        
        if playcount == 0 and listeners == 0:
            score += 0.3
        elif playcount < 50000:
            score += 0.4
        elif playcount < 500000:
            score += 0.3
        elif playcount < 5000000:
            score += 0.2
        else:
            score += 0.1
        
        # Underground indicators in tags
        tags = candidate.get('tags', [])
        if tags is None:
            tags = []
        
        underground_indicators = ['underground', 'hidden gem', 'obscure', 'cult', 'rare']
        for tag in tags:
            if tag and any(indicator in str(tag).lower() for indicator in underground_indicators):
                score += 0.3
        
        # Artist name length (longer names often indicate less mainstream artists)
        artist = candidate.get('artist', '')
        if artist and len(str(artist)) > 15:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_similarity_score(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any]
    ) -> float:
        """Calculate similarity score to user preferences."""
        score = 0.0
        
        # Extract target preferences
        target_artists = self._extract_target_artists(entities)
        target_genres = self._extract_target_genres(entities)
        
        candidate_tags = candidate.get('tags', [])
        candidate_artist = candidate.get('artist', '').lower()
        
        # Artist similarity (if this is a similar artist)
        source_artist = candidate.get('source_artist', '')
        if source_artist and any(target.lower() in source_artist.lower() for target in target_artists):
            score += 0.4
        
        # Genre similarity
        for genre in target_genres:
            if any(genre.lower() in tag.lower() for tag in candidate_tags):
                score += 0.3
        
        # Multi-hop similarity bonus
        if candidate.get('source') == 'multi_hop_similarity':
            score += 0.2
        
        return min(score, 1.0)
    
    async def _filter_for_discovery(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter candidates for discovery criteria."""
        filtered = []
        
        for candidate in scored_candidates:
            # Quality threshold check (lower for discovery)
            quality_score = candidate.get('quality_score', 0)
            if quality_score is None:
                quality_score = 0
            if quality_score < self.quality_threshold:
                continue
            
            # Novelty threshold check
            novelty_score = candidate.get('novelty_score', 0)
            if novelty_score is None:
                novelty_score = 0
            if novelty_score < self.novelty_threshold:
                continue
            
            # Underground bias check
            underground_score = candidate.get('underground_score', 0)
            if underground_score is None:
                underground_score = 0
            if underground_score < (self.underground_bias * 0.5):
                continue
            
            filtered.append(candidate)
        
        # Ensure diversity and novelty
        filtered = self._ensure_discovery_diversity(filtered)
        
        return filtered
    
    def _ensure_discovery_diversity(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure diversity in discovery recommendations."""
        seen_artists = set()
        seen_genres = set()
        diverse_candidates = []
        
        for candidate in candidates:
            artist = candidate.get('artist', '').lower()
            tags = candidate.get('tags', [])
            
            # Skip if we already have this artist
            if artist in seen_artists:
                continue
            
            # Limit genre repetition
            candidate_genres = [tag.lower() for tag in tags[:3]]
            genre_overlap = len(set(candidate_genres) & seen_genres)
            if genre_overlap > 1:  # Allow some overlap but not too much
                continue
            
            seen_artists.add(artist)
            seen_genres.update(candidate_genres)
            diverse_candidates.append(candidate)
            
            # Limit to prevent over-filtering
            if len(diverse_candidates) >= self.final_recommendations * 2:
                break
        
        return diverse_candidates
    
    async def _create_discovery_recommendations(
        self,
        candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[TrackRecommendation]:
        """Create final discovery recommendations."""
        recommendations = []
        
        for i, candidate in enumerate(candidates):
            try:
                # Generate discovery-focused reasoning
                reasoning = await self._generate_discovery_reasoning(
                    candidate, entities, intent_analysis, i + 1
                )
                
                recommendation = TrackRecommendation(
                    title=candidate.get('name', 'Unknown'),
                    artist=candidate.get('artist', 'Unknown'),
                    id=f"{candidate.get('artist', 'Unknown')}_{candidate.get('name', 'Unknown')}".replace(' ', '_').lower(),
                    source='discovery_agent',
                    track_url=candidate.get('url', ''),
                    album_title=candidate.get('album', ''),
                    genres=self._extract_discovery_genres(candidate, entities),
                    moods=self._extract_discovery_tags(candidate, entities, intent_analysis),
                    confidence=candidate.get('combined_score', 0.5),
                    explanation=reasoning,
                    novelty_score=candidate.get('novelty_score', 0.0),
                    quality_score=candidate.get('quality_score', 0.0),
                    advocate_source_agent='discovery_agent'
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                self.logger.warning(f"Failed to create discovery recommendation: {e}")
                continue
        
        return recommendations
    
    async def _generate_discovery_reasoning(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        rank: int
    ) -> str:
        """Generate discovery-focused reasoning using shared LLM utils."""
        try:
            # Create discovery reasoning prompt
            target_artists = self._extract_target_artists(entities)
            target_genres = self._extract_target_genres(entities)
            
            # Ensure listeners is a valid number for formatting
            listeners = candidate.get('listeners', 0)
            if listeners is None:
                listeners = 0
            elif not isinstance(listeners, (int, float)):
                try:
                    listeners = int(listeners)
                except (ValueError, TypeError):
                    listeners = 0
            
            prompt = f"""Explain why "{candidate.get('name')}" by {candidate.get('artist')} is a great discovery.

Target artists: {', '.join(target_artists) if target_artists else 'Open to discovery'}
Target genres: {', '.join(target_genres) if target_genres else 'Any'}
Track tags: {', '.join(candidate.get('tags', [])[:5])}
Novelty score: {candidate.get('novelty_score', 0):.2f}
Underground score: {candidate.get('underground_score', 0):.2f}
Listeners: {listeners:,}
Rank: #{rank}

Provide a brief, engaging explanation (2-3 sentences) focusing on discovery value and uniqueness."""
            
            reasoning = await self.llm_utils.call_llm(prompt)
            return reasoning.strip()
            
        except Exception as e:
            self.logger.debug(f"LLM reasoning failed, using fallback: {e}")
            return self._create_discovery_fallback_reasoning(candidate, entities, intent_analysis, rank)
    
    def _create_discovery_fallback_reasoning(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        rank: int
    ) -> str:
        """Create fallback discovery reasoning when LLM is unavailable."""
        name = candidate.get('name', 'This track')
        artist = candidate.get('artist', 'the artist')
        listeners = candidate.get('listeners', 0)
        tags = candidate.get('tags', [])[:3]
        
        # Ensure listeners is a valid number
        if listeners is None:
            listeners = 0
        elif not isinstance(listeners, (int, float)):
            try:
                listeners = int(listeners)
            except (ValueError, TypeError):
                listeners = 0
        
        reasoning_parts = [f"#{rank}: {name} by {artist}"]
        
        # Highlight discovery aspects
        if listeners < 10000:
            reasoning_parts.append("Hidden gem with limited exposure")
        elif listeners < 100000:
            reasoning_parts.append("Underground favorite")
        
        if tags:
            reasoning_parts.append(f"Tagged as {', '.join(tags)}")
        
        novelty_score = candidate.get('novelty_score', 0)
        if novelty_score is not None and novelty_score > 0.7:
            reasoning_parts.append("High novelty discovery")
        
        return ". ".join(reasoning_parts) + "."
    
    def _extract_target_artists(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target artists from entities."""
        musical_entities = entities.get('musical_entities', {})
        artists = musical_entities.get('artists', {})
        
        target_artists = []
        target_artists.extend(artists.get('primary', []))
        target_artists.extend(artists.get('similar_to', []))
        
        return list(set(target_artists))  # Remove duplicates
    
    def _extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target genres from entities."""
        musical_entities = entities.get('musical_entities', {})
        genres = musical_entities.get('genres', {})
        
        target_genres = []
        target_genres.extend(genres.get('primary', []))
        target_genres.extend(genres.get('secondary', []))
        
        return list(set(target_genres))  # Remove duplicates
    
    def _extract_discovery_genres(self, candidate: Dict[str, Any], entities: Dict[str, Any]) -> List[str]:
        """Extract genres for discovery recommendation."""
        tags = candidate.get('tags', [])
        
        # Filter for genre-like tags, prioritizing unique/underground genres
        genre_tags = []
        for tag in tags[:5]:
            if len(tag) > 2 and not tag.isdigit():
                genre_tags.append(tag)
        
        # Add discovery-specific genre indicators
        if candidate.get('underground_score', 0) > 0.7:
            genre_tags.append('underground')
        if candidate.get('novelty_score', 0) > 0.7:
            genre_tags.append('experimental')
        
        return genre_tags
    
    def _extract_discovery_tags(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract tags for discovery recommendation."""
        tags = candidate.get('tags', [])
        
        # Start with candidate tags
        discovery_tags = tags[:3]
        
        # Add discovery-specific tags
        if candidate.get('novelty_score', 0) > 0.6:
            discovery_tags.append('hidden_gem')
        if candidate.get('underground_score', 0) > 0.6:
            discovery_tags.append('underground')
        if candidate.get('source') == 'multi_hop_similarity':
            discovery_tags.append('similarity_discovery')
        
        # Add listener count category
        listeners = candidate.get('listeners', 0)
        
        # Ensure listeners is a valid number
        if listeners is None:
            listeners = 0
        elif not isinstance(listeners, (int, float)):
            try:
                listeners = int(listeners)
            except (ValueError, TypeError):
                listeners = 0
        
        if listeners < 10000:
            discovery_tags.append('rare_find')
        elif listeners < 100000:
            discovery_tags.append('cult_favorite')
        
        return list(set(discovery_tags))  # Remove duplicates 