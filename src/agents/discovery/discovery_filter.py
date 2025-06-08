"""
Discovery Filter for BeatDebate

Handles discovery-specific filtering logic including:
- Quality threshold filtering
- Novelty threshold filtering
- Genre matching for discovery context
- Artist similarity filtering
- Diversity and variety management

This module extracts filtering logic from DiscoveryAgent for better modularity.
"""

from typing import Dict, List, Any
import structlog

logger = structlog.get_logger(__name__)


class DiscoveryFilter:
    """
    Discovery-specific filtering system for quality, novelty, and diversity.
    
    This filter focuses on discovery-specific criteria that complement the
    general recommendation filtering.
    """
    
    def __init__(self):
        """Initialize discovery filter."""
        self.logger = logger.bind(component="DiscoveryFilter")
        self.logger.info("Discovery Filter initialized")
    
    async def filter_for_discovery(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        quality_threshold: float = 0.3,
        novelty_threshold: float = 0.4,
        context_override: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates for discovery-specific criteria.
        
        Args:
            scored_candidates: List of scored candidate tracks
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent
            quality_threshold: Minimum quality score threshold
            novelty_threshold: Minimum novelty score threshold
            context_override: Optional context override for filtering
            
        Returns:
            List of filtered candidates suitable for discovery
        """
        if not scored_candidates:
            return []
        
        intent = intent_analysis.get('intent', '').lower()
        self.logger.info(f"Filtering {len(scored_candidates)} candidates for discovery intent: {intent}")
        
        # Apply intent-specific filtering
        if intent == 'by_artist':
            return await self._filter_for_by_artist(scored_candidates, entities, intent_analysis)
        elif intent == 'by_artist_underground':
            return await self._filter_for_by_artist_underground(scored_candidates, entities, intent_analysis)
        elif intent == 'artist_genre':
            return await self._filter_for_artist_genre(scored_candidates, entities, intent_analysis)
        elif intent in ['artist_similarity', 'similarity_primary']:
            return await self._filter_for_similarity_primary_hybrid(scored_candidates, entities, intent_analysis)
        elif intent == 'hybrid_similarity_genre':
            return await self._filter_for_similarity_primary_hybrid(scored_candidates, entities, intent_analysis)
        elif intent == 'discovery':
            return await self._filter_for_pure_discovery(scored_candidates, entities, intent_analysis, quality_threshold, novelty_threshold)
        elif intent == 'genre_mood':
            return await self._filter_for_genre_mood(scored_candidates, entities, intent_analysis, quality_threshold)
        elif intent == 'contextual':
            return await self._filter_for_contextual(scored_candidates, entities, intent_analysis, quality_threshold)
        elif intent == 'hybrid':
            # Legacy fallback - redirect to hybrid_similarity_genre
            return await self._filter_for_hybrid_similarity_genre(scored_candidates, entities, intent_analysis, quality_threshold, novelty_threshold, context_override)
        else:
            # Default discovery filtering
            return await self._filter_for_pure_discovery(scored_candidates, entities, intent_analysis, quality_threshold, novelty_threshold)
    
    async def _filter_for_by_artist(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter for by_artist intent - focus on target artist tracks."""
        target_artists = self._extract_target_artists(entities)
        if not target_artists:
            return scored_candidates[:50]  # Fallback if no target artist
        
        filtered = []
        for candidate in scored_candidates:
            candidate_artist = candidate.get('artist', '').lower()
            
            # Check if candidate is from target artist
            is_target_artist = any(
                target.lower() in candidate_artist or candidate_artist in target.lower()
                for target in target_artists
            )
            
            if is_target_artist:
                filtered.append(candidate)
        
        self.logger.info(f"By-artist filtering: {len(filtered)} tracks from target artists")
        return filtered[:50]  # Limit to reasonable number
    
    async def _filter_for_by_artist_underground(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter for by_artist_underground intent - focus on underground tracks by target artist."""
        target_artists = self._extract_target_artists(entities)
        if not target_artists:
            return scored_candidates[:30]
        
        filtered = []
        for candidate in scored_candidates:
            candidate_artist = candidate.get('artist', '').lower()
            
            # Check if candidate is from target artist
            is_target_artist = any(
                target.lower() in candidate_artist or candidate_artist in target.lower()
                for target in target_artists
            )
            
            # Check underground criteria
            underground_score = candidate.get('underground_score', 0)
            listeners = candidate.get('listeners', 0) or 0
            
            if is_target_artist and (underground_score > 0.4 or listeners < 100000):
                filtered.append(candidate)
        
        self.logger.info(f"By-artist-underground filtering: {len(filtered)} underground tracks")
        return filtered[:30]
    
    async def _filter_for_artist_genre(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter for artist_genre intent - focus on target artist tracks that match specific genres."""
        target_artists = self._extract_target_artists(entities)
        required_genres = self._extract_required_genres(entities)
        
        if not target_artists:
            return scored_candidates[:30]  # Fallback if no target artist
        
        if not required_genres:
            # If no genres specified, fall back to by_artist filtering
            return await self._filter_for_by_artist(scored_candidates, entities, intent_analysis)
        
        filtered = []
        for candidate in scored_candidates:
            candidate_artist = candidate.get('artist', '').lower()
            
            # Check if candidate is from target artist
            is_target_artist = any(
                target.lower() in candidate_artist or candidate_artist in target.lower()
                for target in target_artists
            )
            
            if not is_target_artist:
                continue
            
            # Check if candidate matches required genres
            genre_match = await self._strict_genre_match(candidate, required_genres)
            if genre_match:
                filtered.append(candidate)
        
        self.logger.info(f"Artist-genre filtering: {len(filtered)} tracks from {target_artists} matching {required_genres}")
        return filtered[:40]  # Allow more tracks since we're filtering by both artist and genre
    
    async def _filter_for_similarity_primary_hybrid(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates for similarity-focused intents with hybrid approach.
        
        Uses relaxed thresholds but maintains quality standards.
        """
        filtered = []
        target_artists = self._extract_target_artists(entities)
        
        # 🔍 DEBUG: Log input candidates
        self.logger.info(f"🔍 SIMILARITY FILTER INPUT: {len(scored_candidates)} candidates")
        if scored_candidates:
            sample_input = scored_candidates[:3]
            for i, candidate in enumerate(sample_input):
                self.logger.info(
                    f"🔍 Input {i+1}: {candidate.get('artist', 'Unknown')} - {candidate.get('name', 'Unknown')} "
                    f"(listeners: {candidate.get('listeners', 0)}, source: {candidate.get('source', 'unknown')})"
                )
        
        for candidate in scored_candidates:
            # 🚨 CRITICAL: Detect and reject fake/fallback tracks
            listeners = candidate.get('listeners', 0)
            playcount = candidate.get('playcount', 0)
            
            # Tracks with 0 listeners AND 0 playcount are clearly fake fallback data
            if listeners == 0 and playcount == 0:
                self.logger.warning(
                    f"🚨 REJECTING FAKE CANDIDATE: {candidate.get('artist', 'Unknown')} - {candidate.get('name', 'Unknown')} "
                    f"(listeners: {listeners}, playcount: {playcount}) - Zero popularity indicates fallback/fake data"
                )
                continue  # Skip this fake candidate
            
            # Quality threshold (relaxed for similarity)
            quality_score = candidate.get('quality_score', 0)
            if quality_score < 0.2:
                continue
            
            # Artist similarity check
            if target_artists:
                similarity_score = candidate.get('similarity_score', 0)
                if similarity_score < 0.2:  # Relaxed similarity threshold
                    continue
            
            # Novelty check (relaxed for similarity)
            novelty_score = candidate.get('novelty_score', 0.5)
            if novelty_score > 0.8:  # Allow less novel tracks for similarity
                continue
            
            # Contextual relevance check
            contextual_score = candidate.get('contextual_relevance', 0.5)
            if contextual_score < 0.3:  # Relaxed contextual threshold
                continue
            
            filtered.append(candidate)
        
        # 🔍 DEBUG: Log filtered candidates
        self.logger.info(f"🔍 SIMILARITY FILTER OUTPUT: {len(filtered)} candidates (rejected {len(scored_candidates) - len(filtered)} fake/low-quality)")
        if filtered:
            sample_output = filtered[:3]
            for i, candidate in enumerate(sample_output):
                self.logger.info(
                    f"🔍 Output {i+1}: {candidate.get('artist', 'Unknown')} - {candidate.get('name', 'Unknown')} "
                    f"(listeners: {candidate.get('listeners', 0)}, quality: {candidate.get('quality_score', 0):.3f})"
                )
        
        # Sort by combined score and return top candidates
        filtered.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        return filtered[:40]
    
    async def _filter_for_pure_discovery(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        quality_threshold: float,
        novelty_threshold: float
    ) -> List[Dict[str, Any]]:
        """Filter for pure discovery intent - focus on novelty and quality."""
        filtered = []
        
        for candidate in scored_candidates:
            # Quality threshold
            quality_score = candidate.get('quality_score', 0)
            if quality_score < quality_threshold:
                continue
            
            # Novelty threshold
            novelty_score = candidate.get('novelty_score', 0)
            if novelty_score < novelty_threshold:
                continue
            
            # Underground preference for discovery
            underground_score = candidate.get('underground_score', 0)
            if underground_score < 0.3:  # Prefer underground tracks
                continue
            
            filtered.append(candidate)
        
        self.logger.info(f"Pure discovery filtering: {len(filtered)} novel tracks")
        return filtered[:30]
    
    async def _filter_for_genre_mood(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        quality_threshold: float
    ) -> List[Dict[str, Any]]:
        """Filter for genre_mood intent - focus on genre/mood matching."""
        required_genres = self._extract_required_genres(entities)
        
        filtered = []
        for candidate in scored_candidates:
            # Quality threshold
            quality_score = candidate.get('quality_score', 0)
            if quality_score < quality_threshold:
                continue
            
            # Genre matching if specified
            if required_genres:
                genre_match = await self._strict_genre_match(candidate, required_genres)
                if not genre_match:
                    continue
            
            filtered.append(candidate)
        
        self.logger.info(f"Genre-mood filtering: {len(filtered)} matching tracks")
        return filtered[:35]
    
    async def _filter_for_contextual(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        quality_threshold: float
    ) -> List[Dict[str, Any]]:
        """Filter for contextual intent - focus on reliability and quality."""
        filtered = []
        
        for candidate in scored_candidates:
            # Higher quality threshold for contextual
            quality_score = candidate.get('quality_score', 0)
            if quality_score < quality_threshold:
                continue
            
            # Prefer more reliable (less underground) tracks for context
            underground_score = candidate.get('underground_score', 0)
            if underground_score > 0.7:  # Skip very underground tracks
                continue
            
            filtered.append(candidate)
        
        self.logger.info(f"Contextual filtering: {len(filtered)} reliable tracks")
        return filtered[:35]
    
    async def _filter_for_hybrid(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        quality_threshold: float,
        novelty_threshold: float,
        context_override: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Filter for hybrid intent - balanced approach."""
        hybrid_subtype = intent_analysis.get('hybrid_subtype', '')
        
        if hybrid_subtype == 'similarity_primary':
            return await self._filter_for_similarity_primary_hybrid(scored_candidates, entities, intent_analysis)
        elif hybrid_subtype == 'discovery_primary':
            return await self._filter_for_pure_discovery(scored_candidates, entities, intent_analysis, quality_threshold, novelty_threshold)
        else:
            # Balanced hybrid filtering
            filtered = []
            for candidate in scored_candidates:
                quality_score = candidate.get('quality_score', 0)
                if quality_score < quality_threshold:
                    continue
                
                # Moderate novelty requirement
                novelty_score = candidate.get('novelty_score', 0)
                if novelty_score < (novelty_threshold * 0.7):  # Relaxed novelty
                    continue
                
                filtered.append(candidate)
            
            self.logger.info(f"Hybrid filtering: {len(filtered)} balanced tracks")
            return filtered[:35]
    
    async def _strict_genre_match(self, candidate: Dict[str, Any], required_genres: List[str]) -> bool:
        """Check if candidate matches required genres strictly."""
        candidate_genres = [g.lower() for g in candidate.get('genres', [])]
        
        for required_genre in required_genres:
            if await self._check_genre_match(candidate, required_genre):
                return True
        
        return False
    
    async def _check_genre_match(self, candidate: Dict[str, Any], target_genre: str) -> bool:
        """Check if candidate matches target genre."""
        candidate_genres = [g.lower() for g in candidate.get('genres', [])]
        target_lower = target_genre.lower()
        
        # Direct match
        if target_lower in candidate_genres:
            return True
        
        # Partial match
        for genre in candidate_genres:
            if target_lower in genre or genre in target_lower:
                return True
        
        # Fallback genre matching
        return self._fallback_genre_match(candidate, target_genre)
    
    def _fallback_genre_match(self, candidate: Dict[str, Any], target_genre: str) -> bool:
        """Fallback genre matching using tags and broader categories."""
        candidate_tags = [t.lower() for t in candidate.get('tags', [])]
        target_lower = target_genre.lower()
        
        # Check tags for genre indicators
        for tag in candidate_tags:
            if target_lower in tag or tag in target_lower:
                return True
        
        # Genre family matching
        genre_families = {
            'rock': ['alternative', 'indie', 'punk', 'metal', 'grunge'],
            'electronic': ['techno', 'house', 'ambient', 'idm', 'experimental'],
            'hip-hop': ['rap', 'trap', 'boom bap', 'conscious'],
            'jazz': ['fusion', 'bebop', 'smooth', 'contemporary'],
            'folk': ['acoustic', 'singer-songwriter', 'americana', 'country']
        }
        
        for family, subgenres in genre_families.items():
            if target_lower == family:
                return any(sub in ' '.join(candidate.get('genres', [])).lower() for sub in subgenres)
            elif target_lower in subgenres and family in ' '.join(candidate.get('genres', [])).lower():
                return True
        
        return False
    
    def _extract_target_artists(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target artists from entities."""
        artists = []
        
        musical_entities = entities.get('musical_entities', {})
        artist_entities = musical_entities.get('artists', {})
        
        # Get primary artists
        primary_artists = artist_entities.get('primary', [])
        if isinstance(primary_artists, list):
            artists.extend(primary_artists)
        
        # Get similar_to artists
        similar_artists = artist_entities.get('similar_to', [])
        if isinstance(similar_artists, list):
            artists.extend(similar_artists)
        
        return [artist for artist in artists if artist and isinstance(artist, str)]
    
    def _extract_required_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract required genres from entities."""
        genres = []
        
        musical_entities = entities.get('musical_entities', {})
        genre_entities = musical_entities.get('genres', {})
        
        # Get primary genres
        primary_genres = genre_entities.get('primary', [])
        if isinstance(primary_genres, list):
            genres.extend(primary_genres)
        
        return [genre for genre in genres if genre and isinstance(genre, str)] 