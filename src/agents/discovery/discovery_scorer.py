"""
Discovery Scorer for BeatDebate

Handles discovery-specific scoring logic including:
- Novelty scoring for underground appeal
- Underground detection scoring
- Similarity scoring for discovery context
- Combined discovery score calculation

This module extracts scoring logic from DiscoveryAgent for better modularity.
"""

from typing import Dict, List, Any
import structlog

logger = structlog.get_logger(__name__)


class DiscoveryScorer:
    """
    Discovery-specific scoring system for novelty, underground appeal, and similarity.
    
    This scorer focuses on discovery-specific metrics that complement the
    general quality scoring system.
    """
    
    def __init__(self):
        """Initialize discovery scorer."""
        self.logger = logger.bind(component="DiscoveryScorer")
        
        # Scoring weights for discovery metrics
        self.discovery_weights = {
            'novelty': 0.4,
            'underground': 0.3,
            'similarity': 0.3
        }
        
        self.logger.info("Discovery Scorer initialized")
    
    async def score_discovery_candidates(
        self,
        candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        quality_scorer=None
    ) -> List[Dict[str, Any]]:
        """
        Score candidates with discovery-specific metrics.
        
        Args:
            candidates: List of candidate tracks
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent
            quality_scorer: Optional quality scorer for base quality
            
        Returns:
            List of candidates with discovery scores added
        """
        scored_candidates = []
        
        for candidate in candidates:
            try:
                # Get base quality score if scorer provided
                quality_score = 0.0
                if quality_scorer:
                    quality_score = await quality_scorer.calculate_quality_score(
                        candidate, entities, intent_analysis
                    )
                    # Ensure quality_score is a number
                    if quality_score is None:
                        quality_score = 0.0
                    elif not isinstance(quality_score, (int, float)):
                        quality_score = 0.0
                
                # Calculate discovery-specific scores
                novelty_score = self.calculate_novelty_score(candidate, entities, intent_analysis)
                underground_score = self.calculate_underground_score(candidate)
                similarity_score = self.calculate_similarity_score(candidate, entities)
                
                # Combined discovery score
                discovery_score = (
                    novelty_score * self.discovery_weights['novelty'] +
                    underground_score * self.discovery_weights['underground'] +
                    similarity_score * self.discovery_weights['similarity']
                )
                
                # Add scores to candidate
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
    
    def calculate_novelty_score(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate novelty score for discovery.
        
        Args:
            candidate: Candidate track data
            entities: Extracted entities
            intent_analysis: Intent analysis
            
        Returns:
            Novelty score (0.0 to 1.0)
        """
        score = 0.0
        
        # Get intent to adjust novelty criteria
        intent = intent_analysis.get('intent', '').lower()
        
        # Detect both pure artist_similarity and similarity-primary hybrids
        is_artist_similarity = False
        if intent == 'artist_similarity':
            is_artist_similarity = True
        elif intent == 'hybrid':
            # Check for similarity-primary hybrid
            reasoning = intent_analysis.get('reasoning', '')
            hybrid_subtype = intent_analysis.get('hybrid_subtype', '')
            if (hybrid_subtype == 'similarity_primary' or 
                'similarity_primary' in reasoning):
                is_artist_similarity = True
        
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
        
        # For artist similarity, be more lenient with popularity thresholds
        if is_artist_similarity:
            if listeners == 0:
                score += 0.5
            elif listeners < 100000:  # Raised from 10k
                score += 0.4
            elif listeners < 1000000:  # Raised from 100k
                score += 0.3
            elif listeners < 5000000:  # Raised from 500k
                score += 0.2
            else:
                score += 0.1
        else:
            # Standard novelty thresholds for discovery
            if listeners == 0:
                score += 0.6
            elif listeners < 10000:
                score += 0.5
            elif listeners < 100000:
                score += 0.4
            elif listeners < 500000:
                score += 0.3
            elif listeners < 2000000:
                score += 0.2
            else:
                score += 0.1
        
        # Boost for underground sources
        source = candidate.get('source', '')
        if 'underground' in source.lower():
            score += 0.3
        elif 'serendipitous' in source.lower():
            score += 0.2
        
        # Penalize very mainstream indicators
        if listeners > 10000000:
            score *= 0.7
        
        # Boost for rare genres or experimental tags
        genres = candidate.get('genres', [])
        tags = candidate.get('tags', [])
        
        experimental_indicators = [
            'experimental', 'avant-garde', 'noise', 'drone', 'ambient',
            'post-rock', 'math rock', 'krautrock', 'shoegaze', 'lo-fi'
        ]
        
        for indicator in experimental_indicators:
            if any(indicator in str(item).lower() for item in genres + tags):
                score += 0.1
                break
        
        return min(score, 1.0)
    
    def calculate_underground_score(self, candidate: Dict[str, Any]) -> float:
        """
        Calculate underground appeal score.
        
        Args:
            candidate: Candidate track data
            
        Returns:
            Underground score (0.0 to 1.0)
        """
        score = 0.0
        
        # Primary factor: listener count (inverse relationship)
        listeners = candidate.get('listeners', 0)
        if listeners is None:
            listeners = 0
        elif not isinstance(listeners, (int, float)):
            try:
                listeners = int(listeners)
            except (ValueError, TypeError):
                listeners = 0
        
        # Underground scoring based on listener count
        if listeners == 0:
            score += 0.8  # Very underground
        elif listeners < 1000:
            score += 0.7
        elif listeners < 10000:
            score += 0.6
        elif listeners < 50000:
            score += 0.5
        elif listeners < 200000:
            score += 0.4
        elif listeners < 1000000:
            score += 0.3
        elif listeners < 5000000:
            score += 0.2
        else:
            score += 0.1  # Mainstream
        
        # Boost for underground source indicators
        source = candidate.get('source', '').lower()
        if 'underground' in source:
            score += 0.2
        elif 'deep' in source or 'hidden' in source:
            score += 0.15
        
        # Genre-based underground indicators
        genres = candidate.get('genres', [])
        underground_genres = [
            'experimental', 'noise', 'drone', 'dark ambient', 'harsh noise',
            'power electronics', 'black metal', 'doom metal', 'sludge',
            'post-punk', 'coldwave', 'minimal wave', 'dungeon synth'
        ]
        
        for genre in genres:
            if any(ug in str(genre).lower() for ug in underground_genres):
                score += 0.1
                break
        
        # Tag-based underground indicators
        tags = candidate.get('tags', [])
        underground_tags = [
            'underground', 'obscure', 'cult', 'rare', 'limited edition',
            'demo', 'rehearsal', 'bootleg', 'unreleased'
        ]
        
        for tag in tags:
            if any(ut in str(tag).lower() for ut in underground_tags):
                score += 0.1
                break
        
        return min(score, 1.0)
    
    def calculate_similarity_score(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity score for discovery context.
        
        Args:
            candidate: Candidate track data
            entities: Extracted entities
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        score = 0.0
        
        # Extract target artists and genres
        target_artists = self._extract_target_artists(entities)
        target_genres = self._extract_target_genres(entities)
        
        candidate_artist = candidate.get('artist', '').lower()
        candidate_genres = [g.lower() for g in candidate.get('genres', [])]
        
        # Artist similarity
        if target_artists:
            for target_artist in target_artists:
                if target_artist.lower() in candidate_artist:
                    score += 0.5  # Direct artist match
                    break
                elif any(word in candidate_artist for word in target_artist.lower().split()):
                    score += 0.3  # Partial artist match
                    break
        
        # Genre similarity
        if target_genres and candidate_genres:
            genre_matches = 0
            for target_genre in target_genres:
                for candidate_genre in candidate_genres:
                    if target_genre.lower() in candidate_genre:
                        genre_matches += 1
                        break
            
            if genre_matches > 0:
                score += min(0.4, genre_matches * 0.2)
        
        # Tag similarity (if available)
        target_tags = entities.get('musical_entities', {}).get('moods', {}).get('primary', [])
        candidate_tags = [t.lower() for t in candidate.get('tags', [])]
        
        if target_tags and candidate_tags:
            tag_matches = 0
            for target_tag in target_tags:
                for candidate_tag in candidate_tags:
                    if target_tag.lower() in candidate_tag:
                        tag_matches += 1
                        break
            
            if tag_matches > 0:
                score += min(0.3, tag_matches * 0.1)
        
        return min(score, 1.0)
    
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
    
    def _extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target genres from entities."""
        genres = []
        
        musical_entities = entities.get('musical_entities', {})
        genre_entities = musical_entities.get('genres', {})
        
        # Get primary genres
        primary_genres = genre_entities.get('primary', [])
        if isinstance(primary_genres, list):
            genres.extend(primary_genres)
        
        return [genre for genre in genres if genre and isinstance(genre, str)] 