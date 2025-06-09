"""
TagGenerator Component

Handles tag generation, extraction, and enhancement logic for the GenreMoodAgent.
Centralizes all tag-related operations including candidate scoring and recommendation enhancement.
"""

from typing import Dict, List, Any
import structlog

logger = structlog.get_logger(__name__)


class TagGenerator:
    """
    Handles tag generation, extraction, and enhancement for GenreMoodAgent.
    
    Responsibilities:
    - Tag extraction from candidates
    - Tag-based scoring and matching
    - Enhanced tag generation for recommendations
    - Tag validation and filtering
    """
    
    def __init__(self):
        """Initialize TagGenerator."""
        self.logger = logger.bind(component="TagGenerator")
        
        # Common music tags for bonus scoring
        self.common_music_tags = [
            'rock', 'indie', 'electronic', 'pop', 'alternative', 
            'experimental', 'ambient', 'jazz', 'classical', 'folk',
            'hip hop', 'rap', 'metal', 'punk', 'blues', 'country'
        ]
        
        self.logger.info("TagGenerator initialized")
    
    def extract_tags_from_candidate(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Extract and enhance tags for a candidate recommendation.
        
        Args:
            candidate: Track candidate
            entities: Extracted entities from query
            intent_analysis: Intent analysis results
            
        Returns:
            List of enhanced tags
        """
        tags = candidate.get('tags', [])
        
        # Start with top candidate tags
        enhanced_tags = tags[:3] if tags else []
        
        # Add genre-related tags
        genre_tags = self._extract_genre_tags(candidate, entities)
        enhanced_tags.extend(genre_tags[:2])  # Add top 2 genre tags
        
        # Add mood-related tags (delegated to MoodAnalyzer in practice)
        mood_tags = self._extract_mood_tags(entities, intent_analysis)
        enhanced_tags.extend(mood_tags[:2])  # Add top 2 mood tags
        
        # Remove duplicates and return
        unique_tags = list(set(enhanced_tags))
        
        self.logger.debug(f"Enhanced tags for {candidate.get('name')}: {unique_tags}")
        return unique_tags
    
    def calculate_tag_based_score(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate tag-based relevance score for a candidate.
        
        Args:
            candidate: Track candidate to score
            entities: Extracted entities from query
            intent_analysis: Intent analysis results
            
        Returns:
            Tag-based relevance score (0.0 to 1.0)
        """
        score = 0.0
        candidate_tags = candidate.get('tags', [])
        
        if not candidate_tags:
            return score
        
        # Bonus for having any relevant music tags
        for tag in candidate_tags:
            if any(music_tag in tag.lower() for music_tag in self.common_music_tags):
                score += 0.1
                self.logger.debug(f"Music tag bonus: {tag}")
                break  # Only add bonus once
        
        # Score based on tag quality and relevance
        score += self._calculate_tag_quality_score(candidate_tags)
        
        # Score based on tag diversity
        score += self._calculate_tag_diversity_score(candidate_tags)
        
        return min(score, 1.0)
    
    def validate_tags(self, tags: List[str]) -> List[str]:
        """
        Validate and filter tags for quality.
        
        Args:
            tags: List of tags to validate
            
        Returns:
            List of validated tags
        """
        validated_tags = []
        
        for tag in tags:
            if self._is_valid_tag(tag):
                validated_tags.append(tag)
            else:
                self.logger.debug(f"Filtered out invalid tag: {tag}")
        
        return validated_tags
    
    def enhance_recommendation_tags(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        mood_analyzer=None
    ) -> List[str]:
        """
        Create enhanced tags for final recommendation display.
        
        Args:
            candidate: Track candidate
            entities: Extracted entities from query
            intent_analysis: Intent analysis results
            mood_analyzer: Optional MoodAnalyzer instance for mood tags
            
        Returns:
            List of enhanced tags for recommendation
        """
        # Start with candidate tags
        base_tags = candidate.get('tags', [])[:3]
        
        # Add genre tags
        genre_tags = self._extract_genre_tags(candidate, entities)
        
        # Add mood tags if mood_analyzer is provided
        mood_tags = []
        if mood_analyzer:
            mood_tags = mood_analyzer.get_mood_tags_for_candidate(candidate, entities, intent_analysis)
        else:
            mood_tags = self._extract_mood_tags(entities, intent_analysis)
        
        # Combine and deduplicate
        all_tags = base_tags + genre_tags[:2] + mood_tags[:2]
        unique_tags = list(set(all_tags))
        
        # Validate and return
        validated_tags = self.validate_tags(unique_tags)
        
        self.logger.debug(f"Final enhanced tags for {candidate.get('name')}: {validated_tags}")
        return validated_tags
    
    def _extract_genre_tags(self, candidate: Dict[str, Any], entities: Dict[str, Any]) -> List[str]:
        """Extract genre-related tags from candidate and entities."""
        genre_tags = []
        
        # From candidate tags
        candidate_tags = candidate.get('tags', [])
        for tag in candidate_tags:
            if any(music_tag in tag.lower() for music_tag in self.common_music_tags):
                genre_tags.append(tag)
        
        # From target genres in entities
        musical_entities = entities.get('musical_entities', {})
        genres = musical_entities.get('genres', {})
        target_genres = genres.get('primary', []) + genres.get('secondary', [])
        genre_tags.extend(target_genres)
        
        return list(set(genre_tags))  # Remove duplicates
    
    def _extract_mood_tags(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[str]:
        """Extract mood-related tags from entities and intent analysis."""
        mood_tags = []
        
        # From entities
        contextual_entities = entities.get('contextual_entities', {})
        mood_entities = contextual_entities.get('moods', {})
        mood_tags.extend(mood_entities.get('energy', []))
        mood_tags.extend(mood_entities.get('emotion', []))
        
        # From intent analysis
        mood_tags.extend(intent_analysis.get('mood_indicators', []))
        
        return list(set(mood_tags))  # Remove duplicates
    
    def _calculate_tag_quality_score(self, tags: List[str]) -> float:
        """Calculate quality score based on tag characteristics."""
        if not tags:
            return 0.0
        
        quality_score = 0.0
        
        # Score based on tag length (prefer meaningful tags)
        avg_length = sum(len(tag) for tag in tags) / len(tags)
        if avg_length > 3:  # Prefer tags longer than 3 characters
            quality_score += 0.1
        
        # Score based on tag count (prefer tracks with multiple tags)
        if len(tags) >= 3:
            quality_score += 0.1
        elif len(tags) >= 5:
            quality_score += 0.2
        
        return quality_score
    
    def _calculate_tag_diversity_score(self, tags: List[str]) -> float:
        """Calculate diversity score based on tag variety."""
        if not tags:
            return 0.0
        
        # Check for diversity in tag types
        has_genre = any(music_tag in tag.lower() for tag in tags for music_tag in self.common_music_tags)
        has_descriptive = any(len(tag) > 6 for tag in tags)  # Longer descriptive tags
        has_short = any(len(tag) <= 6 for tag in tags)  # Shorter categorical tags
        
        diversity_count = sum([has_genre, has_descriptive, has_short])
        
        # Score based on diversity
        if diversity_count >= 3:
            return 0.2
        elif diversity_count >= 2:
            return 0.1
        else:
            return 0.05
    
    def _is_valid_tag(self, tag: str) -> bool:
        """Check if a tag is valid for use."""
        if not tag or not isinstance(tag, str):
            return False
        
        # Filter out very short tags
        if len(tag) < 2:
            return False
        
        # Filter out numeric-only tags
        if tag.isdigit():
            return False
        
        # Filter out tags with special characters (basic validation)
        if any(char in tag for char in ['<', '>', '{', '}', '[', ']']):
            return False
        
        return True
    
    def get_common_music_tags(self) -> List[str]:
        """Get the list of common music tags."""
        return self.common_music_tags.copy()
    
    def add_common_music_tag(self, tag: str) -> None:
        """
        Add a new common music tag.
        
        Args:
            tag: Music tag to add
        """
        if tag not in self.common_music_tags:
            self.common_music_tags.append(tag)
            self.logger.info(f"Added common music tag: {tag}")
    
    def remove_common_music_tag(self, tag: str) -> None:
        """
        Remove a common music tag.
        
        Args:
            tag: Music tag to remove
        """
        if tag in self.common_music_tags:
            self.common_music_tags.remove(tag)
            self.logger.info(f"Removed common music tag: {tag}") 