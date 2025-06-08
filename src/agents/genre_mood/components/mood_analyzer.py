"""
MoodAnalyzer Component

Handles mood detection, analysis, and mapping logic for the GenreMoodAgent.
Centralizes all mood-related processing and energy level analysis.
"""

from typing import Dict, List, Any
import structlog

logger = structlog.get_logger(__name__)


class MoodAnalyzer:
    """
    Handles mood detection, analysis, and mapping for GenreMoodAgent.
    
    Responsibilities:
    - Mood extraction from entities and intent analysis
    - Energy level detection and mapping
    - Mood-to-tag mapping and scoring
    - Mood-based candidate scoring
    """
    
    def __init__(self):
        """Initialize MoodAnalyzer with mood and energy mappings."""
        self.logger = logger.bind(component="MoodAnalyzer")
        
        # Initialize mood and energy mappings
        self.mood_mappings = self._initialize_mood_mappings()
        self.energy_mappings = self._initialize_energy_mappings()
        
        self.logger.info("MoodAnalyzer initialized with mood and energy mappings")
    
    def extract_target_moods(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[str]:
        """
        Extract target moods from entities and intent analysis.
        
        Args:
            entities: Extracted entities from query
            intent_analysis: Intent analysis results
            
        Returns:
            List of target mood strings
        """
        moods = []
        
        # From entities
        contextual_entities = entities.get('contextual_entities', {})
        mood_entities = contextual_entities.get('moods', {})
        moods.extend(mood_entities.get('energy', []))
        moods.extend(mood_entities.get('emotion', []))
        
        # From intent analysis
        moods.extend(intent_analysis.get('mood_indicators', []))
        
        unique_moods = list(set(moods))  # Remove duplicates
        
        self.logger.debug(f"Extracted target moods: {unique_moods}")
        return unique_moods
    
    def extract_energy_level(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> str:
        """
        Extract energy level from entities and intent analysis.
        
        Args:
            entities: Extracted entities from query
            intent_analysis: Intent analysis results
            
        Returns:
            Energy level string ('high', 'medium', 'low')
        """
        # Check for energy indicators in moods
        target_moods = self.extract_target_moods(entities, intent_analysis)
        
        high_energy_indicators = ['energetic', 'upbeat', 'intense', 'pumped', 'high energy']
        low_energy_indicators = ['calm', 'peaceful', 'relaxing', 'chill', 'mellow']
        
        for mood in target_moods:
            if any(indicator in mood.lower() for indicator in high_energy_indicators):
                self.logger.debug(f"Detected high energy from mood: {mood}")
                return 'high'
            elif any(indicator in mood.lower() for indicator in low_energy_indicators):
                self.logger.debug(f"Detected low energy from mood: {mood}")
                return 'low'
        
        self.logger.debug("No specific energy indicators found, defaulting to medium")
        return 'medium'  # Default
    
    def calculate_mood_score(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate mood-based relevance score for a candidate.
        
        Args:
            candidate: Track candidate to score
            entities: Extracted entities from query
            intent_analysis: Intent analysis results
            
        Returns:
            Mood relevance score (0.0 to 1.0)
        """
        score = 0.0
        candidate_tags = candidate.get('tags', [])
        
        # Score based on mood matching
        target_moods = self.extract_target_moods(entities, intent_analysis)
        for mood in target_moods:
            mood_tags = self.mood_mappings.get(mood, [])
            for mood_tag in mood_tags:
                if any(mood_tag.lower() in tag.lower() for tag in candidate_tags):
                    score += 0.2
                    self.logger.debug(f"Mood match found: {mood_tag} in {candidate_tags}")
        
        # Score based on energy level
        energy_level = self.extract_energy_level(entities, intent_analysis)
        energy_tags = self.energy_mappings.get(energy_level, [])
        for energy_tag in energy_tags:
            if any(energy_tag.lower() in tag.lower() for tag in candidate_tags):
                score += 0.3
                self.logger.debug(f"Energy match found: {energy_tag} in {candidate_tags}")
        
        return min(score, 1.0)
    
    def get_mood_tags_for_candidate(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Get enhanced mood tags for a candidate recommendation.
        
        Args:
            candidate: Track candidate
            entities: Extracted entities from query
            intent_analysis: Intent analysis results
            
        Returns:
            List of mood-related tags
        """
        tags = candidate.get('tags', [])
        
        # Add mood and energy tags
        target_moods = self.extract_target_moods(entities, intent_analysis)
        energy_level = self.extract_energy_level(entities, intent_analysis)
        
        enhanced_tags = tags[:3]  # Start with top 3 candidate tags
        enhanced_tags.extend(target_moods[:2])  # Add top 2 target moods
        enhanced_tags.append(f"{energy_level}_energy")  # Add energy level
        
        unique_tags = list(set(enhanced_tags))  # Remove duplicates
        
        self.logger.debug(f"Enhanced mood tags for {candidate.get('name')}: {unique_tags}")
        return unique_tags
    
    def validate_mood_match(
        self,
        candidate: Dict[str, Any],
        target_moods: List[str],
        energy_level: str = None
    ) -> Dict[str, Any]:
        """
        Validate if a candidate matches target moods and energy level.
        
        Args:
            candidate: Track candidate to validate
            target_moods: List of target moods to match
            energy_level: Optional target energy level
            
        Returns:
            Dictionary with match information
        """
        candidate_tags = candidate.get('tags', [])
        matches = {
            'mood_matches': [],
            'energy_matches': [],
            'total_score': 0.0,
            'has_mood_match': False,
            'has_energy_match': False
        }
        
        # Check mood matches
        for mood in target_moods:
            mood_tags = self.mood_mappings.get(mood, [])
            for mood_tag in mood_tags:
                if any(mood_tag.lower() in tag.lower() for tag in candidate_tags):
                    matches['mood_matches'].append({
                        'target_mood': mood,
                        'matched_tag': mood_tag,
                        'candidate_tags': [tag for tag in candidate_tags if mood_tag.lower() in tag.lower()]
                    })
                    matches['total_score'] += 0.2
        
        # Check energy matches
        if energy_level:
            energy_tags = self.energy_mappings.get(energy_level, [])
            for energy_tag in energy_tags:
                if any(energy_tag.lower() in tag.lower() for tag in candidate_tags):
                    matches['energy_matches'].append({
                        'target_energy': energy_level,
                        'matched_tag': energy_tag,
                        'candidate_tags': [tag for tag in candidate_tags if energy_tag.lower() in tag.lower()]
                    })
                    matches['total_score'] += 0.3
        
        matches['has_mood_match'] = len(matches['mood_matches']) > 0
        matches['has_energy_match'] = len(matches['energy_matches']) > 0
        matches['total_score'] = min(matches['total_score'], 1.0)
        
        return matches
    
    def _initialize_mood_mappings(self) -> Dict[str, List[str]]:
        """
        Initialize mood to tag mappings.
        
        Returns:
            Dictionary mapping mood names to related tags
        """
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
        """
        Initialize energy level to tag mappings.
        
        Returns:
            Dictionary mapping energy levels to related tags
        """
        return {
            'high': ['energetic', 'upbeat', 'intense', 'pumped', 'high energy', 'fast'],
            'medium': ['moderate', 'balanced', 'steady', 'medium energy'],
            'low': ['calm', 'peaceful', 'relaxing', 'chill', 'mellow', 'slow']
        }
    
    def get_mood_mappings(self) -> Dict[str, List[str]]:
        """Get the current mood mappings."""
        return self.mood_mappings.copy()
    
    def get_energy_mappings(self) -> Dict[str, List[str]]:
        """Get the current energy mappings."""
        return self.energy_mappings.copy()
    
    def add_mood_mapping(self, mood: str, tags: List[str]) -> None:
        """
        Add a new mood mapping.
        
        Args:
            mood: Mood name
            tags: List of tags associated with the mood
        """
        self.mood_mappings[mood] = tags
        self.logger.info(f"Added mood mapping: {mood} -> {tags}")
    
    def add_energy_mapping(self, energy_level: str, tags: List[str]) -> None:
        """
        Add a new energy level mapping.
        
        Args:
            energy_level: Energy level name
            tags: List of tags associated with the energy level
        """
        self.energy_mappings[energy_level] = tags
        self.logger.info(f"Added energy mapping: {energy_level} -> {tags}") 