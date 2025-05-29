"""
Mood Logic Helpers for Genre Mood Agent

Provides specialized mood mapping, energy level analysis, and mood-based
tag generation for the genre/mood recommendation agent.
"""

from typing import Dict, List, Any, Tuple
import structlog

logger = structlog.get_logger(__name__)


class MoodLogic:
    """
    Specialized mood logic for genre/mood-based recommendations.
    
    Provides:
    - Advanced mood mapping and analysis
    - Energy level detection and classification
    - Mood-based tag generation
    - Context-aware mood interpretation
    """
    
    def __init__(self):
        self.mood_mappings = self._initialize_advanced_mood_mappings()
        self.energy_mappings = self._initialize_energy_mappings()
        self.context_mood_modifiers = self._initialize_context_modifiers()
        self.mood_combinations = self._initialize_mood_combinations()
        
        logger.debug("MoodLogic initialized with advanced mood mappings")
    
    def analyze_mood_request(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        context_factors: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze mood requirements from entities and context.
        
        Args:
            entities: Extracted entities from query
            intent_analysis: Intent analysis results
            context_factors: Additional context factors
            
        Returns:
            Comprehensive mood analysis
        """
        mood_analysis = {
            'primary_moods': [],
            'secondary_moods': [],
            'energy_level': 'medium',
            'mood_intensity': 0.5,
            'context_modifiers': [],
            'mood_tags': [],
            'confidence': 0.5
        }
        
        # Extract moods from entities
        contextual_entities = entities.get('contextual_entities', {})
        mood_entities = contextual_entities.get('moods', {})
        
        # Primary mood detection
        primary_moods = mood_entities.get('emotion', [])
        energy_moods = mood_entities.get('energy', [])
        
        # Combine and prioritize moods
        all_detected_moods = primary_moods + energy_moods
        
        if all_detected_moods:
            mood_analysis['primary_moods'] = all_detected_moods[:2]  # Top 2
            mood_analysis['secondary_moods'] = all_detected_moods[2:4]  # Next 2
            mood_analysis['confidence'] = min(0.9, len(all_detected_moods) * 0.3)
        
        # Energy level analysis
        mood_analysis['energy_level'] = self._determine_energy_level(
            all_detected_moods, intent_analysis, context_factors
        )
        
        # Mood intensity analysis
        mood_analysis['mood_intensity'] = self._calculate_mood_intensity(
            all_detected_moods, intent_analysis
        )
        
        # Context modifiers
        if context_factors:
            mood_analysis['context_modifiers'] = self._apply_context_modifiers(
                all_detected_moods, context_factors
            )
        
        # Generate mood-specific tags
        mood_analysis['mood_tags'] = self._generate_mood_tags(mood_analysis)
        
        logger.debug("Mood analysis completed", analysis=mood_analysis)
        return mood_analysis
    
    def _determine_energy_level(
        self, 
        detected_moods: List[str], 
        intent_analysis: Dict[str, Any],
        context_factors: List[str] = None
    ) -> str:
        """Determine energy level from moods and context."""
        energy_scores = {'high': 0, 'medium': 0, 'low': 0}
        
        # Score based on detected moods
        for mood in detected_moods:
            mood_lower = mood.lower()
            for energy_level, indicators in self.energy_mappings.items():
                if any(indicator in mood_lower for indicator in indicators):
                    energy_scores[energy_level] += 1
        
        # Context-based energy adjustment
        if context_factors:
            for context in context_factors:
                context_lower = context.lower()
                if any(word in context_lower for word in ['workout', 'exercise', 'pump']):
                    energy_scores['high'] += 2
                elif any(word in context_lower for word in ['relax', 'chill', 'calm']):
                    energy_scores['low'] += 2
                elif any(word in context_lower for word in ['work', 'focus', 'study']):
                    energy_scores['medium'] += 1
        
        # Intent-based energy adjustment
        primary_intent = intent_analysis.get('primary_intent', '')
        if primary_intent == 'concentration':
            energy_scores['medium'] += 1
        elif primary_intent == 'energy':
            energy_scores['high'] += 2
        elif primary_intent == 'relaxation':
            energy_scores['low'] += 2
        
        # Return highest scoring energy level
        return max(energy_scores, key=energy_scores.get) if any(energy_scores.values()) else 'medium'
    
    def _calculate_mood_intensity(
        self, 
        detected_moods: List[str], 
        intent_analysis: Dict[str, Any]
    ) -> float:
        """Calculate mood intensity from 0.0 to 1.0."""
        intensity = 0.5  # Base intensity
        
        # Intensity indicators in moods
        high_intensity_words = ['intense', 'extreme', 'powerful', 'strong', 'heavy']
        low_intensity_words = ['gentle', 'soft', 'mild', 'subtle', 'light']
        
        for mood in detected_moods:
            mood_lower = mood.lower()
            if any(word in mood_lower for word in high_intensity_words):
                intensity += 0.2
            elif any(word in mood_lower for word in low_intensity_words):
                intensity -= 0.2
        
        # Intent-based intensity adjustment
        primary_intent = intent_analysis.get('primary_intent', '')
        if primary_intent in ['energy', 'mood_enhancement']:
            intensity += 0.1
        elif primary_intent in ['background', 'relaxation']:
            intensity -= 0.1
        
        return max(0.0, min(1.0, intensity))
    
    def _apply_context_modifiers(
        self, 
        detected_moods: List[str], 
        context_factors: List[str]
    ) -> List[str]:
        """Apply context-based mood modifiers."""
        modifiers = []
        
        for context in context_factors:
            context_lower = context.lower()
            
            # Activity-based modifiers
            if 'workout' in context_lower or 'exercise' in context_lower:
                modifiers.extend(['motivational', 'energizing', 'pumping'])
            elif 'work' in context_lower or 'study' in context_lower:
                modifiers.extend(['focused', 'productive', 'concentration'])
            elif 'party' in context_lower or 'social' in context_lower:
                modifiers.extend(['social', 'upbeat', 'danceable'])
            elif 'drive' in context_lower or 'travel' in context_lower:
                modifiers.extend(['driving', 'journey', 'road'])
            elif 'sleep' in context_lower or 'bedtime' in context_lower:
                modifiers.extend(['sleepy', 'dreamy', 'peaceful'])
        
        return list(set(modifiers))  # Remove duplicates
    
    def _generate_mood_tags(self, mood_analysis: Dict[str, Any]) -> List[str]:
        """Generate mood-specific tags for search."""
        tags = []
        
        # Primary mood tags
        for mood in mood_analysis['primary_moods']:
            if mood in self.mood_mappings:
                tags.extend(self.mood_mappings[mood][:3])  # Top 3 tags per mood
        
        # Energy level tags
        energy_level = mood_analysis['energy_level']
        if energy_level in self.energy_mappings:
            tags.extend(self.energy_mappings[energy_level][:2])  # Top 2 energy tags
        
        # Context modifier tags
        tags.extend(mood_analysis.get('context_modifiers', [])[:2])
        
        # Intensity-based tags
        intensity = mood_analysis['mood_intensity']
        if intensity > 0.7:
            tags.extend(['intense', 'powerful'])
        elif intensity < 0.3:
            tags.extend(['gentle', 'subtle'])
        
        return list(set(tags))  # Remove duplicates
    
    def get_mood_combinations(self, primary_mood: str) -> List[str]:
        """Get compatible mood combinations for a primary mood."""
        return self.mood_combinations.get(primary_mood, [])
    
    def _initialize_advanced_mood_mappings(self) -> Dict[str, List[str]]:
        """Initialize comprehensive mood to tag mappings."""
        return {
            'energetic': [
                'energetic', 'upbeat', 'high energy', 'pumped', 'intense',
                'powerful', 'driving', 'motivational', 'dynamic', 'vigorous'
            ],
            'calm': [
                'calm', 'peaceful', 'relaxing', 'chill', 'mellow',
                'serene', 'tranquil', 'soothing', 'gentle', 'quiet'
            ],
            'happy': [
                'happy', 'joyful', 'uplifting', 'cheerful', 'positive',
                'bright', 'sunny', 'optimistic', 'feel good', 'euphoric'
            ],
            'melancholic': [
                'sad', 'melancholic', 'depressing', 'somber', 'moody',
                'melancholy', 'wistful', 'nostalgic', 'bittersweet', 'pensive'
            ],
            'aggressive': [
                'aggressive', 'angry', 'intense', 'heavy', 'brutal',
                'fierce', 'powerful', 'hard', 'raw', 'edgy'
            ],
            'romantic': [
                'romantic', 'love', 'intimate', 'sensual', 'passionate',
                'tender', 'sweet', 'dreamy', 'soft', 'emotional'
            ],
            'nostalgic': [
                'nostalgic', 'vintage', 'retro', 'classic', 'old school',
                'throwback', 'timeless', 'memories', 'reminiscent', 'sentimental'
            ],
            'mysterious': [
                'mysterious', 'dark', 'atmospheric', 'moody', 'enigmatic',
                'haunting', 'ethereal', 'ambient', 'cinematic', 'brooding'
            ],
            'playful': [
                'playful', 'fun', 'quirky', 'whimsical', 'lighthearted',
                'bouncy', 'silly', 'amusing', 'carefree', 'jovial'
            ]
        }
    
    def _initialize_energy_mappings(self) -> Dict[str, List[str]]:
        """Initialize energy level to tag mappings."""
        return {
            'high': [
                'energetic', 'upbeat', 'intense', 'pumped', 'high energy',
                'fast', 'driving', 'powerful', 'dynamic', 'vigorous'
            ],
            'medium': [
                'moderate', 'balanced', 'steady', 'medium energy', 'flowing',
                'consistent', 'rhythmic', 'groovy', 'smooth', 'comfortable'
            ],
            'low': [
                'calm', 'peaceful', 'relaxing', 'chill', 'mellow',
                'slow', 'gentle', 'soft', 'quiet', 'ambient'
            ]
        }
    
    def _initialize_context_modifiers(self) -> Dict[str, List[str]]:
        """Initialize context-based mood modifiers."""
        return {
            'workout': ['motivational', 'energizing', 'pumping', 'driving'],
            'work': ['focused', 'productive', 'concentration', 'background'],
            'study': ['focused', 'calm', 'concentration', 'instrumental'],
            'party': ['social', 'upbeat', 'danceable', 'fun'],
            'driving': ['driving', 'journey', 'road', 'cruising'],
            'relaxation': ['peaceful', 'calming', 'soothing', 'gentle']
        }
    
    def _initialize_mood_combinations(self) -> Dict[str, List[str]]:
        """Initialize compatible mood combinations."""
        return {
            'energetic': ['happy', 'aggressive', 'playful'],
            'calm': ['romantic', 'nostalgic', 'mysterious'],
            'happy': ['energetic', 'playful', 'romantic'],
            'melancholic': ['nostalgic', 'mysterious', 'romantic'],
            'aggressive': ['energetic', 'mysterious'],
            'romantic': ['calm', 'happy', 'melancholic'],
            'nostalgic': ['melancholic', 'calm', 'romantic'],
            'mysterious': ['melancholic', 'aggressive', 'calm'],
            'playful': ['happy', 'energetic']
        } 