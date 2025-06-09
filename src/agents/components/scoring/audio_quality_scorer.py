"""
Audio Quality Scorer for BeatDebate

Scores tracks based on audio feature analysis, evaluating energy, danceability, 
valence, and other audio characteristics to determine musical quality and appropriateness.
"""

from typing import Dict
import structlog

logger = structlog.get_logger(__name__)


class AudioQualityScorer:
    """
    Scores tracks based on audio feature analysis.
    
    Evaluates energy, danceability, valence, and other audio characteristics
    to determine musical quality and appropriateness.
    """
    
    def __init__(self):
        """Initialize audio quality scorer with feature weights."""
        self.logger = logger.bind(component="AudioQualityScorer")
        
        # Feature weights for audio quality calculation
        self.feature_weights = {
            'energy': 0.20,           # Energy level appropriateness
            'danceability': 0.15,     # Rhythmic quality
            'valence': 0.15,          # Emotional positivity
            'acousticness': 0.10,     # Acoustic vs electronic balance
            'instrumentalness': 0.10,  # Vocal vs instrumental balance
            'liveness': 0.10,         # Live recording quality
            'speechiness': 0.10,      # Speech content appropriateness
            'tempo_consistency': 0.10  # Tempo stability
        }
        
        self.logger.info("Audio Quality Scorer initialized")
    
    def calculate_audio_quality_score(self, track_features: Dict, intent_analysis: Dict) -> float:
        """
        Calculate quality score based on audio features.
        
        Args:
            track_features: Audio features from Last.fm or Spotify
            intent_analysis: User intent for context-aware scoring
            
        Returns:
            Audio quality score (0.0 - 1.0)
        """
        try:
            quality_score = 0.0
            
            # Energy optimization based on intent
            energy_score = self._score_energy(track_features, intent_analysis)
            quality_score += energy_score * self.feature_weights['energy']
            
            # Danceability scoring
            danceability_score = self._score_danceability(track_features, intent_analysis)
            quality_score += danceability_score * self.feature_weights['danceability']
            
            # Valence (mood) scoring
            valence_score = self._score_valence(track_features, intent_analysis)
            quality_score += valence_score * self.feature_weights['valence']
            
            # Acousticness balance
            acousticness_score = self._score_acousticness(track_features, intent_analysis)
            quality_score += acousticness_score * self.feature_weights['acousticness']
            
            # Instrumentalness balance
            instrumentalness_score = self._score_instrumentalness(track_features, intent_analysis)
            quality_score += instrumentalness_score * self.feature_weights['instrumentalness']
            
            # Liveness quality
            liveness_score = self._score_liveness(track_features)
            quality_score += liveness_score * self.feature_weights['liveness']
            
            # Speechiness appropriateness
            speechiness_score = self._score_speechiness(track_features, intent_analysis)
            quality_score += speechiness_score * self.feature_weights['speechiness']
            
            # Tempo consistency
            tempo_score = self._score_tempo(track_features, intent_analysis)
            quality_score += tempo_score * self.feature_weights['tempo_consistency']
            
            final_score = min(1.0, max(0.0, quality_score))
            
            self.logger.debug(
                "Audio quality calculated",
                score=final_score,
                energy=energy_score,
                danceability=danceability_score,
                valence=valence_score
            )
            
            return final_score
            
        except Exception as e:
            self.logger.warning("Audio quality calculation failed", error=str(e))
            return 0.5  # Default neutral score
    
    def _score_energy(self, features: Dict, intent: Dict) -> float:
        """Score energy level based on intent context."""
        energy = features.get('energy', 0.5)
        
        # Get activity context for energy preferences
        primary_intent = intent.get('primary_intent', 'discovery')
        
        # Energy preferences by intent
        energy_preferences = {
            'concentration': 0.3,  # Lower energy for focus
            'relaxation': 0.2,     # Very low energy for relaxation
            'energy': 0.8,         # High energy for energetic activities
            'discovery': 0.5,      # Balanced energy for discovery
            'workout': 0.9,        # Very high energy for workouts
            'study': 0.3           # Lower energy for studying
        }
        
        target_energy = energy_preferences.get(primary_intent, 0.5)
        
        # Score based on distance from target
        energy_distance = abs(energy - target_energy)
        energy_score = 1.0 - (energy_distance / 1.0)  # Normalize to 0-1
        
        return max(0.0, energy_score)
    
    def _score_danceability(self, features: Dict, intent: Dict) -> float:
        """Score danceability based on context."""
        danceability = features.get('danceability', 0.5)
        
        # Higher danceability is generally positive
        # But adjust based on intent
        primary_intent = intent.get('primary_intent', 'discovery')
        
        if primary_intent in ['workout', 'energy', 'party']:
            # High danceability preferred
            return danceability
        elif primary_intent in ['concentration', 'study', 'relaxation']:
            # Moderate danceability preferred
            return 1.0 - abs(danceability - 0.4) / 0.6
        else:
            # Balanced approach
            return danceability * 0.8 + 0.2
    
    def _score_valence(self, features: Dict, intent: Dict) -> float:
        """Score valence (musical positivity) based on intent."""
        valence = features.get('valence', 0.5)
        
        primary_intent = intent.get('primary_intent', 'discovery')
        
        # Valence preferences by intent
        if primary_intent in ['energy', 'workout', 'party']:
            # Higher valence preferred for energetic activities
            return valence
        elif primary_intent in ['relaxation', 'study']:
            # Moderate valence preferred
            return 1.0 - abs(valence - 0.4) / 0.6
        else:
            # Avoid extremes, prefer moderate positivity
            return 1.0 - abs(valence - 0.6) / 0.6
    
    def _score_acousticness(self, features: Dict, intent: Dict) -> float:
        """Score acousticness based on context preferences."""
        acousticness = features.get('acousticness', 0.5)
        
        primary_intent = intent.get('primary_intent', 'discovery')
        
        if primary_intent in ['relaxation', 'study', 'concentration']:
            # Higher acousticness preferred for calm activities
            return acousticness
        elif primary_intent in ['workout', 'energy', 'party']:
            # Lower acousticness (more electronic) preferred
            return 1.0 - acousticness
        else:
            # Balanced preference
            return 1.0 - abs(acousticness - 0.5) / 0.5
    
    def _score_instrumentalness(self, features: Dict, intent: Dict) -> float:
        """Score instrumentalness based on context."""
        instrumentalness = features.get('instrumentalness', 0.5)
        
        primary_intent = intent.get('primary_intent', 'discovery')
        
        if primary_intent in ['concentration', 'study']:
            # Higher instrumentalness preferred for focus
            return instrumentalness
        else:
            # Generally prefer some vocals
            return 1.0 - (instrumentalness * 0.6)
    
    def _score_liveness(self, features: Dict) -> float:
        """Score liveness - generally prefer studio recordings."""
        liveness = features.get('liveness', 0.1)
        
        # Prefer studio recordings (lower liveness)
        # But don't penalize too heavily
        return 1.0 - (liveness * 0.5)
    
    def _score_speechiness(self, features: Dict, intent: Dict) -> float:
        """Score speechiness based on context."""
        speechiness = features.get('speechiness', 0.1)
        
        primary_intent = intent.get('primary_intent', 'discovery')
        
        if primary_intent in ['concentration', 'study']:
            # Lower speechiness preferred for focus
            return 1.0 - speechiness
        else:
            # Moderate speechiness is fine
            return 1.0 - (speechiness * 0.3)
    
    def _score_tempo(self, features: Dict, intent: Dict) -> float:
        """Score tempo appropriateness."""
        tempo = features.get('tempo', 120)
        
        primary_intent = intent.get('primary_intent', 'discovery')
        
        # Tempo preferences by intent
        tempo_preferences = {
            'concentration': (80, 110),   # Slower tempo for focus
            'relaxation': (60, 100),      # Very slow tempo
            'energy': (120, 180),         # Fast tempo for energy
            'workout': (130, 170),        # High tempo for workouts
            'study': (80, 120),           # Moderate tempo for studying
            'discovery': (90, 140)        # Wide range for discovery
        }
        
        min_tempo, max_tempo = tempo_preferences.get(primary_intent, (90, 140))
        
        if min_tempo <= tempo <= max_tempo:
            return 1.0
        elif tempo < min_tempo:
            # Too slow
            distance = min_tempo - tempo
            return max(0.0, 1.0 - (distance / 50))
        else:
            # Too fast
            distance = tempo - max_tempo
            return max(0.0, 1.0 - (distance / 50)) 