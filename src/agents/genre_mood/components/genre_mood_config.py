"""
GenreMoodConfig Component

Handles intent parameter management and configuration for the GenreMoodAgent.
Centralizes all intent-specific parameter configurations and adaptation logic.
"""

from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class GenreMoodConfig:
    """
    Manages intent-specific configuration and parameter adaptation for GenreMoodAgent.
    
    Responsibilities:
    - Intent parameter configuration management
    - Parameter adaptation based on detected intent
    - Configuration validation and defaults
    - Intent-specific threshold and weight management
    """
    
    def __init__(self):
        """Initialize GenreMoodConfig with intent parameter mappings."""
        self.logger = logger.bind(component="GenreMoodConfig")
        
        # Base configuration
        self.target_candidates = 200
        self.final_recommendations = 20
        self.quality_threshold = 0.4
        
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
        
        # Current adapted parameters (set by adapt_to_intent)
        self.current_genre_weight = 0.5
        self.current_mood_weight = 0.5
        self.current_audio_feature_weight = 0.5
        self.current_max_per_genre = 4
        self.current_candidate_focus = 'balanced_style'
        
        self.logger.info("GenreMoodConfig initialized with intent-aware parameters")
    
    def adapt_to_intent(self, intent: str) -> None:
        """
        Adapt agent parameters based on detected intent.
        
        Args:
            intent: The detected intent to adapt parameters for
        """
        if intent in self.intent_parameters:
            params = self.intent_parameters[intent]
            
            self.quality_threshold = params['quality_threshold']
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
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration parameters.
        
        Returns:
            Dictionary containing current configuration values
        """
        return {
            'quality_threshold': self.quality_threshold,
            'genre_weight': self.current_genre_weight,
            'mood_weight': self.current_mood_weight,
            'audio_feature_weight': self.current_audio_feature_weight,
            'max_per_genre': self.current_max_per_genre,
            'candidate_focus': self.current_candidate_focus,
            'target_candidates': self.target_candidates,
            'final_recommendations': self.final_recommendations
        }
    
    def get_intent_parameters(self, intent: str) -> Dict[str, Any]:
        """
        Get parameters for a specific intent without adapting current config.
        
        Args:
            intent: The intent to get parameters for
            
        Returns:
            Dictionary containing intent-specific parameters
        """
        return self.intent_parameters.get(intent, {
            'quality_threshold': 0.4,
            'genre_weight': 0.5,
            'mood_weight': 0.5,
            'audio_feature_weight': 0.5,
            'max_per_genre': 4,
            'candidate_focus': 'balanced_style'
        })
    
    def validate_config(self) -> bool:
        """
        Validate current configuration parameters.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check threshold bounds
            if not (0.0 <= self.quality_threshold <= 1.0):
                self.logger.error(f"Invalid quality_threshold: {self.quality_threshold}")
                return False
            
            # Check weight bounds
            weights = [
                self.current_genre_weight,
                self.current_mood_weight,
                self.current_audio_feature_weight
            ]
            
            for weight in weights:
                if not (0.0 <= weight <= 1.0):
                    self.logger.error(f"Invalid weight value: {weight}")
                    return False
            
            # Check positive integers
            if self.current_max_per_genre <= 0:
                self.logger.error(f"Invalid max_per_genre: {self.current_max_per_genre}")
                return False
            
            if self.target_candidates <= 0:
                self.logger.error(f"Invalid target_candidates: {self.target_candidates}")
                return False
            
            if self.final_recommendations <= 0:
                self.logger.error(f"Invalid final_recommendations: {self.final_recommendations}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.quality_threshold = 0.4
        self.current_genre_weight = 0.5
        self.current_mood_weight = 0.5
        self.current_audio_feature_weight = 0.5
        self.current_max_per_genre = 4
        self.current_candidate_focus = 'balanced_style'
        
        self.logger.info("Configuration reset to defaults") 