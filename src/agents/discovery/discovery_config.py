"""
Discovery Configuration for BeatDebate

Centralized configuration for discovery-specific parameters including:
- Intent-specific parameter mappings
- Quality and novelty thresholds
- Underground bias settings
- Similarity depth configurations
- Candidate focus strategies

This module extracts configuration logic from DiscoveryAgent for better maintainability.
"""

from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class DiscoveryConfig:
    """
    Centralized configuration for discovery agent parameters.
    
    This class manages all intent-specific configurations and provides
    a clean interface for parameter adaptation.
    """
    
    def __init__(self):
        """Initialize discovery configuration."""
        self.logger = logger.bind(component="DiscoveryConfig")
        
        # Base configuration
        self.base_config = {
            'target_candidates': 200,
            'final_recommendations': 20,
            'quality_threshold': 0.3,
            'novelty_threshold': 0.4,
            'underground_bias': 0.7,
            'similarity_depth': 2
        }
        
        # Intent-specific parameter configurations
        self.intent_parameters = {
            'by_artist': {
                'novelty_threshold': 0.0,     # No novelty filtering for by_artist
                'quality_threshold': 0.25,    # Moderate quality for artist discography
                'underground_bias': 0.0,      # No underground bias for by_artist
                'similarity_depth': 1,        # Simple matching for target artist
                'max_per_artist': 20,         # Allow many tracks from target artist
                'candidate_focus': 'target_artist_discography'
            },
            'by_artist_underground': {
                'novelty_threshold': 0.6,     # High novelty filtering for underground tracks
                'quality_threshold': 0.15,    # Lower quality threshold for underground gems
                'underground_bias': 0.8,      # Strong underground bias
                'similarity_depth': 1,        # Simple matching for target artist
                'max_per_artist': 20,         # Allow many tracks from target artist
                'candidate_focus': 'target_artist_deep_cuts'
            },
            'artist_similarity': {
                'target_candidates': 200,     # Large pool for diverse recommendations
                'final_recommendations': 20,  # Target 20 recommendations
                'novelty_threshold': 0.15,    # Very relaxed for similar artists
                'quality_threshold': 0.2,     # Lower quality for broader similarity search
                'underground_bias': 0.1,      # Slight underground preference
                'similarity_depth': 2,        # Deep similarity search
                'max_per_artist': 5,          # Allow more tracks per similar artist for diversity
                'candidate_focus': 'similar_artists'
            },
            'discovery': {
                'novelty_threshold': 0.6,     # Strict novelty requirement
                'quality_threshold': 0.25,    # Lower quality to find gems
                'underground_bias': 0.8,      # High underground preference
                'similarity_depth': 1,        # Less similarity depth
                'max_per_artist': 3,          # Increased from 1 to 3 for better exploration
                'candidate_focus': 'underground_gems'
            },
            'discovering_serendipity': {
                'novelty_threshold': 0.1,     # Very low novelty filtering for maximum surprise
                'quality_threshold': 0.15,    # Lower quality threshold for unexpected finds
                'underground_bias': 0.2,      # Light underground preference
                'similarity_depth': 0,        # No similarity constraints for pure serendipity
                'max_per_artist': 3,          # Allow more tracks per artist for serendipitous exploration
                'candidate_focus': 'serendipitous_exploration'
            },
            'genre_mood': {
                'novelty_threshold': 0.3,     # Moderate novelty
                'quality_threshold': 0.35,    # Balanced quality
                'underground_bias': 0.5,      # Balanced underground bias
                'similarity_depth': 2,        # Standard depth
                'max_per_artist': 2,          # Moderate diversity
                'candidate_focus': 'style_match'
            },
            'contextual': {
                'novelty_threshold': 0.2,     # Relaxed for functional music
                'quality_threshold': 0.45,    # Higher quality for context
                'underground_bias': 0.4,      # Less underground for reliability
                'similarity_depth': 1,        # Simple matching
                'max_per_artist': 2,          # Moderate diversity
                'candidate_focus': 'functional_fit'
            },
            'hybrid': {
                'novelty_threshold': 0.25,    # Moderate novelty
                'quality_threshold': 0.35,    # Balanced approach
                'underground_bias': 0.6,      # Moderate underground bias
                'similarity_depth': 2,        # Standard depth
                'max_per_artist': 2,          # Balanced diversity
                'candidate_focus': 'balanced'
            }
        }
        
        # Followup-specific adjustments
        self.followup_adjustments = {
            'load_more': {
                'quality_threshold_adjustment': -0.1,  # More permissive
                'novelty_threshold_adjustment': -0.05,
                'candidate_multiplier': 1.2
            },
            'artist_deep_dive': {
                'underground_bias_override': 0.2,
                'novelty_threshold_override': 0.1,
                'max_per_artist_multiplier': 1.5
            },
            'genre_exploration': {
                'similarity_depth_adjustment': 1,
                'underground_bias_adjustment': 0.1,
                'candidate_multiplier': 1.3
            }
        }
        
        self.logger.info("Discovery Configuration initialized")
    
    def get_intent_parameters(self, intent: str) -> Dict[str, Any]:
        """
        Get parameters for a specific intent.
        
        Args:
            intent: Intent string
            
        Returns:
            Dictionary of intent-specific parameters
        """
        if intent in self.intent_parameters:
            params = self.intent_parameters[intent].copy()
            self.logger.debug(f"Retrieved parameters for intent: {intent}")
            return params
        else:
            self.logger.warning(f"Unknown intent: {intent}, using default parameters")
            return self.intent_parameters['discovery'].copy()
    
    def adapt_for_effective_intent(
        self, 
        effective_intent: Dict[str, Any], 
        base_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Adapt parameters based on effective intent from IntentOrchestrationService.
        
        Args:
            effective_intent: Effective intent dictionary
            base_params: Optional base parameters to modify
            
        Returns:
            Adapted parameters dictionary
        """
        intent_str = effective_intent.get('intent', 'discovery')
        
        # Start with intent-specific parameters
        if base_params is None:
            params = self.get_intent_parameters(intent_str)
        else:
            params = base_params.copy()
        
        # Apply followup adjustments if this is a followup
        if effective_intent.get('is_followup'):
            followup_type = effective_intent.get('followup_type')
            if followup_type in self.followup_adjustments:
                adjustments = self.followup_adjustments[followup_type]
                params = self._apply_followup_adjustments(params, adjustments)
                self.logger.info(f"Applied followup adjustments for: {followup_type}")
        
        # Adjust based on confidence level
        confidence = effective_intent.get('confidence', 0.8)
        if confidence < 0.6:
            # Lower confidence - be more permissive and generate more candidates
            params['quality_threshold'] = max(0.1, params.get('quality_threshold', 0.3) - 0.1)
            params['novelty_threshold'] = max(0.0, params.get('novelty_threshold', 0.4) - 0.1)
            params['target_candidates'] = int(params.get('target_candidates', 200) * 1.3)
            self.logger.info(f"Adjusted for low confidence ({confidence})")
        
        # Apply context-specific adjustments
        context_type = effective_intent.get('context_type')
        if context_type == 'continuation':
            # For continuation contexts, be more permissive
            params['quality_threshold'] = max(0.1, params.get('quality_threshold', 0.3) - 0.05)
            params['novelty_threshold'] = max(0.0, params.get('novelty_threshold', 0.4) - 0.05)
        
        self.logger.debug(
            "Adapted parameters for effective intent",
            intent=intent_str,
            is_followup=effective_intent.get('is_followup', False),
            confidence=confidence
        )
        
        return params
    
    def _apply_followup_adjustments(
        self, 
        params: Dict[str, Any], 
        adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply followup-specific adjustments to parameters."""
        adjusted_params = params.copy()
        
        # Apply threshold adjustments
        if 'quality_threshold_adjustment' in adjustments:
            current_quality = adjusted_params.get('quality_threshold', 0.3)
            adjusted_params['quality_threshold'] = max(
                0.1, current_quality + adjustments['quality_threshold_adjustment']
            )
        
        if 'novelty_threshold_adjustment' in adjustments:
            current_novelty = adjusted_params.get('novelty_threshold', 0.4)
            adjusted_params['novelty_threshold'] = max(
                0.0, current_novelty + adjustments['novelty_threshold_adjustment']
            )
        
        # Apply bias adjustments
        if 'underground_bias_adjustment' in adjustments:
            current_bias = adjusted_params.get('underground_bias', 0.7)
            adjusted_params['underground_bias'] = min(
                1.0, max(0.0, current_bias + adjustments['underground_bias_adjustment'])
            )
        
        # Apply overrides
        if 'underground_bias_override' in adjustments:
            adjusted_params['underground_bias'] = adjustments['underground_bias_override']
        
        if 'novelty_threshold_override' in adjustments:
            adjusted_params['novelty_threshold'] = adjustments['novelty_threshold_override']
        
        # Apply multipliers
        if 'candidate_multiplier' in adjustments:
            current_candidates = adjusted_params.get('target_candidates', 200)
            adjusted_params['target_candidates'] = int(
                current_candidates * adjustments['candidate_multiplier']
            )
        
        if 'max_per_artist_multiplier' in adjustments:
            current_max = adjusted_params.get('max_per_artist', 2)
            adjusted_params['max_per_artist'] = int(
                current_max * adjustments['max_per_artist_multiplier']
            )
        
        # Apply depth adjustments
        if 'similarity_depth_adjustment' in adjustments:
            current_depth = adjusted_params.get('similarity_depth', 2)
            adjusted_params['similarity_depth'] = max(
                1, current_depth + adjustments['similarity_depth_adjustment']
            )
        
        return adjusted_params
    
    def get_candidate_focus_strategy(self, intent: str) -> str:
        """
        Get the candidate focus strategy for an intent.
        
        Args:
            intent: Intent string
            
        Returns:
            Candidate focus strategy string
        """
        params = self.get_intent_parameters(intent)
        return params.get('candidate_focus', 'balanced')
    
    def get_diversity_parameters(self, intent: str) -> Dict[str, Any]:
        """
        Get diversity-specific parameters for an intent.
        
        Args:
            intent: Intent string
            
        Returns:
            Dictionary of diversity parameters
        """
        params = self.get_intent_parameters(intent)
        
        diversity_params = {
            'max_per_artist': params.get('max_per_artist', 2),
            'candidate_limit': 30,  # Default limit
            'variety_emphasis': 0.5  # Default variety emphasis
        }
        
        # Intent-specific diversity adjustments
        if intent == 'discovery':
            diversity_params.update({
                'candidate_limit': 50,   # Increased from 30 to 50 for more discovery options
                'variety_emphasis': 0.8  # High variety for discovery
            })
        elif intent in ['by_artist', 'by_artist_underground']:
            diversity_params.update({
                'candidate_limit': 50,
                'variety_emphasis': 0.2  # Low variety for artist focus
            })
        elif intent == 'artist_similarity':
            diversity_params.update({
                'candidate_limit': 40,
                'variety_emphasis': 0.4  # Moderate variety for similarity
            })
        
        return diversity_params
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize parameters.
        
        Args:
            params: Parameters dictionary to validate
            
        Returns:
            Validated parameters dictionary
        """
        validated = params.copy()
        
        # Ensure thresholds are within valid ranges
        validated['quality_threshold'] = max(0.0, min(1.0, validated.get('quality_threshold', 0.3)))
        validated['novelty_threshold'] = max(0.0, min(1.0, validated.get('novelty_threshold', 0.4)))
        validated['underground_bias'] = max(0.0, min(1.0, validated.get('underground_bias', 0.7)))
        
        # Ensure positive integers
        validated['target_candidates'] = max(10, validated.get('target_candidates', 200))
        validated['similarity_depth'] = max(1, validated.get('similarity_depth', 2))
        validated['max_per_artist'] = max(1, validated.get('max_per_artist', 2))
        
        return validated 