"""
Models Module

Unified data models and schemas for the BeatDebate system.
Provides consistent data structures across all services and components.
"""

from .metadata_models import (
    MetadataSource,
    UnifiedTrackMetadata,
    UnifiedArtistMetadata,
    SearchResult,
    merge_track_metadata,
    calculate_quality_scores
)

# Import existing models for backward compatibility
try:
    from .agent_models import (
        SystemConfig,
        MusicRecommenderState,
        RecommendationRequest,
        RecommendationResult
    )
except ImportError:
    # Handle case where agent_models doesn't exist yet
    pass

try:
    from .recommendation_models import (
        TrackRecommendation,
        RecommendationContext,
        RecommendationMetrics
    )
except ImportError:
    # Handle case where recommendation_models doesn't exist yet
    pass

__all__ = [
    # Unified metadata models
    "MetadataSource",
    "UnifiedTrackMetadata", 
    "UnifiedArtistMetadata",
    "SearchResult",
    "merge_track_metadata",
    "calculate_quality_scores",
    
    # Agent models (if available)
    "SystemConfig",
    "MusicRecommenderState",
    "RecommendationRequest",
    "RecommendationResult",
    
    # Recommendation models (if available)
    "TrackRecommendation",
    "RecommendationContext", 
    "RecommendationMetrics",
]
