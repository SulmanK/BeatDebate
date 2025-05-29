"""
Services Module

Enhanced services layer with unified API access and eliminated duplication.
Provides centralized business logic and streamlined workflows.
"""

# Enhanced services (Primary)
from .api_service import (
    APIService,
    get_api_service,
    close_api_service
)

from .enhanced_recommendation_service import (
    EnhancedRecommendationService,
    RecommendationRequest,
    RecommendationResponse,
    get_recommendation_service,
    close_recommendation_service
)

from .metadata_service import (
    MetadataService,
    get_metadata_service,
    close_metadata_service
)

from .conversation_context_service import (
    ConversationContextManager
)

# Existing services (maintained for backward compatibility)
from .smart_context_manager import SmartContextManager
from .cache_manager import CacheManager, get_cache_manager

# Deprecated services (Phase 5: marked for removal)
# TODO: Remove after migration to enhanced services
# from .recommendation_engine import RecommendationEngine

__all__ = [
    # Enhanced services (Primary)
    "APIService",
    "get_api_service", 
    "close_api_service",
    "EnhancedRecommendationService",
    "RecommendationRequest",
    "RecommendationResponse", 
    "get_recommendation_service",
    "close_recommendation_service",
    "MetadataService",
    "get_metadata_service",
    "close_metadata_service",
    "ConversationContextManager",
    
    # Existing services
    "SmartContextManager",
    "CacheManager",
    "get_cache_manager",
]
