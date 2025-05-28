"""
Services Module

Enhanced services layer with unified API access and eliminated duplication.
Provides centralized business logic and streamlined workflows.
"""

# Enhanced services
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

# Existing services (maintained for backward compatibility)
# TODO: Fix import issues in these modules
# from .recommendation_engine import RecommendationEngine
# from .smart_context_manager import SmartContextManager
from .cache_manager import CacheManager, get_cache_manager, close_cache_manager

__all__ = [
    # Enhanced services
    "APIService",
    "get_api_service", 
    "close_api_service",
    "EnhancedRecommendationService",
    "RecommendationRequest",
    "RecommendationResponse",
    "get_recommendation_service",
    "close_recommendation_service",
    
    # Legacy services (backward compatibility)
    # "RecommendationEngine",
    # "SmartContextManager",
    "CacheManager",
    "get_cache_manager",
    "close_cache_manager",
]
