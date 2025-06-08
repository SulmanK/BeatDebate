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

from .recommendation_service import (
    RecommendationService,
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

# ConversationContextManager functionality moved to SessionManagerService

# Phase 1 Enhanced Services
from .session_manager_service import (
    SessionManagerService,
    OriginalQueryContext,
    CandidatePool,
    ContextState
)

from .intent_orchestration_service import (
    IntentOrchestrationService,
    FollowUpType
)

# Existing services (maintained for backward compatibility)
from .cache_manager import CacheManager, get_cache_manager

# Deprecated services (Phase 5: marked for removal)
# TODO: Remove after migration to enhanced services
# from .recommendation_engine import RecommendationEngine

__all__ = [
    # Enhanced services (Primary)
    "APIService",
    "get_api_service", 
    "close_api_service",
    "RecommendationService",
    "RecommendationRequest",
    "RecommendationResponse", 
    "get_recommendation_service",
    "close_recommendation_service",
    "MetadataService",
    "get_metadata_service",
    "close_metadata_service",
    
    # Phase 1 Enhanced Services
    "SessionManagerService",
    "OriginalQueryContext", 
    "CandidatePool",
    "ContextState",
    "IntentOrchestrationService",
    "FollowUpType",
    
    # Existing services
    "CacheManager",
    "get_cache_manager",
]
