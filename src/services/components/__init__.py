"""
Service Components

Modular components for the BeatDebate recommendation system.
Each component follows single responsibility principle for better maintainability.
"""

# API Service Components (no circular dependencies)
from .client_manager import ClientManager
from .track_operations import TrackOperations
from .artist_operations import ArtistOperations
from .genre_analyzer import GenreAnalyzer

# Enhanced Recommendation Service Components (may have agent dependencies)
# Import these separately to avoid circular imports when only API service is needed
try:
    from .context_handler import ContextHandler
    from .agent_coordinator import AgentCoordinator
    from .workflow_orchestrator import WorkflowOrchestrator
    from .state_manager import StateManager
    _RECOMMENDATION_COMPONENTS_AVAILABLE = True
except ImportError:
    _RECOMMENDATION_COMPONENTS_AVAILABLE = False

__all__ = [
    # API Service Components (always available)
    "ClientManager",
    "TrackOperations",
    "ArtistOperations", 
    "GenreAnalyzer"
]

# Add recommendation service components if available
if _RECOMMENDATION_COMPONENTS_AVAILABLE:
    __all__.extend([
        "ContextHandler",
        "AgentCoordinator", 
        "WorkflowOrchestrator",
        "StateManager"
    ]) 