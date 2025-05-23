"""
Models Package for BeatDebate Multi-Agent System

Contains Pydantic models for state management and data structures.
"""

from .agent_models import (
    MusicRecommenderState,
    AgentStrategy,
    TaskAnalysis,
    AgentCoordinationStrategy,
    EvaluationFramework,
    TrackRecommendation,
    AgentDeliberation,
    ReasoningChain,
    FinalRecommendationResponse,
    AgentConfig,
    SystemConfig
)

__all__ = [
    "MusicRecommenderState",
    "AgentStrategy",
    "TaskAnalysis",
    "AgentCoordinationStrategy",
    "EvaluationFramework",
    "TrackRecommendation",
    "AgentDeliberation",
    "ReasoningChain",
    "FinalRecommendationResponse",
    "AgentConfig",
    "SystemConfig",
]
