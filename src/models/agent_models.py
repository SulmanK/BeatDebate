"""
Agent Models for BeatDebate Multi-Agent Music Recommendation System

Pydantic models for state management, agent communication, and data structures.
"""

from typing import Dict, List, Any, Optional, Annotated
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


def keep_first(x: Any, y: Any) -> Any:
    """Reducer that keeps the first non-None value"""
    # If x is None or empty, use y
    if x is None or (isinstance(x, (list, dict, str)) and len(x) == 0):
        return y
    # Otherwise keep x (the first/existing value)
    return x


def list_replace_reducer(x: List, y: List) -> List:
    """Reducer that replaces the list entirely if y is not empty"""
    if y is None:
        return x if x is not None else []
    if len(y) > 0:
        return y
    return x if x is not None else []


def list_append_reducer(x: List, y: List) -> List:
    """Reducer that appends new items to existing list"""
    if x is None:
        x = []
    if y is None:
        y = []
    return x + y


def dict_update_reducer(x: Dict, y: Dict) -> Dict:
    """Reducer that updates dictionary with new values"""
    if x is None:
        x = {}
    if y is None:
        y = {}
    result = x.copy()
    result.update(y)
    return result


class MusicRecommenderState(BaseModel):
    """Shared state across all agents in the LangGraph workflow"""
    
    # Input - these should not change after initial setting
    user_query: Annotated[str, keep_first] = Field(..., description="Original user query for music recommendation")
    user_profile: Annotated[Optional[Dict[str, Any]], keep_first] = Field(default=None, description="User preferences and history")
    max_recommendations: Annotated[int, keep_first] = Field(default=10, description="Maximum number of recommendations to return")
    
    # Planning phase - set once by planner
    planning_strategy: Annotated[Optional[Dict[str, Any]], keep_first] = Field(default=None, description="Strategy created by PlannerAgent")
    execution_plan: Annotated[Optional[Dict[str, Any]], keep_first] = Field(default=None, description="Execution monitoring plan")
    coordination_strategy: Annotated[Optional[Dict[str, Any]], keep_first] = Field(default=None, description="Enhanced coordination strategy with confidence-based selection")
    agent_coordination: Annotated[Optional[Dict[str, Any]], keep_first] = Field(default=None, description="Agent coordination plan")
    
    # Enhanced Planning Phase - Entity Recognition (NEW)
    entities: Annotated[Optional[Dict[str, Any]], keep_first] = Field(default=None, description="Extracted entities from user query")
    intent_analysis: Annotated[Optional[Dict[str, Any]], keep_first] = Field(default=None, description="Intent analysis from user query")
    query_understanding: Annotated[Optional[Any], keep_first] = Field(default=None, description="Pure LLM query understanding result")
    conversation_context: Annotated[Optional[Dict[str, Any]], dict_update_reducer] = Field(default=None, description="Session conversation context")
    entity_reasoning: Annotated[List[Dict], list_append_reducer] = Field(default_factory=list, description="Entity extraction reasoning steps")
    context_decision: Annotated[Optional[Dict[str, Any]], keep_first] = Field(default=None, description="Context analysis decision")
    
    # Advocate phase - these will be updated by parallel agents
    genre_mood_recommendations: Annotated[List[Dict], list_append_reducer] = Field(default_factory=list, description="GenreMoodAgent recommendations")
    discovery_recommendations: Annotated[List[Dict], list_append_reducer] = Field(default_factory=list, description="DiscoveryAgent recommendations")
    
    # Judge phase - set by judge agent
    final_recommendations: Annotated[List[Dict], list_replace_reducer] = Field(default_factory=list, description="Final selected recommendations")
    
    # Reasoning transparency - can be updated by any agent
    reasoning_log: Annotated[List[str], list_append_reducer] = Field(default_factory=list, description="Step-by-step reasoning log")
    agent_deliberations: Annotated[List[Dict], list_append_reducer] = Field(default_factory=list, description="Agent decision records")
    
    # Error handling - can be set by any agent
    error_info: Annotated[Optional[Dict[str, str]], keep_first] = Field(default=None, description="Error information if workflow fails")
    
    # Metadata - set once at start, updated at end
    processing_start_time: Annotated[Optional[float], keep_first] = Field(default=None, description="Processing start timestamp")
    total_processing_time: Annotated[Optional[float], keep_first] = Field(default=None, description="Total processing time in seconds")
    session_id: Annotated[Optional[str], keep_first] = Field(default=None, description="Unique session identifier")
    confidence: Annotated[Optional[float], keep_first] = Field(default=None, description="Overall confidence in query understanding")


class AgentStrategy(BaseModel):
    """Strategy object passed between agents"""
    
    task_analysis: Dict[str, Any] = Field(..., description="Analysis of the user query and task")
    coordination_strategy: Dict[str, Any] = Field(..., description="Strategy for each advocate agent")
    evaluation_framework: Dict[str, Any] = Field(..., description="Criteria for judge evaluation")
    execution_monitoring: Dict[str, Any] = Field(..., description="Monitoring and adaptation protocols")


class TaskAnalysis(BaseModel):
    """Analysis of user query complexity and intent"""
    
    primary_goal: str = Field(..., description="Main intent extracted from query")
    complexity_level: str = Field(..., description="Query complexity: simple, medium, complex")
    context_factors: List[str] = Field(default_factory=list, description="Context clues from query")
    mood_indicators: List[str] = Field(default_factory=list, description="Mood/energy indicators")
    genre_hints: List[str] = Field(default_factory=list, description="Genre preferences or hints")


class AgentCoordinationStrategy(BaseModel):
    """Coordination strategy for advocate agents"""
    
    genre_mood_agent: Dict[str, Any] = Field(..., description="Strategy for GenreMoodAgent")
    discovery_agent: Dict[str, Any] = Field(..., description="Strategy for DiscoveryAgent")


class EvaluationFramework(BaseModel):
    """Framework for judge evaluation"""
    
    primary_weights: Dict[str, float] = Field(..., description="Weights for different criteria")
    diversity_targets: Dict[str, int] = Field(..., description="Diversity targets for recommendations")
    explanation_style: str = Field(..., description="Style for generating explanations")


class TrackRecommendation(BaseModel):
    """Individual track recommendation with reasoning"""
    
    # Track metadata
    title: str = Field(..., description="Track title")
    artist: str = Field(..., description="Artist name")
    album: Optional[str] = Field(default=None, description="Album name")
    year: Optional[int] = Field(default=None, description="Release year")
    
    # External identifiers
    lastfm_url: Optional[str] = Field(default=None, description="Last.fm URL")
    spotify_url: Optional[str] = Field(default=None, description="Spotify URL")
    preview_url: Optional[str] = Field(default=None, description="Audio preview URL")
    
    # Metadata
    genres: List[str] = Field(default_factory=list, description="Genre tags")
    tags: List[str] = Field(default_factory=list, description="Mood/style tags")
    similar_artists: List[str] = Field(default_factory=list, description="Similar artists")
    
    # Recommendation context
    reasoning_chain: str = Field(..., description="Agent's reasoning for this recommendation")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    novelty_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Novelty/underground score")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to user query")
    
    # Agent attribution
    recommending_agent: str = Field(..., description="Agent that made this recommendation")
    strategy_applied: Dict[str, Any] = Field(..., description="Strategy used for this recommendation")


class AgentDeliberation(BaseModel):
    """Record of agent decision-making process"""
    
    agent_name: str = Field(..., description="Name of the agent")
    timestamp: datetime = Field(default_factory=datetime.now, description="When deliberation occurred")
    input_data: Dict[str, Any] = Field(..., description="Input data for the agent")
    reasoning_steps: List[str] = Field(..., description="Step-by-step reasoning process")
    output_data: Dict[str, Any] = Field(..., description="Agent's output/decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent's confidence in decision")
    processing_time: float = Field(..., description="Time taken for deliberation in seconds")


class ReasoningChain(BaseModel):
    """Structured reasoning chain for transparency"""
    
    step_number: int = Field(..., description="Step number in reasoning chain")
    step_type: str = Field(..., description="Type of reasoning step")
    description: str = Field(..., description="Description of reasoning step")
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting this step")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this reasoning step")


class FinalRecommendationResponse(BaseModel):
    """Final response format for the user"""
    
    recommendations: List[TrackRecommendation] = Field(..., description="Final track recommendations")
    explanation: str = Field(..., description="Overall explanation of recommendations")
    planning_summary: str = Field(..., description="Summary of planning process")
    agent_coordination_summary: str = Field(..., description="Summary of agent coordination")
    
    # Metadata
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    agents_involved: List[str] = Field(..., description="List of agents that participated")
    reasoning_transparency: List[AgentDeliberation] = Field(..., description="Full reasoning transparency")
    session_id: str = Field(..., description="Session identifier")
    
    # Quality metrics
    diversity_score: float = Field(..., ge=0.0, le=1.0, description="Diversity of recommendations")
    novelty_score: float = Field(..., ge=0.0, le=1.0, description="Average novelty score")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")


class AgentConfig(BaseModel):
    """Configuration for individual agents"""
    
    agent_name: str = Field(..., description="Name of the agent")
    agent_type: str = Field(..., description="Type of agent (planner, advocate, judge)")
    llm_model: str = Field(default="gemini-2.0-flash-exp", description="LLM model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=1000, description="Maximum tokens for LLM response")
    timeout_seconds: int = Field(default=30, description="Timeout for agent processing")
    
    # Agent-specific configuration
    specialty_config: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific configuration")


class SystemConfig(BaseModel):
    """Overall system configuration"""
    
    # API configurations
    gemini_api_key: str = Field(..., description="Gemini API key")
    lastfm_api_key: str = Field(..., description="Last.fm API key")
    spotify_client_id: Optional[str] = Field(default=None, description="Spotify client ID")
    spotify_client_secret: Optional[str] = Field(default=None, description="Spotify client secret")
    
    # Rate limiting
    gemini_rate_limit: int = Field(default=15, description="Gemini requests per minute")
    lastfm_rate_limit: float = Field(default=3.0, description="Last.fm requests per second")
    spotify_rate_limit: int = Field(default=50, description="Spotify requests per hour")
    
    # Caching
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl_hours: int = Field(default=24, description="Cache TTL in hours")
    cache_directory: str = Field(default="data/cache", description="Cache directory path")
    
    # Agent configurations
    agent_configs: Dict[str, AgentConfig] = Field(default_factory=dict, description="Configuration for each agent")
    
    # Performance settings
    max_concurrent_agents: int = Field(default=2, description="Maximum concurrent agent executions")
    total_timeout_minutes: int = Field(default=5, description="Total workflow timeout in minutes")


class QueryIntent(Enum):
    """Primary intent types for music queries."""
    ARTIST_SIMILARITY = "artist_similarity"      # "Music like X"
    GENRE_EXPLORATION = "genre_exploration"      # "I want jazz music"
    MOOD_MATCHING = "mood_matching"              # "Happy music for workout"
    ACTIVITY_CONTEXT = "activity_context"        # "Music for studying"
    DISCOVERY = "discovery"                      # "Something new and different"
    PLAYLIST_BUILDING = "playlist_building"      # "Songs for my road trip"
    SPECIFIC_REQUEST = "specific_request"        # "Play Bohemian Rhapsody"


class SimilarityType(Enum):
    """Types of similarity for artist-based queries."""
    STYLISTIC = "stylistic"        # Similar sound/production
    GENRE = "genre"               # Same genre family
    ERA = "era"                   # Same time period
    MOOD = "mood"                 # Similar emotional feel
    ENERGY = "energy"             # Similar energy level


@dataclass
class QueryUnderstanding:
    """Structured representation of understood query."""
    intent: QueryIntent
    confidence: float
    
    # Core entities
    artists: List[str]
    genres: List[str]
    moods: List[str]
    activities: List[str]
    
    # Intent-specific details
    similarity_type: Optional[SimilarityType] = None
    exploration_level: str = "moderate"  # strict, moderate, broad
    temporal_context: Optional[str] = None
    energy_level: Optional[str] = None
    
    # Agent coordination hints
    primary_agent: str = "genre_mood"  # Which agent should lead
    agent_weights: Dict[str, float] = None
    search_strategy: Dict[str, Any] = None
    
    # Original query context
    original_query: str = ""
    normalized_query: str = ""
    reasoning: str = "" 