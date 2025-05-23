"""
Agent Models for BeatDebate Multi-Agent Music Recommendation System

Pydantic models for state management, agent communication, and data structures.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class MusicRecommenderState(BaseModel):
    """Shared state across all agents in the LangGraph workflow"""
    
    # Input
    user_query: str = Field(..., description="Original user query for music recommendation")
    user_profile: Optional[Dict[str, Any]] = Field(default=None, description="User preferences and history")
    
    # Planning phase
    planning_strategy: Optional[Dict[str, Any]] = Field(default=None, description="Strategy created by PlannerAgent")
    execution_plan: Optional[Dict[str, Any]] = Field(default=None, description="Execution monitoring plan")
    
    # Advocate phase  
    genre_mood_recommendations: List[Dict] = Field(default_factory=list, description="GenreMoodAgent recommendations")
    discovery_recommendations: List[Dict] = Field(default_factory=list, description="DiscoveryAgent recommendations")
    
    # Judge phase
    final_recommendations: List[Dict] = Field(default_factory=list, description="Final selected recommendations")
    
    # Reasoning transparency
    reasoning_log: List[str] = Field(default_factory=list, description="Step-by-step reasoning log")
    agent_deliberations: List[Dict] = Field(default_factory=list, description="Agent decision records")
    
    # Metadata
    processing_start_time: Optional[float] = Field(default=None, description="Processing start timestamp")
    total_processing_time: Optional[float] = Field(default=None, description="Total processing time in seconds")
    session_id: Optional[str] = Field(default=None, description="Unique session identifier")


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