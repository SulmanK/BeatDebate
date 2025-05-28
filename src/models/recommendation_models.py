from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TrackRecommendation(BaseModel):
    """
    Represents a music track recommendation, including its metadata,
    descriptive attributes, scorable features, and judge-added evaluations.
    """
    # --- Essential Metadata ---
    title: str = Field(..., description="The title of the track.")
    artist: str = Field(..., description="The primary artist of the track.")
    id: str = Field(
        ..., 
        description=(
            "A unique identifier for the track "
            "(e.g., from Last.fm, Spotify, or internal)."
        )
    )
    source: str = Field(
        ..., 
        description="The origin of the track data (e.g., 'lastfm', 'spotify')."
    )

    # --- Optional Rich Metadata ---
    track_url: Optional[str] = Field(
        None, 
        description="A URL to the track's page on the source platform."
    )
    preview_url: Optional[str] = Field(
        None, 
        description="A URL to an audio preview of the track."
    )
    album_title: Optional[str] = Field(
        None, 
        description="The title of the album the track belongs to."
    )
    album_art_url: Optional[str] = Field(
        None, 
        description="A URL to the album artwork."
    )

    # --- Descriptive Attributes (for diversity, filtering, and scoring) ---
    genres: List[str] = Field(
        default_factory=list, 
        description=(
            "A list of genres associated with the track. "
            "Can be used for diversity."
        )
    )
    era: Optional[str] = Field(
        None, 
        description=(
            "The era of the track (e.g., '1990s', '2020s'). "
            "Can be used for diversity."
        )
    )
    moods: List[str] = Field(
        default_factory=list, 
        description=(
            "A list of moods associated with the track "
            "(e.g., ['chill', 'upbeat'])."
        )
    )
    energy: Optional[str] = Field(
        None, 
        description=(
            "Categorical energy level (e.g., 'low', 'medium', 'high'). "
            "Aligns with diversity targets if 'energy' is a key."
        )
    )
    # Alternative for energy, if a normalized numerical value is preferred:
    # energy_value: Optional[float] = Field(
    #     None, ge=0, le=1, description="Numerical energy level (0-1)."
    # )
    instrumental: Optional[bool] = Field(
        None, 
        description="Indicates if the track is instrumental."
    )

    # --- Scorable Attributes (normalized 0-1 by advocates) ---
    # These keys should align with criteria in PlannerAgent's 
    # evaluation_framework.primary_weights
    concentration_friendliness_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=1, 
        description="Score indicating suitability for concentration (0-1)."
    )
    novelty_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=1, 
        description=(
            "Score indicating how novel or undiscovered "
            "the track might be (0-1)."
        )
    )
    quality_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=1, 
        description="An overall quality score for the track (0-1)."
    )
    
    # For flexibility with other scores defined by PlannerAgent or Advocates:
    additional_scores: Dict[str, Any] = Field(
        default_factory=dict, 
        description=(
            "A dictionary for any other named scores and metadata "
            "relevant to evaluation. Can contain numeric scores (0-1) "
            "and string metadata like source types, quality tiers, etc."
        )
    )

    # --- Advocate Agent Information (Optional) ---
    advocate_source_agent: Optional[str] = Field(
        None, 
        description=(
            "Name of the advocate agent that proposed this track "
            "(e.g., 'GenreMoodAgent')."
        )
    )
    # advocate_confidence: Optional[float] = Field(
    #     None, ge=0, le=1, 
    #     description=(
    #         "Advocate's confidence in this specific recommendation, "
    #         "if provided directly."
    #     )
    # )

    # --- Fields to be populated by the JudgeAgent ---
    judge_score: Optional[float] = Field(
        None, 
        description="The final weighted score assigned by the JudgeAgent."
    )
    explanation: Optional[str] = Field(
        None, 
        description=(
            "The JudgeAgent's explanation for why this track was selected."
        )
    )
    confidence: Optional[float] = Field(
        None, 
        ge=0, 
        le=1, 
        description="Overall confidence score for this recommendation."
    )

    # --- Raw data (Optional, for debugging or deeper analysis) ---
    raw_source_data: Optional[Dict[str, Any]] = Field(
        None, 
        description="Original raw data from the source API, if needed."
    )

    class Config:
        str_strip_whitespace = True
        validate_assignment = True
        # Consider adding example data here for documentation 
        # if using FastAPI's automatic docs
        # schema_extra = {
        #     "example": {
        #         "title": "Example Track",
        #         "artist": "Example Artist",
        #         # ... other fields
        #     }
        # }


class RecommendationResponse(BaseModel):
    """
    Complete response model for music recommendations from the 4-agent system.
    """
    recommendations: List[TrackRecommendation] = Field(
        ..., 
        description="List of recommended tracks"
    )
    reasoning_log: List[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning from all agents"
    )
    agent_coordination_log: List[str] = Field(
        default_factory=list,
        description="Log of agent coordination and communication"
    )
    session_id: str = Field(
        ...,
        description="Session identifier for this recommendation request"
    )
    response_time: float = Field(
        ...,
        description="Total time taken to generate recommendations (seconds)"
    )
    planning_strategy: Optional[Dict[str, Any]] = Field(
        None,
        description="The planning strategy used by PlannerAgent"
    )
    
    class Config:
        str_strip_whitespace = True
        validate_assignment = True

# Example of how it might be used by an advocate agent:
# track_data = {
#     "title": "Solitude",
#     "artist": "Photek",
#     "id": "lastfm_track_123",
#     "source": "lastfm",
#     "genres": ["Drum and Bass", "Ambient"],
#     "era": "1990s",
#     "moods": ["atmospheric", "dark", "introspective"],
#     "energy": "medium",
#     "instrumental": True,
#     "novelty_score": 0.8,
#     "quality_score": 0.9,
#     "advocate_source_agent": "DiscoveryAgent"
# }
# validated_track = TrackRecommendation(**track_data) 