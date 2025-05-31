"""
FastAPI Backend for BeatDebate Music Recommendation System

This module provides REST API endpoints for the 4-agent music recommendation
system, exposing the Enhanced Recommendation Service functionality via HTTP endpoints.
"""

import time
from typing import Dict, Optional
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Updated imports to use new enhanced services
from ..services.enhanced_recommendation_service import (
    EnhancedRecommendationService,
    RecommendationRequest as ServiceRequest,
    get_recommendation_service
)
from ..services.api_service import get_api_service, close_api_service
from ..services.cache_manager import get_cache_manager
from ..services.smart_context_manager import SmartContextManager
from ..models.agent_models import SystemConfig, AgentConfig
from ..models.metadata_models import UnifiedTrackMetadata
from .logging_middleware import LoggingMiddleware, PerformanceLoggingMiddleware

# Setup logger - will be initialized after logging setup
logger = None

# Global service instances
recommendation_service: Optional[EnhancedRecommendationService] = None
api_service = None
cache_manager = None
context_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global recommendation_service, api_service, cache_manager, context_manager, logger
    
    # Initialize logging
    from ..utils.logging_config import setup_logging, get_logger
    setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
    logger = get_logger(__name__)
    
    # Startup
    logger.info("Initializing BeatDebate enhanced recommendation service...")
    try:
        # Get API keys from environment variables with fallbacks
        gemini_api_key = os.getenv("GEMINI_API_KEY", "demo_gemini_key")
        lastfm_api_key = os.getenv("LASTFM_API_KEY", "demo_lastfm_key")
        spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID", "demo_spotify_id")
        spotify_client_secret = os.getenv(
            "SPOTIFY_CLIENT_SECRET", 
            "demo_spotify_secret"
        )
        
        # Create system configuration
        system_config = SystemConfig(
            gemini_api_key=gemini_api_key,
            lastfm_api_key=lastfm_api_key,
            spotify_client_id=spotify_client_id,
            spotify_client_secret=spotify_client_secret,
            agent_configs={
                "planner": AgentConfig(
                    agent_name="PlannerAgent", 
                    agent_type="planner"
                ),
                "genre_mood": AgentConfig(
                    agent_name="GenreMoodAgent", 
                    agent_type="advocate"
                ),
                "discovery": AgentConfig(
                    agent_name="DiscoveryAgent", 
                    agent_type="advocate"
                ),
                "judge": AgentConfig(
                    agent_name="JudgeAgent", 
                    agent_type="judge"
                )
            }
        )
        
        # Initialize shared services
        cache_manager = get_cache_manager()
        api_service = get_api_service(cache_manager=cache_manager)
        context_manager = SmartContextManager()
        
        # Initialize enhanced recommendation service
        recommendation_service = get_recommendation_service(
            system_config=system_config,
            api_service=api_service,
            cache_manager=cache_manager
        )
        
        # Initialize agents within the service
        await recommendation_service.initialize_agents()
        
        logger.info(
            "BeatDebate enhanced recommendation service initialized successfully"
        )
    except Exception as e:
        logger.error(f"Failed to initialize recommendation service: {e}")
        # For demo purposes, continue without the service
        logger.warning("Continuing without recommendation service for demo")
        recommendation_service = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down BeatDebate recommendation service...")
    if recommendation_service:
        await recommendation_service.close()
    if api_service:
        await close_api_service()
    recommendation_service = None
    api_service = None
    cache_manager = None
    context_manager = None


# Create FastAPI app
app = FastAPI(
    title="BeatDebate API",
    description="4-Agent Music Recommendation System with Strategic Planning",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.add_middleware(LoggingMiddleware, exclude_paths=["/health", "/docs", "/openapi.json"])
app.add_middleware(PerformanceLoggingMiddleware, slow_request_threshold=5.0)


# Request/Response Models
class RecommendationRequest(BaseModel):
    """Request model for music recommendations."""
    query: str = Field(..., description="User's music preference query")
    session_id: Optional[str] = Field(
        None, 
        description="Session identifier for conversation history"
    )
    max_recommendations: int = Field(
        3, 
        ge=1, 
        le=10, 
        description="Maximum number of recommendations"
    )
    include_previews: bool = Field(
        True, 
        description="Whether to include audio previews"
    )
    chat_context: Optional[Dict] = Field(
        None,
        description="Previous chat context for continuity"
    )


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str
    components: Dict[str, str]


class PlanningResponse(BaseModel):
    """Response model for planning strategy."""
    strategy: Dict  # Using Dict instead of PlanningStrategy for now
    execution_time: float
    session_id: str


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        components={
            "recommendation_service": (
                "active" if recommendation_service else "inactive"
            ),
            "lastfm_client": "configured",
            "spotify_client": "configured"
        }
    )


def transform_unified_to_ui_format(unified_track: UnifiedTrackMetadata) -> Dict:
    """Transform UnifiedTrackMetadata to UI-expected format."""
    return {
        "title": unified_track.name,
        "artist": unified_track.artist,
        "album": unified_track.album,
        "confidence": unified_track.recommendation_score or 0.0,
        "explanation": unified_track.recommendation_reason or "",
        "source": unified_track.agent_source or "unknown",
        "genres": unified_track.genres,
        "moods": unified_track.tags,  # tags are used as moods
        "preview_url": unified_track.preview_url,
        "spotify_url": unified_track.external_urls.get("spotify") if unified_track.external_urls else None,
        "quality_score": unified_track.quality_score,
        "novelty_score": unified_track.underground_score,  # underground_score is used as novelty
        # Additional metadata
        "popularity": unified_track.popularity,
        "listeners": unified_track.listeners,
        "playcount": unified_track.playcount
    }


@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """
    Get music recommendations using the 4-agent system.
    
    This endpoint orchestrates the complete recommendation workflow:
    1. PlannerAgent creates strategic plan
    2. GenreMoodAgent and DiscoveryAgent execute searches
    3. JudgeAgent selects final recommendations
    """
    if not recommendation_service:
        # Return demo response when service is not available
        demo_tracks = [
            UnifiedTrackMetadata(
                name="Demo Track 1",
                artist="Demo Artist",
                recommendation_score=0.85,
                recommendation_reason="This is a demo recommendation",
                agent_source="demo"
            ),
            UnifiedTrackMetadata(
                name="Demo Track 2", 
                artist="Demo Artist 2",
                recommendation_score=0.78,
                recommendation_reason="This is another demo recommendation",
                agent_source="demo"
            )
        ]
        
        # Transform to UI format and return as dict
        return {
            "recommendations": [transform_unified_to_ui_format(track) for track in demo_tracks],
            "strategy_used": {"type": "demo", "reason": "Service not available"},
            "reasoning": [
                "Demo: PlannerAgent analyzed request",
                "Demo: GenreMoodAgent found tracks",
                "Demo: JudgeAgent selected recommendations"
            ],
            "session_id": request.session_id or "demo_session",
            "processing_time": 1.5,
            "metadata": {
                "demo_mode": True,
                "agents_used": ["demo"],
                "total_candidates": 2,
                "final_count": 2
            }
        }
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing recommendation request: {request.query}")
        
        # Execute recommendation workflow
        service_request = ServiceRequest(
            query=request.query,
            session_id=request.session_id,
            max_recommendations=request.max_recommendations,
            include_audio_features=request.include_previews,
            context=request.chat_context
        )
        
        result = await recommendation_service.get_recommendations(service_request)
        
        execution_time = time.time() - start_time
        logger.info(f"Recommendation completed in {execution_time:.2f}s")
        
        # Transform recommendations to UI format
        transformed_recommendations = []
        for track in result.recommendations:
            transformed_recommendations.append(transform_unified_to_ui_format(track))
        
        # Return as dictionary for JSON serialization
        return {
            "recommendations": transformed_recommendations,
            "strategy_used": result.strategy_used,
            "reasoning": result.reasoning,
            "session_id": request.session_id or result.session_id,
            "processing_time": execution_time,
            "metadata": result.metadata
        }
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Recommendation failed: {str(e)}"
        )


@app.post("/planning")
async def get_planning_strategy(request: RecommendationRequest):
    """
    Get the planning strategy without executing full recommendations.
    
    This endpoint is useful for demonstrating the PlannerAgent's strategic
    thinking process in the UI.
    """
    if not recommendation_service:
        # Return demo planning response
        demo_strategy = {
            "task_analysis": {
                "primary_goal": "Demo music discovery",
                "complexity_level": "medium",
                "context_factors": ["demo", "testing"]
            },
            "coordination_strategy": {
                "genre_mood_agent": {"focus": "Demo genre search"},
                "discovery_agent": {"focus": "Demo discovery search"}
            },
            "evaluation_framework": {
                "primary_weights": {"quality": 0.5, "novelty": 0.5}
            }
        }
        
        return PlanningResponse(
            strategy=demo_strategy,
            execution_time=0.5,
            session_id=request.session_id or "demo_session"
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"Generating planning strategy for: {request.query}")
        
        # Get planning strategy from PlannerAgent
        strategy = await recommendation_service.get_planning_strategy(
            query=request.query,
            session_id=request.session_id
        )
        
        execution_time = time.time() - start_time
        
        return PlanningResponse(
            strategy=(
                strategy.dict() if hasattr(strategy, 'dict') else strategy
            ),
            execution_time=execution_time,
            session_id=request.session_id or "default"
        )
        
    except Exception as e:
        logger.error(f"Planning strategy generation failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Planning failed: {str(e)}"
        )


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    if not recommendation_service:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation service not available"
        )
    
    try:
        # This would integrate with session management
        # For now, return placeholder
        return {
            "session_id": session_id,
            "history": [],
            "message": "Session history feature coming soon"
        }
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get session history: {str(e)}"
        )


@app.post("/feedback")
async def submit_feedback(
    session_id: str,
    recommendation_id: str,
    feedback: str,  # "thumbs_up" or "thumbs_down"
    background_tasks: BackgroundTasks
):
    """Submit user feedback for recommendations."""
    if feedback not in ["thumbs_up", "thumbs_down"]:
        raise HTTPException(status_code=400, detail="Invalid feedback value")
    
    # Add background task to process feedback
    background_tasks.add_task(
        process_feedback,
        session_id=session_id,
        recommendation_id=recommendation_id,
        feedback=feedback
    )
    
    return {"message": "Feedback submitted successfully"}


async def process_feedback(
    session_id: str, 
    recommendation_id: str, 
    feedback: str
):
    """Background task to process user feedback."""
    logger.info(
        f"Processing feedback: {feedback} for recommendation "
        f"{recommendation_id} in session {session_id}"
    )
    # This would integrate with learning/improvement systems
    # For now, just log the feedback


@app.get("/sessions/{session_id}/context")
async def get_session_context(session_id: str):
    """Get smart context status for a session."""
    if not recommendation_service:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation service not available"
        )
    
    try:
        context_summary = await recommendation_service.smart_context_manager.get_context_summary(session_id)
        return {
            "session_id": session_id,
            "context_summary": context_summary,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get context summary: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get context summary: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 