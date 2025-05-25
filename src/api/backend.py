"""
FastAPI Backend for BeatDebate Music Recommendation System

This module provides REST API endpoints for the 4-agent music recommendation
system, exposing the RecommendationEngine functionality via HTTP endpoints.
"""

import logging
import time
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.recommendation_engine import (
    RecommendationEngine, 
    create_recommendation_engine
)
from ..models.recommendation_models import RecommendationResponse
from ..models.agent_models import SystemConfig, AgentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
recommendation_engine: Optional[RecommendationEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global recommendation_engine
    
    # Startup
    logger.info("Initializing BeatDebate recommendation engine...")
    try:
        # Create system configuration
        system_config = SystemConfig(
            lastfm_api_key="demo_key",  # Will be replaced with actual API key
            spotify_client_id="demo_client_id",
            spotify_client_secret="demo_client_secret",
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
        
        # Initialize recommendation engine using factory
        recommendation_engine = await create_recommendation_engine(system_config)
        
        logger.info("BeatDebate recommendation engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recommendation engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down BeatDebate recommendation engine...")
    recommendation_engine = None


# Create FastAPI app
app = FastAPI(
    title="BeatDebate API",
    description="4-Agent Music Recommendation System with Strategic Planning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            "recommendation_engine": (
                "active" if recommendation_engine else "inactive"
            ),
            "lastfm_client": "configured",
            "spotify_client": "configured"
        }
    )


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get music recommendations using the 4-agent system.
    
    This endpoint orchestrates the complete recommendation workflow:
    1. PlannerAgent creates strategic plan
    2. GenreMoodAgent and DiscoveryAgent execute searches
    3. JudgeAgent selects final recommendations
    """
    if not recommendation_engine:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation engine not available"
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing recommendation request: {request.query}")
        
        # Execute recommendation workflow
        result = await recommendation_engine.get_recommendations(
            query=request.query,
            session_id=request.session_id,
            max_recommendations=request.max_recommendations
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Recommendation completed in {execution_time:.2f}s")
        
        # Add execution metadata
        result.response_time = execution_time
        result.session_id = request.session_id or result.session_id
        
        return result
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Recommendation failed: {str(e)}"
        )


@app.post("/planning", response_model=PlanningResponse)
async def get_planning_strategy(request: RecommendationRequest):
    """
    Get the planning strategy without executing full recommendations.
    
    This endpoint is useful for demonstrating the PlannerAgent's strategic
    thinking process in the UI.
    """
    if not recommendation_engine:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation engine not available"
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"Generating planning strategy for: {request.query}")
        
        # Get planning strategy from PlannerAgent
        strategy = await recommendation_engine.get_planning_strategy(
            query=request.query,
            session_id=request.session_id
        )
        
        execution_time = time.time() - start_time
        
        return PlanningResponse(
            strategy=strategy.dict() if hasattr(strategy, 'dict') else strategy,
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
    if not recommendation_engine:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation engine not available"
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


async def process_feedback(session_id: str, recommendation_id: str, feedback: str):
    """Background task to process user feedback."""
    logger.info(
        f"Processing feedback: {feedback} for recommendation "
        f"{recommendation_id} in session {session_id}"
    )
    # This would integrate with learning/improvement systems
    # For now, just log the feedback


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