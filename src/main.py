"""
BeatDebate - Main Application Entry Point

Multi-agent music recommendation system using sophisticated planning behavior.
Built for the AgentX competition.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    environment: str
    api_keys_configured: Dict[str, bool]
    dependencies: Dict[str, str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🎵 Starting BeatDebate application")
    
    # Startup validation
    required_env_vars = ["GEMINI_API_KEY", "LASTFM_API_KEY", "SPOTIFY_CLIENT_ID"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(
            "Missing environment variables",
            missing_vars=missing_vars
        )
    else:
        logger.info("✅ All required API keys configured")
    
    yield
    
    logger.info("🎵 Shutting down BeatDebate application")


# Create FastAPI application
app = FastAPI(
    title="BeatDebate",
    description="Multi-Agent Music Recommendation System",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Root endpoint with basic information."""
    return {
        "name": "BeatDebate",
        "description": "Multi-Agent Music Recommendation System",
        "version": "0.1.0",
        "status": "active",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Comprehensive health check endpoint."""
    try:
        # Check environment configuration
        api_keys_configured = {
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "lastfm": bool(os.getenv("LASTFM_API_KEY")),
            "spotify": bool(os.getenv("SPOTIFY_CLIENT_ID") and os.getenv("SPOTIFY_CLIENT_SECRET")),
        }
        
        # Check dependencies (basic import test)
        dependencies = {}
        try:
            import langchain
            dependencies["langchain"] = langchain.__version__
        except ImportError:
            dependencies["langchain"] = "not_available"
            
        try:
            import chromadb
            dependencies["chromadb"] = chromadb.__version__
        except ImportError:
            dependencies["chromadb"] = "not_available"
            
        try:
            import sentence_transformers
            dependencies["sentence_transformers"] = sentence_transformers.__version__
        except ImportError:
            dependencies["sentence_transformers"] = "not_available"
        
        environment = os.getenv("ENVIRONMENT", "development")
        
        logger.info(
            "Health check completed",
            api_keys=api_keys_configured,
            dependencies=dependencies,
            environment=environment
        )
        
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            environment=environment,
            api_keys_configured=api_keys_configured,
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/api/test")
async def test_endpoint() -> Dict[str, Any]:
    """Test endpoint for development."""
    return {
        "message": "BeatDebate API is working!",
        "timestamp": "2025-01-25T10:00:00Z",
        "agents": {
            "planner": "ready",
            "genre_mood": "ready", 
            "discovery": "ready",
            "judge": "ready"
        },
        "data_sources": {
            "lastfm": "configured" if os.getenv("LASTFM_API_KEY") else "missing",
            "spotify": "configured" if os.getenv("SPOTIFY_CLIENT_ID") else "missing"
        }
    }


if __name__ == "__main__":
    # Get configuration from environment
    port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    debug = os.getenv("DEBUG_MODE", "true").lower() == "true"
    
    logger.info(
        "Starting BeatDebate server",
        port=port,
        debug=debug,
        environment=os.getenv("ENVIRONMENT", "development")
    )
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=debug,
        log_level="info"
    ) 