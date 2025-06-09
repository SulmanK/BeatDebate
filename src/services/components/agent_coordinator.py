"""
Agent Coordinator for Enhanced Recommendation Service

Manages agent initialization, configuration, and coordination.
Extracted from EnhancedRecommendationService to improve modularity and maintainability.
"""

import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...agents.planner.agent import PlannerAgent
    from ...agents.genre_mood.agent import GenreMoodAgent
    from ...agents.discovery.agent import DiscoveryAgent
    from ...agents.judge.agent import JudgeAgent
import structlog

# Handle imports gracefully
try:
    from ...models.agent_models import AgentConfig
    from ..api_service import APIService
    from ..metadata_service import MetadataService
    from ..session_manager_service import SessionManagerService
    from ...api.rate_limiter import UnifiedRateLimiter
except ImportError:
    # Fallback imports for testing
    import sys
    sys.path.append('src')
    from models.agent_models import AgentConfig
    from services.api_service import APIService
    from services.metadata_service import MetadataService
    from services.session_manager_service import SessionManagerService
    from api.rate_limiter import UnifiedRateLimiter

logger = structlog.get_logger(__name__)


def create_gemini_client(api_key: str):
    """
    Create a Gemini client for LLM interactions.
    
    Args:
        api_key: Gemini API key
        
    Returns:
        Configured Gemini client or None if creation fails
    """
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Create generative model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("Gemini client created successfully")
        return model
        
    except ImportError:
        logger.warning("google-generativeai not available, LLM features will be disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to create Gemini client: {e}")
        return None


class AgentCoordinator:
    """
    Coordinates agent initialization and configuration for the Enhanced Recommendation Service.
    
    Responsibilities:
    - Agent initialization with shared services
    - LLM client and rate limiter configuration
    - Agent configuration management
    - Agent lifecycle management
    """
    
    def __init__(
        self,
        api_service: APIService,
        session_manager: SessionManagerService
    ):
        self.api_service = api_service
        self.session_manager = session_manager
        self.logger = structlog.get_logger(__name__)
        
        # Agent instances
        self.planner_agent: Optional["PlannerAgent"] = None
        self.genre_mood_agent: Optional["GenreMoodAgent"] = None
        self.discovery_agent: Optional["DiscoveryAgent"] = None
        self.judge_agent: Optional["JudgeAgent"] = None
        
        # Shared resources
        self.gemini_client = None
        self.gemini_rate_limiter = None
        self.metadata_service = None
        
        # Initialization status
        self._agents_initialized = False
    
    async def initialize_agents(self) -> bool:
        """
        Initialize all agents with shared API service and rate limiting.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self._agents_initialized:
            return True
        
        try:
            # Initialize shared resources
            await self._initialize_shared_resources()
            
            # Create agent configuration
            agent_config = self._create_agent_config()
            
            # Initialize individual agents
            await self._initialize_individual_agents(agent_config)
            
            self._agents_initialized = True
            self.logger.info("All agents initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize agents", error=str(e))
            return False
    
    async def _initialize_shared_resources(self):
        """Initialize shared resources for all agents."""
        
        # Create Gemini client for LLM interactions
        gemini_api_key = os.getenv('GEMINI_API_KEY', 'demo_gemini_key')
        self.gemini_client = create_gemini_client(gemini_api_key)
        
        if not self.gemini_client:
            self.logger.warning("Failed to create Gemini client, agents will have limited functionality")
        
        # Create rate limiter for Gemini API (free tier: 10 requests per minute)
        self.gemini_rate_limiter = UnifiedRateLimiter.for_gemini(calls_per_minute=8)  # Conservative limit
        self.logger.info("Gemini rate limiter created", calls_per_minute=8)
        
        # Get shared clients from API service
        lastfm_client = await self.api_service.get_lastfm_client()
        
        # Create metadata service with shared client
        self.metadata_service = MetadataService(lastfm_client=lastfm_client)
        
        self.logger.info("Shared resources initialized")
    
    def _create_agent_config(self) -> AgentConfig:
        """Create standard agent configuration."""
        return AgentConfig(
            agent_name="default",
            agent_type="enhanced",
            llm_model="gemini-2.0-flash-exp",
            temperature=0.7,
            max_tokens=1000
        )
    
    async def _initialize_individual_agents(self, agent_config: AgentConfig):
        """Initialize individual agent instances."""
        
        # Import agents dynamically to avoid circular imports
        try:
            from ...agents.planner.agent import PlannerAgent
            from ...agents.genre_mood.agent import GenreMoodAgent
            from ...agents.discovery.agent import DiscoveryAgent
            from ...agents.judge.agent import JudgeAgent
        except ImportError:
            from agents.planner.agent import PlannerAgent
            from agents.genre_mood.agent import GenreMoodAgent
            from agents.discovery.agent import DiscoveryAgent
            from agents.judge.agent import JudgeAgent
        
        # Initialize PlannerAgent
        self.planner_agent = PlannerAgent(
            config=agent_config,
            llm_client=self.gemini_client,
            api_service=self.api_service,
            metadata_service=self.metadata_service,
            rate_limiter=self.gemini_rate_limiter
        )
        self.logger.debug("PlannerAgent initialized")
        
        # Initialize GenreMoodAgent
        self.genre_mood_agent = GenreMoodAgent(
            config=agent_config,
            llm_client=self.gemini_client,
            api_service=self.api_service,
            metadata_service=self.metadata_service,
            rate_limiter=self.gemini_rate_limiter,
            session_manager=self.session_manager  # Phase 3: For candidate pool persistence
        )
        self.logger.debug("GenreMoodAgent initialized")
        
        # Initialize DiscoveryAgent
        self.discovery_agent = DiscoveryAgent(
            config=agent_config,
            llm_client=self.gemini_client,
            api_service=self.api_service,
            metadata_service=self.metadata_service,
            rate_limiter=self.gemini_rate_limiter,
            session_manager=self.session_manager  # Phase 3: For candidate pool persistence
        )
        self.logger.debug("DiscoveryAgent initialized")
        
        # Initialize JudgeAgent
        self.judge_agent = JudgeAgent(
            config=agent_config,
            llm_client=self.gemini_client,
            api_service=self.api_service,
            metadata_service=self.metadata_service,
            rate_limiter=self.gemini_rate_limiter,
            session_manager=self.session_manager  # Phase 3: For candidate pool retrieval
        )
        self.logger.debug("JudgeAgent initialized")
    
    def get_planner_agent(self) -> Optional["PlannerAgent"]:
        """Get the planner agent instance."""
        return self.planner_agent
    
    def get_genre_mood_agent(self) -> Optional["GenreMoodAgent"]:
        """Get the genre mood agent instance."""
        return self.genre_mood_agent
    
    def get_discovery_agent(self) -> Optional["DiscoveryAgent"]:
        """Get the discovery agent instance."""
        return self.discovery_agent
    
    def get_judge_agent(self) -> Optional["JudgeAgent"]:
        """Get the judge agent instance."""
        return self.judge_agent
    
    def get_gemini_client(self):
        """Get the shared Gemini client."""
        return self.gemini_client
    
    def get_rate_limiter(self):
        """Get the shared rate limiter."""
        return self.gemini_rate_limiter
    
    def are_agents_initialized(self) -> bool:
        """Check if agents are initialized."""
        return self._agents_initialized
    
    async def close(self):
        """Clean up agent resources."""
        # Close individual agents if they have cleanup methods
        for agent in [self.planner_agent, self.genre_mood_agent, self.discovery_agent, self.judge_agent]:
            if agent and hasattr(agent, 'close'):
                try:
                    await agent.close()
                except Exception as e:
                    self.logger.warning(f"Error closing agent: {e}")
        
        self.logger.info("Agent coordinator closed") 