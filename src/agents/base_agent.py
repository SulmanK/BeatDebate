"""
Base Agent Class for BeatDebate Multi-Agent Music Recommendation System

Provides common functionality for all agents including LLM integration,
logging, error handling, and reasoning chain management.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import structlog
from datetime import datetime

from ..models.agent_models import (
    MusicRecommenderState,
    AgentDeliberation,
    ReasoningChain,
    AgentConfig
)

logger = structlog.get_logger(__name__)


class BaseAgent(ABC):
    """
    Base class for all agents in the BeatDebate system.
    
    Provides common functionality:
    - LLM integration with Gemini
    - Reasoning chain management
    - Error handling and logging
    - Strategy processing utilities
    - Performance monitoring
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize base agent with configuration.
        
        Args:
            config: Agent configuration including LLM settings
        """
        self.config = config
        self.agent_name = config.agent_name
        self.agent_type = config.agent_type
        self.logger = logger.bind(agent=self.agent_name)
        
        # Initialize LLM client (will be set up in subclasses)
        self.llm_client = None
        
        # Performance tracking
        self.processing_times: List[float] = []
        self.success_count = 0
        self.error_count = 0
        
        self.logger.info(
            "Agent initialized",
            agent_type=self.agent_type,
            llm_model=config.llm_model,
            temperature=config.temperature
        )
    
    @abstractmethod
    async def process(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Main processing method that each agent must implement.
        
        Args:
            state: Current state of the music recommendation workflow
            
        Returns:
            Updated state after agent processing
        """
        pass
    
    async def execute_with_monitoring(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Execute agent processing with performance monitoring and error handling.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after processing
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting agent processing", user_query=state.user_query)
            
            # Execute main processing
            updated_state = await self._execute_with_timeout(state)
            
            # Record successful execution
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.success_count += 1
            
            # Add deliberation record
            deliberation = AgentDeliberation(
                agent_name=self.agent_name,
                timestamp=datetime.now(),
                input_data={"user_query": state.user_query},
                reasoning_steps=self._extract_reasoning_steps(updated_state),
                output_data=self._extract_output_data(updated_state),
                confidence=self._calculate_confidence(updated_state),
                processing_time=processing_time
            )
            
            updated_state.agent_deliberations.append(deliberation.dict())
            
            self.logger.info(
                "Agent processing completed successfully",
                processing_time=processing_time,
                confidence=deliberation.confidence
            )
            
            return updated_state
            
        except asyncio.TimeoutError:
            self.error_count += 1
            self.logger.error(
                "Agent processing timed out",
                timeout_seconds=self.config.timeout_seconds
            )
            # Return state with error information
            return self._handle_timeout_error(state)
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(
                "Agent processing failed",
                error=str(e),
                error_type=type(e).__name__
            )
            # Return state with error information
            return self._handle_processing_error(state, e)
    
    async def _execute_with_timeout(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """Execute processing with timeout."""
        return await asyncio.wait_for(
            self.process(state),
            timeout=self.config.timeout_seconds
        )
    
    def _extract_reasoning_steps(self, state: MusicRecommenderState) -> List[str]:
        """Extract reasoning steps from the updated state."""
        # Get the most recent reasoning log entries added by this agent
        if hasattr(self, '_reasoning_steps'):
            return self._reasoning_steps
        return ["Processing completed"]
    
    def _extract_output_data(self, state: MusicRecommenderState) -> Dict[str, Any]:
        """Extract output data specific to this agent."""
        return {"status": "completed"}
    
    def _calculate_confidence(self, state: MusicRecommenderState) -> float:
        """Calculate confidence score for this agent's processing."""
        # Default implementation - subclasses should override
        return 0.8
    
    def _handle_timeout_error(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """Handle timeout error by adding error information to state."""
        error_msg = f"{self.agent_name} processing timed out after {self.config.timeout_seconds}s"
        state.reasoning_log.append(f"ERROR: {error_msg}")
        return state
    
    def _handle_processing_error(self, state: MusicRecommenderState, error: Exception) -> MusicRecommenderState:
        """Handle processing error by adding error information to state."""
        error_msg = f"{self.agent_name} processing failed: {str(error)}"
        state.reasoning_log.append(f"ERROR: {error_msg}")
        return state
    
    def add_reasoning_step(self, step: str, evidence: List[str] = None, confidence: float = 0.8):
        """
        Add a reasoning step for transparency.
        
        Args:
            step: Description of the reasoning step
            evidence: Supporting evidence for this step
            confidence: Confidence in this reasoning step
        """
        if not hasattr(self, '_reasoning_steps'):
            self._reasoning_steps = []
        
        self._reasoning_steps.append(step)
        
        if evidence:
            self._reasoning_steps.append(f"Evidence: {', '.join(evidence)}")
        
        self.logger.debug(
            "Reasoning step added",
            step=step,
            confidence=confidence
        )
    
    def log_strategy_application(self, strategy: Dict[str, Any], step: str):
        """
        Log how strategy is being applied.
        
        Args:
            strategy: Strategy object being applied
            step: Description of strategy application step
        """
        self.logger.info(
            "Applying strategy",
            step=step,
            strategy_keys=list(strategy.keys()) if strategy else []
        )
    
    async def call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """
        Call LLM with proper error handling and logging.
        
        Args:
            prompt: User prompt for the LLM
            system_prompt: System prompt (optional)
            
        Returns:
            LLM response text
        """
        if not self.llm_client:
            raise RuntimeError(f"LLM client not initialized for {self.agent_name}")
        
        try:
            self.logger.debug(
                "Calling LLM",
                prompt_length=len(prompt),
                model=self.config.llm_model
            )
            
            # This will be implemented by subclasses with actual LLM integration
            response = await self._make_llm_call(prompt, system_prompt)
            
            self.logger.debug(
                "LLM response received",
                response_length=len(response)
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "LLM call failed",
                error=str(e),
                prompt_length=len(prompt)
            )
            raise
    
    async def _make_llm_call(self, prompt: str, system_prompt: str = None) -> str:
        """
        Make actual LLM call - to be implemented by subclasses.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            LLM response
        """
        raise NotImplementedError("Subclasses must implement _make_llm_call")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this agent.
        
        Returns:
            Dictionary of performance metrics
        """
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )
        
        total_requests = self.success_count + self.error_count
        success_rate = self.success_count / total_requests if total_requests > 0 else 0
        
        return {
            "agent_name": self.agent_name,
            "total_requests": total_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "avg_processing_time": avg_processing_time,
            "processing_times": self.processing_times[-10:]  # Last 10 times
        }
    
    def validate_strategy(self, strategy: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that strategy contains required keys.
        
        Args:
            strategy: Strategy object to validate
            required_keys: List of required keys
            
        Returns:
            True if strategy is valid, False otherwise
        """
        if not strategy:
            self.logger.warning("Strategy is None or empty")
            return False
        
        missing_keys = [key for key in required_keys if key not in strategy]
        if missing_keys:
            self.logger.warning(
                "Strategy missing required keys",
                missing_keys=missing_keys,
                available_keys=list(strategy.keys())
            )
            return False
        
        return True
    
    def extract_strategy_for_agent(self, full_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract strategy specific to this agent from full strategy.
        
        Args:
            full_strategy: Complete strategy from PlannerAgent
            
        Returns:
            Strategy specific to this agent
        """
        if not full_strategy:
            return {}
        
        coordination_strategy = full_strategy.get("coordination_strategy", {})
        
        # Map agent names to strategy keys
        agent_strategy_map = {
            "GenreMoodAgent": "genre_mood_agent",
            "DiscoveryAgent": "discovery_agent",
            "JudgeAgent": "evaluation_framework"
        }
        
        strategy_key = agent_strategy_map.get(self.agent_name)
        if strategy_key and strategy_key in coordination_strategy:
            return coordination_strategy[strategy_key]
        
        # Return evaluation framework for JudgeAgent
        if self.agent_name == "JudgeAgent":
            return full_strategy.get("evaluation_framework", {})
        
        return {}
    
    def format_reasoning_chain(self, steps: List[str]) -> str:
        """
        Format reasoning steps into a coherent chain.
        
        Args:
            steps: List of reasoning steps
            
        Returns:
            Formatted reasoning chain
        """
        if not steps:
            return "No reasoning steps recorded."
        
        formatted_steps = []
        for i, step in enumerate(steps, 1):
            formatted_steps.append(f"{i}. {step}")
        
        return "\n".join(formatted_steps) 