"""
Refactored Planner Agent - Phase 4.3

Modularized planner agent using extracted components for better architecture.
Reduced from 49KB (1,276 lines) to ~25KB (~500 lines) through component extraction.

Components:
- QueryAnalyzer: Query understanding and parsing
- ContextAnalyzer: Context interpretation and transformation  
- StrategyPlanner: Agent strategy and parameter planning
- EntityProcessor: Entity extraction and processing
"""

from typing import Dict, Any
import structlog

from ...models.agent_models import (
    MusicRecommenderState, 
    QueryUnderstanding, 
    QueryIntent
)
from ...services.api_service import APIService
from ...services.metadata_service import MetadataService
from ..base_agent import BaseAgent
from ..components.llm_utils import LLMUtils
from ..components.query_analysis_utils import QueryAnalysisUtils

# Imported components
from .query_analyzer import QueryAnalyzer
from .context_analyzer import ContextAnalyzer
from .strategy_planner import StrategyPlanner
from .entity_processor import EntityProcessor

logger = structlog.get_logger(__name__)


class PlannerAgent(BaseAgent):
    """
    Refactored Planner Agent with modular component architecture.
    
    Phase 4.3: Dramatically reduced size through component extraction.
    
    Responsibilities:
    - Orchestrating query understanding workflow
    - Coordinating context analysis and entity processing
    - Creating comprehensive planning strategies
    - Managing agent coordination plans
    
    Uses specialized components:
    - QueryAnalyzer for query understanding and complexity analysis
    - ContextAnalyzer for context interpretation and effective intent handling
    - StrategyPlanner for strategy creation and agent coordination
    - EntityProcessor for standardized entity manipulation
    """
    
    def __init__(
        self,
        config,
        llm_client,
        api_service: APIService,
        metadata_service: MetadataService,
        rate_limiter=None
    ):
        """
        Initialize refactored planner agent with component architecture.
        
        Args:
            config: Agent configuration
            llm_client: LLM client for query understanding
            api_service: Unified API service
            metadata_service: Unified metadata service
            rate_limiter: Rate limiter for LLM API calls
        """
        super().__init__(
            config=config, 
            llm_client=llm_client, 
            api_service=api_service,
            metadata_service=metadata_service,
            rate_limiter=rate_limiter
        )
        
        # Initialize specialized components
        self.query_analyzer = QueryAnalyzer(llm_client, rate_limiter=rate_limiter)
        self.context_analyzer = ContextAnalyzer()
        self.strategy_planner = StrategyPlanner()
        self.entity_processor = EntityProcessor()
        
        # Shared utilities (for backward compatibility)
        self.query_utils = QueryAnalysisUtils()
        
        self.logger.info(
            "Refactored PlannerAgent initialized with component architecture",
            components=['QueryAnalyzer', 'ContextAnalyzer', 'StrategyPlanner', 'EntityProcessor']
        )
    
    async def process(
        self, state: MusicRecommenderState
    ) -> MusicRecommenderState:
        """
        Process user query to create planning strategy using modular components.
        
        Phase 4.3: Simplified orchestration with component delegation.
        
        Args:
            state: Current recommender state
            
        Returns:
            Updated state with planning strategy
        """
        try:
            self.logger.info("Starting refactored planner agent processing")
            
            # Phase 1: Query Understanding & Entity Extraction
            query_understanding, entities = await self._handle_query_understanding(state)
            state.query_understanding = query_understanding
            state.entities = entities
            
            # Phase 2: Task Analysis
            task_analysis = await self._analyze_task_complexity(
                state.user_query, query_understanding
            )
            state.intent_analysis = task_analysis
            
            # Phase 3: Planning Strategy Creation
            planning_strategy = await self._create_planning_strategy(
                query_understanding, task_analysis
            )
            state.planning_strategy = planning_strategy
            
            # Phase 4: Agent Coordination Planning
            coordination_plan = await self._plan_agent_coordination(
                state.user_query, task_analysis
            )
            state.agent_coordination = coordination_plan
            
            self.logger.info(
                "Refactored planner agent processing completed",
                intent=query_understanding.intent.value,
                complexity=task_analysis.get('complexity_level', 'unknown'),
                strategy_components=len(planning_strategy) if planning_strategy else 0
            )
            
            return state
            
        except Exception as e:
            self.logger.error("Refactored planner agent processing failed", error=str(e))
            # Return state with minimal planning strategy
            state.planning_strategy = self.strategy_planner.create_fallback_strategy()
            return state
    
    async def _handle_query_understanding(
        self, state: MusicRecommenderState
    ) -> tuple[QueryUnderstanding, Dict[str, Any]]:
        """
        Handle query understanding using appropriate method based on available context.
        
        Phase 4.3: Simplified logic with component delegation.
        
        Args:
            state: Current recommender state
            
        Returns:
            Tuple of (QueryUnderstanding, entities)
        """
        # Phase 2: Use effective intent if available
        if hasattr(state, 'effective_intent') and state.effective_intent:
            self.logger.info("🎯 Using effective intent from IntentOrchestrationService")
            
            query_understanding = self.context_analyzer.create_understanding_from_effective_intent(
                state.user_query, state.effective_intent
            )
            entities = self.context_analyzer.create_entities_from_effective_intent(
                state.effective_intent
            )
            
            return query_understanding, entities
        
        # Legacy: Check for context override (Phase 1 functionality)
        context_override = getattr(state, 'context_override', None)
        if (context_override and 
            self.context_analyzer.is_followup_with_preserved_context(context_override)):
            
            self.logger.info("🔧 Using preserved context override")
            
            query_understanding = self.context_analyzer.create_understanding_from_context(
                state.user_query, context_override
            )
            entities = self.context_analyzer.create_entities_from_context(context_override)
            
            return query_understanding, entities
        
        # Default: Traditional query understanding
        self.logger.info("🔍 Using traditional query understanding")
        
        query_understanding = await self.query_analyzer.understand_user_query(state.user_query)
        entities = self.query_analyzer.convert_understanding_to_entities(query_understanding)
        
        return query_understanding, entities
    
    async def _analyze_task_complexity(
        self, user_query: str, understanding: QueryUnderstanding
    ) -> Dict[str, Any]:
        """
        Analyze task complexity using QueryAnalyzer component.
        
        Args:
            user_query: User's music request
            understanding: Query understanding results
            
        Returns:
            Task analysis dictionary
        """
        return await self.query_analyzer.analyze_task_complexity(user_query, understanding)
    
    async def _create_planning_strategy(
        self, understanding: QueryUnderstanding, task_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create planning strategy using StrategyPlanner component.
        
        Args:
            understanding: Query understanding results
            task_analysis: Task complexity analysis
            
        Returns:
            Planning strategy dictionary
        """
        return await self.strategy_planner.create_planning_strategy(understanding, task_analysis)
    
    async def _plan_agent_coordination(
        self, user_query: str, task_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Plan agent coordination using StrategyPlanner component.
        
        Args:
            user_query: User's music request
            task_analysis: Task analysis dictionary
            
        Returns:
            Agent coordination plan dictionary
        """
        return await self.strategy_planner.plan_agent_coordination(user_query, task_analysis)
    
    # Backward compatibility methods for existing code
    
    async def _understand_user_query(self, user_query: str) -> QueryUnderstanding:
        """Backward compatibility wrapper for query understanding."""
        return await self.query_analyzer.understand_user_query(user_query)
    
    def _convert_understanding_to_entities(self, understanding: QueryUnderstanding) -> Dict[str, Any]:
        """Backward compatibility wrapper for entity conversion."""
        return self.query_analyzer.convert_understanding_to_entities(understanding)
    
    def _is_followup_with_preserved_context(self, context_override: Dict) -> bool:
        """Backward compatibility wrapper for context validation."""
        return self.context_analyzer.is_followup_with_preserved_context(context_override)
    
    def _create_understanding_from_context(self, user_query: str, context_override: Dict) -> QueryUnderstanding:
        """Backward compatibility wrapper for context understanding."""
        return self.context_analyzer.create_understanding_from_context(user_query, context_override)
    
    def _create_entities_from_context(self, context_override: Dict) -> Dict[str, Any]:
        """Backward compatibility wrapper for context entities."""
        return self.context_analyzer.create_entities_from_context(context_override)
    
    def _extract_entity_names(self, entity_list: list) -> list[str]:
        """Backward compatibility wrapper for entity name extraction."""
        return self.entity_processor.extract_entity_names(entity_list)
    
    def _create_understanding_from_effective_intent(
        self, user_query: str, effective_intent: Dict[str, Any]
    ) -> QueryUnderstanding:
        """Backward compatibility wrapper for effective intent understanding."""
        return self.context_analyzer.create_understanding_from_effective_intent(
            user_query, effective_intent
        )
    
    def _create_entities_from_effective_intent(self, effective_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility wrapper for effective intent entities."""
        return self.context_analyzer.create_entities_from_effective_intent(effective_intent)
    
    def _extract_artists_from_effective_intent(self, entities: Dict[str, Any]) -> list[str]:
        """Backward compatibility wrapper for artist extraction."""
        return self.entity_processor.extract_artists_from_effective_intent(entities)
    
    def _extract_genres_from_effective_intent(self, entities: Dict[str, Any]) -> list[str]:
        """Backward compatibility wrapper for genre extraction."""
        return self.entity_processor.extract_genres_from_effective_intent(entities)
    
    def _extract_moods_from_effective_intent(self, entities: Dict[str, Any]) -> list[str]:
        """Backward compatibility wrapper for mood extraction."""
        return self.entity_processor.extract_moods_from_effective_intent(entities)
    
    def _extract_activities_from_effective_intent(self, entities: Dict[str, Any]) -> list[str]:
        """Backward compatibility wrapper for activity extraction."""
        return self.entity_processor.extract_activities_from_effective_intent(entities)
    
    def _create_fallback_strategy(self) -> Dict[str, Any]:
        """Backward compatibility wrapper for fallback strategy."""
        return self.strategy_planner.create_fallback_strategy()
    
    def _create_fallback_coordination(self) -> Dict[str, Any]:
        """Backward compatibility wrapper for fallback coordination."""
        return self.strategy_planner.create_fallback_coordination()
    
    def _should_generate_large_pool(
        self, understanding: QueryUnderstanding, task_analysis: Dict[str, Any]
    ) -> bool:
        """Backward compatibility wrapper for large pool decision."""
        return self.strategy_planner.should_generate_large_pool(understanding, task_analysis)
    
    def _determine_pool_size_multiplier(
        self, understanding: QueryUnderstanding, task_analysis: Dict[str, Any]
    ) -> int:
        """Backward compatibility wrapper for pool size multiplier."""
        return self.strategy_planner.determine_pool_size_multiplier(understanding, task_analysis)
    
    def _get_timestamp(self) -> str:
        """Utility method for timestamp generation."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def _make_llm_call(self, prompt: str, system_prompt: str = None) -> str:
        """
        Backward compatibility wrapper for LLM calls.
        Delegates to LLMUtils for consistency.
        """
        try:
            if system_prompt:
                response = await self.llm_utils.call_llm_with_json_response(
                    user_prompt=prompt,
                    system_prompt=system_prompt
                )
                return str(response) if response else "{}"
            else:
                # Simple text response
                response = await self.llm_utils.call_llm(prompt)
                return response if response else "{}"
        except Exception as e:
            self.logger.error("LLM call failed", error=str(e))
            return "{}"
    
    # Component access methods for advanced usage
    
    def get_query_analyzer(self) -> QueryAnalyzer:
        """Get the QueryAnalyzer component for direct access."""
        return self.query_analyzer
    
    def get_context_analyzer(self) -> ContextAnalyzer:
        """Get the ContextAnalyzer component for direct access."""
        return self.context_analyzer
    
    def get_strategy_planner(self) -> StrategyPlanner:
        """Get the StrategyPlanner component for direct access."""
        return self.strategy_planner
    
    def get_entity_processor(self) -> EntityProcessor:
        """Get the EntityProcessor component for direct access."""
        return self.entity_processor
    
    def get_component_status(self) -> Dict[str, bool]:
        """
        Get status of all components for health checking.
        
        Returns:
            Dictionary indicating component health status
        """
        return {
            'query_analyzer': self.query_analyzer is not None,
            'context_analyzer': self.context_analyzer is not None,
            'strategy_planner': self.strategy_planner is not None,
            'entity_processor': self.entity_processor is not None
        } 