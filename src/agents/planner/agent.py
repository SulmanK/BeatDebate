"""
Simplified Planner Agent

Refactored to use dependency injection and shared components, eliminating:
- Client instantiation duplication
- LLM calling duplication  
- JSON parsing duplication
- Rate limiting duplication
"""

import json
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
from .query_understanding_engine import QueryUnderstandingEngine

logger = structlog.get_logger(__name__)


class PlannerAgent(BaseAgent):
    """
    Simplified Planner Agent with dependency injection.
    
    Responsibilities:
    - Query understanding and entity extraction
    - Task analysis and complexity assessment
    - Planning strategy creation
    - Agent coordination planning
    
    Uses shared components to eliminate duplication:
    - LLMUtils for all LLM interactions
    - APIService for unified API access
    - MetadataService for metadata operations
    """
    
    def __init__(
        self,
        config,
        llm_client,
        api_service: APIService,
        metadata_service: MetadataService
    ):
        """
        Initialize simplified planner agent with injected dependencies.
        
        Args:
            config: Agent configuration
            llm_client: LLM client for query understanding
            api_service: Unified API service
            metadata_service: Unified metadata service
        """
        super().__init__(
            config=config, 
            llm_client=llm_client, 
            agent_name="PlannerAgent",
            api_service=api_service,
            metadata_service=metadata_service
        )
        
        # Shared utilities
        self.llm_utils = LLMUtils(llm_client)
        self.query_utils = QueryAnalysisUtils()
        
        # Specialized components
        self.query_understanding_engine = QueryUnderstandingEngine(
            llm_client
        )
        
        self.logger.info(
            "Simplified PlannerAgent initialized with dependency injection"
        )
    
    async def process(
        self, state: MusicRecommenderState
    ) -> MusicRecommenderState:
        """
        Process user query to create planning strategy.
        
        Args:
            state: Current recommender state
            
        Returns:
            Updated state with planning strategy
        """
        try:
            self.logger.info("Starting planner agent processing")
            
            # Phase 1: Query Understanding
            query_understanding = await self._understand_user_query(
                state.user_query
            )
            state.query_understanding = query_understanding
            
            # Phase 2: Task Analysis
            task_analysis = await self._analyze_task_complexity(
                state.user_query, query_understanding
            )
            
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
                "Planner agent processing completed",
                intent=query_understanding.intent.value,
                complexity=task_analysis.get('complexity_level', 'unknown'),
                strategy_components=len(planning_strategy) 
                if planning_strategy else 0
            )
            
            return state
            
        except Exception as e:
            self.logger.error("Planner agent processing failed", error=str(e))
            # Return state with minimal planning strategy
            state.planning_strategy = self._create_fallback_strategy()
            return state
    
    async def _understand_user_query(
        self, user_query: str
    ) -> QueryUnderstanding:
        """
        Understand user query using shared query understanding engine.
        
        Args:
            user_query: User's music request
            
        Returns:
            QueryUnderstanding object
        """
        try:
            understanding = await self.query_understanding_engine.understand_query(
                user_query
            )
            
            self.logger.debug(
                "Query understanding completed",
                intent=understanding.intent.value,
                confidence=understanding.confidence,
                has_entities=bool(understanding.artists or understanding.genres)
            )
            
            return understanding
            
        except Exception as e:
            self.logger.error("Query understanding failed", error=str(e))
            # Create fallback understanding
            return QueryUnderstanding(
                intent=QueryIntent.DISCOVERY,
                confidence=0.3,
                artists=[],
                genres=[],
                moods=[],
                activities=[],
                original_query=user_query,
                normalized_query=user_query.lower(),
                reasoning="Fallback understanding due to processing error"
            )
    
    async def _analyze_task_complexity(
        self, user_query: str, understanding: QueryUnderstanding
    ) -> Dict[str, Any]:
        """
        Analyze task complexity using shared utilities and LLM.
        
        Args:
            user_query: User's music request
            understanding: Query understanding results
            
        Returns:
            Task analysis dictionary
        """
        try:
            # Use shared query analysis utilities
            complexity_analysis = self.query_utils.analyze_query_complexity(
                user_query
            )
            
            # Enhanced analysis using LLM for complex queries
            if complexity_analysis['complexity_level'] == 'complex':
                llm_analysis = await self._llm_task_analysis(
                    user_query, understanding
                )
                # Merge analyses
                task_analysis = self._merge_task_analyses(
                    complexity_analysis, llm_analysis
                )
            else:
                task_analysis = complexity_analysis
            
            # Add understanding-based factors
            task_analysis['intent_complexity'] = (
                self._assess_intent_complexity(understanding)
            )
            task_analysis['entity_complexity'] = (
                self._assess_entity_complexity(understanding)
            )
            
            self.logger.debug(
                "Task complexity analyzed",
                complexity_level=task_analysis['complexity_level'],
                intent_complexity=task_analysis['intent_complexity'],
                entity_complexity=task_analysis['entity_complexity']
            )
            
            return task_analysis
            
        except Exception as e:
            self.logger.error("Task analysis failed", error=str(e))
            return {'complexity_level': 'medium', 'confidence': 0.3}
    
    async def _llm_task_analysis(
        self, user_query: str, understanding: QueryUnderstanding
    ) -> Dict[str, Any]:
        """Use shared LLM utilities for enhanced task analysis."""
        system_prompt = """You are a strategic music recommendation planner. 
Analyze the user's query to understand their intent, mood, and context.

Return a JSON object with this structure:
{
    "primary_goal": "brief description of main intent",
    "complexity_level": "simple|medium|complex",
    "context_factors": ["factor1", "factor2"],
    "mood_indicators": ["mood1", "mood2"],
    "genre_hints": ["genre1", "genre2"],
    "urgency_level": "low|medium|high",
    "specificity": "vague|moderate|specific"
}"""
        
        user_prompt = f"""Analyze this music request:
"{user_query}"

Current understanding:
- Intent: {understanding.intent.value}
- Confidence: {understanding.confidence}
- Has entities: {bool(understanding.artists or understanding.genres)}

Provide enhanced analysis in the specified JSON format."""
        
        try:
            # Use shared LLM utilities
            llm_data = await self.llm_utils.call_llm_with_json_response(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_retries=2
            )
            
            # Validate structure
            required_keys = [
                'primary_goal', 'complexity_level', 'context_factors'
            ]
            optional_keys = [
                'mood_indicators', 'genre_hints', 'urgency_level', 
                'specificity'
            ]
            
            validated_data = self.llm_utils.validate_json_structure(
                llm_data, required_keys, optional_keys
            )
            
            return validated_data
            
        except Exception as e:
            self.logger.warning("LLM task analysis failed", error=str(e))
            return {}
    
    def _merge_task_analyses(
        self, complexity_analysis: Dict[str, Any], llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge complexity analysis with LLM analysis."""
        merged = complexity_analysis.copy()
        
        # Add LLM insights
        if llm_analysis:
            merged.update({
                'primary_goal': llm_analysis.get(
                    'primary_goal', 'music_discovery'
                ),
                'urgency_level': llm_analysis.get('urgency_level', 'medium'),
                'specificity': llm_analysis.get('specificity', 'moderate'),
                'llm_mood_indicators': llm_analysis.get(
                    'mood_indicators', []
                ),
                'llm_genre_hints': llm_analysis.get('genre_hints', [])
            })
            
            # Override complexity if LLM has different assessment
            llm_complexity = llm_analysis.get('complexity_level')
            if (llm_complexity and 
                llm_complexity != merged['complexity_level']):
                merged['complexity_level'] = llm_complexity
                merged['complexity_source'] = 'llm_override'
        
        return merged
    
    def _assess_intent_complexity(
        self, understanding: QueryUnderstanding
    ) -> str:
        """Assess complexity based on intent type."""
        intent_complexity_map = {
            'discovery': 'medium',
            'similarity': 'simple',
            'mood_based': 'simple',
            'activity_based': 'simple',
            'genre_specific': 'simple'
        }
        
        base_complexity = intent_complexity_map.get(
            understanding.intent.value, 'medium'
        )
        
        # Increase complexity if multiple factors present
        if (understanding.similarity_type and 
            len(understanding.activities) > 2):
            if base_complexity == 'simple':
                return 'medium'
            elif base_complexity == 'medium':
                return 'complex'
        
        return base_complexity
    
    def _assess_entity_complexity(
        self, understanding: QueryUnderstanding
    ) -> str:
        """Assess complexity based on extracted entities."""
        entity_count = (
            len(understanding.artists) + 
            len(understanding.genres) + 
            len(understanding.moods) + 
            len(understanding.activities)
        )
        
        if entity_count == 0:
            return 'simple'
        elif entity_count <= 3:
            return 'medium'
        else:
            return 'complex'
    
    async def _create_planning_strategy(
        self, understanding: QueryUnderstanding, task_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create planning strategy based on understanding and analysis.
        
        Args:
            understanding: Query understanding results
            task_analysis: Task complexity analysis
            
        Returns:
            Planning strategy dictionary
        """
        try:
            strategy = {
                'intent': understanding.intent.value,
                'complexity_level': task_analysis.get(
                    'complexity_level', 'medium'
                ),
                'confidence': understanding.confidence,
                'agent_sequence': self._determine_agent_sequence(
                    understanding, task_analysis
                ),
                'quality_thresholds': self._determine_quality_thresholds(
                    task_analysis
                ),
                'diversity_targets': self._determine_diversity_targets(
                    understanding
                ),
                'explanation_style': self._determine_explanation_style(
                    task_analysis
                ),
                'fallback_strategies': self._create_fallback_strategies(
                    understanding
                )
            }
            
            self.logger.debug(
                "Planning strategy created",
                agent_sequence=strategy['agent_sequence'],
                quality_thresholds=strategy['quality_thresholds']
            )
            
            return strategy
            
        except Exception as e:
            self.logger.error(
                "Planning strategy creation failed", error=str(e)
            )
            return self._create_fallback_strategy()
    
    def _determine_agent_sequence(
        self, understanding: QueryUnderstanding, task_analysis: Dict[str, Any]
    ) -> list:
        """Determine optimal agent sequence based on intent and complexity."""
        intent = understanding.intent.value
        complexity = task_analysis.get('complexity_level', 'medium')
        
        # Base sequences by intent
        intent_sequences = {
            'discovery': ['genre_mood', 'discovery', 'judge'],
            'similarity': ['genre_mood', 'judge'],
            'mood_based': ['genre_mood', 'judge'],
            'activity_based': ['genre_mood', 'judge'],
            'genre_specific': ['genre_mood', 'judge']
        }
        
        base_sequence = intent_sequences.get(
            intent, ['genre_mood', 'judge']
        )
        
        # Adjust based on complexity
        if complexity == 'complex':
            # Add discovery agent for complex queries
            if 'discovery' not in base_sequence:
                base_sequence.insert(-1, 'discovery')
        elif complexity == 'simple':
            # Simplify sequence for simple queries
            if len(base_sequence) > 2:
                base_sequence = base_sequence[:2]
        
        return base_sequence
    
    def _determine_quality_thresholds(
        self, task_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Determine quality thresholds based on task analysis."""
        complexity = task_analysis.get('complexity_level', 'medium')
        urgency = task_analysis.get('urgency_level', 'medium')
        
        # Base thresholds
        thresholds = {
            'confidence': 0.6,
            'novelty_score': 0.5,
            'quality_score': 0.7,
            'concentration_friendliness_score': 0.6
        }
        
        # Adjust based on complexity
        if complexity == 'complex':
            thresholds['confidence'] = 0.7
            thresholds['quality_score'] = 0.8
        elif complexity == 'simple':
            thresholds['confidence'] = 0.5
            thresholds['quality_score'] = 0.6
        
        # Adjust based on urgency
        if urgency == 'high':
            # Lower thresholds for faster results
            for key in thresholds:
                thresholds[key] *= 0.9
        
        return thresholds
    
    def _determine_diversity_targets(
        self, understanding: QueryUnderstanding
    ) -> Dict[str, Any]:
        """Determine diversity targets based on understanding."""
        # Base diversity targets
        targets = {
            'attributes': ['genres', 'artist'],
            'genres': 2,
            'era': 2,
            'energy': 1,
            'artist': 3
        }
        
        # Adjust based on intent
        if understanding.intent.value == 'discovery':
            targets['genres'] = 3
            targets['artist'] = 4
            targets['attributes'].append('era')
        elif understanding.intent.value == 'similarity':
            targets['genres'] = 1
            targets['artist'] = 2
        
        return targets
    
    def _determine_explanation_style(
        self, task_analysis: Dict[str, Any]
    ) -> str:
        """Determine explanation style based on task analysis."""
        complexity = task_analysis.get('complexity_level', 'medium')
        specificity = task_analysis.get('specificity', 'moderate')
        
        if complexity == 'complex' or specificity == 'specific':
            return 'detailed'
        elif complexity == 'simple':
            return 'concise'
        else:
            return 'casual'
    
    def _create_fallback_strategies(
        self, understanding: QueryUnderstanding
    ) -> list:
        """Create fallback strategies for error handling."""
        fallbacks = [
            {
                'trigger': 'low_confidence',
                'action': 'expand_search_terms',
                'threshold': 0.3
            },
            {
                'trigger': 'no_results',
                'action': 'broaden_criteria',
                'fallback_intent': 'discovery'
            },
            {
                'trigger': 'api_failure',
                'action': 'use_cached_results',
                'cache_duration': 3600
            }
        ]
        
        return fallbacks
    
    async def _plan_agent_coordination(
        self, user_query: str, task_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Plan agent coordination and evaluation framework.
        
        Args:
            user_query: User's music request
            task_analysis: Task complexity analysis
            
        Returns:
            Agent coordination plan
        """
        try:
            # Use shared LLM utilities for coordination planning
            system_prompt = """You are a music recommendation coordinator. 
Create an evaluation framework for agent coordination.

Return a JSON object with this structure:
{
    "primary_weights": {
        "confidence": 0.0-1.0,
        "novelty_score": 0.0-1.0,
        "quality_score": 0.0-1.0,
        "concentration_friendliness_score": 0.0-1.0
    },
    "diversity_targets": {
        "attributes": ["genres", "artist"],
        "genres": 1-3,
        "era": 1-3,
        "energy": 1-2,
        "artist": 2-3
    },
    "explanation_style": "detailed|concise|technical|casual",
    "selection_criteria": ["criterion1", "criterion2"]
}"""
            
            user_prompt = f"""User Query: "{user_query}"
Task Analysis: {json.dumps(task_analysis, indent=2)}

Create evaluation framework for this music recommendation task."""
            
            coordination_data = await self.llm_utils.call_llm_with_json_response(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_retries=2
            )
            
            # Validate and enhance coordination plan
            required_keys = [
                'primary_weights', 'diversity_targets', 'explanation_style'
            ]
            optional_keys = ['selection_criteria']
            
            validated_plan = self.llm_utils.validate_json_structure(
                coordination_data, required_keys, optional_keys
            )
            
            # Add execution metadata
            validated_plan['created_timestamp'] = self._get_timestamp()
            validated_plan['task_complexity'] = task_analysis.get(
                'complexity_level', 'medium'
            )
            
            return validated_plan
            
        except Exception as e:
            self.logger.warning(
                "Agent coordination planning failed", error=str(e)
            )
            return self._create_fallback_coordination()
    
    def _create_fallback_strategy(self) -> Dict[str, Any]:
        """Create fallback planning strategy."""
        return {
            'intent': 'discovery',
            'complexity_level': 'medium',
            'confidence': 0.3,
            'agent_sequence': ['genre_mood', 'judge'],
            'quality_thresholds': {
                'confidence': 0.5,
                'novelty_score': 0.4,
                'quality_score': 0.6,
                'concentration_friendliness_score': 0.5
            },
            'diversity_targets': {
                'attributes': ['genres', 'artist'],
                'genres': 2,
                'era': 1,
                'energy': 1,
                'artist': 2
            },
            'explanation_style': 'casual',
            'fallback_strategies': []
        }
    
    def _create_fallback_coordination(self) -> Dict[str, Any]:
        """Create fallback coordination plan."""
        return {
            'primary_weights': {
                'confidence': 0.3,
                'novelty_score': 0.2,
                'quality_score': 0.3,
                'concentration_friendliness_score': 0.2
            },
            'diversity_targets': {
                'attributes': ['genres', 'artist'],
                'genres': 2,
                'era': 1,
                'energy': 1,
                'artist': 2
            },
            'explanation_style': 'casual',
            'selection_criteria': ['relevance', 'quality'],
            'created_timestamp': self._get_timestamp(),
            'task_complexity': 'medium'
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def _make_llm_call(
        self, prompt: str, system_prompt: str = None
    ) -> str:
        """Use shared LLM utilities for LLM calls."""
        return await self.llm_utils.call_llm(prompt, system_prompt) 