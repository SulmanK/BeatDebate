"""
Simplified Planner Agent

Refactored to use dependency injection and shared components, eliminating:
- Client instantiation duplication
- LLM calling duplication  
- JSON parsing duplication
- Rate limiting duplication
"""

import json
from typing import Dict, Any, List
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
        metadata_service: MetadataService,
        rate_limiter=None
    ):
        """
        Initialize simplified planner agent with injected dependencies.
        
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
        
        # Specialized components (LLMUtils now initialized in parent with rate limiter)
        self.query_understanding_engine = QueryUnderstandingEngine(
            llm_client, rate_limiter=rate_limiter
        )
        
        # Shared utilities
        self.query_utils = QueryAnalysisUtils()
        
        self.logger.info(
            "Simplified PlannerAgent initialized with dependency injection"
        )
    
    async def process(
        self, state: MusicRecommenderState
    ) -> MusicRecommenderState:
        """
        Process user query to create planning strategy.
        
        Phase 2: Simplified to use effective intent from IntentOrchestrationService.
        
        Args:
            state: Current recommender state
            
        Returns:
            Updated state with planning strategy
        """
        try:
            self.logger.info("Starting planner agent processing (Phase 2)")
            
            # Phase 2: Use effective intent if available, otherwise fall back to query understanding
            if hasattr(state, 'effective_intent') and state.effective_intent:
                self.logger.info("🎯 Phase 2: Using effective intent from IntentOrchestrationService")
                query_understanding = self._create_understanding_from_effective_intent(
                    state.user_query, state.effective_intent
                )
                entities = self._create_entities_from_effective_intent(state.effective_intent)
                state.query_understanding = query_understanding
                state.entities = entities
            else:
                # Fallback: Use traditional query understanding for backward compatibility
                self.logger.info("🔧 Phase 2: No effective intent available, using traditional query understanding")
                query_understanding = await self._understand_user_query(state.user_query)
                entities = self._convert_understanding_to_entities(query_understanding)
                state.query_understanding = query_understanding
                state.entities = entities
            
            # Phase 2: Task Analysis
            task_analysis = await self._analyze_task_complexity(
                state.user_query, query_understanding
            )
            state.intent_analysis = task_analysis  # Also set intent_analysis for agents
            
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
    
    def _convert_understanding_to_entities(self, understanding: QueryUnderstanding) -> Dict[str, Any]:
        """
        Convert QueryUnderstanding object to the entities structure expected by agents.
        
        Args:
            understanding: QueryUnderstanding object from query understanding engine
            
        Returns:
            Entities dictionary with proper nested structure
        """
        # Create properly structured entities dict
        entities = {
            "musical_entities": {
                "artists": {
                    "primary": understanding.artists,
                    "similar_to": []  # All artists from similarity queries go to primary for now
                },
                "genres": {
                    "primary": understanding.genres,
                    "secondary": []
                },
                "tracks": {
                    "primary": [],
                    "referenced": []
                },
                "moods": {
                    "primary": understanding.moods,
                    "energy": [],
                    "emotion": []
                }
            },
            "contextual_entities": {
                "activities": {
                    "physical": understanding.activities,
                    "mental": [],
                    "social": []
                },
                "temporal": {
                    "decades": [],
                    "periods": []
                }
            },
            "confidence_scores": {
                "overall": understanding.confidence
            },
            "extraction_method": "llm_query_understanding",
            "intent_analysis": {
                "intent": understanding.intent.value,
                "similarity_type": understanding.similarity_type.value if understanding.similarity_type else None,
                "confidence": understanding.confidence
            }
        }
        
        # For similarity queries, put artists in similar_to instead of primary
        if understanding.intent == QueryIntent.ARTIST_SIMILARITY and understanding.artists:
            entities["musical_entities"]["artists"]["similar_to"] = understanding.artists
            entities["musical_entities"]["artists"]["primary"] = []
        
        self.logger.debug(
            "Converted QueryUnderstanding to entities",
            artists_count=len(understanding.artists),
            genres_count=len(understanding.genres),
            moods_count=len(understanding.moods),
            intent=understanding.intent.value
        )
        
        return entities
    
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
                ),
                # Phase 3: Candidate pool persistence strategy
                'generate_large_pool': self._should_generate_large_pool(
                    understanding, task_analysis
                ),
                'pool_size_multiplier': self._determine_pool_size_multiplier(
                    understanding, task_analysis
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
        
        # Intent-aware sequences from design document
        intent_sequences = {
            'by_artist': ['discovery', 'judge'],     # 🔧 NEW: by_artist uses discovery for artist tracks
            'by_artist_underground': ['discovery', 'judge'],  # 🔧 NEW: underground discovery by artist
            'artist_similarity': ['discovery', 'judge'],
            'discovery': ['discovery', 'judge'],
            'genre_mood': ['genre_mood', 'discovery', 'judge'],
            'contextual': ['genre_mood', 'discovery', 'judge'],
            'hybrid': ['discovery', 'genre_mood', 'judge'],
            
            # Legacy compatibility mappings
            'similarity': ['discovery', 'judge'],
            'genre_exploration': ['genre_mood', 'discovery', 'judge'],
            'mood_matching': ['genre_mood', 'discovery', 'judge'],
            'activity_context': ['genre_mood', 'judge'],
            'mood_based': ['genre_mood', 'judge'],
            'activity_based': ['genre_mood', 'judge'],
            'genre_specific': ['genre_mood', 'judge']
        }
        
        base_sequence = intent_sequences.get(
            intent, ['genre_mood', 'judge']  # Default fallback
        )
        
        # Adjust based on complexity for edge cases
        if complexity == 'complex':
            # For complex queries, ensure discovery agent is included
            if 'discovery' not in base_sequence and intent != 'contextual':
                base_sequence.insert(-1, 'discovery')
        elif complexity == 'simple':
            # For simple queries, we can sometimes simplify but keep intent focus
            if intent == 'discovery' and len(base_sequence) > 2:
                # For simple discovery, just discovery + judge
                base_sequence = ['discovery', 'judge']
            elif intent == 'contextual' and len(base_sequence) > 2:
                # For simple contextual, just genre_mood + judge
                base_sequence = ['genre_mood', 'judge']
        
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
    
    def _should_generate_large_pool(
        self, understanding: QueryUnderstanding, task_analysis: Dict[str, Any]
    ) -> bool:
        """
        Determine if a large candidate pool should be generated for follow-up queries.
        
        Phase 3: This method decides when to generate 3x more candidates for persistence.
        
        Args:
            understanding: Query understanding results
            task_analysis: Task complexity analysis
            
        Returns:
            True if large pool should be generated
        """
        intent = understanding.intent.value
        
        # Intents that benefit from large pools for follow-ups
        pool_beneficial_intents = ['by_artist', 'artist_similarity', 'genre_exploration', 'discovery']
        
        # Generate large pool if:
        # 1. Intent benefits from pools
        # 2. High confidence query (likely to have follow-ups)
        # 3. Not a follow-up query itself (check if this is original)
        
        if intent in pool_beneficial_intents and understanding.confidence > 0.7:
            self.logger.info(
                f"Phase 3: Large pool generation recommended for intent '{intent}' "
                f"with confidence {understanding.confidence}"
            )
            return True
        
        return False
    
    def _determine_pool_size_multiplier(
        self, understanding: QueryUnderstanding, task_analysis: Dict[str, Any]
    ) -> int:
        """
        Determine the multiplier for candidate pool size.
        
        Phase 3: This method determines how much larger the pool should be.
        
        Args:
            understanding: Query understanding results
            task_analysis: Task complexity analysis
            
        Returns:
            Pool size multiplier (default 3)
        """
        intent = understanding.intent.value
        confidence = understanding.confidence
        
        # Higher multiplier for high-confidence artist queries (more likely to have follow-ups)
        if intent in ['by_artist', 'artist_similarity'] and confidence > 0.8:
            return 4  # 4x larger pool for artist queries
        
        # Standard multiplier for other pool-beneficial intents
        if intent in ['genre_exploration', 'discovery']:
            return 3  # 3x larger pool
        
        # Default (no large pool)
        return 1
    
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
        """Make an LLM call with the provided prompt."""
        try:
            if not self.llm_client:
                self.logger.warning("No LLM client available")
                return "{}"
            
            response = await self.llm_client.generate_response(
                prompt, system_prompt=system_prompt
            )
            return response
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return "{}"

    def _is_followup_with_preserved_context(self, context_override: Dict) -> bool:
        """
        Check if context override contains preserved entities that should skip query understanding.
        
        Following clean architecture principles, this method acts as a domain rule
        to determine when we should use preserved context vs fresh query understanding.
        
        Returns True for follow-ups with preserved entities like:
        - Artist deep dives with preserved genres ('hybrid_artist_genre')
        - Style continuations with preserved context ('style_continuation')
        - Artist refinements with preserved filters ('artist_style_refinement')
        - Artist similarity follow-ups with target entity ('artist_similarity')
        """
        if not isinstance(context_override, dict):
            return False
        
        # Check for follow-up indicators
        is_followup = context_override.get('is_followup', False)
        has_preserved_entities = 'preserved_entities' in context_override
        has_intent_override = 'intent_override' in context_override
        has_target_entity = context_override.get('target_entity') is not None
        
        # Define which intent overrides should use preserved context
        followup_types_with_context = [
            'hybrid_artist_genre', 'artist_style_refinement', 'style_continuation', 'by_artist'
        ]
        
        intent_override = context_override.get('intent_override')
        
        # Two types of valid follow-ups:
        # 1. Complex follow-ups with preserved entities (hybrid scenarios)
        # 2. Simple artist follow-ups with target entity (artist_similarity scenarios)
        
        complex_followup = (is_followup and 
                           has_preserved_entities and 
                           has_intent_override and
                           intent_override in followup_types_with_context)
        
        simple_artist_followup = (is_followup and
                                 has_target_entity and
                                 has_intent_override and
                                 intent_override in ['artist_similarity', 'by_artist'])
        
        result = complex_followup or simple_artist_followup
        
        self.logger.debug(
            "🔍 Context override validation",
            is_followup=is_followup,
            has_preserved_entities=has_preserved_entities,
            has_intent_override=has_intent_override,
            has_target_entity=has_target_entity,
            intent_override=intent_override,
            complex_followup=complex_followup,
            simple_artist_followup=simple_artist_followup,
            should_use_context=result
        )
        
        return result

    def _create_understanding_from_context(self, user_query: str, context_override: Dict) -> QueryUnderstanding:
        """
        Create QueryUnderstanding from preserved context override.
        
        This method implements the domain logic for converting preserved conversation
        context into a proper QueryUnderstanding object, ensuring consistency with
        the rest of the system.
        """
        preserved_entities = context_override.get('preserved_entities', {})
        intent_override = context_override.get('intent_override', 'discovery')
        confidence = context_override.get('confidence', 0.9)
        target_entity = context_override.get('target_entity')
        
        # For artist similarity follow-ups, create entities from target_entity
        if intent_override == 'artist_similarity' and target_entity and not preserved_entities:
            # Simple artist follow-up: "More tracks" after "Music by Mk.gee"
            artists = [target_entity]
            genres = []
            moods = []
            self.logger.info(
                f"🎯 Artist similarity follow-up: Creating entities from target_entity='{target_entity}'"
            )
        else:
            # Complex follow-up with preserved entities
            artists = self._extract_entity_names(
                preserved_entities.get('artists', {}).get('primary', [])
            )
            genres = self._extract_entity_names(
                preserved_entities.get('genres', {}).get('primary', [])
            )
            moods = self._extract_entity_names(
                preserved_entities.get('moods', {}).get('primary', [])
            )
        
        # Map intent override to QueryIntent enum - domain rule mapping
        intent_mapping = {
            'hybrid_artist_genre': QueryIntent.HYBRID,
            'artist_style_refinement': QueryIntent.HYBRID, 
            'style_continuation': QueryIntent.GENRE_MOOD,
            'artist_deep_dive': QueryIntent.ARTIST_SIMILARITY,
            'artist_similarity': QueryIntent.ARTIST_SIMILARITY,  # Similar artists
            'by_artist': QueryIntent.BY_ARTIST  # ✅ NEW: More tracks by the same artist
        }
        
        intent = intent_mapping.get(intent_override, QueryIntent.DISCOVERY)
        
        self.logger.info(
            f"🎯 Created understanding from context",
            intent=intent.value,
            artists=artists,
            genres=genres,
            confidence=confidence,
            override_type=intent_override
        )
        
        return QueryUnderstanding(
            intent=intent,
            confidence=confidence,
            artists=artists,
            genres=genres,
            moods=moods,
            activities=[],
            original_query=user_query,
            normalized_query=user_query.lower(),
            reasoning=f"Context override: {intent_override} follow-up with preserved entities"
        )

    def _create_entities_from_context(self, context_override: Dict) -> Dict[str, Any]:
        """
        Create entities structure from context override.
        
        This method transforms preserved conversation context into the standardized
        entities structure expected by downstream agents, maintaining architectural
        consistency.
        """
        preserved_entities = context_override.get('preserved_entities', {})
        intent_override = context_override.get('intent_override', 'discovery')
        target_entity = context_override.get('target_entity')
        
        # For artist similarity follow-ups, create entities from target_entity
        if intent_override == 'artist_similarity' and target_entity and not preserved_entities:
            # Simple artist follow-up: "More tracks" after "Music by Mk.gee"
            artists_primary = [target_entity]
            genres_primary = []
            moods_primary = []
            self.logger.info(
                f"🎯 Artist similarity follow-up: Creating entities structure from target_entity='{target_entity}'"
            )
        else:
            # Complex follow-up with preserved entities
            # Extract preserved entity data with safe navigation
            artists_data = preserved_entities.get('artists', {})
            genres_data = preserved_entities.get('genres', {})
            moods_data = preserved_entities.get('moods', {})
            
            artists_primary = self._extract_entity_names(artists_data.get('primary', []))
            genres_primary = self._extract_entity_names(genres_data.get('primary', []))
            moods_primary = self._extract_entity_names(moods_data.get('primary', []))
        
        # Convert to proper entities structure following established schema
        entities = {
            "musical_entities": {
                "artists": {
                    "primary": artists_primary,
                    "similar_to": []
                },
                "genres": {
                    "primary": genres_primary,
                    "secondary": []
                },
                "tracks": {
                    "primary": [],
                    "referenced": []
                },
                "moods": {
                    "primary": moods_primary,
                    "energy": [],
                    "emotion": []
                }
            },
            "contextual_entities": {
                "activities": {
                    "physical": [],
                    "mental": [],
                    "social": []
                },
                "temporal": {
                    "decades": [],
                    "periods": []
                }
            },
            "confidence_scores": {
                "overall": context_override.get('confidence', 0.9)
            },
            "extraction_method": "context_override_preserved",
            "intent_analysis": {
                "intent": intent_override,
                "confidence": context_override.get('confidence', 0.9),
                "context_override_applied": True
            }
        }
        
        self.logger.info(
            f"🎯 Created entities from context",
            artists_count=len(entities['musical_entities']['artists']['primary']),
            genres_count=len(entities['musical_entities']['genres']['primary']),
            moods_count=len(entities['musical_entities']['moods']['primary']),
            extraction_method=entities['extraction_method']
        )
        
        return entities

    def _extract_entity_names(self, entity_list: List) -> List[str]:
        """
        Extract names from entity list that may contain dicts or strings.
        
        This utility method handles the data transformation needed to work with
        various entity formats from preserved context.
        """
        names = []
        for item in entity_list:
            if isinstance(item, dict):
                # Handle confidence score format: {'name': 'Artist', 'confidence': 0.8}
                names.append(item.get('name', str(item)))
            elif isinstance(item, str):
                names.append(item)
            else:
                names.append(str(item))
        return names
    
    # Phase 2: New methods for handling effective intent from IntentOrchestrationService
    
    def _create_understanding_from_effective_intent(
        self, user_query: str, effective_intent: Dict[str, Any]
    ) -> QueryUnderstanding:
        """
        Create QueryUnderstanding from effective intent provided by IntentOrchestrationService.
        
        Phase 2: Simplified approach that trusts the intent orchestrator's resolution.
        """
        intent_str = effective_intent.get('intent', 'discovery')
        entities = effective_intent.get('entities', {})
        confidence = effective_intent.get('confidence', 0.8)
        
        # Extract entities from effective intent
        artists = self._extract_artists_from_effective_intent(entities)
        genres = self._extract_genres_from_effective_intent(entities)
        moods = self._extract_moods_from_effective_intent(entities)
        activities = self._extract_activities_from_effective_intent(entities)
        
        # Map intent string to QueryIntent enum
        intent_mapping = {
            'artist_similarity': QueryIntent.ARTIST_SIMILARITY,
            'genre_exploration': QueryIntent.GENRE_MOOD,
            'mood_matching': QueryIntent.GENRE_MOOD,
            'activity_context': QueryIntent.GENRE_MOOD,
            'discovery': QueryIntent.DISCOVERY,
            'hybrid': QueryIntent.HYBRID,
            'by_artist': QueryIntent.BY_ARTIST
        }
        
        intent = intent_mapping.get(intent_str, QueryIntent.DISCOVERY)
        
        reasoning = "Phase 2: Effective intent from IntentOrchestrationService"
        if effective_intent.get('is_followup'):
            reasoning += f" (follow-up: {effective_intent.get('followup_type', 'unknown')})"
        
        self.logger.info(
            "Phase 2: Created understanding from effective intent",
            intent=intent.value,
            artists=artists,
            genres=genres,
            confidence=confidence,
            is_followup=effective_intent.get('is_followup', False)
        )
        
        return QueryUnderstanding(
            intent=intent,
            confidence=confidence,
            artists=artists,
            genres=genres,
            moods=moods,
            activities=activities,
            original_query=user_query,
            normalized_query=user_query.lower(),
            reasoning=reasoning
        )
    
    def _create_entities_from_effective_intent(self, effective_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create entities structure from effective intent.
        
        Phase 2: Simplified approach that uses the entities resolved by IntentOrchestrationService.
        """
        entities = effective_intent.get('entities', {})
        
        # Extract entities using helper methods
        artists = self._extract_artists_from_effective_intent(entities)
        genres = self._extract_genres_from_effective_intent(entities)
        moods = self._extract_moods_from_effective_intent(entities)
        activities = self._extract_activities_from_effective_intent(entities)
        
        # Create standardized entities structure
        result = {
            "musical_entities": {
                "artists": {
                    "primary": artists,
                    "similar_to": []
                },
                "genres": {
                    "primary": genres,
                    "secondary": []
                },
                "tracks": {
                    "primary": [],
                    "referenced": []
                },
                "moods": {
                    "primary": moods,
                    "energy": [],
                    "emotion": []
                }
            },
            "contextual_entities": {
                "activities": {
                    "physical": activities,
                    "mental": [],
                    "social": []
                },
                "temporal": {
                    "decades": [],
                    "periods": []
                }
            },
            "confidence_scores": {
                "overall": effective_intent.get('confidence', 0.8)
            },
            "extraction_method": "effective_intent_phase2",
            "intent_analysis": {
                "intent": effective_intent.get('intent', 'discovery'),
                "confidence": effective_intent.get('confidence', 0.8),
                "is_followup": effective_intent.get('is_followup', False),
                "followup_type": effective_intent.get('followup_type'),
                "preserves_original_context": effective_intent.get('preserves_original_context', False)
            }
        }
        
        self.logger.info(
            "Phase 2: Created entities from effective intent",
            artists_count=len(artists),
            genres_count=len(genres),
            moods_count=len(moods),
            is_followup=effective_intent.get('is_followup', False)
        )
        
        return result
    
    def _extract_artists_from_effective_intent(self, entities: Dict[str, Any]) -> List[str]:
        """Extract artist names from effective intent entities."""
        artists = []
        
        # Handle different entity structures
        if 'artists' in entities:
            artist_data = entities['artists']
            if isinstance(artist_data, list):
                artists.extend(self._extract_entity_names(artist_data))
        
        # Also check musical_entities structure
        if 'musical_entities' in entities and 'artists' in entities['musical_entities']:
            artist_data = entities['musical_entities']['artists']
            if isinstance(artist_data, dict) and 'primary' in artist_data:
                artists.extend(self._extract_entity_names(artist_data['primary']))
            elif isinstance(artist_data, list):
                artists.extend(self._extract_entity_names(artist_data))
        
        return list(set(artists))  # Remove duplicates
    
    def _extract_genres_from_effective_intent(self, entities: Dict[str, Any]) -> List[str]:
        """Extract genre names from effective intent entities."""
        genres = []
        
        if 'genres' in entities:
            genre_data = entities['genres']
            if isinstance(genre_data, dict):
                for genre_list in [genre_data.get('primary', []), genre_data.get('secondary', [])]:
                    genres.extend(self._extract_entity_names(genre_list))
            elif isinstance(genre_data, list):
                genres.extend(self._extract_entity_names(genre_data))
        
        # Also check musical_entities structure
        if 'musical_entities' in entities and 'genres' in entities['musical_entities']:
            genre_data = entities['musical_entities']['genres']
            if isinstance(genre_data, dict):
                for genre_list in [genre_data.get('primary', []), genre_data.get('secondary', [])]:
                    genres.extend(self._extract_entity_names(genre_list))
        
        return list(set(genres))  # Remove duplicates
    
    def _extract_moods_from_effective_intent(self, entities: Dict[str, Any]) -> List[str]:
        """Extract mood names from effective intent entities."""
        moods = []
        
        if 'moods' in entities:
            mood_data = entities['moods']
            if isinstance(mood_data, dict):
                for mood_list in [mood_data.get('primary', []), mood_data.get('secondary', [])]:
                    moods.extend(self._extract_entity_names(mood_list))
            elif isinstance(mood_data, list):
                moods.extend(self._extract_entity_names(mood_data))
        
        # Also check musical_entities structures
        if 'musical_entities' in entities and 'moods' in entities['musical_entities']:
            mood_data = entities['musical_entities']['moods']
            if isinstance(mood_data, dict):
                for mood_list in [mood_data.get('primary', []), mood_data.get('energy', []), mood_data.get('emotion', [])]:
                    moods.extend(self._extract_entity_names(mood_list))
        
        return list(set(moods))  # Remove duplicates
    
    def _extract_activities_from_effective_intent(self, entities: Dict[str, Any]) -> List[str]:
        """Extract activity names from effective intent entities."""
        activities = []
        
        if 'activities' in entities:
            activity_data = entities['activities']
            if isinstance(activity_data, list):
                activities.extend(self._extract_entity_names(activity_data))
        
        # Also check contextual_entities structure
        if 'contextual_entities' in entities and 'activities' in entities['contextual_entities']:
            activity_data = entities['contextual_entities']['activities']
            if isinstance(activity_data, dict):
                for activity_list in [activity_data.get('physical', []), activity_data.get('mental', []), activity_data.get('social', [])]:
                    activities.extend(self._extract_entity_names(activity_list))
        
        return list(set(activities))  # Remove duplicates