"""
Query Analyzer Component for Planner Agent

Handles query understanding and parsing logic for the planner agent.
Extracted from PlannerAgent for better modularization and single responsibility.
"""

from typing import Dict, Any
import structlog

from ...models.agent_models import QueryUnderstanding, QueryIntent
from ..components.llm_utils import LLMUtils
from ..components.query_analysis_utils import QueryAnalysisUtils
from .query_understanding_engine import QueryUnderstandingEngine

logger = structlog.get_logger(__name__)


class QueryAnalyzer:
    """
    Handles query understanding and analysis for the planner agent.
    
    Responsibilities:
    - Understanding user queries using QueryUnderstandingEngine
    - Analyzing task complexity
    - Performing LLM-based enhanced analysis
    - Converting understanding to standardized formats
    """
    
    def __init__(self, llm_client, rate_limiter=None):
        """
        Initialize the QueryAnalyzer.
        
        Args:
            llm_client: LLM client for query understanding
            rate_limiter: Rate limiter for LLM API calls
        """
        self.llm_utils = LLMUtils(llm_client, rate_limiter)
        self.query_understanding_engine = QueryUnderstandingEngine(
            llm_client, rate_limiter=rate_limiter
        )
        self.query_utils = QueryAnalysisUtils()
        self.logger = logger
        
        self.logger.info("QueryAnalyzer initialized")
    
    async def understand_user_query(self, user_query: str) -> QueryUnderstanding:
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
    
    def convert_understanding_to_entities(self, understanding: QueryUnderstanding) -> Dict[str, Any]:
        """
        Convert QueryUnderstanding object to the entities structure expected by agents.
        
        Args:
            understanding: QueryUnderstanding object from query understanding engine
            
        Returns:
            Entities dictionary in standard format
        """
        entities = {
            "musical_entities": {
                "artists": {
                    "primary": understanding.artists,
                    "similar_to": []
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
            "extraction_method": "query_understanding_engine",
            "intent_analysis": {
                "intent": understanding.intent.value,
                "confidence": understanding.confidence,
                "reasoning": understanding.reasoning
            }
        }
        
        self.logger.debug(
            "Converted QueryUnderstanding to entities",
            artists_count=len(understanding.artists),
            genres_count=len(understanding.genres),
            moods_count=len(understanding.moods),
            intent=understanding.intent.value
        )
        
        return entities
    
    async def analyze_task_complexity(
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
            
            # Fix: Include the actual intent information that discovery agent needs
            task_analysis['intent'] = understanding.intent.value
            task_analysis['query_understanding'] = understanding
            
            self.logger.debug(
                "Task complexity analyzed",
                complexity_level=task_analysis['complexity_level'],
                intent_complexity=task_analysis['intent_complexity'],
                entity_complexity=task_analysis['entity_complexity'],
                intent=understanding.intent.value
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
    
    def _assess_intent_complexity(self, understanding: QueryUnderstanding) -> str:
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
    
    def _assess_entity_complexity(self, understanding: QueryUnderstanding) -> str:
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