"""
Simplified Entity Recognizer for Planner Agent

Uses shared components to eliminate duplication and provide focused
entity recognition for planning purposes.
"""

from typing import Dict, Any, Optional
import structlog

from ..components.llm_utils import LLMUtils
from ..components.entity_extraction_utils import EntityExtractionUtils
from ..components.query_analysis_utils import QueryAnalysisUtils

logger = structlog.get_logger(__name__)


class EntityRecognizer:
    """
    Simplified entity recognizer that leverages shared components.
    
    Focused on entity extraction for planning purposes, using:
    - Shared LLM utilities for consistent LLM interactions
    - Shared entity extraction utilities for pattern-based extraction
    - Shared query analysis utilities for context understanding
    """
    
    def __init__(self, llm_client):
        """
        Initialize entity recognizer with shared components.
        
        Args:
            llm_client: LLM client for entity extraction
        """
        self.llm_utils = LLMUtils(llm_client)
        self.entity_utils = EntityExtractionUtils()
        self.query_utils = QueryAnalysisUtils()
        self.logger = logger.bind(component="PlannerEntityRecognizer")
        
        self.logger.info("Simplified Entity Recognizer initialized")
    
    async def extract_entities(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract entities using shared components with fallback strategy.
        
        Args:
            query: User's music request
            conversation_context: Previous conversation context
            
        Returns:
            Extracted entities with confidence scores
        """
        try:
            # Always try LLM-based extraction first (most accurate)
            self.logger.debug("Using LLM extraction for entity extraction")
            try:
                llm_entities = await self._llm_entity_extraction(query, conversation_context)
                final_entities = llm_entities
                
                # Add extraction metadata
                final_entities['extraction_method'] = 'llm_primary'
                
            except Exception as llm_error:
                self.logger.warning("LLM extraction failed, falling back to pattern-based", error=str(llm_error))
                # Fallback to pattern-based extraction
                pattern_entities = self.entity_utils.validate_and_enhance_entities({}, query)
                final_entities = pattern_entities
                final_entities['extraction_method'] = 'pattern_fallback'
            
            query_complexity = self.query_utils.analyze_query_complexity(query)
            final_entities['complexity_level'] = query_complexity['complexity_level']
            
            self.logger.debug(
                "Entity extraction completed",
                method=final_entities['extraction_method'],
                complexity=query_complexity['complexity_level'],
                total_entities=self._count_entities(final_entities)
            )
            
            return final_entities
            
        except Exception as e:
            self.logger.warning("Entity extraction failed, using fallback", error=str(e))
            return self._create_fallback_entities(query)
    
    async def _llm_entity_extraction(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Extract entities using shared LLM utilities."""
        system_prompt = """Extract musical entities from the user query.
        
Return JSON with this structure:
{
    "musical_entities": {
        "artists": {"primary": [], "similar_to": []},
        "genres": {"primary": [], "secondary": []},
        "tracks": {"primary": [], "referenced": []},
        "albums": {"primary": []}
    },
    "contextual_entities": {
        "moods": {"energy": [], "emotion": []},
        "activities": {"physical": [], "mental": []},
        "temporal": {"decades": [], "periods": []}
    },
    "conversation_entities": {
        "session_references": [],
        "similarity_requests": []
    },
    "confidence_scores": {
        "overall": 0.0
    }
}"""
        
        context_info = ""
        if conversation_context:
            recent_tracks = conversation_context.get('recommendation_history', [])
            if recent_tracks:
                last_track = recent_tracks[-1].get('tracks', [{}])[0]
                context_info = f"\nRecent context: {last_track.get('name', 'Unknown')} by {last_track.get('artist', 'Unknown')}"
        
        user_prompt = f"""Query: "{query}"{context_info}
        
Extract entities following the JSON format above. Focus on:
- Musical entities (artists, genres, tracks, albums)
- Context (moods, activities, time periods)
- Conversation references (session references, similarity requests)
"""
        
        try:
            llm_data = await self.llm_utils.call_llm_with_json_response(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_retries=2
            )
            
            # Validate structure using shared utilities
            required_keys = ['musical_entities', 'contextual_entities']
            optional_keys = ['conversation_entities', 'confidence_scores']
            
            validated_data = self.llm_utils.validate_json_structure(
                llm_data, required_keys, optional_keys
            )
            
            return validated_data
            
        except Exception as e:
            self.logger.error("LLM entity extraction failed", error=str(e))
            raise
    
    def _merge_entity_results(
        self, 
        pattern_entities: Dict[str, Any], 
        llm_entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge pattern-based and LLM-based entity extraction results."""
        merged = pattern_entities.copy()
        
        # Merge musical entities
        if 'musical_entities' in llm_entities:
            pattern_musical = merged.get('musical_entities', {})
            llm_musical = llm_entities['musical_entities']
            
            for entity_type in ['artists', 'genres', 'tracks', 'albums']:
                if entity_type in llm_musical:
                    if entity_type not in pattern_musical:
                        pattern_musical[entity_type] = {}
                    
                    for category in llm_musical[entity_type]:
                        if category in pattern_musical[entity_type]:
                            # Combine and deduplicate
                            existing = pattern_musical[entity_type][category]
                            new_items = llm_musical[entity_type][category]
                            combined = list(dict.fromkeys(existing + new_items))
                            pattern_musical[entity_type][category] = combined
                        else:
                            pattern_musical[entity_type][category] = llm_musical[entity_type][category]
        
        # Merge contextual entities
        if 'contextual_entities' in llm_entities:
            pattern_contextual = merged.get('contextual_entities', {})
            llm_contextual = llm_entities['contextual_entities']
            
            for context_type in ['moods', 'activities', 'temporal']:
                if context_type in llm_contextual:
                    if context_type not in pattern_contextual:
                        pattern_contextual[context_type] = {}
                    
                    for category in llm_contextual[context_type]:
                        if category in pattern_contextual[context_type]:
                            existing = pattern_contextual[context_type][category]
                            new_items = llm_contextual[context_type][category]
                            combined = list(dict.fromkeys(existing + new_items))
                            pattern_contextual[context_type][category] = combined
                        else:
                            pattern_contextual[context_type][category] = llm_contextual[context_type][category]
        
        # Add conversation entities from LLM
        if 'conversation_entities' in llm_entities:
            merged['conversation_entities'] = llm_entities['conversation_entities']
        
        # Use higher confidence score
        pattern_confidence = merged.get('confidence_scores', {}).get('overall', 0.5)
        llm_confidence = llm_entities.get('confidence_scores', {}).get('overall', 0.5)
        
        if 'confidence_scores' not in merged:
            merged['confidence_scores'] = {}
        merged['confidence_scores']['overall'] = max(pattern_confidence, llm_confidence)
        
        return merged
    
    def _count_entities(self, entities: Dict[str, Any]) -> int:
        """Count total number of extracted entities."""
        count = 0
        
        # Count musical entities
        musical = entities.get('musical_entities', {})
        for entity_type in ['artists', 'genres', 'tracks', 'albums']:
            if entity_type in musical:
                for category in musical[entity_type]:
                    if isinstance(musical[entity_type][category], list):
                        count += len(musical[entity_type][category])
        
        # Count contextual entities
        contextual = entities.get('contextual_entities', {})
        for context_type in ['moods', 'activities', 'temporal']:
            if context_type in contextual:
                for category in contextual[context_type]:
                    if isinstance(contextual[context_type][category], list):
                        count += len(contextual[context_type][category])
        
        return count
    
    def _create_fallback_entities(self, query: str) -> Dict[str, Any]:
        """Create fallback entities when extraction fails."""
        # Use shared entity extraction utilities for fallback
        fallback_entities = self.entity_utils.validate_and_enhance_entities({}, query)
        
        # Ensure minimum structure
        if 'confidence_scores' not in fallback_entities:
            fallback_entities['confidence_scores'] = {'overall': 0.3}
        
        fallback_entities['extraction_method'] = 'fallback'
        
        self.logger.info("Created fallback entities", entity_count=self._count_entities(fallback_entities))
        
        return fallback_entities 