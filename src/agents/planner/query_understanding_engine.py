"""
Query Understanding Engine for Planner Agent

Moved from root agents directory and simplified to use shared components.
Eliminates duplicate LLM and JSON parsing logic.
"""

import json
from typing import Dict, Any, Optional
import structlog

from ...models.agent_models import MusicRecommenderState, QueryIntent, SimilarityType, QueryUnderstanding
from ..components.llm_utils import LLMUtils
from ..components.entity_extraction_utils import EntityExtractionUtils
from ..components.query_analysis_utils import QueryAnalysisUtils

logger = structlog.get_logger(__name__)


class QueryUnderstandingEngine:
    """
    Simplified query understanding engine using shared components.
    
    Eliminates duplicate LLM calling and JSON parsing patterns by using:
    - LLMUtils for all LLM interactions
    - EntityExtractionUtils for entity extraction
    - QueryAnalysisUtils for query analysis
    """
    
    def __init__(self, llm_client):
        """
        Initialize query understanding engine.
        
        Args:
            llm_client: LLM client for query understanding
        """
        self.llm_utils = LLMUtils(llm_client)
        self.entity_utils = EntityExtractionUtils()
        self.query_utils = QueryAnalysisUtils()
        self.logger = logger.bind(component="QueryUnderstandingEngine")
        
        # System prompt for LLM-based understanding
        self.system_prompt = """You are an expert music recommendation query analyzer. 
Your task is to understand user queries about music and extract structured information.

Analyze the query and return a JSON object with the following structure:
{
    "intent": "discovery|similarity|mood_based|activity_based|genre_specific",
    "similarity_type": "exact|moderate|loose|null",
    "musical_entities": {
        "artists": {
            "primary": ["artist1", "artist2"],
            "similar_to": ["artist3", "artist4"]
        },
        "genres": {
            "primary": ["genre1", "genre2"],
            "secondary": ["genre3"]
        },
        "tracks": {
            "primary": ["track1", "track2"]
        },
        "moods": {
            "primary": ["mood1", "mood2"]
        }
    },
    "context_factors": ["factor1", "factor2"],
    "complexity_level": "simple|medium|complex",
    "confidence": 0.0-1.0
}

Focus on extracting clear, actionable information for music recommendation."""
    
    async def understand_query(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> QueryUnderstanding:
        """
        Understand user query using hybrid approach with shared components.
        
        Args:
            query: User's music query
            conversation_context: Optional conversation context
            
        Returns:
            QueryUnderstanding object with extracted information
        """
        self.logger.info("Starting query understanding", query_length=len(query))
        
        try:
            # Phase 1: Pattern-based analysis using shared utilities
            pattern_analysis = self._pattern_based_analysis(query)
            
            # Phase 2: LLM-based understanding for complex queries
            if pattern_analysis['complexity_level'] in ['medium', 'complex']:
                llm_analysis = await self._llm_based_understanding(query)
                # Merge pattern and LLM analysis
                final_analysis = self._merge_analyses(pattern_analysis, llm_analysis)
            else:
                final_analysis = pattern_analysis
            
            # Phase 3: Validate and enhance with shared utilities
            final_analysis = self.entity_utils.validate_and_enhance_entities(
                final_analysis, query
            )
            
            # Convert to QueryUnderstanding object
            understanding = self._convert_to_understanding(final_analysis, query)
            
            self.logger.info(
                "Query understanding completed",
                intent=understanding.intent.value,
                confidence=understanding.confidence,
                entity_count=len(understanding.artists)
            )
            
            return understanding
            
        except Exception as e:
            self.logger.error("Query understanding failed", error=str(e))
            # Return fallback understanding
            return self._create_fallback_understanding(query)
    
    def _pattern_based_analysis(self, query: str) -> Dict[str, Any]:
        """Use shared utilities for pattern-based analysis."""
        # Use shared query analysis utilities
        comprehensive_analysis = self.query_utils.create_comprehensive_analysis(query)
        
        # Extract entities using shared utilities - fix the entity extraction
        try:
            entities = self.entity_utils.validate_and_enhance_entities({}, query)
        except Exception as e:
            self.logger.warning("Entity extraction failed, using fallback", error=str(e))
            entities = {"musical_entities": {"artists": {"primary": []}, "genres": {"primary": []}, "tracks": {"primary": []}, "moods": {"primary": []}}}
        
        # Extract similarity indicators
        try:
            similarity_info = self.entity_utils.extract_similarity_indicators(query)
        except Exception as e:
            self.logger.warning("Similarity extraction failed, using fallback", error=str(e))
            similarity_info = {"similarity_type": None}
        
        # Combine all analyses
        pattern_analysis = {
            'intent': comprehensive_analysis['intent_analysis']['primary_intent'],
            'similarity_type': similarity_info.get('similarity_type'),
            'musical_entities': entities.get('musical_entities', {}),
            'context_factors': comprehensive_analysis['context_factors'],
            'complexity_level': comprehensive_analysis['complexity_analysis']['complexity_level'],
            'confidence': 0.7,  # Base confidence for pattern matching
            'mood_indicators': comprehensive_analysis['mood_indicators'],
            'genre_hints': comprehensive_analysis['genre_hints']
        }
        
        return pattern_analysis
    
    async def _llm_based_understanding(self, query: str) -> Dict[str, Any]:
        """Use shared LLM utilities for comprehensive understanding."""
        user_prompt = f"""Analyze this music query and return the structured JSON response:

Query: "{query}"

Remember to return ONLY the JSON object with no additional text."""
        
        try:
            # Use shared LLM utilities with JSON parsing
            llm_data = await self.llm_utils.call_llm_with_json_response(
                user_prompt=user_prompt,
                system_prompt=self.system_prompt,
                max_retries=2
            )
            
            # Validate JSON structure using shared utilities
            required_keys = ['intent', 'musical_entities', 'context_factors', 'complexity_level']
            optional_keys = ['similarity_type', 'confidence']
            
            validated_data = self.llm_utils.validate_json_structure(
                llm_data, required_keys, optional_keys
            )
            
            return validated_data
            
        except Exception as e:
            self.logger.warning("LLM understanding failed", error=str(e))
            raise e
    
    def _merge_analyses(
        self, pattern_analysis: Dict[str, Any], llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge pattern-based and LLM-based analyses."""
        merged = pattern_analysis.copy()
        
        # Use LLM intent if it has higher confidence
        llm_confidence = llm_analysis.get('confidence', 0.5)
        if llm_confidence > merged.get('confidence', 0.0):
            merged['intent'] = llm_analysis.get('intent', merged['intent'])
            merged['confidence'] = llm_confidence
        
        # Merge musical entities
        llm_entities = llm_analysis.get('musical_entities', {})
        pattern_entities = merged.get('musical_entities', {})
        
        for entity_type in ['artists', 'genres', 'tracks', 'moods']:
            if entity_type in llm_entities:
                if entity_type not in pattern_entities:
                    pattern_entities[entity_type] = {}
                
                for category in ['primary', 'secondary', 'similar_to']:
                    if category in llm_entities[entity_type]:
                        # Combine and deduplicate
                        existing = pattern_entities[entity_type].get(category, [])
                        new_items = llm_entities[entity_type][category]
                        combined = list(dict.fromkeys(existing + new_items))
                        pattern_entities[entity_type][category] = combined
        
        # Merge context factors
        llm_context = llm_analysis.get('context_factors', [])
        pattern_context = merged.get('context_factors', [])
        merged['context_factors'] = list(dict.fromkeys(pattern_context + llm_context))
        
        # Use LLM similarity type if available
        if llm_analysis.get('similarity_type'):
            merged['similarity_type'] = llm_analysis['similarity_type']
        
        return merged
    
    def _convert_to_understanding(
        self, analysis: Dict[str, Any], original_query: str
    ) -> QueryUnderstanding:
        """Convert analysis to QueryUnderstanding object."""
        try:
            # Ensure original_query is a string
            if isinstance(original_query, dict):
                original_query = original_query.get('query', str(original_query))
            elif not isinstance(original_query, str):
                original_query = str(original_query)
            
            # Extract and validate intent
            intent_str = analysis.get('intent', 'discovery')
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                self.logger.warning("Invalid intent", intent=intent_str)
                intent = QueryIntent.DISCOVERY
            
            # Extract similarity type if present
            similarity_type = None
            if analysis.get('similarity_type'):
                try:
                    similarity_type = SimilarityType(analysis['similarity_type'])
                except ValueError:
                    self.logger.warning("Invalid similarity_type", similarity_type=analysis['similarity_type'])
            
            # Extract musical entities and convert to separate lists
            musical_entities = analysis.get('musical_entities', {})
            
            # Helper function to extract names from entity lists
            def extract_names(entity_list):
                """Extract names from entity list that may contain dicts or strings."""
                names = []
                if isinstance(entity_list, list):
                    for item in entity_list:
                        if isinstance(item, dict):
                            # Handle confidence score format: {'name': 'Artist', 'confidence': 0.8}
                            names.append(item.get('name', str(item)))
                        elif isinstance(item, str):
                            names.append(item)
                        else:
                            names.append(str(item))
                return names
            
            # Extract artists from musical entities
            artists = []
            if 'artists' in musical_entities:
                artists_data = musical_entities['artists']
                if isinstance(artists_data, dict):
                    artists.extend(extract_names(artists_data.get('primary', [])))
                    artists.extend(extract_names(artists_data.get('similar_to', [])))
                elif isinstance(artists_data, list):
                    artists.extend(extract_names(artists_data))
            
            # Extract genres from musical entities
            genres = []
            if 'genres' in musical_entities:
                genres_data = musical_entities['genres']
                if isinstance(genres_data, dict):
                    genres.extend(extract_names(genres_data.get('primary', [])))
                    genres.extend(extract_names(genres_data.get('secondary', [])))
                elif isinstance(genres_data, list):
                    genres.extend(extract_names(genres_data))
            
            # Extract moods from musical entities
            moods = []
            if 'moods' in musical_entities:
                moods_data = musical_entities['moods']
                if isinstance(moods_data, dict):
                    moods.extend(extract_names(moods_data.get('primary', [])))
                    # Also check other mood categories
                    moods.extend(extract_names(moods_data.get('energy', [])))
                    moods.extend(extract_names(moods_data.get('emotion', [])))
                elif isinstance(moods_data, list):
                    moods.extend(extract_names(moods_data))
            
            # Extract activities (if any)
            activities = []
            if 'activities' in musical_entities:
                activities_data = musical_entities['activities']
                if isinstance(activities_data, dict):
                    activities.extend(extract_names(activities_data.get('primary', [])))
                    activities.extend(extract_names(activities_data.get('physical', [])))
                    activities.extend(extract_names(activities_data.get('mental', [])))
                elif isinstance(activities_data, list):
                    activities.extend(extract_names(activities_data))
            
            # Also check contextual entities for activities
            contextual_entities = analysis.get('contextual_entities', {})
            if 'activities' in contextual_entities:
                activities_data = contextual_entities['activities']
                if isinstance(activities_data, dict):
                    activities.extend(extract_names(activities_data.get('physical', [])))
                    activities.extend(extract_names(activities_data.get('mental', [])))
                elif isinstance(activities_data, list):
                    activities.extend(extract_names(activities_data))
            
            # Extract confidence
            confidence = analysis.get('confidence', 0.5)
            if isinstance(confidence, dict):
                confidence = confidence.get('overall', 0.5)
            
            # Create QueryUnderstanding object with correct parameters
            understanding = QueryUnderstanding(
                intent=intent,
                confidence=confidence,
                artists=artists,
                genres=genres,
                moods=moods,
                activities=activities,
                similarity_type=similarity_type,
                original_query=original_query,
                normalized_query=original_query.lower().strip(),
                reasoning=f"Analysis completed with {confidence:.1%} confidence"
            )
            
            return understanding
            
        except Exception as e:
            self.logger.error("Failed to convert analysis to understanding", error=str(e))
            return self._create_fallback_understanding(original_query)
    
    def _create_fallback_understanding(self, query: str) -> QueryUnderstanding:
        """Create fallback understanding when analysis fails."""
        # Ensure query is a string
        if isinstance(query, dict):
            query = query.get('query', str(query))
        elif not isinstance(query, str):
            query = str(query)
        
        return QueryUnderstanding(
            intent=QueryIntent.DISCOVERY,
            confidence=0.3,
            artists=[],
            genres=[],
            moods=[],
            activities=[],
            similarity_type=None,
            original_query=query,
            normalized_query=query.lower(),
            reasoning="Fallback understanding due to processing error"
        )
    
    def get_understanding_summary(self, understanding: QueryUnderstanding) -> Dict[str, Any]:
        """Get summary of understanding for logging/debugging."""
        return {
            "intent": understanding.intent.value,
            "similarity_type": understanding.similarity_type.value if understanding.similarity_type else None,
            "confidence": understanding.confidence,
            "complexity": getattr(understanding, 'complexity_level', 'unknown'),
            "has_artists": bool(understanding.artists),
            "has_genres": bool(understanding.genres),
            "has_moods": bool(understanding.moods),
            "has_activities": bool(understanding.activities)
        } 