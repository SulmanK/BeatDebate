"""
Enhanced Entity Recognition System for BeatDebate

LLM-powered entity recognition with comprehensive fallback mechanisms
for extracting musical, contextual, preference, and conversational entities.
"""

import json
import re
from typing import Dict, List, Any, Optional
import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)


class EnhancedEntityRecognizer:
    """
    LLM-powered entity recognition with fallback mechanisms.
    
    Extracts comprehensive entities from user queries including:
    - Musical entities (artists, tracks, albums, genres)
    - Contextual entities (moods, activities, temporal)
    - Preference entities (similarity requests, discovery preferences)
    - Conversation entities (session references, preference evolution)
    """
    
    def __init__(self, gemini_client):
        """
        Initialize entity recognizer with LLM client.
        
        Args:
            gemini_client: Gemini LLM client for entity extraction
        """
        self.llm_client = gemini_client
        self.entity_cache = {}
        self.fallback_extractor = FallbackEntityExtractor()
        self.logger = logger.bind(component="EntityRecognizer")
        
        self.logger.info("Enhanced Entity Recognizer initialized")
    
    async def extract_entities(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract comprehensive entities from user query with conversation context.
        
        Args:
            query: User's music request
            conversation_context: Previous conversation context
            
        Returns:
            Extracted entities with confidence scores
        """
        try:
            # Phase 2: Enhanced context-aware extraction
            entities = await self._extract_entities_with_context(query, conversation_context)
            
            # Phase 2: Detect and analyze contextual modifications
            contextual_modifications = await self._detect_contextual_modifications(query, entities)
            if contextual_modifications:
                entities["contextual_modifications"] = contextual_modifications
            
            # Phase 2: Analyze style modifications
            style_modifications = await self._analyze_style_modifications(query, entities)
            if style_modifications:
                entities["style_modifications"] = style_modifications
            
            # Validate and enhance entities
            entities = await self._validate_and_enhance_entities(entities, query)
            
            # Add extraction metadata
            entities["extraction_method"] = "llm_enhanced"
            
            return entities
            
        except Exception as e:
            self.logger.warning("Enhanced entity extraction failed, using fallback", error=str(e))
            return self._fallback_entity_extraction(query)

    async def extract_entities_optimized(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Phase 3: Optimized entity extraction with confidence-based method selection.
        
        Args:
            query: User's music request
            conversation_context: Previous conversation context
            confidence_threshold: Minimum confidence for detailed extraction
            
        Returns:
            Extracted entities with confidence scores
        """
        try:
            # Phase 3: Determine extraction complexity needed
            query_complexity = self._assess_query_complexity(query)
            
            # Phase 3: Choose extraction method based on complexity and confidence needs
            if query_complexity["complexity_score"] > 0.7 or query_complexity["requires_context"]:
                # Use full detailed extraction for complex queries
                self.logger.debug("Using detailed extraction for complex query")
                return await self.extract_entities(query, conversation_context)
            else:
                # Use optimized extraction for simple queries
                self.logger.debug("Using optimized extraction for simple query")
                return await self._extract_entities_optimized(query, conversation_context)
                
        except Exception as e:
            self.logger.warning("Optimized entity extraction failed, using fallback", error=str(e))
            return self._fallback_entity_extraction(query)

    def _assess_query_complexity(self, query: str) -> Dict[str, Any]:
        """Phase 3: Assess query complexity to determine extraction method."""
        query_lower = query.lower()
        
        complexity_indicators = {
            "multi_faceted": ["but", "and", "with", "for", "during"],
            "conversational": ["like the last", "that song", "that artist", "before"],
            "style_modification": ["jazzier", "more upbeat", "less heavy", "but more"],
            "contextual": ["for working out", "while studying", "when driving"]
        }
        
        detected_patterns = []
        for pattern_type, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                detected_patterns.append(pattern_type)
        
        complexity_score = len(detected_patterns) / len(complexity_indicators)
        requires_context = "conversational" in detected_patterns
        
        return {
            "complexity_score": complexity_score,
            "detected_patterns": detected_patterns,
            "requires_context": requires_context,
            "extraction_method": "detailed" if complexity_score > 0.5 else "optimized"
        }

    async def _extract_entities_optimized(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Phase 3: Optimized entity extraction for simple queries."""
        # Use optimized prompts
        system_prompt = self._create_optimized_system_prompt()
        user_prompt = self._create_optimized_extraction_prompt(query, conversation_context)
        
        try:
            response = await self._make_llm_call(user_prompt, system_prompt)
            entities = self._parse_json_response(response)
            
            # Phase 3: Add enhanced confidence scoring
            entities = self._add_enhanced_confidence_scores(entities, query)
            entities["extraction_method"] = "llm_optimized"
            
            return entities
            
        except Exception as e:
            self.logger.error("Optimized LLM entity extraction failed", error=str(e))
            raise

    async def _extract_entities_with_context(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract entities using LLM with enhanced conversation context awareness.
        
        Args:
            query: User's music request
            conversation_context: Previous conversation context
            
        Returns:
            Extracted entities with context awareness
        """
        # Create context-aware prompt
        system_prompt = self._create_enhanced_system_prompt()
        user_prompt = await self._create_context_aware_extraction_prompt(query, conversation_context)
        
        try:
            response = await self._make_llm_call(user_prompt, system_prompt)
            entities = self._parse_json_response(response)
            
            # Phase 2: Post-process entities with context
            entities = await self._post_process_entities_with_context(entities, conversation_context)
            
            return entities
            
        except Exception as e:
            self.logger.error("LLM entity extraction failed", error=str(e))
            # Fall back to basic extraction
            return self._fallback_entity_extraction(query)

    def _create_enhanced_system_prompt(self) -> str:
        """Create enhanced system prompt for Phase 2 entity recognition."""
        return """
You are an advanced music entity recognition system with conversation context awareness.

ENHANCED CAPABILITIES (Phase 2):
1. Contextual Modification Detection: Identify "but jazzier", "more upbeat", "for working out"
2. Session Reference Resolution: Handle "like the last song", "that artist"
3. Style Modification Analysis: Detect genre fusion, energy adjustments, mood shifts
4. Multi-faceted Query Decomposition: Parse complex queries with multiple intents

ENTITY CATEGORIES:
1. Musical: artists, tracks, albums, genres (with modification indicators)
2. Contextual: moods, activities, temporal references
3. Preferences: similarity requests, discovery preferences, quality preferences
4. Conversational: session references, preference evolution, conversation flow

CONTEXT PROCESSING:
- Use conversation history to resolve ambiguous references
- Track preference evolution across interactions
- Identify multi-faceted intents and contextual modifications

OUTPUT: Structured JSON with confidence scores, relationships, and modification indicators.
"""

    def _create_optimized_system_prompt(self) -> str:
        """Phase 3: Optimized system prompt for better efficiency and accuracy."""
        return """Extract music entities from user queries. Focus on:

ENTITIES:
- Musical: artists, genres, tracks, albums
- Context: mood, activity, time
- Preferences: similarity, discovery, style
- References: "last song", "that artist"

MODIFICATIONS:
- Style: "jazzier", "more upbeat" 
- Context: "for workout", "while studying"
- Session: "like before", "similar to earlier"

OUTPUT: JSON with entities, confidence scores, relationships.
Be concise but comprehensive."""

    async def _create_context_aware_extraction_prompt(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> str:
        """Create context-aware extraction prompt for entity recognition."""
        if conversation_context:
            context_section = f"""
CONVERSATION CONTEXT:
Previous Interactions: {len(conversation_context.get('interaction_history', []))}
Last Recommendation: {conversation_context.get('recommendation_history', [{}])[-1] if conversation_context.get('recommendation_history') else 'None'}
User Preferences: {conversation_context.get('preference_profile', {})}
"""
        else:
            context_section = "CONVERSATION CONTEXT: None (first interaction)"

        return f"""
QUERY: "{query}"

{context_section}

Extract entities in this JSON format:
{{
    "musical_entities": {{
        "artists": {{"primary": [], "similar_to": [], "avoid": []}},
        "tracks": {{"specific": [], "referenced": [], "style_reference": []}},
        "albums": {{"specific": [], "era": [], "type": []}},
        "genres": {{"primary": [], "sub_genres": [], "fusion": [], "avoid": []}}
    }},
    "contextual_entities": {{
        "moods": {{"energy": [], "emotion": [], "atmosphere": []}},
        "activities": {{"physical": [], "mental": [], "social": [], "temporal": []}},
        "temporal": {{"decades": [], "eras": [], "periods": []}}
    }},
    "preference_entities": {{
        "similarity_requests": [],
        "discovery_preferences": [],
        "quality_preferences": []
    }},
    "conversation_entities": {{
        "session_references": [],
        "preference_evolution": [],
        "conversation_flow": []
    }},
    "confidence_scores": {{
        "overall": 0.0-1.0,
        "entity_specific": {{"entity_name": 0.0-1.0}}
    }},
    "relationships": [
        {{"source": "entity1", "target": "entity2", "relationship": "similar_to"}}
    ]
}}
"""

    def _create_optimized_extraction_prompt(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> str:
        """Phase 3: Optimized extraction prompt for better efficiency."""
        context_info = ""
        if conversation_context:
            recent_tracks = conversation_context.get('recommendation_history', [])
            if recent_tracks:
                context_info = f"Recent: {recent_tracks[-1].get('tracks', [{}])[0].get('name', 'Unknown')} by {recent_tracks[-1].get('tracks', [{}])[0].get('artist', 'Unknown')}"
        
        return f"""QUERY: "{query}"
{f"CONTEXT: {context_info}" if context_info else ""}

Extract entities as JSON:
{{
    "musical_entities": {{"artists": {{"primary": [], "similar_to": []}}, "genres": {{"primary": []}}}},
    "contextual_entities": {{"moods": {{"energy": []}}, "activities": {{"physical": [], "mental": []}}}},
    "conversation_entities": {{"session_references": []}},
    "confidence_scores": {{"overall": 0.0}}
}}

Focus on key entities. Be precise and concise."""

    async def _detect_contextual_modifications(
        self, 
        query: str, 
        entities: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect contextual modifications in the query.
        
        Args:
            query: User's music request
            entities: Extracted entities
            
        Returns:
            Contextual modification analysis
        """
        query_lower = query.lower()
        
        # Detect modification patterns
        modification_patterns = {
            "activity_context": {
                "patterns": ["for working out", "while studying", "for the gym", "during", "when"],
                "activities": []
            },
            "energy_adjustment": {
                "patterns": ["more upbeat", "more energetic", "calmer", "mellower", "higher energy"],
                "adjustments": []
            },
            "mood_shift": {
                "patterns": ["happier", "sadder", "darker", "brighter", "more emotional"],
                "shifts": []
            },
            "genre_fusion": {
                "patterns": ["but jazzier", "more electronic", "with rock elements", "acoustic version"],
                "fusions": []
            }
        }
        
        detected_modifications = {}
        
        for mod_type, config in modification_patterns.items():
            for pattern in config["patterns"]:
                if pattern in query_lower:
                    if mod_type not in detected_modifications:
                        detected_modifications[mod_type] = []
                    
                    detected_modifications[mod_type].append({
                        "pattern": pattern,
                        "context": self._extract_modification_context(query, pattern),
                        "intensity": self._estimate_modification_intensity(pattern)
                    })
        
        if detected_modifications:
            return {
                "modifications": detected_modifications,
                "modification_count": sum(len(mods) for mods in detected_modifications.values()),
                "primary_modification": max(detected_modifications.keys(), key=lambda k: len(detected_modifications[k])),
                "confidence": min(len(detected_modifications) / 2.0, 1.0)
            }
        
        return None

    async def _analyze_style_modifications(
        self, 
        query: str, 
        entities: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze style modification requests in the query.
        
        Args:
            query: User's music request
            entities: Extracted entities
            
        Returns:
            Style modification analysis
        """
        query_lower = query.lower()
        
        style_indicators = {
            "genre_shift": {
                "jazz": ["jazzier", "with jazz elements", "jazz-influenced"],
                "rock": ["rockier", "more rock", "rock version"],
                "electronic": ["more electronic", "electronic version", "with synths"],
                "acoustic": ["acoustic", "unplugged", "stripped down"]
            },
            "energy_modification": {
                "increase": ["more upbeat", "more energetic", "higher energy", "more intense"],
                "decrease": ["calmer", "mellower", "more chill", "less intense", "quieter"]
            },
            "production_style": {
                "polished": ["more polished", "studio quality", "well-produced"],
                "raw": ["rawer", "more raw", "lo-fi", "unpolished"],
                "ambient": ["more ambient", "atmospheric", "spacey"]
            }
        }
        
        detected_styles = {}
        
        for style_category, style_types in style_indicators.items():
            for style_type, indicators in style_types.items():
                for indicator in indicators:
                    if indicator in query_lower:
                        if style_category not in detected_styles:
                            detected_styles[style_category] = {}
                        
                        detected_styles[style_category][style_type] = {
                            "indicator": indicator,
                            "confidence": self._calculate_style_confidence(indicator, query),
                            "modification_type": self._classify_modification_type(indicator)
                        }
        
        if detected_styles:
            return {
                "style_modifications": detected_styles,
                "modification_complexity": len(detected_styles),
                "primary_style_category": max(detected_styles.keys(), key=lambda k: len(detected_styles[k])),
                "overall_confidence": sum(
                    mod["confidence"] for category in detected_styles.values() 
                    for mod in category.values()
                ) / sum(len(category) for category in detected_styles.values())
            }
        
        return None

    async def _post_process_entities_with_context(
        self, 
        entities: Dict[str, Any], 
        conversation_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Post-process entities with conversation context for Phase 2 enhancements.
        
        Args:
            entities: Raw extracted entities
            conversation_context: Conversation context
            
        Returns:
            Enhanced entities with context processing
        """
        if not conversation_context:
            return entities
        
        # Enhance session references with context
        session_refs = entities.get("conversation_entities", {}).get("session_references", [])
        if session_refs:
            enhanced_refs = []
            for ref in session_refs:
                enhanced_ref = await self._enhance_session_reference(ref, conversation_context)
                enhanced_refs.append(enhanced_ref)
            entities["conversation_entities"]["session_references"] = enhanced_refs
        
        # Add contextual continuity indicators
        if "conversation_entities" not in entities:
            entities["conversation_entities"] = {}
        
        entities["conversation_entities"]["contextual_continuity"] = await self._analyze_contextual_continuity(
            entities, conversation_context
        )
        
        return entities

    async def _enhance_session_reference(
        self, 
        reference: Dict[str, Any], 
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance session reference with conversation context."""
        enhanced_ref = reference.copy()
        
        # Add context from previous interactions
        interaction_history = conversation_context.get("interaction_history", [])
        if interaction_history:
            last_interaction = interaction_history[-1]
            enhanced_ref["context"] = {
                "last_query": last_interaction.get("query", ""),
                "last_entities": last_interaction.get("extracted_entities", {}),
                "interaction_count": len(interaction_history)
            }
        
        return enhanced_ref

    async def _analyze_contextual_continuity(
        self, 
        entities: Dict[str, Any], 
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze contextual continuity across conversation turns."""
        continuity_analysis = {
            "genre_continuity": False,
            "mood_continuity": False,
            "activity_continuity": False,
            "preference_continuity": False,
            "continuity_score": 0.0
        }
        
        interaction_history = conversation_context.get("interaction_history", [])
        if not interaction_history:
            return continuity_analysis
        
        last_interaction = interaction_history[-1]
        last_entities = last_interaction.get("extracted_entities", {})
        
        # Check genre continuity
        current_genres = entities.get("musical_entities", {}).get("genres", {}).get("primary", [])
        last_genres = last_entities.get("musical_entities", {}).get("genres", {}).get("primary", [])
        if current_genres and last_genres:
            genre_overlap = len(set(current_genres) & set(last_genres)) / len(set(current_genres) | set(last_genres))
            continuity_analysis["genre_continuity"] = genre_overlap > 0.3
        
        # Check mood continuity
        current_moods = entities.get("contextual_entities", {}).get("moods", {}).get("energy", [])
        last_moods = last_entities.get("contextual_entities", {}).get("moods", {}).get("energy", [])
        if current_moods and last_moods:
            mood_overlap = len(set(current_moods) & set(last_moods)) / len(set(current_moods) | set(last_moods))
            continuity_analysis["mood_continuity"] = mood_overlap > 0.3
        
        # Calculate overall continuity score
        continuity_count = sum(1 for key, value in continuity_analysis.items() if key.endswith("_continuity") and value)
        total_continuity_checks = sum(1 for key in continuity_analysis.keys() if key.endswith("_continuity"))
        continuity_analysis["continuity_score"] = continuity_count / total_continuity_checks if total_continuity_checks > 0 else 0.0
        
        return continuity_analysis

    def _extract_modification_context(self, query: str, pattern: str) -> str:
        """Extract context around a modification pattern."""
        pattern_index = query.lower().find(pattern)
        if pattern_index == -1:
            return ""
        
        # Extract 20 characters before and after the pattern
        start = max(0, pattern_index - 20)
        end = min(len(query), pattern_index + len(pattern) + 20)
        return query[start:end].strip()

    def _estimate_modification_intensity(self, pattern: str) -> float:
        """Estimate the intensity of a modification pattern."""
        intensity_map = {
            "more": 0.7,
            "much more": 0.9,
            "slightly": 0.3,
            "way more": 1.0,
            "less": 0.3,
            "much less": 0.1
        }
        
        for intensity_word, score in intensity_map.items():
            if intensity_word in pattern.lower():
                return score
        
        return 0.5  # Default moderate intensity

    def _calculate_style_confidence(self, indicator: str, query: str) -> float:
        """Calculate confidence score for style modification detection."""
        # Base confidence on specificity and context
        base_confidence = 0.6
        
        # Increase confidence for specific indicators
        if "ier" in indicator:  # "jazzier", "rockier"
            base_confidence += 0.2
        
        # Increase confidence if surrounded by relevant context
        if any(word in query.lower() for word in ["but", "more", "with", "version"]):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)

    def _classify_modification_type(self, indicator: str) -> str:
        """Classify the type of style modification."""
        if "ier" in indicator:
            return "genre_fusion"
        elif "more" in indicator or "less" in indicator:
            return "intensity_adjustment"
        elif "version" in indicator:
            return "style_variant"
        else:
            return "general_modification"

    async def _validate_and_enhance_entities(
        self, 
        entities: Dict[str, Any], 
        query: str
    ) -> Dict[str, Any]:
        """
        Validate and enhance extracted entities.
        
        Args:
            entities: Extracted entities
            query: User's music request
            
        Returns:
            Enhanced entities
        """
        # Validate entity structure
        entities = self._validate_entity_structure(entities)
        
        # Enhance with confidence scores
        entities = self._add_confidence_scores(entities, query)
        
        return entities

    def _validate_entity_structure(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and ensure proper entity structure."""
        # Ensure all required top-level keys exist
        required_keys = [
            "musical_entities", "contextual_entities", 
            "preference_entities", "conversation_entities"
        ]
        
        for key in required_keys:
            if key not in entities:
                entities[key] = {}
        
        # Ensure musical_entities structure
        musical_defaults = {
            "artists": {"primary": [], "similar_to": [], "avoid": []},
            "tracks": {"specific": [], "referenced": [], "style_reference": []},
            "albums": {"specific": [], "era": [], "type": []},
            "genres": {"primary": [], "sub_genres": [], "fusion": [], "avoid": []}
        }
        
        for key, default_value in musical_defaults.items():
            if key not in entities["musical_entities"]:
                entities["musical_entities"][key] = default_value
            else:
                # Fix malformed data - if it's a list instead of dict, convert it
                current_value = entities["musical_entities"][key]
                if isinstance(current_value, list):
                    # Convert list to proper dict structure
                    entities["musical_entities"][key] = {
                        "primary": current_value,
                        "similar_to": [],
                        "avoid": []
                    }
                elif not isinstance(current_value, dict):
                    entities["musical_entities"][key] = default_value
        
        # Ensure contextual_entities structure
        contextual_defaults = {
            "moods": {"energy": [], "emotion": [], "atmosphere": []},
            "activities": {"physical": [], "mental": [], "social": [], "temporal": []},
            "temporal": {"decades": [], "eras": [], "periods": []}
        }
        
        for key, default_value in contextual_defaults.items():
            if key not in entities["contextual_entities"]:
                entities["contextual_entities"][key] = default_value
            else:
                # Fix malformed data - if it's a list instead of dict, convert it
                current_value = entities["contextual_entities"][key]
                if isinstance(current_value, list):
                    # Convert list to proper dict structure based on category
                    if key == "moods":
                        entities["contextual_entities"][key] = {
                            "energy": current_value,
                            "emotion": [],
                            "atmosphere": []
                        }
                    elif key == "activities":
                        entities["contextual_entities"][key] = {
                            "physical": current_value,
                            "mental": [],
                            "social": [],
                            "temporal": []
                        }
                    elif key == "temporal":
                        entities["contextual_entities"][key] = {
                            "decades": current_value,
                            "eras": [],
                            "periods": []
                        }
                elif not isinstance(current_value, dict):
                    entities["contextual_entities"][key] = default_value
        
        # Ensure other structures
        if "preference_entities" not in entities:
            entities["preference_entities"] = {
                "similarity_requests": [],
                "discovery_preferences": [],
                "quality_preferences": []
            }
        
        if "conversation_entities" not in entities:
            entities["conversation_entities"] = {
                "session_references": [],
                "preference_evolution": [],
                "conversation_flow": []
            }
        
        # Ensure confidence scores
        if "confidence_scores" not in entities:
            entities["confidence_scores"] = {"overall": 0.5, "entity_specific": {}}
        
        # Ensure relationships
        if "relationships" not in entities:
            entities["relationships"] = []
        
        return entities
    
    def _add_confidence_scores(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Add confidence scores to extracted entities."""
        # Calculate entity-specific confidence scores
        entity_specific_scores = {}
        
        # Musical entities confidence
        musical_entities = entities.get("musical_entities", {})
        if musical_entities.get("artists", {}).get("primary"):
            entity_specific_scores["artists"] = 0.8
        if musical_entities.get("genres", {}).get("primary"):
            entity_specific_scores["genres"] = 0.7
        
        # Contextual entities confidence
        contextual_entities = entities.get("contextual_entities", {})
        if contextual_entities.get("moods"):
            entity_specific_scores["moods"] = 0.6
        if contextual_entities.get("activities"):
            entity_specific_scores["activities"] = 0.7
        
        # Calculate overall confidence
        if entity_specific_scores:
            overall_confidence = sum(entity_specific_scores.values()) / len(entity_specific_scores)
        else:
            overall_confidence = 0.3  # Low confidence if no entities found
        
        entities["confidence_scores"] = {
            "overall": overall_confidence,
            "entity_specific": entity_specific_scores
        }
        
        return entities

    def _add_enhanced_confidence_scores(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Phase 3: Enhanced confidence scoring with multiple factors."""
        # Initialize confidence tracking
        confidence_factors = {
            "entity_count": 0.0,
            "entity_specificity": 0.0,
            "query_match": 0.0,
            "context_coherence": 0.0,
            "extraction_method": 0.0
        }
        
        # Factor 1: Entity count and richness
        entity_count = self._count_total_entities(entities)
        if entity_count > 0:
            confidence_factors["entity_count"] = min(1.0, entity_count / 5.0)  # Normalize to max 5 entities
        
        # Factor 2: Entity specificity (how specific vs generic)
        specificity_score = self._calculate_entity_specificity(entities, query)
        confidence_factors["entity_specificity"] = specificity_score
        
        # Factor 3: Query-entity match quality
        match_score = self._calculate_query_entity_match(entities, query)
        confidence_factors["query_match"] = match_score
        
        # Factor 4: Context coherence (do entities make sense together)
        coherence_score = self._calculate_context_coherence(entities)
        confidence_factors["context_coherence"] = coherence_score
        
        # Factor 5: Extraction method reliability
        extraction_method = entities.get("extraction_method", "fallback")
        method_confidence = {
            "llm_enhanced": 0.9,
            "llm_optimized": 0.8,
            "llm_basic": 0.7,
            "fallback": 0.4
        }
        confidence_factors["extraction_method"] = method_confidence.get(extraction_method, 0.4)
        
        # Calculate weighted overall confidence
        weights = {
            "entity_count": 0.15,
            "entity_specificity": 0.25,
            "query_match": 0.25,
            "context_coherence": 0.20,
            "extraction_method": 0.15
        }
        
        overall_confidence = sum(
            confidence_factors[factor] * weights[factor] 
            for factor in confidence_factors
        )
        
        # Calculate entity-specific confidence scores
        entity_specific_scores = self._calculate_entity_specific_confidence(entities, confidence_factors)
        
        entities["confidence_scores"] = {
            "overall": round(overall_confidence, 3),
            "entity_specific": entity_specific_scores,
            "confidence_factors": confidence_factors,
            "confidence_breakdown": {
                factor: round(score * weights[factor], 3) 
                for factor, score in confidence_factors.items()
            }
        }
        
        return entities

    def _count_total_entities(self, entities: Dict[str, Any]) -> int:
        """Count total number of extracted entities."""
        count = 0
        
        # Count musical entities
        musical = entities.get("musical_entities", {})
        for category in ["artists", "genres", "tracks", "albums"]:
            if category in musical:
                category_data = musical[category]
                # Handle case where category_data might be a list instead of dict
                if isinstance(category_data, dict):
                    for subcategory in category_data.values():
                        if isinstance(subcategory, list):
                            count += len(subcategory)
                elif isinstance(category_data, list):
                    count += len(category_data)
        
        # Count contextual entities
        contextual = entities.get("contextual_entities", {})
        for category in ["moods", "activities", "temporal"]:
            if category in contextual:
                category_data = contextual[category]
                # Handle case where category_data might be a list instead of dict
                if isinstance(category_data, dict):
                    for subcategory in category_data.values():
                        if isinstance(subcategory, list):
                            count += len(subcategory)
                elif isinstance(category_data, list):
                    count += len(category_data)
        
        return count

    def _calculate_entity_specificity(self, entities: Dict[str, Any], query: str) -> float:
        """Calculate how specific (vs generic) the extracted entities are."""
        specificity_score = 0.0
        entity_count = 0
        
        # Check musical entity specificity
        musical = entities.get("musical_entities", {})
        
        # Artists - specific names vs generic terms
        artists = musical.get("artists", {}).get("primary", [])
        for artist in artists:
            entity_count += 1
            if len(artist) > 3 and not artist.lower() in ["rock", "pop", "jazz", "classical"]:
                specificity_score += 1.0
            else:
                specificity_score += 0.3
        
        # Genres - specific subgenres vs broad categories
        genres = musical.get("genres", {}).get("primary", [])
        for genre in genres:
            entity_count += 1
            if any(specific in genre.lower() for specific in ["indie", "alternative", "progressive", "experimental"]):
                specificity_score += 0.8
            elif genre.lower() in ["rock", "pop", "jazz", "classical", "electronic"]:
                specificity_score += 0.5
            else:
                specificity_score += 0.7
        
        return specificity_score / entity_count if entity_count > 0 else 0.0

    def _calculate_query_entity_match(self, entities: Dict[str, Any], query: str) -> float:
        """Calculate how well extracted entities match the original query."""
        query_lower = query.lower()
        match_score = 0.0
        total_entities = 0
        
        # Check if extracted artists appear in query
        musical = entities.get("musical_entities", {})
        artists = musical.get("artists", {}).get("primary", [])
        for artist in artists:
            total_entities += 1
            if artist.lower() in query_lower:
                match_score += 1.0
            elif any(word in query_lower for word in artist.lower().split()):
                match_score += 0.7
            else:
                match_score += 0.3
        
        # Check if extracted genres appear in query
        genres = musical.get("genres", {}).get("primary", [])
        for genre in genres:
            total_entities += 1
            if genre.lower() in query_lower:
                match_score += 1.0
            elif any(word in query_lower for word in genre.lower().split()):
                match_score += 0.8
            else:
                match_score += 0.4
        
        return match_score / total_entities if total_entities > 0 else 0.0

    def _calculate_context_coherence(self, entities: Dict[str, Any]) -> float:
        """Calculate how coherent the extracted entities are together."""
        # Simple coherence check - could be enhanced with more sophisticated logic
        musical = entities.get("musical_entities", {})
        contextual = entities.get("contextual_entities", {})
        
        coherence_score = 0.5  # Base coherence
        
        # Check if genres and moods are coherent
        genres = musical.get("genres", {}).get("primary", [])
        moods = contextual.get("moods", {}).get("energy", [])
        
        # Example coherence rules (could be expanded)
        if "rock" in [g.lower() for g in genres] and "energetic" in [m.lower() for m in moods]:
            coherence_score += 0.3
        if "classical" in [g.lower() for g in genres] and "relaxing" in [m.lower() for m in moods]:
            coherence_score += 0.3
        if "jazz" in [g.lower() for g in genres] and "sophisticated" in [m.lower() for m in moods]:
            coherence_score += 0.3
        
        return min(1.0, coherence_score)

    def _calculate_entity_specific_confidence(
        self, 
        entities: Dict[str, Any], 
        confidence_factors: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate confidence scores for specific entity categories."""
        entity_specific = {}
        
        # Musical entities
        musical = entities.get("musical_entities", {})
        if musical.get("artists", {}).get("primary"):
            entity_specific["artists"] = min(1.0, confidence_factors["query_match"] + confidence_factors["entity_specificity"])
        if musical.get("genres", {}).get("primary"):
            entity_specific["genres"] = min(1.0, confidence_factors["query_match"] + 0.1)
        
        # Contextual entities
        contextual = entities.get("contextual_entities", {})
        if contextual.get("moods"):
            entity_specific["moods"] = min(1.0, confidence_factors["context_coherence"] + 0.2)
        if contextual.get("activities"):
            entity_specific["activities"] = min(1.0, confidence_factors["context_coherence"] + 0.1)
        
        return entity_specific

    async def _make_llm_call(self, prompt: str, system_prompt: str = None) -> str:
        """
        Make LLM call using Gemini client.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            LLM response
        """
        if not self.llm_client:
            raise RuntimeError("Gemini client not initialized")
        
        try:
            # Combine system and user prompts
            full_prompt = (
                f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            )
            
            # Call Gemini - handle both sync and async mocks
            response = self.llm_client.generate_content(full_prompt)
            
            # If it's a coroutine (async mock), await it
            if hasattr(response, '__await__'):
                response = await response
            
            return response.text
            
        except Exception as e:
            self.logger.error("Gemini API call failed", error=str(e))
            raise

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, handling common formatting issues."""
        try:
            # Clean up response - remove markdown formatting and explanatory text
            cleaned = response.strip()
            
            # Remove any text before the JSON starts
            json_start = cleaned.find('{')
            if json_start > 0:
                cleaned = cleaned[json_start:]
            
            # Find the end of the JSON object by counting braces
            brace_count = 0
            json_end = -1
            for i, char in enumerate(cleaned):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break
            
            if json_end > 0:
                cleaned = cleaned[:json_end + 1]
            
            # Remove markdown code blocks
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*$', '', cleaned)
            cleaned = cleaned.strip()
            
            # Additional cleanup for common LLM response issues
            cleaned = re.sub(r'^[^{]*', '', cleaned)  # Remove any text before first {
            cleaned = re.sub(r'}[^}]*$', '}', cleaned)  # Remove any text after last }
            
            return json.loads(cleaned)
            
        except json.JSONDecodeError as e:
            self.logger.warning("Failed to parse JSON response", error=str(e), response=response[:500])
            # Try to extract JSON using a more aggressive approach
            try:
                # Look for JSON pattern with regex
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, response, re.DOTALL)
                if matches:
                    # Try the largest match first
                    largest_match = max(matches, key=len)
                    return json.loads(largest_match)
            except:
                pass
            
            # If all else fails, raise the original error
            raise

    def _fallback_entity_extraction(self, query: str) -> Dict[str, Any]:
        """Fallback entity extraction using regex patterns."""
        fallback_extractor = FallbackEntityExtractor()
        entities = fallback_extractor.extract_entities(query)
        entities["extraction_method"] = "fallback"
        return entities


class FallbackEntityExtractor:
    """
    Regex-based fallback for when LLM extraction fails.
    """
    
    def __init__(self):
        """Initialize fallback patterns."""
        self.patterns = {
            "artist_similarity": [
                r"(?i)(?:like|similar to)\s+([A-Z][a-zA-Z0-9\s&\-']+)",
                r"(?i)sounds?\s+like\s+([A-Z][a-zA-Z0-9\s&\-']+)",
                r"(?i)reminds?\s+me\s+of\s+([A-Z][a-zA-Z0-9\s&\-']+)"
            ],
            "genres": [
                r"(?i)\b(rock|jazz|pop|hip.hop|electronic|classical|country|blues|folk|metal|indie|alternative|progressive|experimental|ambient)\b",
            ],
            "decades": [
                r"(?i)\b(60s|70s|80s|90s|2000s|2010s|sixties|seventies|eighties|nineties)\b",
            ],
            "activities": [
                r"(?i)\b(workout|exercise|running|studying|party|driving|cooking|gym|work|sleep|relax|focus|coding|reading)\b",
            ],
            "moods": [
                r"(?i)\b(chill|energetic|calm|upbeat|mellow|intense|peaceful|aggressive|happy|sad|nostalgic|romantic)\b",
            ],
            "session_references": [
                r"(?i)\b(last song|previous track|that song|the one before|earlier)\b",
            ]
        }
        
        self.logger = logger.bind(component="FallbackExtractor")
    
    def extract_entities(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract entities using regex patterns.
        
        Args:
            query: User query
            conversation_context: Previous context
            
        Returns:
            Extracted entities dictionary
        """
        self.logger.info("Starting fallback entity extraction", query=query)
        
        entities = {
            "musical_entities": {
                "artists": {"primary": [], "similar_to": [], "avoid": []},
                "tracks": {"specific": [], "referenced": [], "style_reference": []},
                "albums": {"specific": [], "era": [], "type": []},
                "genres": {"primary": [], "sub_genres": [], "fusion": [], "avoid": []}
            },
            "contextual_entities": {
                "moods": {"energy": [], "emotion": [], "atmosphere": []},
                "activities": {"physical": [], "mental": [], "social": [], "temporal": []},
                "temporal": {"decades": [], "eras": [], "periods": []}
            },
            "preference_entities": {
                "similarity_requests": [],
                "discovery_preferences": [],
                "quality_preferences": []
            },
            "conversation_entities": {
                "session_references": [],
                "preference_evolution": [],
                "conversation_flow": []
            },
            "confidence_scores": {"overall": 0.6, "entity_specific": {}},
            "relationships": []
        }
        
        # Extract artist similarities
        for pattern in self.patterns["artist_similarity"]:
            matches = re.findall(pattern, query)
            for match in matches:
                artist = match.strip()
                if artist and len(artist) > 1:
                    entities["musical_entities"]["artists"]["similar_to"].append(artist)
                    entities["preference_entities"]["similarity_requests"].append({
                        "type": "artist_similarity",
                        "target": artist,
                        "relationship": "similar_to",
                        "intensity": "moderate"
                    })
        
        # Extract genres
        for pattern in self.patterns["genres"]:
            matches = re.findall(pattern, query)
            for match in matches:
                genre = match.lower()
                if genre not in entities["musical_entities"]["genres"]["primary"]:
                    entities["musical_entities"]["genres"]["primary"].append(genre)
        
        # Extract decades
        for pattern in self.patterns["decades"]:
            matches = re.findall(pattern, query)
            for match in matches:
                decade = match.lower()
                if decade not in entities["contextual_entities"]["temporal"]["decades"]:
                    entities["contextual_entities"]["temporal"]["decades"].append(decade)
        
        # Extract activities
        for pattern in self.patterns["activities"]:
            matches = re.findall(pattern, query)
            for match in matches:
                activity = match.lower()
                # Categorize activity
                if activity in ["workout", "exercise", "running", "gym"]:
                    entities["contextual_entities"]["activities"]["physical"].append(activity)
                elif activity in ["studying", "work", "focus", "coding", "reading"]:
                    entities["contextual_entities"]["activities"]["mental"].append(activity)
                elif activity in ["party", "driving", "cooking"]:
                    entities["contextual_entities"]["activities"]["social"].append(activity)
        
        # Extract moods
        for pattern in self.patterns["moods"]:
            matches = re.findall(pattern, query)
            for match in matches:
                mood = match.lower()
                # Categorize mood
                if mood in ["energetic", "upbeat", "intense", "aggressive"]:
                    entities["contextual_entities"]["moods"]["energy"].append(mood)
                elif mood in ["happy", "sad", "nostalgic", "romantic"]:
                    entities["contextual_entities"]["moods"]["emotion"].append(mood)
                elif mood in ["chill", "calm", "mellow", "peaceful"]:
                    entities["contextual_entities"]["moods"]["atmosphere"].append(mood)
        
        # Extract session references
        for pattern in self.patterns["session_references"]:
            matches = re.findall(pattern, query)
            for match in matches:
                entities["conversation_entities"]["session_references"].append({
                    "type": "previous_track",
                    "reference": match,
                    "target": "last_recommended_track"
                })
        
        # Calculate confidence based on entities found
        total_entities = sum([
            len(entities["musical_entities"]["artists"]["similar_to"]),
            len(entities["musical_entities"]["genres"]["primary"]),
            len(entities["contextual_entities"]["temporal"]["decades"]),
            sum(len(activities) for activities in entities["contextual_entities"]["activities"].values()),
            sum(len(moods) for moods in entities["contextual_entities"]["moods"].values()),
            len(entities["conversation_entities"]["session_references"])
        ])
        
        if total_entities > 0:
            entities["confidence_scores"]["overall"] = min(0.8, 0.4 + (total_entities * 0.1))
        else:
            entities["confidence_scores"]["overall"] = 0.2
        
        self.logger.info("Fallback extraction completed", entity_count=total_entities)
        
        return entities 