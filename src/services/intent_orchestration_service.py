"""
Intent Orchestration Service for BeatDebate

Centralizes intent resolution logic, especially for complex follow-up scenarios.
Works with SessionManagerService to provide accurate intent interpretation
by considering original query context and conversation history.

This service addresses the "hacky code" concern by creating a single source of truth
for intent resolution, especially for complex follow-up scenarios.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
import structlog
import re

from .session_manager_service import SessionManagerService, OriginalQueryContext

logger = structlog.get_logger(__name__)


class FollowUpType(Enum):
    """Types of follow-up queries."""
    ARTIST_DEEP_DIVE = "artist_deep_dive"  # More tracks from same artist
    STYLE_CONTINUATION = "style_continuation"  # More of same style
    ARTIST_STYLE_REFINEMENT = "artist_style_refinement"  # Same artist + style constraint
    SIMILARITY_EXPLORATION = "similarity_exploration"  # Similar to previous
    VARIATION_REQUEST = "variation_request"  # Something different but related
    MORE_CONTENT = "more_content"  # Generic "more" request


class IntentOrchestrationService:
    """
    Centralizes intent resolution logic for accurate follow-up interpretation.
    
    Key Features:
    - Accurate follow-up detection using original query context
    - Intent resolution that considers conversation history
    - Centralized logic for handling variations in follow-up phrasing
    - Clear "effective intent" generation for agents
    """
    
    def __init__(self, session_manager: SessionManagerService, llm_utils=None):
        """Initialize intent orchestration service."""
        self.session_manager = session_manager
        self.llm_utils = llm_utils
        self.logger = logger.bind(component="IntentOrchestration")
        
        # Follow-up detection patterns
        self.followup_patterns = {
            FollowUpType.MORE_CONTENT: [
                r"more\s+(tracks?|songs?|music)",
                r"give\s+me\s+more",
                r"show\s+me\s+more",
                r"i\s+want\s+more"
            ],
            FollowUpType.STYLE_CONTINUATION: [
                r"more\s+like\s+(this|that|these|those)",
                r"similar\s+(to\s+)?(this|that|these|those)",
                r"in\s+the\s+same\s+(style|vein)",
                r"keep\s+going",
                r"continue"
            ],
            FollowUpType.ARTIST_DEEP_DIVE: [
                r"more\s+(.+?)\s+(tracks?|songs?|music)",
                r"other\s+(.+?)\s+(tracks?|songs?)",
                r"(.+?)\s+discography",
                r"explore\s+(.+?)(?:\s|$)"
            ],
            FollowUpType.SIMILARITY_EXPLORATION: [
                r"similar\s+artists?",
                r"artists?\s+like\s+(.+)",
                r"bands?\s+like\s+(.+)",
                r"music\s+like\s+(.+)"
            ],
            FollowUpType.VARIATION_REQUEST: [
                r"something\s+(else|different)",
                r"what\s+about",
                r"try\s+something",
                r"mix\s+it\s+up"
            ]
        }
        
        self.logger.info("Intent Orchestration Service initialized")
    
    async def resolve_effective_intent(
        self,
        current_query: str,
        session_id: str,
        llm_understanding: Optional[Dict] = None,
        context_override: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Resolve the effective intent for the current query, considering conversation context.
        
        Args:
            current_query: User's current query
            session_id: Session identifier
            llm_understanding: LLM's understanding of the query
            context_override: Context override from context handler (for follow-ups)
            
        Returns:
            Effective intent with all necessary context for agents
        """
        self.logger.info("Resolving effective intent", session_id=session_id)
        
        # 🔧 PRIORITY 1: If context_override indicates follow-up, use it directly
        if context_override and context_override.get('is_followup'):
            self.logger.info(
                "🎯 Using context override for follow-up query",
                target_entity=context_override.get('target_entity'),
                intent_override=context_override.get('intent_override'),
                followup_type=context_override.get('followup_type')
            )
            
            # Convert context_override to effective_intent format
            effective_intent = {
                "intent": context_override.get('intent_override', 'discovery'),
                "entities": context_override.get('entities', {}),
                "is_followup": True,
                "followup_type": context_override.get('followup_type', 'more_content'),
                "target_entity": context_override.get('target_entity'),
                "confidence": context_override.get('confidence', 0.9),
                "query": current_query,
                "reasoning": f"Follow-up query for {context_override.get('target_entity', 'previous context')}"
            }
            return effective_intent
        
        # 🔧 PRIORITY 2: If no context override, do LLM analysis for fresh queries
        if llm_understanding is None and self.llm_utils:
            self.logger.info("Performing LLM analysis for query", query=current_query)
            try:
                llm_understanding = await self._analyze_query_with_llm(current_query)
                self.logger.info(
                    "Generated LLM understanding",
                    intent=llm_understanding.get("intent") if llm_understanding else None,
                    entities_count=len(llm_understanding.get("entities", {})) if llm_understanding else 0,
                    confidence=llm_understanding.get("confidence") if llm_understanding else 0
                )
            except Exception as e:
                self.logger.warning("LLM analysis failed, using fallback", error=str(e))
                llm_understanding = None
        else:
            self.logger.warning("No LLM utils available for query analysis")
        
        # 🔧 PRIORITY 3: For fresh queries (no context_override), create fresh intent
        return self._create_fresh_intent(llm_understanding, context_decision={})
    
    async def _resolve_followup_intent(
        self,
        current_query: str,
        session_id: str,
        llm_understanding: Optional[Dict],
        context_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve intent for follow-up queries using original context."""
        
        # Get original query context
        original_context = await self.session_manager.get_original_query_context(session_id)
        if not original_context:
            self.logger.warning("No original context found for follow-up", session_id=session_id)
            return self._create_fresh_intent(llm_understanding, context_decision)
        
        # Detect specific follow-up type
        followup_type = self._detect_followup_type(current_query, original_context)
        
        # Create effective intent based on follow-up type
        effective_intent = await self._create_followup_intent(
            followup_type=followup_type,
            current_query=current_query,
            original_context=original_context,
            llm_understanding=llm_understanding,
            context_decision=context_decision
        )
        
        self.logger.info(
            "Resolved follow-up intent",
            session_id=session_id,
            followup_type=followup_type.value,
            effective_intent=effective_intent["intent"],
            preserves_original=effective_intent.get("preserves_original_context", False)
        )
        
        return effective_intent
    
    def _detect_followup_type(self, query: str, original_context: OriginalQueryContext) -> FollowUpType:
        """Detect the specific type of follow-up query."""
        query_lower = query.lower().strip()
        
        # 🎯 PRIORITY 1: Context-aware detection - preserve original intent when appropriate
        # If original intent was by_artist and query is generic "more", continue as artist deep dive
        if (original_context.intent == "by_artist" and 
            re.search(r"^(more|give\s+me\s+more|show\s+me\s+more|i\s+want\s+more)\s*(tracks?|songs?|music)?$", query_lower)):
            self.logger.debug(
                "🎯 Context-aware detection: preserving by_artist intent for generic 'more' request",
                original_intent=original_context.intent,
                query=query_lower
            )
            return FollowUpType.ARTIST_DEEP_DIVE
        
        # 🎯 PRIORITY 2: Check for artist-specific patterns (explicit artist mentions)
        original_artists = self._extract_artist_names_from_entities(original_context.entities)
        
        for artist in original_artists:
            artist_lower = artist.lower()
            # Check if query mentions the same artist
            if artist_lower in query_lower:
                # Check for style refinement patterns
                style_keywords = ["upbeat", "electronic", "acoustic", "fast", "slow", "happy", "sad"]
                if any(keyword in query_lower for keyword in style_keywords):
                    return FollowUpType.ARTIST_STYLE_REFINEMENT
                else:
                    return FollowUpType.ARTIST_DEEP_DIVE
        
        # 🎯 PRIORITY 3: Check specific follow-up patterns (ordered by specificity)
        # Check MORE_CONTENT patterns first (most specific)
        for pattern in self.followup_patterns[FollowUpType.MORE_CONTENT]:
            if re.search(pattern, query_lower):
                # If original was by_artist, treat generic "more" as artist continuation
                if original_context.intent == "by_artist":
                    return FollowUpType.ARTIST_DEEP_DIVE
                else:
                    return FollowUpType.MORE_CONTENT
        
        # Check other patterns in order of specificity
        pattern_order = [
            FollowUpType.SIMILARITY_EXPLORATION,
            FollowUpType.VARIATION_REQUEST,
            FollowUpType.STYLE_CONTINUATION
        ]
        
        for followup_type in pattern_order:
            patterns = self.followup_patterns[followup_type]
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return followup_type
        
        # 🎯 PRIORITY 4: Context-based default
        # If original was by_artist, default to artist deep dive, otherwise style continuation
        if original_context.intent == "by_artist":
            return FollowUpType.ARTIST_DEEP_DIVE
        else:
            return FollowUpType.STYLE_CONTINUATION
    
    async def _create_followup_intent(
        self,
        followup_type: FollowUpType,
        current_query: str,
        original_context: OriginalQueryContext,
        llm_understanding: Optional[Dict],
        context_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create effective intent for follow-up queries."""
        
        base_intent = {
            "intent": original_context.intent,
            "query": current_query,
            "is_followup": True,
            "followup_type": followup_type.value,
            "original_query": original_context.query,
            "original_intent": original_context.intent,
            "preserves_original_context": True,
            "context_decision": context_decision,
            "confidence": 0.9
        }
        
        if followup_type == FollowUpType.ARTIST_DEEP_DIVE:
            # More tracks from the same artist
            base_intent.update({
                "entities": original_context.entities,
                "constraint_overrides": {
                    "focus_on_artist": True,
                    "expand_discography": True
                },
                "reasoning": "Follow-up requesting more tracks from the same artist"
            })
            
        elif followup_type == FollowUpType.STYLE_CONTINUATION:
            # More of the same style/genre/mood
            base_intent.update({
                "entities": original_context.entities,
                "constraint_overrides": {
                    "preserve_style": True,
                    "vary_artists": True
                },
                "reasoning": "Follow-up requesting more tracks in the same style"
            })
            
        elif followup_type == FollowUpType.ARTIST_STYLE_REFINEMENT:
            # Same artist with style modifications
            style_modifiers = self._extract_style_modifiers(current_query)
            modified_entities = self._apply_style_modifiers(original_context.entities, style_modifiers)
            
            base_intent.update({
                "entities": modified_entities,
                "constraint_overrides": {
                    "focus_on_artist": True,
                    "apply_style_refinement": True,
                    "style_modifiers": style_modifiers
                },
                "reasoning": f"Follow-up requesting same artist with style refinement: {style_modifiers}"
            })
            
        elif followup_type == FollowUpType.SIMILARITY_EXPLORATION:
            # Similar artists or tracks
            base_intent.update({
                "intent": "artist_similarity",  # Override to similarity intent
                "entities": original_context.entities,
                "constraint_overrides": {
                    "find_similar_artists": True,
                    "preserve_style_elements": True
                },
                "reasoning": "Follow-up requesting similar artists or tracks"
            })
            
        elif followup_type == FollowUpType.VARIATION_REQUEST:
            # Something different but related
            base_intent.update({
                "entities": self._create_variation_entities(original_context.entities),
                "constraint_overrides": {
                    "introduce_variation": True,
                    "maintain_some_elements": True
                },
                "reasoning": "Follow-up requesting variation while maintaining some elements"
            })
            
        else:  # MORE_CONTENT or default
            # Generic "more" request
            base_intent.update({
                "entities": original_context.entities,
                "constraint_overrides": {
                    "continue_exploration": True
                },
                "reasoning": "Generic follow-up requesting more content"
            })
        
        return base_intent
    
    def _create_fresh_intent(self, llm_understanding: Optional[Dict], context_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Create intent for fresh (non-follow-up) queries."""
        if not llm_understanding:
            return {
                "intent": "discovery",
                "entities": {},
                "is_followup": False,
                "context_decision": context_decision,
                "confidence": 0.5,
                "reasoning": "No LLM understanding available, defaulting to discovery"
            }
        
        return {
            "intent": llm_understanding.get("intent", "discovery"),
            "entities": llm_understanding.get("entities", {}),
            "query": llm_understanding.get("query", ""),
            "is_followup": False,
            "preserves_original_context": False,
            "context_decision": context_decision,
            "confidence": llm_understanding.get("confidence", 0.8),
            "reasoning": "Fresh query with LLM understanding"
        }
    
    def _create_contextual_intent(self, llm_understanding: Optional[Dict], context_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Create intent that uses context but isn't a direct follow-up."""
        base_intent = self._create_fresh_intent(llm_understanding, context_decision)
        base_intent.update({
            "uses_context": True,
            "reasoning": "Query uses conversation context but isn't a direct follow-up"
        })
        return base_intent
    
    def _extract_artist_names_from_entities(self, entities: Dict[str, Any]) -> List[str]:
        """Extract artist names from entities structure."""
        artists = []
        
        # Handle different entity structures
        if "artists" in entities:
            artist_data = entities["artists"]
            if isinstance(artist_data, list):
                for artist in artist_data:
                    if isinstance(artist, str):
                        artists.append(artist)
                    elif isinstance(artist, dict) and "name" in artist:
                        artists.append(artist["name"])
        
        # Also check musical_entities structure
        if "musical_entities" in entities and "artists" in entities["musical_entities"]:
            artist_data = entities["musical_entities"]["artists"]
            if isinstance(artist_data, list):
                for artist in artist_data:
                    if isinstance(artist, str):
                        artists.append(artist)
                    elif isinstance(artist, dict) and "name" in artist:
                        artists.append(artist["name"])
        
        return artists
    
    def _extract_style_modifiers(self, query: str) -> List[str]:
        """Extract style modification keywords from query."""
        query_lower = query.lower()
        
        style_modifiers = []
        
        # Energy modifiers
        if any(word in query_lower for word in ["upbeat", "energetic", "fast", "high energy"]):
            style_modifiers.append("high_energy")
        elif any(word in query_lower for word in ["slow", "chill", "relaxed", "mellow"]):
            style_modifiers.append("low_energy")
        
        # Mood modifiers
        if any(word in query_lower for word in ["happy", "cheerful", "positive"]):
            style_modifiers.append("positive_mood")
        elif any(word in query_lower for word in ["sad", "melancholy", "dark"]):
            style_modifiers.append("negative_mood")
        
        # Genre modifiers
        genre_keywords = ["electronic", "acoustic", "rock", "pop", "jazz", "classical", "hip hop", "rap"]
        for genre in genre_keywords:
            if genre in query_lower:
                style_modifiers.append(f"genre_{genre.replace(' ', '_')}")
        
        return style_modifiers
    
    def _apply_style_modifiers(self, original_entities: Dict[str, Any], style_modifiers: List[str]) -> Dict[str, Any]:
        """Apply style modifiers to original entities."""
        modified_entities = original_entities.copy()
        
        # Add style modifiers to moods or create mood section
        if "moods" not in modified_entities:
            modified_entities["moods"] = {"primary": [], "secondary": []}
        
        moods = modified_entities["moods"]
        if not isinstance(moods, dict):
            moods = {"primary": [], "secondary": []}
            modified_entities["moods"] = moods
        
        # Convert style modifiers to mood/genre entities
        for modifier in style_modifiers:
            if modifier.startswith("genre_"):
                genre_name = modifier.replace("genre_", "").replace("_", " ")
                if "genres" not in modified_entities:
                    modified_entities["genres"] = {"primary": [], "secondary": []}
                modified_entities["genres"]["secondary"].append({"name": genre_name, "confidence": 0.8})
            else:
                # Add as mood modifier
                mood_name = modifier.replace("_", " ")
                moods["secondary"].append({"name": mood_name, "confidence": 0.8})
        
        return modified_entities
    
    def _create_variation_entities(self, original_entities: Dict[str, Any]) -> Dict[str, Any]:
        """Create entities for variation requests that maintain some original elements."""
        variation_entities = {}
        
        # Keep some genres but reduce their weight
        if "genres" in original_entities:
            original_genres = original_entities["genres"]
            if isinstance(original_genres, dict):
                variation_entities["genres"] = {
                    "primary": [],
                    "secondary": original_genres.get("primary", [])[:2]  # Keep only first 2
                }
        
        # Keep artists but mark for variation
        if "artists" in original_entities:
            variation_entities["artists"] = original_entities["artists"]
            variation_entities["constraint_overrides"] = {
                "vary_from_artists": True,
                "find_related_artists": True
            }
        
        # Modify moods slightly
        if "moods" in original_entities:
            original_moods = original_entities["moods"]
            if isinstance(original_moods, dict):
                variation_entities["moods"] = {
                    "primary": [],
                    "secondary": original_moods.get("primary", [])[:1]  # Keep only one mood
                }
        
        return variation_entities
    
    async def get_intent_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of intent resolution for the session."""
        session_context = await self.session_manager.get_session_context(session_id)
        if not session_context:
            return {"error": "No session found"}
        
        original_context = await self.session_manager.get_original_query_context(session_id)
        interaction_history = session_context.get("interaction_history", [])
        
        return {
            "session_id": session_id,
            "original_intent": original_context.intent if original_context else None,
            "original_query": original_context.query if original_context else None,
            "interaction_count": len(interaction_history),
            "followup_count": sum(1 for i in interaction_history if not i.get("is_original_query", True)),
            "context_state": session_context.get("context_state"),
            "last_updated": session_context.get("last_updated")
        } 
    
    async def _analyze_query_with_llm(self, query: str) -> Dict[str, Any]:
        """
        Analyze query using LLM to extract intent and entities.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dict with intent, entities, and confidence
        """
        if not self.llm_utils:
            raise ValueError("LLM utils not available for query analysis")
        
        system_prompt = """You are a music recommendation AI that analyzes user queries to extract intent and entities.

Your task is to analyze music queries and return structured JSON with:
1. Intent classification (by_artist, by_artist_underground, artist_genre, genre_mood, discovery, artist_similarity, discovering_serendpity, contextual, hybrid_similarity_genre)
2. Musical entities (artists, genres, moods, tracks)
3. Contextual entities (activities, temporal context)
4. Confidence level

Intent Guidelines:
- "by_artist": User wants popular/well-known music by specific artist(s) - e.g. "Music by The Beatles", "Songs by Radiohead"
- "by_artist_underground": User wants underground/lesser-known tracks by specific artist(s) - e.g. "Discover underground tracks by Kendrick Lamar", "Find deep cuts by The Beatles", "Hidden gems by Radiohead"
- "artist_genre": User wants tracks by specific artist(s) filtered by genre - e.g. "Songs by Michael Jackson that are R&B", "Electronic tracks by Radiohead", "Jazz songs by Miles Davis"
- "genre_mood": User wants specific genre/mood - e.g. "Jazz music", "Happy songs", "Chill vibes"
- "discovery": User wants to discover new music without specific artist - e.g. "Discover new music", "Find me something new", "Recommend music"
- "discovering_serendipity": User wants UNEXPECTED, SURPRISING, RANDOM music - e.g. "Surprise me", "Something completely different", "Something completely new and different", "Random music", "Blow my mind", "Shock me", "Something unexpected", "Anything", "Dealer's choice"
- "artist_similarity": User wants similar artists - e.g. "Artists like Radiohead", "Similar to The Beatles"
- "contextual": User wants music for specific activity/context - e.g. "Workout music", "Study playlist", "Music for coding", "Focus music", "Party songs"
- "hybrid_similarity_genre": Mix of artist similarity + genre filtering - e.g. "Music like Kendrick Lamar but Jazzy", "Songs similar to Bon Iver but upbeat"

Activity Classification Guidelines:
- "physical": workout, exercise, gym, running, dancing, sports, fitness, training
- "mental": study, studying, work, working, coding, focus, concentration, reading, learning
- "social": party, partying, celebration, gathering, friends, social, dancing (social context)

IMPORTANT: If the query contains words like "completely different", "unexpected", "surprise", "random", "shock", "blow my mind", or "anything" - classify as "discovering_serendipity" NOT "discovery".

Return ONLY valid JSON in this format:
{
  "intent": "intent_name",
  "entities": {
    "artists": ["artist1", "artist2"],
    "genres": ["genre1", "genre2"],
    "moods": ["mood1", "mood2"],
    "tracks": ["track1", "track2"]
  },
  "contextual_entities": {
    "activities": {
      "physical": ["workout", "exercise"],
      "mental": ["study", "coding", "focus"],
      "social": ["party", "dance"]
    },
    "temporal": {
      "decades": ["90s", "2000s"],
      "periods": ["morning", "evening"]
    }
  },
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}"""
        
        user_prompt = f'Analyze this music query: "{query}"'
        
        try:
            response = await self.llm_utils.call_llm_with_json_response(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_retries=2
            )
            
            # Validate and normalize response
            if not isinstance(response, dict):
                raise ValueError("LLM response is not a dict")
            
            intent = response.get('intent', 'discovery')
            entities = response.get('entities', {})
            contextual_entities = response.get('contextual_entities', {})
            confidence = float(response.get('confidence', 0.8))
            reasoning = response.get('reasoning', 'LLM analysis')
            
            # Normalize entities to expected format
            normalized_entities = {}
            for entity_type in ['artists', 'genres', 'moods', 'tracks']:
                entity_list = entities.get(entity_type, [])
                if entity_list:
                    normalized_entities[entity_type] = entity_list
            
            # Add contextual entities if present
            if contextual_entities:
                normalized_entities['contextual_entities'] = contextual_entities
            
            result = {
                'intent': intent,
                'entities': normalized_entities,
                'confidence': confidence,
                'reasoning': reasoning,
                'query': query
            }
            
            self.logger.debug(
                "LLM query analysis completed",
                intent=intent,
                entities_found=list(normalized_entities.keys()),
                contextual_activities=contextual_entities.get('activities', {}) if contextual_entities else {},
                confidence=confidence
            )
            
            return result
            
        except Exception as e:
            self.logger.error("LLM query analysis failed", error=str(e))
            raise e 