"""
Context Handler for Enhanced Recommendation Service

Manages context analysis, conversation history processing, and intent resolution.
Extracted from EnhancedRecommendationService to improve modularity and maintainability.
"""

import asyncio
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import structlog

# Handle imports gracefully
try:
    from ...models.agent_models import MusicRecommenderState
    from ...agents.components.llm_utils import LLMUtils
    # SmartContextManager functionality moved to SessionManagerService
    from ..session_manager_service import SessionManagerService
    from ..intent_orchestration_service import IntentOrchestrationService
except ImportError:
    # Fallback imports for testing
    import sys
    sys.path.append('src')
    from models.agent_models import MusicRecommenderState
    from agents.components.llm_utils import LLMUtils
    # SmartContextManager functionality moved to SessionManagerService
    from services.session_manager_service import SessionManagerService
    from services.intent_orchestration_service import IntentOrchestrationService

logger = structlog.get_logger(__name__)


@dataclass
class ContextOverride:
    """Context override information for recommendation constraints."""
    is_followup: bool
    intent_override: Optional[str] = None
    target_entity: Optional[str] = None
    confidence: float = 0.0
    constraint_overrides: Optional[Dict[str, Any]] = None


class ContextAwareIntentAnalyzer:
    """
    Analyzes conversation context to detect follow-up queries and override intents.
    
    Supports:
    - Simple artist followups: "More Mk.gee tracks"
    - Style continuation: "More like this"
    - Artist-style refinement: "Mk.gee tracks that are more electronic"
    """
    
    def __init__(self, llm_client, rate_limiter=None):
        self.llm_client = llm_client
        self.rate_limiter = rate_limiter
        self.logger = structlog.get_logger(__name__)
        
        # LLM utils for context analysis
        self.llm_utils = LLMUtils(llm_client, rate_limiter)
    
    async def analyze_context(self, query: str, conversation_history: List[Dict]) -> Dict:
        """
        Analyze query context to detect follow-up intents.
        
        Returns:
        {
            'is_followup': bool,
            'intent_override': str,  # artist_similarity, artist_style_refinement, style_continuation
            'target_entity': str,    # artist name
            'style_modifier': str,   # style/genre constraint (if applicable)
            'confidence': float,     # 0.0-1.0
            'constraint_overrides': Dict
        }
        """
        # Default return structure
        default_result = {
            'is_followup': False,
            'intent_override': None,
            'target_entity': None,
            'style_modifier': None,
            'confidence': 0.0,
            'constraint_overrides': None,
            'entities': {}  # 🔧 FIX: Add empty entities for non-follow-up queries
        }
        
        if not conversation_history:
            return default_result
        
        try:
            # 🎯 PRIMARY: LLM analysis for complex patterns
            llm_result = await self._analyze_followup_with_llm(query, conversation_history)
            
            # 🔧 FIXED: Accept confidence >= 0.7 (was > 0.7) to prioritize LLM analysis
            if llm_result.get('is_followup') and llm_result.get('confidence', 0) >= 0.7:
                return self._create_context_override_from_llm(llm_result, conversation_history)
            
        except Exception as e:
            self.logger.warning(f"LLM context analysis failed: {e}")
        
        # 🔧 FALLBACK: Regex pattern matching
        fallback_result = self._analyze_with_regex_fallback(query, conversation_history)
        
        # 🔧 ADDITIONAL FIX: If LLM detected follow-up but confidence was < 0.7,
        # still use LLM if it provides better target entity than regex
        try:
            if llm_result.get('is_followup') and llm_result.get('target_entity'):
                # LLM provided specific artist, prefer over generic regex result
                if fallback_result.get('target_entity') == 'previous recommendations':
                    self.logger.info("🎯 Using LLM target entity over generic regex fallback")
                    return self._create_context_override_from_llm(llm_result, conversation_history)
        except:
            pass  # llm_result might not exist
        
        # Ensure fallback result has all required keys
        if fallback_result.get('is_followup', False):
            return fallback_result
        else:
            return default_result
    
    async def _analyze_followup_with_llm(self, query: str, conversation_history: List[Dict]) -> Dict:
        """Use LLM to detect followup patterns including artist-style refinement."""
        
        # Extract previous context for analysis
        previous_artists = self._extract_artists_from_history(conversation_history)
        original_intent = self._extract_original_intent_from_history(conversation_history)
        recent_query = conversation_history[-1].get('query', '') if conversation_history else ''
        was_artist_focused = original_intent == 'by_artist' or len(previous_artists) > 0
        
        prompt = f"""
        Analyze this query to determine if it's a follow-up request and what type:

        Previous query: "{recent_query}"
        Previous artists mentioned: {previous_artists}
        Original session intent: {original_intent}
        Artist-focused session: {was_artist_focused}
        Current query: "{query}"

        🎯 CRITICAL RULES FOR NEW PRIMARY QUERIES:
        1. If the current query mentions a DIFFERENT artist than previous artists, it is NEVER a follow-up
        2. Queries like "Music by X", "Songs by X", "Tracks by X" are PRIMARY queries, not follow-ups
        3. Only treat as follow-up if query explicitly references previous context ("more", "like this", "similar")

        🎯 FOLLOW-UP DETECTION RULES:
        1. Generic "more" requests in artist-focused sessions → artist_deep_dive (preserve artist context)
        2. "More like this" or "similar" → style_continuation
        3. "More [same artist] tracks" → artist_deep_dive

        Examples:
        ✅ FOLLOW-UP: "More tracks" (after "Music by Kendrick Lamar") → artist_deep_dive
        ✅ FOLLOW-UP: "More like this" → style_continuation  
        ✅ FOLLOW-UP: "More Kendrick tracks" (same artist) → artist_deep_dive
        ❌ NOT FOLLOW-UP: "Music by Kendrick Lamar" (new primary query, even if different from previous)
        ❌ NOT FOLLOW-UP: "Songs by The Beatles" (new primary query)
        ❌ NOT FOLLOW-UP: "Tracks by Drake" (new primary query)

        🚨 IMPORTANT: If current query is a complete standalone request about a different artist, 
        it should ALWAYS be is_followup=false, regardless of conversation history!

        Return JSON:
        {{
            "is_followup": true/false,
            "followup_type": "artist_deep_dive" | "style_continuation" | "artist_style_refinement" | "none",
            "target_entity": "artist name from previous context (only if is_followup=true)",
            "style_modifier": "style/genre/mood constraint" or null,
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }}

        Follow-up types:
        - "artist_deep_dive": More tracks from SAME artist OR generic "more" in artist-focused session
        - "style_continuation": More of same style without artist reference in non-artist sessions
        - "artist_style_refinement": More tracks from SAME artist with style constraint
        """

        response = await self.llm_utils.call_llm_with_json_response(
            user_prompt=prompt, 
            system_prompt="You are an expert at analyzing conversational context for music recommendations. Be conservative - only detect follow-ups when there are clear references to previous context."
        )
        
        self.logger.debug(f"🎯 LLM context analysis: {response}")
        return response

    def _create_context_override_from_llm(self, llm_result: Dict, history: List) -> Dict:
        """Create context override based on LLM analysis."""
        
        # Extract values with defaults to prevent KeyError
        target_entity = llm_result.get('target_entity', None)
        style_modifier = llm_result.get('style_modifier', None)
        followup_type = llm_result.get('followup_type', 'artist_deep_dive')
        confidence = llm_result.get('confidence', 0.8)
        
        # 🎯 CONTEXT-AWARE: Check original intent for style_continuation
        original_intent = self._extract_original_intent_from_history(history)
        
        # Map followup types to intent overrides with context awareness
        if followup_type == 'artist_deep_dive' and original_intent == 'discovering_serendipity':
            # 🔧 FIX: Preserve discovering_serendipity intent for follow-ups
            intent_override = 'discovering_serendipity'
            target_entity = 'serendipitous discovery'
            followup_type = 'more_content'
        elif followup_type == 'style_continuation' and original_intent == 'discovering_serendipity':
            # 🔧 FIX: Preserve discovering_serendipity intent for style continuation follow-ups
            intent_override = 'discovering_serendipity'
            target_entity = 'serendipitous discovery'
            followup_type = 'more_content'
        elif followup_type == 'style_continuation' and original_intent == 'artist_similarity':
            # Preserve artist similarity intent for style continuation follow-ups
            intent_override = 'artist_similarity'
            target_entity = 'similar artists'
            followup_type = 'artist_similarity_continuation'
        elif followup_type == 'artist_deep_dive' and original_intent == 'artist_similarity':
            # 🔧 FIX: Preserve artist similarity intent for "more tracks" after "music like X"
            intent_override = 'artist_similarity'
            target_entity = 'similar artists'
            followup_type = 'artist_similarity_continuation'
        elif followup_type == 'artist_deep_dive' and original_intent == 'by_artist_underground':
            # 🔧 FIX: Preserve by_artist_underground intent for follow-ups after underground discovery
            intent_override = 'by_artist_underground'
            target_entity = target_entity  # Keep the target artist
            followup_type = 'artist_deep_dive'
        elif followup_type == 'artist_deep_dive' and original_intent == 'by_artist':
            # 🔧 FIX: Preserve by_artist intent for "more songs" after "music by X"
            intent_override = 'by_artist'
            target_entity = target_entity  # Keep the target artist
            followup_type = 'artist_deep_dive'
        elif followup_type == 'artist_deep_dive' and original_intent == 'genre_mood':
            # 🔧 FIX: Preserve genre_mood intent for "more tracks" after genre/mood queries
            intent_override = 'genre_mood'
            target_entity = 'genre/mood exploration'
            followup_type = 'more_content'
        elif followup_type == 'style_continuation' and original_intent == 'genre_mood':
            # 🔧 FIX: Preserve genre_mood intent for style continuation follow-ups
            intent_override = 'genre_mood'
            target_entity = 'genre/mood exploration'
            followup_type = 'more_content'
        elif followup_type == 'artist_deep_dive' and original_intent == 'artist_genre':
            # 🔧 FIX: Preserve artist_genre intent for "more tracks" after artist+genre queries
            intent_override = 'artist_genre'
            target_entity = 'artist genre filtering'
            followup_type = 'more_content'
        elif followup_type == 'style_continuation' and original_intent == 'artist_genre':
            # 🔧 FIX: Preserve artist_genre intent for style continuation follow-ups
            intent_override = 'artist_genre'
            target_entity = 'artist genre filtering'
            followup_type = 'more_content'
        elif followup_type == 'artist_deep_dive' and original_intent == 'hybrid_similarity_genre':
            # 🔧 FIX: Preserve hybrid_similarity_genre intent for "more tracks" after hybrid queries
            intent_override = 'hybrid_similarity_genre'
            target_entity = 'similar artists with genre filtering'
            followup_type = 'more_content'
        elif followup_type == 'style_continuation' and original_intent == 'hybrid_similarity_genre':
            # 🔧 FIX: Preserve hybrid_similarity_genre intent for style continuation follow-ups
            intent_override = 'hybrid_similarity_genre'
            target_entity = 'similar artists with genre filtering'
            followup_type = 'more_content'
        else:
            # Standard mapping for other cases
            intent_mapping = {
                'artist_deep_dive': 'by_artist',
                'style_continuation': 'style_continuation', 
                'artist_style_refinement': 'artist_style_refinement'
            }
            intent_override = intent_mapping.get(followup_type, 'artist_similarity')
        
        # Create constraint overrides for style refinement
        constraint_overrides = None
        if style_modifier and followup_type == 'artist_style_refinement':
            constraint_overrides = {
                'style_filter': style_modifier,
                'preserve_artist': target_entity
            }
        
        # Create entities based on context type
        if target_entity and followup_type == 'artist_deep_dive':
            # For artist deep dive, focus only on the target entity
            entities = {
                'artists': [target_entity],
                'tracks': [],
                'genres': [],
                'moods': []
            }
        elif followup_type == 'artist_similarity_continuation':
            # For artist similarity continuation, extract entities from history
            entities = self._extract_complete_entities_from_history(history)
        else:
            # For other follow-up types, extract complete entities from history
            entities = self._extract_complete_entities_from_history(history)
        
        result = {
            'is_followup': True,
            'intent_override': intent_override,
            'target_entity': target_entity,
            'style_modifier': style_modifier,
            'confidence': confidence,
            'constraint_overrides': constraint_overrides,
            'entities': entities  # Include for context
        }
        
        self.logger.info(f"🎯 LLM Context Override Created: {result}")
        return result
    
    def _analyze_with_regex_fallback(self, query: str, conversation_history: List[Dict]) -> Dict:
        """
        Fallback regex-based analysis for followup detection.
        
        Returns dict with same structure as LLM analysis.
        """
        query_lower = query.lower().strip()
        
        # Extract artists from recent history for context
        previous_artists = self._extract_artists_from_history(conversation_history)
        
        # 🎯 CONTEXT-AWARE: Determine if previous session was artist-focused
        original_intent = self._extract_original_intent_from_history(conversation_history)
        was_artist_focused = original_intent == 'by_artist' or len(previous_artists) > 0
        
        # 🔧 REFINED PATTERNS: More precise regex patterns
        patterns = {
            'simple_more': r'^more\s*$|^more\s+tracks?\s*$|^more\s+songs?\s*$|^more\s+music\s*$',
            'artist_more': r'^more\s+(.+?)\s+(?:tracks?|songs?|music)?\s*$',
            'like_this': r'^more\s+like\s+this|^similar\s+(?:to\s+)?this|^tracks?\s+like\s+this',
            'artist_style': r'^(.+?)\s+(?:tracks?|songs?)\s+(?:that\s+are\s+)?(?:more\s+)?(.+)$',
            'show_more': r'^show\s+more|^give\s+me\s+more|^i\s+want\s+more'
        }
        
        # Check each pattern
        for pattern_name, pattern in patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                self.logger.debug(f"🔧 REGEX: Matched pattern '{pattern_name}' for query: {query}")
                
                if pattern_name == 'simple_more':
                    # 🎯 CONTEXT-AWARE: "more" - preserve original intent context
                    original_intent = self._extract_original_intent_from_history(conversation_history)
                    
                    if original_intent == 'discovering_serendipity':
                        # Preserve discovering_serendipity intent for "more tracks" follow-ups
                        return {
                            'is_followup': True,
                            'intent_override': 'discovering_serendipity',
                            'target_entity': 'serendipitous discovery',
                            'style_modifier': None,
                            'confidence': 0.9,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'more_content'
                        }
                    elif original_intent == 'artist_similarity':
                        # Preserve artist similarity intent for "more tracks" follow-ups
                        return {
                            'is_followup': True,
                            'intent_override': 'artist_similarity',
                            'target_entity': 'similar artists',
                            'style_modifier': None,
                            'confidence': 0.9,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'artist_similarity_continuation'
                        }
                    elif original_intent == 'genre_mood':
                        # Preserve genre_mood intent for "more tracks" follow-ups
                        return {
                            'is_followup': True,
                            'intent_override': 'genre_mood',
                            'target_entity': 'genre/mood exploration',
                            'style_modifier': None,
                            'confidence': 0.9,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'more_content'
                        }
                    elif original_intent == 'contextual' or original_intent == 'activity_context':
                        # Preserve contextual intent for "more tracks" follow-ups
                        return {
                            'is_followup': True,
                            'intent_override': 'contextual',
                            'target_entity': 'contextual activity',
                            'style_modifier': None,
                            'confidence': 0.9,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'more_content'
                        }
                    elif original_intent == 'artist_genre':
                        # Preserve artist_genre intent for "more tracks" follow-ups
                        return {
                            'is_followup': True,
                            'intent_override': 'artist_genre',
                            'target_entity': 'artist genre filtering',
                            'style_modifier': None,
                            'confidence': 0.9,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'more_content'
                        }
                    elif original_intent == 'hybrid_similarity_genre':
                        # Preserve hybrid_similarity_genre intent for "more tracks" follow-ups
                        return {
                            'is_followup': True,
                            'intent_override': 'hybrid_similarity_genre',
                            'target_entity': 'similar artists with genre filtering',
                            'style_modifier': None,
                            'confidence': 0.9,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'more_content'
                        }
                    elif was_artist_focused:
                        # If previous was artist-focused, preserve the original intent
                        target_entity = previous_artists[0] if previous_artists else 'previous artist'
                        intent_to_preserve = original_intent if original_intent in ['by_artist', 'by_artist_underground'] else 'by_artist'
                        return {
                            'is_followup': True,
                            'intent_override': intent_to_preserve,
                            'target_entity': target_entity,
                            'style_modifier': None,
                            'confidence': 0.9,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'artist_deep_dive'
                        }
                    else:
                        # Default to style continuation for non-artist sessions
                        return {
                            'is_followup': True,
                            'intent_override': 'style_continuation',
                            'target_entity': 'previous recommendations',
                            'style_modifier': None,
                            'confidence': 0.9,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'style_continuation'
                        }
                
                elif pattern_name == 'artist_more':
                    # "more X tracks" - check if X matches previous artists
                    candidate_artist = match.group(1).strip()
                    if any(candidate_artist.lower() in prev_artist.lower() or 
                           prev_artist.lower() in candidate_artist.lower() 
                           for prev_artist in previous_artists):
                        return {
                            'is_followup': True,
                            'intent_override': 'artist_similarity',
                            'target_entity': candidate_artist,
                            'style_modifier': None,
                            'confidence': 0.85,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history)
                        }
                
                elif pattern_name == 'like_this':
                    # "more like this", "similar to this"
                    return {
                        'is_followup': True,
                        'intent_override': 'style_continuation',
                        'target_entity': 'previous recommendations',
                        'style_modifier': None,
                        'confidence': 0.9,
                        'constraint_overrides': None,
                        'entities': self._extract_complete_entities_from_history(conversation_history)
                    }
                
                elif pattern_name == 'artist_style':
                    # "X tracks that are more Y" - artist style refinement
                    artist_part = match.group(1).strip()
                    style_part = match.group(2).strip()
                    
                    if any(artist_part.lower() in prev_artist.lower() or 
                           prev_artist.lower() in artist_part.lower() 
                           for prev_artist in previous_artists):
                        return {
                            'is_followup': True,
                            'intent_override': 'artist_style_refinement',
                            'target_entity': artist_part,
                            'style_modifier': style_part,
                            'confidence': 0.8,
                            'constraint_overrides': {
                                'style_filter': style_part,
                                'preserve_artist': artist_part
                            },
                            'entities': self._extract_complete_entities_from_history(conversation_history)
                        }
                
                elif pattern_name == 'show_more':
                    # 🎯 CONTEXT-AWARE: "show more", "give me more"
                    original_intent = self._extract_original_intent_from_history(conversation_history)
                    
                    if original_intent == 'discovering_serendipity':
                        # Preserve discovering_serendipity intent for "show more" follow-ups
                        return {
                            'is_followup': True,
                            'intent_override': 'discovering_serendipity',
                            'target_entity': 'serendipitous discovery',
                            'style_modifier': None,
                            'confidence': 0.85,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'more_content'
                        }
                    elif original_intent == 'artist_similarity':
                        # Preserve artist similarity intent for "show more" follow-ups
                        return {
                            'is_followup': True,
                            'intent_override': 'artist_similarity',
                            'target_entity': 'similar artists',
                            'style_modifier': None,
                            'confidence': 0.85,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'artist_similarity_continuation'
                        }
                    elif original_intent == 'genre_mood':
                        # Preserve genre_mood intent for "show more" follow-ups
                        return {
                            'is_followup': True,
                            'intent_override': 'genre_mood',
                            'target_entity': 'genre/mood exploration',
                            'style_modifier': None,
                            'confidence': 0.85,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'more_content'
                        }
                    elif original_intent == 'contextual' or original_intent == 'activity_context':
                        # Preserve contextual intent for "show more" follow-ups
                        return {
                            'is_followup': True,
                            'intent_override': 'contextual',
                            'target_entity': 'contextual activity',
                            'style_modifier': None,
                            'confidence': 0.85,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'more_content'
                        }
                    elif original_intent == 'artist_genre':
                        # Preserve artist_genre intent for "show more" follow-ups
                        return {
                            'is_followup': True,
                            'intent_override': 'artist_genre',
                            'target_entity': 'artist genre filtering',
                            'style_modifier': None,
                            'confidence': 0.85,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'more_content'
                        }
                    elif was_artist_focused:
                        # If previous was artist-focused (by_artist intent), continue as artist deep dive
                        target_entity = previous_artists[0] if previous_artists else 'previous artist'
                        return {
                            'is_followup': True,
                            'intent_override': 'by_artist',
                            'target_entity': target_entity,
                            'style_modifier': None,
                            'confidence': 0.85,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'artist_deep_dive'
                        }
                    else:
                        # Default to style continuation for non-artist sessions
                        return {
                            'is_followup': True,
                            'intent_override': 'style_continuation',
                            'target_entity': 'previous recommendations',
                            'style_modifier': None,
                            'confidence': 0.85,
                            'constraint_overrides': None,
                            'entities': self._extract_complete_entities_from_history(conversation_history),
                            'followup_type': 'style_continuation'
                        }
        
        # No patterns matched
        return {
            'is_followup': False,
            'intent_override': None,
            'target_entity': None,
            'style_modifier': None,
            'confidence': 0.0,
            'constraint_overrides': None
        }
    
    def _extract_complete_entities_from_history(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Extract entities from current session context only (not entire conversation history)."""
        
        entities = {
            'artists': [],
            'tracks': [],
            'genres': [],
            'moods': []
        }
        
        # 🔧 FIX: Only extract from the MOST RECENT non-follow-up query in the session
        # This prevents contamination from previous sessions
        most_recent_primary_query = None
        
        # Find the most recent non-follow-up query (working backwards)
        for conversation in reversed(conversation_history):
            query = conversation.get('query', '').lower()
            
            # Skip follow-up queries
            if any(followup_word in query for followup_word in ['more tracks', 'more songs', 'more music', 'show more', 'give me more']):
                continue
                
            # This is a primary query - use its recommendations
            most_recent_primary_query = conversation
            break
        
        # Extract entities only from the most recent primary query
        if most_recent_primary_query:
            recommendations = most_recent_primary_query.get('recommendations', [])
            
            for rec in recommendations:
                if isinstance(rec, dict):
                    artist_name = rec.get('artist', rec.get('artist_name', ''))
                    if artist_name and artist_name not in entities['artists']:
                        entities['artists'].append(artist_name)
                    
                    track_name = rec.get('track', rec.get('track_name', ''))
                    if track_name and track_name not in entities['tracks']:
                        entities['tracks'].append(track_name)
        
        return entities
    
    def _extract_artists_from_history(self, conversation_history: List[Dict]) -> List[str]:
        """Extract artist names from conversation history."""
        artists = []
        
        for conversation in conversation_history:
            recommendations = conversation.get('recommendations', [])
            for rec in recommendations:
                if isinstance(rec, dict):
                    artist_name = rec.get('artist', rec.get('artist_name', ''))
                    if artist_name and artist_name not in artists:
                        artists.append(artist_name)
        
        return artists
    
    def _extract_original_intent_from_history(self, conversation_history: List[Dict]) -> str:
        """Extract the original intent from conversation history."""
        if not conversation_history:
            return 'discovery'
        
        # 🔧 FIX: Look for the MOST RECENT non-follow-up query, not the first
        # This ensures context resets work correctly when a new primary query is made
        for conversation in reversed(conversation_history):
            intent = conversation.get('intent')
            query = conversation.get('query', '').lower()
            
            # Skip follow-up queries (like "more tracks", "more songs", etc.)
            if any(followup_word in query for followup_word in ['more tracks', 'more songs', 'more music', 'show more', 'give me more']):
                continue
                
            if intent:
                return intent
            
            # If no explicit intent found, infer from query pattern
            if any(serendipity_word in query for serendipity_word in ['completely new', 'completely different', 'surprise', 'random', 'unexpected', 'anything', 'shock', 'blow my mind', 'serendipity']):
                return 'discovering_serendipity'
            elif any(similarity_word in query for similarity_word in ['like ', 'similar to', 'similar ', 'music like', 'sounds like']):
                # Check if it's hybrid similarity + genre (e.g., "music like X but Y")
                if any(genre_connector in query for genre_connector in [' but ', ' that are ', ' that is ', ' which are ', ' which is ']):
                    return 'hybrid_similarity_genre'
                else:
                    return 'artist_similarity'
            elif any(underground_word in query for underground_word in ['underground', 'hidden', 'lesser known', 'deep cuts', 'rare']):
                # Check if it's artist-specific underground or general underground
                if any(artist_word in query for artist_word in ['by ', 'from ', 'artist', 'band']):
                    return 'by_artist_underground'
                else:
                    return 'underground'
            elif any(artist_word in query for artist_word in ['by ', 'from ', 'artist', 'band']):
                # Check if it's artist + genre filtering (e.g., "songs by X that are Y")
                if any(genre_filter_word in query for genre_filter_word in ['that are', 'that is', 'which are', 'which is']):
                    return 'artist_genre'
                else:
                    return 'by_artist'
            elif any(contextual_word in query for contextual_word in ['for ', 'while ', 'during ', 'coding', 'study', 'workout', 'work', 'relax', 'sleep', 'drive', 'party']):
                return 'contextual'
            elif any(genre_word in query for genre_word in ['genre', 'style', 'mood', 'vibe']):
                return 'genre_mood'
        
        # Fallback to discovery if no clear intent found
        return 'discovery'


class ContextHandler:
    """
    Handles all context-related operations for the Enhanced Recommendation Service.
    
    Responsibilities:
    - Processing conversation history from different formats
    - Context analysis and follow-up detection
    - Recently shown tracks extraction
    - Session context management
    """
    
    def __init__(
        self,
        session_manager: SessionManagerService,
        intent_orchestrator: IntentOrchestrationService
    ):
        self.session_manager = session_manager
        self.intent_orchestrator = intent_orchestrator
        self.logger = structlog.get_logger(__name__)
        
        # Context analyzer will be initialized when LLM client is available
        self.context_analyzer: Optional[ContextAwareIntentAnalyzer] = None
    
    def initialize_context_analyzer(self, llm_client, rate_limiter):
        """Initialize the context analyzer with LLM client."""
        self.context_analyzer = ContextAwareIntentAnalyzer(llm_client, rate_limiter)
        self.logger.info("Context analyzer initialized")
    
    async def process_conversation_history(self, request) -> List[Dict]:
        """
        Process conversation history from various request formats.
        
        Args:
            request: Request object with potential context/chat_context
            
        Returns:
            List of conversation history dictionaries
        """
        conversation_history = []
        
        # Method 1: Check request.context
        if hasattr(request, 'context') and request.context:
            chat_context = request.context
            if 'previous_queries' in chat_context:
                conversation_history = self._convert_chat_context_to_history(chat_context)
                self.logger.info(f"Loaded {len(conversation_history)} conversations from request.context")
        
        # Method 2: Check request.chat_context
        elif hasattr(request, 'chat_context') and request.chat_context:
            chat_context = request.chat_context
            if 'previous_queries' in chat_context:
                conversation_history = self._convert_chat_context_to_history(chat_context)
                self.logger.info(f"Loaded {len(conversation_history)} conversations from request.chat_context")
        
        # Method 3: Check nested context in request dict
        elif hasattr(request, '__dict__') and 'chat_context' in request.__dict__:
            chat_context = request.__dict__['chat_context']
            if isinstance(chat_context, dict) and 'previous_queries' in chat_context:
                conversation_history = self._convert_chat_context_to_history(chat_context)
                self.logger.info(f"Loaded {len(conversation_history)} conversations from request dict")
        
        # 🔧 Method 4: Retrieve from session store using session_id
        # This is crucial for follow-up detection when history isn't passed in request
        elif hasattr(request, 'session_id') and request.session_id:
            try:
                session_context = await self.session_manager.get_session_context(request.session_id)
                if session_context and 'interaction_history' in session_context:
                    interaction_history = session_context['interaction_history']
                    conversation_history = self._convert_session_history_to_conversation(interaction_history)
                    self.logger.info(f"Loaded {len(conversation_history)} conversations from session {request.session_id}")
                else:
                    self.logger.debug(f"No session context found for session_id: {request.session_id}")
            except Exception as e:
                self.logger.warning(f"Failed to retrieve session history: {e}")
        
        # Log detailed conversation data for debugging
        if conversation_history:
            self.logger.debug(
                "Conversation history processed",
                history_data=conversation_history,
                first_query=conversation_history[0].get('query') if conversation_history else None
            )
        else:
            self.logger.debug("No conversation history found in request")
        
        return conversation_history
    
    def _convert_chat_context_to_history(self, chat_context: Dict) -> List[Dict]:
        """Convert chat interface format to conversation history format."""
        previous_queries = chat_context.get('previous_queries', [])
        previous_recommendations = chat_context.get('previous_recommendations', [])
        
        conversation_history = []
        for i, query in enumerate(previous_queries):
            conversation_history.append({
                'query': query,
                'recommendations': previous_recommendations[i] if i < len(previous_recommendations) else []
            })
        
        return conversation_history
    
    async def analyze_context(
        self, 
        query: str, 
        conversation_history: List[Dict],
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Analyze context for follow-up detection and intent resolution.
        
        Args:
            query: Current user query
            conversation_history: Processed conversation history
            session_id: Session identifier
            
        Returns:
            Context override dictionary
        """
        if not self.context_analyzer:
            self.logger.warning("Context analyzer not initialized, returning default context")
            return {
                'is_followup': False,
                'intent_override': None,
                'target_entity': None,
                'confidence': 0.0,
                'constraint_overrides': None,
                'entities': {}  # 🔧 FIX: Add empty entities for non-follow-up queries
            }
        
        # Analyze followup intent
        context_override = await self.context_analyzer.analyze_context(query, conversation_history)
        
        self.logger.info(
            "Context analysis complete",
            followup_detected=context_override['is_followup'],
            target_entity=context_override['target_entity'],
            confidence=context_override['confidence']
        )
        
        return context_override
    
    def extract_recently_shown_tracks(
        self,
        conversation_history: Optional[List[Dict[str, Any]]],
        context_override: Dict[str, Any],
        workflow_state: Optional[MusicRecommenderState] = None
    ) -> List[str]:
        """
        Extract recently shown track IDs to avoid duplicates in follow-up queries.
        
        Args:
            conversation_history: Conversation history data
            context_override: Context analysis results
            workflow_state: Current workflow state (optional)
            
        Returns:
            List of track IDs to avoid recommending again
        """
        track_ids = []
        
        if not self._is_followup_query(context_override):
            return track_ids
        
        self.logger.debug("Processing follow-up query for track extraction")
        
        try:
            # Primary extraction from conversation history
            if conversation_history:
                track_ids.extend(self._extract_from_conversation_history(conversation_history))
            
            # Secondary extraction from session context if available
            if workflow_state:
                track_ids.extend(self._extract_from_session_context(context_override, workflow_state))
            
            # Remove duplicates while preserving order
            unique_track_ids = []
            for track_id in track_ids:
                if track_id not in unique_track_ids:
                    unique_track_ids.append(track_id)
            
            self.logger.info(f"Extracted {len(unique_track_ids)} unique track IDs to avoid")
            return unique_track_ids
            
        except Exception as e:
            self.logger.error(f"Error extracting recently shown tracks: {e}")
            return []
    
    def _is_followup_query(self, context_override: Dict[str, Any]) -> bool:
        """Check if the current query is a follow-up based on context analysis."""
        return (
            context_override and 
            isinstance(context_override, dict) and 
            context_override.get('is_followup', False)
        )
    
    def _extract_from_conversation_history(self, conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Extract track IDs from conversation history."""
        track_ids = []
        
        for conversation in conversation_history:
            recommendations = conversation.get('recommendations', [])
            
            if not recommendations:
                continue
            
            for rec in recommendations:
                if isinstance(rec, dict):
                    # Extract artist and track name with proper field names
                    artist = rec.get('artist', '').strip()
                    # FIXED: Use 'title' field which is the correct field name in the data structure
                    title = rec.get('title', rec.get('track', '')).strip()
                    
                    if artist and title:
                        # FIXED: Use the same format as the filtering logic expects: "artist||title"
                        track_id = f"{artist.lower()}||{title.lower()}"
                        track_ids.append(track_id)
                    else:
                        # Fallback to other ID fields if available
                        fallback_id = rec.get('id') or rec.get('track_id') or rec.get('lastfm_url')
                        if fallback_id:
                            track_ids.append(str(fallback_id))
        
        self.logger.debug(f"Extracted {len(track_ids)} track IDs from conversation history")
        return track_ids
    
    def _extract_from_session_context(
        self, 
        context_override: Dict[str, Any], 
        workflow_state: MusicRecommenderState
    ) -> List[str]:
        """Extract track IDs from session context stored in workflow state."""
        track_ids = []
        
        # Check if there are recently shown tracks already in state
        if hasattr(workflow_state, 'recently_shown_track_ids') and workflow_state.recently_shown_track_ids:
            track_ids.extend(workflow_state.recently_shown_track_ids)
        
        # Check conversation context in state
        if hasattr(workflow_state, 'conversation_context') and workflow_state.conversation_context:
            context = workflow_state.conversation_context
            
            # Extract from previous queries and recommendations
            if isinstance(context, dict):
                previous_recs = context.get('previous_recommendations', [])
                if previous_recs and isinstance(previous_recs, list):
                    for rec_list in previous_recs:
                        if isinstance(rec_list, list):
                            for rec in rec_list:
                                if isinstance(rec, dict):
                                    # Extract artist and track name with proper field names
                                    artist = rec.get('artist', '').strip()
                                    # FIXED: Use 'title' field and same format as conversation history extraction
                                    title = rec.get('title', rec.get('track', '')).strip()
                                    
                                    if artist and title:
                                        # FIXED: Use the same format as the filtering logic expects: "artist||title"
                                        track_id = f"{artist.lower()}||{title.lower()}"
                                        track_ids.append(track_id)
                                    else:
                                        # Fallback to other ID fields if available
                                        fallback_id = (
                                            rec.get('id') or 
                                            rec.get('track_id') or 
                                            rec.get('lastfm_url')
                                        )
                                        if fallback_id:
                                            track_ids.append(str(fallback_id))
        
        self.logger.debug(f"Extracted {len(track_ids)} track IDs from session context")
        return track_ids

    async def get_session_context(self, session_id: str) -> Dict:
        """Get session context from session manager."""
        return await self.session_manager.get_session_context(session_id)

    def _convert_session_history_to_conversation(self, interaction_history: List[Dict]) -> List[Dict]:
        """Convert session interaction history to conversation history format."""
        conversation_history = []
        
        for interaction in interaction_history:
            if isinstance(interaction, dict):
                query = interaction.get('query', '')
                recommendations = interaction.get('recommendations', [])
                
                # Convert UnifiedTrackMetadata to dict format if needed
                formatted_recommendations = []
                for rec in recommendations:
                    if isinstance(rec, dict):
                        # Already in dict format
                        formatted_recommendations.append(rec)
                    else:
                        # Convert from object to dict
                        formatted_rec = {
                            'title': getattr(rec, 'name', getattr(rec, 'title', '')),
                            'artist': getattr(rec, 'artist', ''),
                            'album': getattr(rec, 'album', ''),
                            'confidence': getattr(rec, 'recommendation_score', 0.0),
                            'explanation': getattr(rec, 'recommendation_reason', ''),
                            'source': getattr(rec, 'agent_source', 'discovery_agent')
                        }
                        formatted_recommendations.append(formatted_rec)
                
                if query:  # Only add if we have a query
                    conversation_history.append({
                        'query': query,
                        'recommendations': formatted_recommendations
                    })
        
        self.logger.debug(f"Converted {len(interaction_history)} interactions to {len(conversation_history)} conversation entries")
        return conversation_history 