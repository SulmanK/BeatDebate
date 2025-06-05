"""
Enhanced Recommendation Service

Streamlined recommendation service that uses the unified API service
and eliminates duplicate business logic patterns.
"""

import asyncio
import os
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import structlog
from langgraph.graph import StateGraph, END

# Handle imports gracefully
try:
    from ..models.agent_models import MusicRecommenderState, AgentConfig, SystemConfig
    from ..models.metadata_models import UnifiedTrackMetadata, MetadataSource
    from ..models.recommendation_models import TrackRecommendation
    from ..agents import PlannerAgent, GenreMoodAgent, DiscoveryAgent, JudgeAgent
    from .api_service import APIService, get_api_service
    from .smart_context_manager import SmartContextManager
    from .session_manager_service import SessionManagerService
    from .intent_orchestration_service import IntentOrchestrationService
    from .cache_manager import CacheManager, get_cache_manager
except ImportError:
    # Fallback imports for testing
    import sys
    sys.path.append('src')
    from models.agent_models import MusicRecommenderState, AgentConfig, SystemConfig
    from models.metadata_models import UnifiedTrackMetadata, MetadataSource
    from models.recommendation_models import TrackRecommendation
    from agents import PlannerAgent, GenreMoodAgent, DiscoveryAgent, JudgeAgent
    from services.api_service import APIService, get_api_service
    from services.smart_context_manager import SmartContextManager
    from services.session_manager_service import SessionManagerService
    from services.intent_orchestration_service import IntentOrchestrationService
    from services.cache_manager import CacheManager, get_cache_manager

logger = structlog.get_logger(__name__)


@dataclass
class RecommendationRequest:
    """Request for music recommendations."""
    query: str
    session_id: Optional[str] = None
    max_recommendations: int = 10
    include_audio_features: bool = True
    context: Optional[Dict[str, Any]] = None


@dataclass
class RecommendationResponse:
    """Response containing music recommendations."""
    recommendations: List[UnifiedTrackMetadata]
    strategy_used: Dict[str, Any]
    reasoning: List[str]
    session_id: str
    processing_time: float
    metadata: Dict[str, Any]


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
        from ..agents.components.llm_utils import LLMUtils
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
            'constraint_overrides': None
        }
        
        if not conversation_history:
            return default_result
        
        try:
            # 🎯 PRIMARY: LLM analysis for complex patterns
            llm_result = await self._analyze_followup_with_llm(query, conversation_history)
            
            if llm_result.get('is_followup') and llm_result.get('confidence', 0) > 0.7:
                return self._create_context_override_from_llm(llm_result, conversation_history)
            
        except Exception as e:
            self.logger.warning(f"LLM context analysis failed: {e}")
        
        # 🔧 FALLBACK: Regex pattern matching
        fallback_result = self._analyze_with_regex_fallback(query, conversation_history)
        
        # Ensure fallback result has all required keys
        if fallback_result.get('is_followup', False):
            return fallback_result
        else:
            return default_result
    
    async def _analyze_followup_with_llm(self, query: str, conversation_history: List[Dict]) -> Dict:
        """Use LLM to detect followup patterns including artist-style refinement."""
        
        # Extract previous artists for context
        previous_artists = self._extract_artists_from_history(conversation_history)
        recent_query = conversation_history[-1].get('query', '') if conversation_history else ''
        
        prompt = f"""
        Analyze this query to determine if it's a follow-up request and what type:

        Previous query: "{recent_query}"
        Previous artists mentioned: {previous_artists}
        Current query: "{query}"

        CRITICAL RULES:
        1. If the current query mentions a DIFFERENT artist than previous artists, it is NOT a follow-up (return is_followup=false)
        2. Follow-ups must explicitly reference previous context (e.g., "more", "like this", "similar") 
        3. New standalone queries about different artists should be treated as fresh requests

        Examples:
        ✅ FOLLOW-UP: "More Mk.gee tracks" (same artist + reference word)
        ✅ FOLLOW-UP: "More like this" (reference to previous recommendations)
        ❌ NOT FOLLOW-UP: "Discover underground tracks by Kendrick Lamar" (different artist, no reference)
        ❌ NOT FOLLOW-UP: "Music by The Beatles" (different artist, new request)

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
        - "artist_deep_dive": More tracks from SAME artist ("more Mk.gee songs", "I want more Mk.gee tracks")
        - "style_continuation": More of same style without artist reference ("more like this", "similar tracks") 
        - "artist_style_refinement": More tracks from SAME artist with style constraint 
          ("Mk.gee tracks that are more electronic", "more upbeat Kendrick songs")

        Only return is_followup=true if:
        - The current query contains reference words (more, like this, similar, etc.) AND
        - References the same artist(s) as previous context OR has no specific artist (style continuation)
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
        confidence = llm_result.get('confidence', 0.0)
        followup_type = llm_result.get('followup_type', 'none')
        style_modifier = llm_result.get('style_modifier', None)
        
        # 🎯 NEW: Extract complete original entities to preserve hybrid query structure
        original_entities = self._extract_complete_entities_from_history(history)
        original_genres = []
        original_moods = []
        original_intent = None
        
        if original_entities:
            # Try multiple formats for genre extraction
            
            # Format 1: Direct genres structure (new format)
            if 'genres' in original_entities:
                genres_info = original_entities.get('genres', {})
                if isinstance(genres_info, dict):
                    # Extract from primary/secondary structure
                    primary_genres = genres_info.get('primary', [])
                    secondary_genres = genres_info.get('secondary', [])
                    
                    # Handle both string and dict formats
                    for genre_item in primary_genres + secondary_genres:
                        if isinstance(genre_item, dict) and 'name' in genre_item:
                            original_genres.append(genre_item['name'])
                        elif isinstance(genre_item, str):
                            original_genres.append(genre_item)
                elif isinstance(genres_info, list):
                    # Direct list of genres
                    original_genres.extend(genres_info)
            
            # Format 2: musical_entities nested structure (legacy format)
            musical_entities = original_entities.get("musical_entities", {})
            if musical_entities and 'genres' in musical_entities:
                genres_info = musical_entities.get("genres", {})
                if isinstance(genres_info, dict):
                    primary_genres = genres_info.get("primary", [])
                    secondary_genres = genres_info.get("secondary", [])
                    
                    for genre_item in primary_genres + secondary_genres:
                        if isinstance(genre_item, dict) and 'name' in genre_item:
                            original_genres.append(genre_item['name'])
                        elif isinstance(genre_item, str):
                            original_genres.append(genre_item)
            
            # Extract original moods (similar structure)
            if 'moods' in original_entities:
                moods_info = original_entities.get('moods', {})
                if isinstance(moods_info, dict):
                    primary_moods = moods_info.get('primary', [])
                    for mood_item in primary_moods:
                        if isinstance(mood_item, dict) and 'name' in mood_item:
                            original_moods.append(mood_item['name'])
                        elif isinstance(mood_item, str):
                            original_moods.append(mood_item)
            
            # Also check contextual_entities for moods (legacy format)
            contextual_entities = original_entities.get("contextual_entities", {})
            if contextual_entities and 'moods' in contextual_entities:
                moods_info = contextual_entities.get("moods", {})
                for mood_type, mood_list in moods_info.items():
                    if isinstance(mood_list, list):
                        original_moods.extend(mood_list)
            
            # Extract original intent if available
            if 'intent' in original_entities:
                intent_info = original_entities.get('intent', {})
                if isinstance(intent_info, dict):
                    original_intent = intent_info.get("primary", None)
                elif isinstance(intent_info, str):
                    original_intent = intent_info
            
            self.logger.info(f"🔍 PRESERVING ORIGINAL CONTEXT: genres={original_genres}, moods={original_moods}, intent={original_intent}")
        
        base_override = {
            'is_followup': True,
            'target_entity': target_entity,
            'confidence': confidence,
            'intent_override': None,
            'style_modifier': style_modifier,
            'constraint_overrides': None,
            # 🎯 NEW: Preserve original query structure
            'preserved_entities': original_entities,
            'preserved_genres': original_genres,
            'preserved_moods': original_moods,
            'original_intent': original_intent
        }
        
        if followup_type == 'artist_style_refinement':
            # 🔧 ENHANCED: Artist-style refinement with preserved context
            base_override.update({
                'intent_override': 'artist_style_refinement',
                'style_modifier': style_modifier,
                'constraint_overrides': {
                    'target_artist_priority': True,
                    'style_filtering': style_modifier,
                    'max_per_artist': {target_entity: 8} if target_entity else {},
                    'fallback_similar_style': True,
                    # 🎯 NEW: Preserve original genre filters
                    'required_genres': original_genres,
                    'required_moods': original_moods
                }
            })
            
            self.logger.info(
                f"🎵 Artist-style refinement detected: {target_entity} + {style_modifier} + preserved genres: {original_genres}"
            )
            
        elif followup_type == 'artist_deep_dive':
            # 🔧 ENHANCED: Artist deep dive with preserved hybrid context
            constraint_overrides = {
                'diversity_limits': {'same_artist_limit': 10, 'min_different_artists': 1}
            }
            
            # 🎯 CRITICAL: If original query had genre filters, preserve them!
            if original_genres:
                constraint_overrides['required_genres'] = original_genres
                constraint_overrides['hybrid_query'] = True
                # Change intent to preserve hybrid behavior
                base_override['intent_override'] = 'hybrid_artist_genre'  # NEW: Hybrid intent for artist+genre
                self.logger.info(f"🎯 HYBRID FOLLOW-UP: Preserving artist={target_entity} + genres={original_genres}")
            else:
                base_override['intent_override'] = 'by_artist'  # ✅ FIXED: Use by_artist for "More tracks"
                self.logger.info(f"🎯 Artist deep dive detected: {target_entity} - Using by_artist intent")
            
            # Also preserve moods if present
            if original_moods:
                constraint_overrides['required_moods'] = original_moods
                
            base_override['constraint_overrides'] = constraint_overrides
            
        elif followup_type == 'style_continuation':
            # 🔧 ENHANCED: Style continuation with preserved context
            base_override.update({
                'intent_override': 'style_continuation',
                'constraint_overrides': {
                    'style_consistency_weight': 0.8,
                    # 🎯 NEW: Preserve original filters for better continuity
                    'required_genres': original_genres,
                    'required_moods': original_moods
                }
            })
            
            self.logger.info(f"🎶 Style continuation detected with preserved context: genres={original_genres}, moods={original_moods}")
        
        return base_override
    
    def _analyze_with_regex_fallback(self, query: str, conversation_history: List[Dict]) -> Dict:
        """Regex-based fallback detection."""
        
        # Default return structure for no followup
        default_result = {
            'is_followup': False,
            'intent_override': None,
            'target_entity': None,
            'style_modifier': None,
            'confidence': 0.0,
            'constraint_overrides': None
        }
        
        previous_artists = self._extract_artists_from_history(conversation_history)
        if not previous_artists:
            return default_result
        
        # 🔧 NEW: Artist-style refinement patterns
        artist_style_patterns = [
            r"(.+?)\s+tracks?\s+that\s+are\s+more\s+(.+)",        # "Mk.gee tracks that are more electronic"
            r"more\s+(.+?)\s+but\s+(.+)",                         # "more Mk.gee but electronic"  
            r"(.+?)\s+songs?\s+(?:that\s+are\s+)?(.+?)er",        # "Mk.gee songs that are jazzier"
            r"more\s+(.+?)\s+(?:songs?|tracks?)\s+(.+)",          # "more Mk.gee songs electronic"
            r"(.+?)\s+music\s+that(?:'s|\s+is)\s+more\s+(.+)",    # "Mk.gee music that's more upbeat"
            r"can\s+we\s+get\s+(.+?)\s+tracks?\s+that\s+are\s+more\s+(.+)"  # "Can we get Mk.gee tracks that are more eletronic"
        ]
        
        for pattern in artist_style_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                potential_artist = match.group(1).strip()
                style_modifier = match.group(2).strip()
                
                # 🔧 STRICTER VALIDATION: Exact artist match required (not partial)
                artist_matches = False
                for prev_artist in previous_artists:
                    # Check for exact match (case insensitive) or very close match
                    if (potential_artist.lower() == prev_artist.lower() or 
                        potential_artist.lower() in prev_artist.lower() and len(potential_artist) > 3):
                        artist_matches = True
                        matched_artist = prev_artist
                        break
                
                if artist_matches:
                    self.logger.info(f"🔧 Regex fallback: artist-style refinement {potential_artist} -> {matched_artist} + {style_modifier}")
                    
                    return {
                        'is_followup': True,
                        'intent_override': 'artist_style_refinement',
                        'target_entity': matched_artist,  # Use the exact previous artist name
                        'style_modifier': style_modifier,
                        'confidence': 0.8,  # Lower confidence for regex
                        'constraint_overrides': {
                            'target_artist_priority': True,
                            'style_filtering': style_modifier,
                            'max_per_artist': {matched_artist: 8},
                            'fallback_similar_style': True
                        }
                    }
                else:
                    self.logger.info(f"🔧 Regex fallback: Artist '{potential_artist}' not found in previous artists {previous_artists} for style refinement - NOT a followup")
        
        # Existing simple followup patterns
        simple_more_patterns = [
            r"more\s+(.+?)\s+(?:tracks?|songs?)",
            r"(?:give\s+me\s+)?more\s+(.+)",
            r"(?:i\s+want\s+)?more\s+(.+?)(?:\s+tracks?|\s+songs?|$)",
            r"other\s+(.+?)\s+(?:tracks?|songs?)",
            r"different\s+(.+?)\s+(?:tracks?|songs?)"
        ]
        
        for pattern in simple_more_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                potential_artist = match.group(1).strip()
                
                # 🔧 STRICTER VALIDATION: Exact artist match required (not partial)
                artist_matches = False
                for prev_artist in previous_artists:
                    # Check for exact match (case insensitive) or very close match
                    if (potential_artist.lower() == prev_artist.lower() or 
                        potential_artist.lower() in prev_artist.lower() and len(potential_artist) > 3):
                        artist_matches = True
                        matched_artist = prev_artist
                        break
                
                if artist_matches:
                    self.logger.info(f"🔧 Regex fallback: simple artist followup {potential_artist} -> {matched_artist}")
                    
                    return {
                        'is_followup': True,
                        'intent_override': 'artist_similarity',
                        'target_entity': matched_artist,  # Use the exact previous artist name
                        'style_modifier': None,
                        'confidence': 0.9,
                        'constraint_overrides': {
                            'diversity_limits': {'same_artist_limit': 10, 'min_different_artists': 1}
                        }
                    }
                else:
                    self.logger.info(f"🔧 Regex fallback: Artist '{potential_artist}' not found in previous artists {previous_artists} - NOT a followup")
        
        return default_result
    
    def _extract_complete_entities_from_history(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Extract complete entities structure from conversation history including genres and constraints."""
        if not conversation_history:
            return {}
        
        # Get the most recent entry with entities
        for entry in reversed(conversation_history):
            # Check for the standard extracted_entities format
            if 'extracted_entities' in entry:
                entities = entry['extracted_entities']
                self.logger.debug(f"🔍 Found complete entities in history: {entities}")
                return entities
            
            # Also check for entities stored directly
            elif 'entities' in entry:
                entities = entry['entities']
                self.logger.debug(f"🔍 Found entities in history: {entities}")
                return entities
        
        # If no explicit entities found, try to reconstruct from query analysis
        # This handles cases where the conversation history doesn't have preserved entities
        for entry in reversed(conversation_history):
            query = entry.get('query', '')
            
            # Look for explicit genre mentions in original queries
            if 'r&b' in query.lower() or 'rb' in query.lower():
                reconstructed_entities = {
                    'artists': {'primary': []},
                    'genres': {'primary': [{'name': 'R&B', 'confidence': 0.8}]},
                    'tracks': {'primary': []},
                    'moods': {'primary': []}
                }
                
                # Extract artist names from the query
                if 'michael jackson' in query.lower():
                    reconstructed_entities['artists']['primary'] = [{'name': 'Michael Jackson', 'confidence': 0.9}]
                
                self.logger.info(f"🔧 RECONSTRUCTED entities from query '{query}': {reconstructed_entities}")
                return reconstructed_entities
            
            # Add more genre patterns as needed
            for genre_pattern, genre_name in [
                ('jazz', 'Jazz'),
                ('rock', 'Rock'), 
                ('pop', 'Pop'),
                ('electronic', 'Electronic'),
                ('hip hop', 'Hip Hop'),
                ('country', 'Country'),
                ('blues', 'Blues')
            ]:
                if genre_pattern in query.lower():
                    reconstructed_entities = {
                        'artists': {'primary': []},
                        'genres': {'primary': [{'name': genre_name, 'confidence': 0.8}]},
                        'tracks': {'primary': []},
                        'moods': {'primary': []}
                    }
                    self.logger.info(f"🔧 RECONSTRUCTED entities from query '{query}': {reconstructed_entities}")
                    return reconstructed_entities
        
        self.logger.debug("🔍 No complete entities found in conversation history")
        return {}

    def _extract_artists_from_history(self, conversation_history: List[Dict]) -> List[str]:
        """Extract artist names from conversation history."""
        artists = set()
        
        # First try to get artists from extracted entities
        complete_entities = self._extract_complete_entities_from_history(conversation_history)
        if complete_entities:
            entity_artists = complete_entities.get("musical_entities", {}).get("artists", {}).get("primary", [])
            artists.update(entity_artists)
        
        # Also get artists from recommendations for additional context
        for entry in conversation_history[-3:]:  # Look at last 3 entries
            if 'recommendations' in entry:
                for rec in entry['recommendations']:
                    if 'artist' in rec:
                        artists.add(rec['artist'])
        
        return list(artists)


class EnhancedRecommendationService:
    """
    Enhanced recommendation service with unified API access.
    
    Key improvements:
    - Uses centralized APIService for all external API calls
    - Eliminates duplicate client instantiation across agents
    - Unified metadata handling with UnifiedTrackMetadata
    - Streamlined workflow with better error handling
    - Integrated caching and context management
    """
    
    def __init__(
        self,
        system_config: Optional[SystemConfig] = None,
        api_service: Optional[APIService] = None,
        cache_manager: Optional[CacheManager] = None,
        context_manager: Optional[SmartContextManager] = None,
        session_manager: Optional[SessionManagerService] = None,
        intent_orchestrator: Optional[IntentOrchestrationService] = None
    ):
        """
        Initialize enhanced recommendation service.
        
        Args:
            system_config: System configuration
            api_service: API service instance (optional, will create if not provided)
            cache_manager: Cache manager instance (optional)
            context_manager: Context manager instance (optional)
        """
        self.logger = logger.bind(service="EnhancedRecommendationService")
        self.system_config = system_config
        
        # Initialize services
        self.cache_manager = cache_manager or get_cache_manager()
        self.api_service = api_service or get_api_service(
            cache_manager=self.cache_manager
        )
        self.context_manager = context_manager or SmartContextManager()
        
        # Phase 1 Enhanced Services
        self.session_manager = session_manager or SessionManagerService(cache_manager=self.cache_manager)
        # Intent orchestrator will be initialized after agents are created (needs LLM utils)
        self._intent_orchestrator = intent_orchestrator
        
        # Initialize agents (will be created with shared API service)
        self._agents_initialized = False
        self.planner_agent: Optional[PlannerAgent] = None
        self.genre_mood_agent: Optional[GenreMoodAgent] = None
        self.discovery_agent: Optional[DiscoveryAgent] = None
        self.judge_agent: Optional[JudgeAgent] = None
        
        # Workflow graph
        self.graph: Optional[StateGraph] = None
        
        # Context-aware intent analyzer (will be initialized in initialize_agents)
        self.context_analyzer: Optional[ContextAwareIntentAnalyzer] = None
        
        self.logger.info("Enhanced Recommendation Service initialized")
    
    async def initialize_agents(self):
        """Initialize agents with shared API service and rate limiting."""
        if self._agents_initialized:
            return
        
        try:
            # Create agent configurations
            agent_config = AgentConfig(
                agent_name="default",
                agent_type="enhanced",
                llm_model="gemini-2.0-flash-exp",
                temperature=0.7,
                max_tokens=1000
            )
            
            # Create Gemini client for LLM interactions
            gemini_api_key = os.getenv('GEMINI_API_KEY', 'demo_gemini_key')
            gemini_client = create_gemini_client(gemini_api_key)
            
            if not gemini_client:
                self.logger.warning("Failed to create Gemini client, agents will have limited functionality")
            
            # Create rate limiter for Gemini API (free tier: 10 requests per minute)
            try:
                from ..api.rate_limiter import UnifiedRateLimiter
                gemini_rate_limiter = UnifiedRateLimiter.for_gemini(calls_per_minute=8)  # Conservative limit
                self.logger.info("Gemini rate limiter created", calls_per_minute=8)
            except ImportError:
                from api.rate_limiter import UnifiedRateLimiter
                gemini_rate_limiter = UnifiedRateLimiter.for_gemini(calls_per_minute=8)
                self.logger.info("Gemini rate limiter created", calls_per_minute=8)
            
            # Get shared clients from API service
            lastfm_client = await self.api_service.get_lastfm_client()
            
            # Create metadata service with shared client
            from .metadata_service import MetadataService
            metadata_service = MetadataService(lastfm_client=lastfm_client)
            
            # Initialize agents with Gemini client and rate limiter
            self.planner_agent = PlannerAgent(
                config=agent_config,
                llm_client=gemini_client,
                api_service=self.api_service,
                metadata_service=metadata_service,
                rate_limiter=gemini_rate_limiter
            )
            
            self.genre_mood_agent = GenreMoodAgent(
                config=agent_config,
                llm_client=gemini_client,
                api_service=self.api_service,
                metadata_service=metadata_service,
                rate_limiter=gemini_rate_limiter
            )
            
            self.discovery_agent = DiscoveryAgent(
                config=agent_config,
                llm_client=gemini_client,
                api_service=self.api_service,
                metadata_service=metadata_service,
                rate_limiter=gemini_rate_limiter
            )
            
            self.judge_agent = JudgeAgent(
                config=agent_config,
                llm_client=gemini_client,
                api_service=self.api_service,
                metadata_service=metadata_service,
                rate_limiter=gemini_rate_limiter
            )
            
            # Build workflow graph
            self.graph = self._build_workflow_graph()
            
            # Initialize context-aware intent analyzer
            self.context_analyzer = ContextAwareIntentAnalyzer(gemini_client, gemini_rate_limiter)
            
            # Initialize intent orchestrator if not provided
            if not self._intent_orchestrator:
                from ..agents.components.llm_utils import LLMUtils
                llm_utils = LLMUtils(gemini_client, gemini_rate_limiter)
                self._intent_orchestrator = IntentOrchestrationService(
                    session_manager=self.session_manager,
                    llm_utils=llm_utils
                )
            
            self._agents_initialized = True
            self.logger.info("Agents initialized with shared API service and rate limiting")
            
        except Exception as e:
            self.logger.error("Failed to initialize agents", error=str(e))
            raise
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow with conditional routing."""
        workflow = StateGraph(MusicRecommenderState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("genre_mood_advocate", self._genre_mood_node)
        workflow.add_node("discovery_advocate", self._discovery_node)
        workflow.add_node("judge", self._judge_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # 🔧 INTENT-AWARE ROUTING: Add conditional edges based on planner's agent sequence
        workflow.add_conditional_edges(
            "planner",
            self._route_agents,  # Router function that respects intent-aware sequences
            {
                "discovery_only": "discovery_advocate",
                "genre_mood_only": "genre_mood_advocate", 
                "both_agents": "discovery_advocate",  # 🔧 FIXED: Start with discovery, then genre_mood
                "judge_only": "judge"  # For edge cases
            }
        )
        
        # 🔧 FIXED: Add conditional edges from discovery to either genre_mood or judge
        workflow.add_conditional_edges(
            "discovery_advocate",
            self._route_after_discovery,
            {
                "to_genre_mood": "genre_mood_advocate",
                "to_judge": "judge"
            }
        )
        
        # Add edges from agents to judge
        workflow.add_edge("genre_mood_advocate", "judge")
        workflow.add_edge("judge", END)
        
        return workflow.compile()
    
    def _route_agents(self, state: MusicRecommenderState) -> str:
        """
        Route to appropriate agents based on planner's intent-aware sequence.
        
        Returns:
            Route key for conditional edges
        """
        try:
            # Get the agent sequence from planner's strategy
            planning_strategy = getattr(state, 'planning_strategy', {})
            agent_sequence = planning_strategy.get('agent_sequence', ['discovery', 'genre_mood', 'judge'])
            
            self.logger.info(f"🔧 ROUTER: Agent sequence from planner: {agent_sequence}")
            
            # Route based on which agents are in the sequence (excluding 'judge')
            agents_to_run = [agent for agent in agent_sequence if agent != 'judge']
            
            if 'discovery' in agents_to_run and 'genre_mood' in agents_to_run:
                self.logger.info("🔧 ROUTER: Running both discovery and genre_mood agents")
                return "both_agents"
            elif 'discovery' in agents_to_run:
                self.logger.info("🔧 ROUTER: Running discovery agent only")
                return "discovery_only"
            elif 'genre_mood' in agents_to_run:
                self.logger.info("🔧 ROUTER: Running genre_mood agent only")
                return "genre_mood_only"
            else:
                self.logger.warning("🔧 ROUTER: No valid agents in sequence, defaulting to both")
                return "both_agents"
                
        except Exception as e:
            self.logger.error(f"🔧 ROUTER: Error in routing, defaulting to both agents: {e}")
            return "both_agents"
    
    def _route_after_discovery(self, state: MusicRecommenderState) -> str:
        """
        Route after discovery agent based on planner's intent-aware sequence.
        
        Returns:
            Route key for conditional edges after discovery
        """
        try:
            # Get the agent sequence from planner's strategy
            planning_strategy = getattr(state, 'planning_strategy', {})
            agent_sequence = planning_strategy.get('agent_sequence', ['discovery', 'genre_mood', 'judge'])
            
            self.logger.info(f"🔧 ROUTER: After discovery, checking sequence: {agent_sequence}")
            
            # If genre_mood is in the sequence after discovery, route to it
            if 'genre_mood' in agent_sequence:
                self.logger.info("🔧 ROUTER: Routing from discovery to genre_mood")
                return "to_genre_mood"
            else:
                self.logger.info("🔧 ROUTER: Routing from discovery directly to judge")
                return "to_judge"
                
        except Exception as e:
            self.logger.error(f"🔧 ROUTER: Error in post-discovery routing, defaulting to genre_mood: {e}")
            return "to_genre_mood"
    
    async def _planner_node(self, state: MusicRecommenderState) -> Dict[str, Any]:
        """Execute planner agent and return field updates as a dictionary."""
        try:
            self.logger.info("Executing planner node")
            updated_state = await self.planner_agent.process(state)
            
            # Return only the fields that were updated by the planner agent
            updates = {}
            
            # Extract planner-specific fields
            if hasattr(updated_state, 'planning_strategy') and updated_state.planning_strategy:
                updates["planning_strategy"] = updated_state.planning_strategy
            
            if hasattr(updated_state, 'query_understanding') and updated_state.query_understanding:
                updates["query_understanding"] = updated_state.query_understanding
                
            if hasattr(updated_state, 'agent_coordination') and updated_state.agent_coordination:
                updates["agent_coordination"] = updated_state.agent_coordination
                
            if hasattr(updated_state, 'entities') and updated_state.entities:
                updates["entities"] = updated_state.entities
                
            if hasattr(updated_state, 'intent_analysis') and updated_state.intent_analysis:
                updates["intent_analysis"] = updated_state.intent_analysis
            
            # Extract updated reasoning log
            if hasattr(updated_state, 'reasoning_log') and updated_state.reasoning_log:
                updates["reasoning_log"] = updated_state.reasoning_log
            
            # Extract error info if present
            if hasattr(updated_state, 'error_info') and updated_state.error_info:
                updates["error_info"] = updated_state.error_info
            
            self.logger.info(f"Planner node completed, returning {len(updates)} field updates")
            return updates
            
        except Exception as e:
            self.logger.error("Planner node failed", error=str(e), exc_info=True)
            current_log = list(getattr(state, 'reasoning_log', []))
            current_log.append(f"Planner: Error - {str(e)}")
            return {
                "error_info": {"agent": "PlannerAgent", "message": str(e)},
                "reasoning_log": current_log
            }
    
    async def _genre_mood_node(self, state: MusicRecommenderState) -> Dict[str, Any]:
        """Execute genre/mood agent and return field updates as a dictionary."""
        try:
            self.logger.info("Executing genre/mood advocate node")
            updated_state = await self.genre_mood_agent.process(state)
            
            # Return only the fields that were updated by the genre/mood agent
            updates = {}
            
            # Extract genre/mood recommendations
            if hasattr(updated_state, 'genre_mood_recommendations') and updated_state.genre_mood_recommendations:
                updates["genre_mood_recommendations"] = updated_state.genre_mood_recommendations
                self.logger.debug(f"Genre/mood node returning {len(updated_state.genre_mood_recommendations)} recommendations")
            
            # Extract updated reasoning log
            if hasattr(updated_state, 'reasoning_log') and updated_state.reasoning_log:
                updates["reasoning_log"] = updated_state.reasoning_log
            
            # Extract error info if present
            if hasattr(updated_state, 'error_info') and updated_state.error_info:
                updates["error_info"] = updated_state.error_info
            
            self.logger.info(f"Genre/mood node completed, returning {len(updates)} field updates")
            return updates
            
        except Exception as e:
            self.logger.error("Genre/mood node failed", error=str(e), exc_info=True)
            current_log = list(getattr(state, 'reasoning_log', []))
            current_log.append(f"GenreMood: Error - {str(e)}")
            return {
                "error_info": {"agent": "GenreMoodAgent", "message": str(e)},
                "reasoning_log": current_log
            }
    
    async def _discovery_node(self, state: MusicRecommenderState) -> Dict[str, Any]:
        """Execute discovery agent and return field updates as a dictionary."""
        try:
            self.logger.info("Executing discovery advocate node")
            updated_state = await self.discovery_agent.process(state)
            
            # Return only the fields that were updated by the discovery agent
            updates = {}
            
            # Extract discovery recommendations
            if hasattr(updated_state, 'discovery_recommendations') and updated_state.discovery_recommendations:
                updates["discovery_recommendations"] = updated_state.discovery_recommendations
                self.logger.debug(f"Discovery node returning {len(updated_state.discovery_recommendations)} recommendations")
            
            # Extract updated reasoning log
            if hasattr(updated_state, 'reasoning_log') and updated_state.reasoning_log:
                updates["reasoning_log"] = updated_state.reasoning_log
            
            # Extract error info if present
            if hasattr(updated_state, 'error_info') and updated_state.error_info:
                updates["error_info"] = updated_state.error_info
            
            # 🔧 CRITICAL FIX: Preserve recently_shown_track_ids through the workflow
            # LangGraph only preserves fields that are included in node updates
            
            # 🔧 DEBUG: Check what we actually received in the state
            self.logger.info(f"🔧 DISCOVERY DEBUG: recently_shown_track_ids = {getattr(state, 'recently_shown_track_ids', 'NOT_SET')}")
            self.logger.info(f"🔧 DISCOVERY DEBUG: recently_shown_track_ids type = {type(getattr(state, 'recently_shown_track_ids', None))}")
            self.logger.info(f"🔧 DISCOVERY DEBUG: hasattr check = {hasattr(state, 'recently_shown_track_ids')}")
            
            if hasattr(state, 'recently_shown_track_ids') and state.recently_shown_track_ids:
                updates["recently_shown_track_ids"] = state.recently_shown_track_ids
                self.logger.info(f"🔧 DISCOVERY: Preserving {len(state.recently_shown_track_ids)} recently shown track IDs through workflow")
            else:
                self.logger.info("🎯 NEW QUERY: No recently shown tracks")
            
            self.logger.info(f"Discovery node completed, returning {len(updates)} field updates")
            return updates
            
        except Exception as e:
            self.logger.error("Discovery node failed", error=str(e), exc_info=True)
            current_log = list(getattr(state, 'reasoning_log', []))
            current_log.append(f"Discovery: Error - {str(e)}")
            return {
                "error_info": {"agent": "DiscoveryAgent", "message": str(e)},
                "reasoning_log": current_log
            }
    
    async def _judge_node(self, state: MusicRecommenderState) -> Dict[str, Any]:
        """Execute judge agent and return field updates as a dictionary."""
        try:
            self.logger.info("Executing judge node")
            updated_state = await self.judge_agent.evaluate_and_select(state)
            
            # Debug: log the updated_state details
            self.logger.debug(f"Judge agent returned state type: {type(updated_state)}")
            if hasattr(updated_state, '__dict__'):
                self.logger.debug(f"Judge agent state dict keys: {list(updated_state.__dict__.keys())}")
                if hasattr(updated_state, 'final_recommendations'):
                    self.logger.debug(f"final_recommendations type: {type(updated_state.final_recommendations)}, length: {len(updated_state.final_recommendations) if updated_state.final_recommendations else 0}")
            
            # Return only the fields that were updated by the judge agent
            updates = {}
            
            # Extract final_recommendations - this is the critical field
            if hasattr(updated_state, 'final_recommendations') and updated_state.final_recommendations is not None:
                updates["final_recommendations"] = updated_state.final_recommendations
                self.logger.debug(f"Judge node returning {len(updated_state.final_recommendations)} final recommendations")
            else:
                updates["final_recommendations"] = []
                self.logger.warning("Judge agent did not produce final_recommendations")
            
            # Extract updated reasoning log
            # Always update reasoning log with judge's message
            current_log = list(getattr(state, 'reasoning_log', []))
            if hasattr(updated_state, 'reasoning_log') and updated_state.reasoning_log:
                # If the judge agent added to the reasoning log, use that
                updates["reasoning_log"] = updated_state.reasoning_log
            else:
                # Otherwise, just add our message to the existing log
                current_log.append("Judge: Final recommendations selected")
                updates["reasoning_log"] = current_log
            
            # Extract error info if present
            if hasattr(updated_state, 'error_info') and updated_state.error_info:
                updates["error_info"] = updated_state.error_info
            
            # 🔧 CRITICAL FIX: Preserve recently_shown_track_ids through the workflow
            # LangGraph only preserves fields that are included in node updates
            if hasattr(state, 'recently_shown_track_ids') and state.recently_shown_track_ids:
                updates["recently_shown_track_ids"] = state.recently_shown_track_ids
                self.logger.info(f"🔧 PRESERVING: {len(state.recently_shown_track_ids)} recently shown track IDs through workflow")
            
            self.logger.info(f"Judge node completed, returning {len(updates)} field updates")
            return updates
            
        except Exception as e:
            self.logger.error("Judge node failed", error=str(e), exc_info=True)
            # Return error state as dictionary
            current_log = list(getattr(state, 'reasoning_log', []))
            current_log.append(f"Judge: Error - {str(e)}")
            return {
                "final_recommendations": [],
                "error_info": {"agent": "JudgeAgent", "message": str(e)},
                "reasoning_log": current_log
            }
    
    async def get_recommendations(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """
        Get music recommendations using the enhanced workflow.
        
        Args:
            request: Recommendation request
            
        Returns:
            Recommendation response with unified metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        # Ensure agents are initialized
        await self.initialize_agents()
        
        # NEW: Analyze followup intent using conversation history
        conversation_history = []
        session_context = await self.context_manager.get_session_context(
            request.session_id or "default"
        )
        
        # Integrate chat history from request if available
        if hasattr(request, 'context') and request.context:
            chat_context = request.context
            if 'previous_queries' in chat_context:
                # Convert chat interface format to conversation history format
                previous_queries = chat_context.get('previous_queries', [])
                previous_recommendations = chat_context.get('previous_recommendations', [])
                
                # Merge conversation history
                conversation_history = []
                for i, query in enumerate(previous_queries):
                    conversation_history.append({
                        'query': query,
                        'recommendations': previous_recommendations[i] if i < len(previous_recommendations) else []
                    })
        
        # 🔧 CRITICAL FIX: Also check for chat_context format from UI
        elif hasattr(request, 'chat_context') and request.chat_context:
            chat_context = request.chat_context
            if 'previous_queries' in chat_context:
                # Convert chat interface format to conversation history format
                previous_queries = chat_context.get('previous_queries', [])
                previous_recommendations = chat_context.get('previous_recommendations', [])
                
                # Merge conversation history
                conversation_history = []
                for i, query in enumerate(previous_queries):
                    conversation_history.append({
                        'query': query,
                        'recommendations': previous_recommendations[i] if i < len(previous_recommendations) else []
                    })
                    
                self.logger.info(f"🔧 CHAT CONTEXT FIX: Loaded {len(conversation_history)} conversations from chat_context")
            
        # 🔧 ADDITIONAL FIX: Check if context is nested in request data (common API pattern)
        elif hasattr(request, '__dict__') and 'chat_context' in request.__dict__:
            chat_context = request.__dict__['chat_context']
            if isinstance(chat_context, dict) and 'previous_queries' in chat_context:
                previous_queries = chat_context.get('previous_queries', [])
                previous_recommendations = chat_context.get('previous_recommendations', [])
                
                conversation_history = []
                for i, query in enumerate(previous_queries):
                    conversation_history.append({
                        'query': query,
                        'recommendations': previous_recommendations[i] if i < len(previous_recommendations) else []
                    })
                    
                self.logger.info(f"🔧 DICT CONTEXT FIX: Loaded {len(conversation_history)} conversations from request dict")
        
        # Log conversation history for debugging
        self.logger.info(
            f"Conversation history for followup analysis",
            session_id=request.session_id,
            conversation_length=len(conversation_history),
            current_query=request.query
        )
        
        # Add detailed logging of conversation data
        if conversation_history:
            self.logger.info(
                "Conversation history details",
                history_data=conversation_history,
                first_query=conversation_history[0].get('query') if conversation_history else None
            )
        else:
            self.logger.warning("Empty conversation history despite conversation_length > 0")
            # Check if chat_context exists but wasn't parsed correctly
            if hasattr(request, 'context') and request.context:
                self.logger.info("Raw context data", context_data=request.context)
        
        # Analyze followup intent
        context_override = await self.context_analyzer.analyze_context(
            request.query, conversation_history
        )
        
        self.logger.info(
            "Context analysis complete",
            followup_detected=context_override['is_followup'],
            target_entity=context_override['target_entity'],
            confidence=context_override['confidence']
        )
        
        # 🚀 CRITICAL FIX: Prepare session context FIRST before state initialization
        recently_shown_track_ids = []
        if context_override and self._is_followup_query(context_override):
            self.logger.info("🔧 EXTRACTION DEBUG: Processing follow-up query")
            self.logger.info(f"🔧 EXTRACTION DEBUG: conversation_history = {conversation_history}")
            self.logger.info(f"🔧 EXTRACTION DEBUG: context_override = {context_override}")
            
            recently_shown_track_ids = self._extract_recently_shown_tracks(
                conversation_history, 
                context_override,
                None  # Will be populated after state creation if needed
            )
            
            self.logger.info(
                f"🔧 EXTRACTION DEBUG: Extracted {len(recently_shown_track_ids)} track IDs: "
                f"{recently_shown_track_ids}"
            )
            
            target_entity = context_override.get('target_entity', 'previous recommendations')
            self.logger.info(
                f"🔄 FOLLOW-UP QUERY: Prepared {len(recently_shown_track_ids)} recently shown tracks "
                f"for '{target_entity}' to avoid duplicates"
        )
        
        # Create workflow state with PRE-POPULATED recently_shown_track_ids
        workflow_state = MusicRecommenderState(
            user_query=request.query,
            max_recommendations=request.max_recommendations or 10,
            entities=context_override.get('entities', {}) if context_override else {},
            conversation_context=session_context,
            context_override=context_override,
            session_id=request.session_id or "default",
            recently_shown_track_ids=recently_shown_track_ids  # 🚀 PRE-POPULATED!
        )
        
        # 🎯 STATE VALIDATION: Validate state before workflow execution
        self._validate_state_for_workflow(workflow_state)
        
        try:
            # Execute workflow
            final_state = await self.graph.ainvoke(workflow_state)
            
            # Debug: Check what's actually in the final state
            self.logger.debug(f"Final state type: {type(final_state)}")
            self.logger.debug(f"Is final_state a dict? {isinstance(final_state, dict)}")
            self.logger.debug(f"Is final_state a MusicRecommenderState? {isinstance(final_state, MusicRecommenderState)}")
            
            if isinstance(final_state, dict):
                self.logger.debug(f"Final state dict keys: {list(final_state.keys())}")
                if 'final_recommendations' in final_state:
                    self.logger.debug(f"final_recommendations in dict, length: {len(final_state['final_recommendations'])}")
            elif hasattr(final_state, '__dict__'):
                self.logger.debug(f"Final state attributes: {dir(final_state)}")
                self.logger.debug(f"Final state dict: {final_state.__dict__ if hasattr(final_state, '__dict__') else 'No __dict__'}")
            
            # Try multiple ways to access final_recommendations
            final_recommendations = None
            
            # First check if it's a dictionary
            if isinstance(final_state, dict):
                final_recommendations = final_state.get('final_recommendations', [])
                self.logger.debug(f"Found final_recommendations in dict: {len(final_recommendations) if final_recommendations else 0} items")
            # Then check if it's an object with attributes
            elif hasattr(final_state, 'final_recommendations'):
                final_recommendations = final_state.final_recommendations
                self.logger.debug(f"Found final_recommendations via hasattr: {len(final_recommendations) if final_recommendations else 'None'}")
            else:
                self.logger.warning("final_recommendations not found in final_state (neither dict nor attribute)")
                final_recommendations = []
            
            if not final_recommendations:
                # Fallback: check if recommendations are in other fields
                all_possible_recs = []
                for attr_name in ['final_recommendations', 'recommendations', 'genre_mood_recommendations', 'discovery_recommendations']:
                    attr_value = getattr(final_state, attr_name, None)
                    if attr_value:
                        self.logger.debug(f"Found {len(attr_value)} items in {attr_name}")
                        all_possible_recs.extend(attr_value)
                
                if all_possible_recs:
                    self.logger.warning(f"Using fallback recommendations from other fields: {len(all_possible_recs)} items")
                    final_recommendations = all_possible_recs
            
            # Convert recommendations to unified metadata
            unified_recommendations = await self._convert_to_unified_metadata(
                final_recommendations,
                include_audio_features=request.include_audio_features
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Extract fields from final_state (handle both dict and object)
            if isinstance(final_state, dict):
                strategy_used = final_state.get('planning_strategy', {})
                reasoning_log = final_state.get('reasoning_log', [])
                session_id = final_state.get('session_id', request.session_id or "default")
                query_understanding = final_state.get('query_understanding', None)
            else:
                strategy_used = getattr(final_state, 'planning_strategy', {})
                reasoning_log = getattr(final_state, 'reasoning_log', [])
                session_id = getattr(final_state, 'session_id', request.session_id or "default")
                query_understanding = getattr(final_state, 'query_understanding', None)
            
            response = RecommendationResponse(
                recommendations=unified_recommendations,
                strategy_used=strategy_used,
                reasoning=reasoning_log,
                session_id=session_id,
                processing_time=processing_time,
                metadata={
                    "context_decision": {
                        'is_followup': context_override['is_followup'],
                        'intent_override': context_override['intent_override'],
                        'target_entity': context_override['target_entity'],
                        'confidence': context_override['confidence'],
                        'style_modifier': context_override['style_modifier'],
                        'constraint_overrides': context_override['constraint_overrides']
                    },
                    "agents_used": ["planner", "genre_mood", "discovery", "judge"],
                    "total_candidates": len(getattr(final_state, 'all_recommendations', [])) if not isinstance(final_state, dict) else 0,
                    "final_count": len(unified_recommendations)
                }
            )
            
            # Update context after recommendation
            await self.context_manager.update_context_after_recommendation(
                session_id=response.session_id,
                query=request.query,
                llm_understanding=query_understanding,
                recommendations=[rec.to_dict() for rec in unified_recommendations],
                context_decision={
                    'is_followup': context_override['is_followup'],
                    'intent_override': context_override['intent_override'],
                    'target_entity': context_override['target_entity'],
                    'confidence': context_override['confidence'],
                    'style_modifier': context_override['style_modifier'],
                    'constraint_overrides': context_override['constraint_overrides']
                },
                # 🎯 NEW: Preserve extracted entities for follow-up queries
                extracted_entities=getattr(final_state, 'entities', None) if not isinstance(final_state, dict) else final_state.get('entities', None)
            )
            
            self.logger.info(
                "Recommendations generated successfully",
                query=request.query,
                count=len(unified_recommendations),
                processing_time=processing_time
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in recommendation workflow: {str(e)}")
            raise

    def _validate_state_for_workflow(self, state: MusicRecommenderState) -> None:
        """Validate state has all required data for workflow execution"""
        if not state.user_query:
            raise ValueError("user_query is required")
        
        if not state.session_id:
            raise ValueError("session_id is required")
        
        # Log state preparation for debugging
        recently_shown_count = len(state.recently_shown_track_ids or [])
        self.logger.info(f"🎯 STATE PREPARED: recently_shown={recently_shown_count} tracks")
        
        if state.recently_shown_track_ids:
            self.logger.info("🎯 FOLLOW-UP DETECTED: Will trigger candidate scaling")
            # Log first few track IDs for debugging
            sample_ids = state.recently_shown_track_ids[:3]
            self.logger.debug(f"🎯 SAMPLE TRACK IDS: {sample_ids}")
        else:
            self.logger.info("🎯 NEW QUERY: No recently shown tracks")

    def _is_followup_query(self, context_override: Dict[str, Any]) -> bool:
        """Check if this is a follow-up query that should avoid duplicate recommendations."""
        if not context_override.get('is_followup'):
            return False
            
        # Follow-up intents that should avoid duplicates
        followup_intents = [
            'style_continuation',     # "More like that"
            'artist_deep_dive',       # "More from [Artist X]"
            'artist_similarity',      # "More from [Artist X]" (mapped from artist_deep_dive)
            'artist_style_refinement',  # "More like [Artist] but [modifier]"
            'by_artist'              # ✅ NEW: "More tracks" after "Music by [Artist]"
        ]
        
        intent_override = context_override.get('intent_override')
        return intent_override in followup_intents
    
    def _extract_recently_shown_tracks(
        self, 
        conversation_history: Optional[List[Dict[str, Any]]], 
        context_override: Dict[str, Any],
        workflow_state: MusicRecommenderState
    ) -> List[str]:
        """Extract track IDs from recent conversation history to avoid duplicates."""
        if not conversation_history:
            self.logger.info("🔧 EXTRACT DEBUG: No conversation history provided")
            return []
        
        recently_shown_ids = []
        intent_override = context_override.get('intent_override')
        target_entity = context_override.get('target_entity', '')
        # 🔧 FIX: Handle None target_entity
        target_entity_lower = target_entity.lower() if target_entity else ''
        
        self.logger.info(f"🔧 EXTRACT DEBUG: Processing {len(conversation_history)} conversation turns")
        self.logger.info(f"🔧 EXTRACT DEBUG: Intent override: {intent_override}")
        self.logger.info(f"🔧 EXTRACT DEBUG: Target entity: '{target_entity}' (lower: '{target_entity_lower}')")
        
        # Limit to recent conversation (last 5 turns)
        recent_turns = conversation_history[-5:]
        
        for i, turn in enumerate(recent_turns):
            self.logger.info(f"🔧 EXTRACT DEBUG: Turn {i}: {list(turn.keys())}")
            
            if isinstance(turn, dict) and 'recommendations' in turn:
                recommendations = turn['recommendations']
                self.logger.info(f"🔧 EXTRACT DEBUG: Turn {i} has {len(recommendations)} recommendations")
                
                # 🔧 NEW: Check if this looks like incomplete history from chat interface
                if len(recommendations) <= 2 and i == 0:  # First turn with very few recommendations
                    self.logger.warning(f"🔧 EXTRACT DEBUG: Detected incomplete chat interface history ({len(recommendations)} tracks), will try session context fallback")
                
                for j, rec in enumerate(recommendations):
                    if isinstance(rec, dict) and 'artist' in rec and 'title' in rec:
                        artist = str(rec['artist']).lower().strip()
                        title = str(rec['title']).lower().strip()
                        self.logger.info(f"🔧 EXTRACT DEBUG: Rec {j}: artist='{artist}', title='{title}'")
                    else:
                        self.logger.info(f"🔧 EXTRACT DEBUG: Skipping rec {j} - missing artist or title")
                        continue
                    
                    # Create track ID using artist::title format
                    track_id = f"{artist}::{title}"
                    
                    # Determine if this track should be included based on intent
                    should_include = False
                    
                    if intent_override == 'style_continuation':
                        # For style continuation, include ALL previous recommendations to avoid duplicates
                        # These tracks should be AVOIDED in future recommendations
                        should_include = True
                        self.logger.info(f"🔧 EXTRACT DEBUG: Style continuation - avoiding duplicate {track_id}")
                        
                    elif intent_override == 'artist_deep_dive':
                        # Include only tracks by the target artist for artist deep dive
                        should_include = (artist == target_entity_lower)
                        self.logger.info(f"🔧 EXTRACT DEBUG: Artist deep dive - artist match: {artist} == {target_entity_lower} = {should_include}")
                        
                    elif intent_override == 'artist_similarity':
                        # Include only tracks by the target artist for artist similarity
                        should_include = (artist == target_entity_lower)
                        self.logger.info(f"🔧 EXTRACT DEBUG: Artist similarity - artist match: {artist} == {target_entity_lower} = {should_include}")
                        
                    elif intent_override == 'artist_style_refinement':
                        # Include tracks by the target artist for style refinement
                        should_include = (artist == target_entity_lower)
                        self.logger.info(f"🔧 EXTRACT DEBUG: Artist style refinement - artist match: {artist} == {target_entity_lower} = {should_include}")
                    elif intent_override == 'by_artist':
                        # ✅ NEW: Include tracks by the target artist for by_artist intent
                        should_include = (artist == target_entity_lower)
                        self.logger.info(f"🔧 EXTRACT DEBUG: By artist - artist match: {artist} == {target_entity_lower} = {should_include}")
                    else:
                        self.logger.info(f"🔧 EXTRACT DEBUG: Unknown intent '{intent_override}' - not including {track_id}")
                    
                    if should_include:
                        recently_shown_ids.append(track_id)
                        self.logger.info(f"🔧 EXTRACT DEBUG: ✅ ADDED TO AVOID LIST: {track_id}")
                    else:
                        self.logger.info(f"🔧 EXTRACT DEBUG: ❌ EXCLUDED: {track_id}")
            else:
                self.logger.info(f"🔧 EXTRACT DEBUG: Turn {i} has no recommendations or invalid format")
        
        # 🔧 NEW: Session context fallback if we got very few tracks from chat interface
        if len(recently_shown_ids) <= 2:
            self.logger.info(f"🔧 EXTRACT DEBUG: Only {len(recently_shown_ids)} tracks from chat interface, trying session context fallback")
            session_tracks = self._extract_from_session_context(context_override, workflow_state)
            if session_tracks:
                self.logger.info(f"🔧 EXTRACT DEBUG: Found {len(session_tracks)} additional tracks from session context")
                recently_shown_ids.extend(session_tracks)
        
        # Remove duplicates while preserving order
        unique_ids = []
        seen = set()
        for track_id in recently_shown_ids:
            if track_id not in seen:
                unique_ids.append(track_id)
                seen.add(track_id)
        
        self.logger.info(f"🔧 EXTRACT DEBUG: Final result: {len(unique_ids)} unique track IDs: {unique_ids}")
        return unique_ids
    
    def _extract_from_session_context(self, context_override: Dict[str, Any], workflow_state: MusicRecommenderState) -> List[str]:
        """Extract recently shown tracks from backend session context as fallback."""
        try:
            intent_override = context_override.get('intent_override')
            target_entity = context_override.get('target_entity', '')
            target_entity_lower = target_entity.lower() if target_entity else ''
            
            session_context = workflow_state.conversation_context
            if not session_context:
                self.logger.info("🔧 SESSION FALLBACK: No session context available in workflow state")
                return []
            
            interaction_history = session_context.get('interaction_history', [])
            if not interaction_history:
                self.logger.info("🔧 SESSION FALLBACK: No interaction history in session context")
                return []
            
            self.logger.info(f"🔧 SESSION FALLBACK: Processing {len(interaction_history)} interactions from session context")
            
            session_track_ids = []
            
            # Look at recent interactions (last 2-3 interactions)
            recent_interactions = interaction_history[-3:]
            
            for i, interaction in enumerate(recent_interactions):
                self.logger.info(f"🔧 SESSION FALLBACK: Interaction {i}: {list(interaction.keys())}")
                
                recommendations = interaction.get('recommendations', [])
                if recommendations:
                    self.logger.info(f"🔧 SESSION FALLBACK: Interaction {i} has {len(recommendations)} recommendations")
                    
                    for j, rec in enumerate(recommendations):
                        if isinstance(rec, dict):
                            # Try different possible formats from session context
                            artist = rec.get('artist') or rec.get('artist_name', '')
                            title = rec.get('title') or rec.get('track_name') or rec.get('name', '')
                            
                            if artist and title:
                                artist = str(artist).lower().strip()
                                title = str(title).lower().strip()
                                track_id = f"{artist}::{title}"
                                
                                # Apply same intent filtering logic
                                should_include = False
                                
                                if intent_override == 'style_continuation':
                                    should_include = True
                                elif intent_override in ['artist_deep_dive', 'artist_similarity', 'artist_style_refinement', 'by_artist']:
                                    should_include = (artist == target_entity_lower)
                                
                                if should_include:
                                    session_track_ids.append(track_id)
                                    self.logger.info(f"🔧 SESSION FALLBACK: ✅ INCLUDED: {track_id}")
                                else:
                                    self.logger.info(f"🔧 SESSION FALLBACK: ❌ EXCLUDED: {track_id}")
                            else:
                                self.logger.info(f"🔧 SESSION FALLBACK: Skipping rec {j} - missing artist ({artist}) or title ({title})")
                else:
                    self.logger.info(f"🔧 SESSION FALLBACK: Interaction {i} has no recommendations")
            
            self.logger.info(f"🔧 SESSION FALLBACK: Found {len(session_track_ids)} tracks from session context")
            return session_track_ids
            
        except Exception as e:
            self.logger.error(f"🔧 SESSION FALLBACK: Error extracting from session context: {e}")
            return []
    
    async def _save_recommendations_to_history(
        self, 
        session_id: str, 
        recommendations: List[TrackRecommendation]
    ) -> None:
        """Save recommendations to conversation history for future duplicate avoidance."""
        # This could be implemented to persist recommendations
        # For now, just log the action
        self.logger.info(
            f"Would save {len(recommendations)} recommendations to history for session {session_id}"
        )
        pass
    
    async def _convert_to_unified_metadata(
        self,
        recommendations: List[Union[Dict[str, Any], "TrackRecommendation"]],
        include_audio_features: bool = True
    ) -> List[UnifiedTrackMetadata]:
        """
        Convert agent recommendations to unified metadata format.
        
        Args:
            recommendations: Raw recommendations from agents (can be dicts or TrackRecommendation objects)
            include_audio_features: Whether to include Spotify audio features
            
        Returns:
            List of unified track metadata
        """
        unified_tracks = []
        
        self.logger.debug(f"Converting {len(recommendations)} recommendations to unified metadata")
        for i, rec in enumerate(recommendations):
            self.logger.debug(f"Recommendation {i}: type={type(rec)}, keys={list(rec.keys()) if isinstance(rec, dict) else 'N/A'}")
        
        for rec in recommendations:
            try:
                # Handle both TrackRecommendation objects and dictionaries
                if hasattr(rec, 'title'):  # TrackRecommendation object
                    artist = rec.artist
                    track = rec.title
                    confidence = rec.confidence
                    explanation = rec.explanation
                    source = rec.source
                elif isinstance(rec, dict):  # Dictionary format
                    artist = rec.get('artist', '')
                    track = rec.get('title', '') or rec.get('name', '')
                    confidence = rec.get('confidence', 0.0)
                    explanation = rec.get('explanation', '')
                    source = rec.get('source', 'unknown')
                else:
                    self.logger.warning(f"Unknown recommendation format: {type(rec)}")
                    continue
                
                if not artist or not track:
                    self.logger.debug(f"Skipping recommendation missing artist/title: artist='{artist}', track='{track}'")
                    continue
                
                # Get comprehensive track info using API service
                unified_track = await self.api_service.get_unified_track_info(
                    artist=artist,
                    track=track,
                    include_audio_features=include_audio_features
                )
                
                self.logger.debug(f"API service returned unified_track type: {type(unified_track)}")
                
                if unified_track:
                    self.logger.debug(f"unified_track is not None, type: {type(unified_track)}")
                    self.logger.debug(f"unified_track has __dict__: {hasattr(unified_track, '__dict__')}")
                    if hasattr(unified_track, '__dict__'):
                        self.logger.debug(f"unified_track attributes: {list(unified_track.__dict__.keys())}")
                    
                    # Add recommendation-specific metadata
                    # Handle both object and dictionary formats for metadata
                    try:
                        if hasattr(rec, 'confidence'):  # TrackRecommendation object
                            self.logger.debug("Setting metadata from TrackRecommendation object")
                            unified_track.recommendation_score = rec.confidence
                            unified_track.recommendation_reason = rec.explanation
                            unified_track.agent_source = rec.source
                        elif isinstance(rec, dict):  # Dictionary format
                            self.logger.debug("Setting metadata from dictionary format")
                            unified_track.recommendation_score = rec.get('confidence', 0.0)
                            unified_track.recommendation_reason = rec.get('explanation', '')
                            unified_track.agent_source = rec.get('source', 'unknown')
                        else:
                            # Fallback values
                            self.logger.debug("Setting metadata from fallback values")
                            unified_track.recommendation_score = confidence
                            unified_track.recommendation_reason = explanation
                            unified_track.agent_source = source
                    except Exception as metadata_error:
                        self.logger.error(f"Error setting metadata on unified_track: {metadata_error}")
                        self.logger.error(f"unified_track type: {type(unified_track)}")
                        self.logger.error(f"Trying to set recommendation_score to: {rec.get('confidence', 0.0) if isinstance(rec, dict) else confidence}")
                        raise metadata_error
                    
                    unified_tracks.append(unified_track)
                    self.logger.debug(f"Successfully converted: {artist} - {track}")
                else:
                    self.logger.debug(f"Failed to get unified track info for {artist} - {track}")
                    
            except Exception as e:
                # Safely extract artist and track names for error logging
                if hasattr(rec, 'artist'):  # TrackRecommendation object
                    artist_name = rec.artist
                    track_name = rec.title
                elif isinstance(rec, dict):  # Dictionary format
                    artist_name = rec.get('artist', 'unknown')
                    track_name = rec.get('title', rec.get('name', 'unknown'))
                else:
                    artist_name = 'unknown'
                    track_name = 'unknown'
                    
                self.logger.warning(
                    "Failed to convert recommendation to unified metadata",
                    artist=artist_name,
                    title=track_name,
                    error=str(e)
                )
                continue
        
        self.logger.info(f"Converted {len(unified_tracks)} out of {len(recommendations)} recommendations to unified metadata")
        return unified_tracks
    
    async def _get_fallback_recommendations(
        self,
        query: str,
        max_recommendations: int
    ) -> List[UnifiedTrackMetadata]:
        """
        Get fallback recommendations using direct API service calls.
        
        Args:
            query: User query
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of fallback recommendations
        """
        try:
            # Use API service for direct search
            fallback_tracks = await self.api_service.search_unified_tracks(
                query=query,
                limit=max_recommendations,
                include_spotify=True
            )
            
            self.logger.info(
                "Fallback recommendations generated",
                query=query,
                count=len(fallback_tracks)
            )
            
            return fallback_tracks
            
        except Exception as e:
            self.logger.error("Fallback recommendations failed", error=str(e))
            return []
    
    async def get_similar_tracks(
        self,
        artist: str,
        track: str,
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """
        Get similar tracks using the API service.
        
        Args:
            artist: Artist name
            track: Track name
            limit: Maximum results
            
        Returns:
            List of similar tracks with unified metadata
        """
        return await self.api_service.get_similar_tracks(
            artist=artist,
            track=track,
            limit=limit,
            include_spotify_features=True
        )
    
    async def search_by_tags(
        self,
        tags: List[str],
        limit: int = 20
    ) -> List[UnifiedTrackMetadata]:
        """
        Search tracks by tags using the API service.
        
        Args:
            tags: List of tags
            limit: Maximum results
            
        Returns:
            List of tracks matching tags
        """
        return await self.api_service.search_by_tags(tags=tags, limit=limit)
    
    async def get_planning_strategy(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get planning strategy from PlannerAgent without executing full workflow.
        
        Args:
            query: User query
            session_id: Session identifier
            
        Returns:
            Planning strategy dictionary
        """
        # Ensure agents are initialized
        await self.initialize_agents()
        
        # Create minimal state for planning
        initial_state = MusicRecommenderState(
            user_query=query,
            session_id=session_id or "default"
        )
        
        try:
            # Execute only the planner node
            planning_state = await self._planner_node(initial_state)
            
            return {
                "planning_strategy": getattr(planning_state, 'planning_strategy', {}),
                "query_understanding": getattr(planning_state, 'query_understanding', {}),
                "agent_coordination": getattr(planning_state, 'agent_coordination', {}),
                "session_id": planning_state.session_id
            }
            
        except Exception as e:
            self.logger.error("Planning strategy generation failed", error=str(e))
            return {
                "planning_strategy": {"type": "fallback", "reason": str(e)},
                "query_understanding": {"intent": "discovery", "confidence": 0.3},
                "agent_coordination": {"strategy": "simple"},
                "session_id": session_id or "default"
            }
    
    @property
    def smart_context_manager(self) -> SmartContextManager:
        """
        Access to smart context manager for backward compatibility.
        
        Returns:
            SmartContextManager instance
        """
        return self.context_manager
    
    @property
    def intent_orchestrator(self) -> IntentOrchestrationService:
        """
        Get the IntentOrchestrationService instance.
        
        Returns:
            IntentOrchestrationService instance
        """
        if not self._intent_orchestrator:
            raise RuntimeError("Intent orchestrator not initialized. Call initialize_agents() first.")
        return self._intent_orchestrator
    
    async def close(self):
        """Close all service connections."""
        await self.api_service.close()
        if self.cache_manager:
            self.cache_manager.close()
        
        self.logger.info("Enhanced Recommendation Service closed")


# Global service instance
_global_recommendation_service: Optional[EnhancedRecommendationService] = None


def get_recommendation_service(
    system_config: Optional[SystemConfig] = None,
    api_service: Optional[APIService] = None,
    cache_manager: Optional[CacheManager] = None
) -> EnhancedRecommendationService:
    """
    Get global recommendation service instance.
    
    Args:
        system_config: System configuration (optional)
        api_service: API service instance (optional)
        cache_manager: Cache manager instance (optional)
        
    Returns:
        Global EnhancedRecommendationService instance
    """
    global _global_recommendation_service
    
    if _global_recommendation_service is None:
        _global_recommendation_service = EnhancedRecommendationService(
            system_config=system_config,
            api_service=api_service,
            cache_manager=cache_manager
        )
    
    return _global_recommendation_service


async def close_recommendation_service():
    """Close global recommendation service."""
    global _global_recommendation_service
    
    if _global_recommendation_service:
        await _global_recommendation_service.close()
        _global_recommendation_service = None


def create_gemini_client(api_key: str):
    """
    Create a Gemini client for LLM interactions.
    
    Args:
        api_key: Gemini API key
        
    Returns:
        Configured Gemini client
    """
    try:
        import google.generativeai as genai
        
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Create and return the model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        logger.info("Gemini client created successfully")
        return model
        
    except ImportError:
        logger.error("google-generativeai library not installed")
        return None
    except Exception as e:
        logger.error("Failed to create Gemini client", error=str(e))
        return None 