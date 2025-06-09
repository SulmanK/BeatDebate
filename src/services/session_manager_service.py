"""
Session Manager Service for BeatDebate

Enhanced service that consolidates conversation context management with:
- Original intent and entity storage for accurate follow-up interpretation
- Candidate pool persistence for efficient "load more" functionality
- Session state management and user preference evolution
- Smart context decision making

This service replaces and enhances ConversationContextService and consolidates
logic from SmartContextManager.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import structlog
from dataclasses import dataclass, asdict

from ..models.metadata_models import UnifiedTrackMetadata

logger = structlog.get_logger(__name__)


class ContextState(Enum):
    """States of conversation context."""
    NEW_SESSION = "new_session"
    CONTINUING = "continuing"
    INTENT_SWITCH = "intent_switch"
    PREFERENCE_REFINEMENT = "preference_refinement"
    RESET_NEEDED = "reset_needed"


@dataclass
class OriginalQueryContext:
    """Stores the original query's parsed intent and entities for follow-up resolution."""
    query: str
    intent: str
    entities: Dict[str, Any]
    timestamp: datetime
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OriginalQueryContext':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class CandidatePool:
    """Stores a larger pool of candidates for efficient follow-up queries."""
    candidates: List[UnifiedTrackMetadata]
    generated_for_intent: str
    generated_for_entities: Dict[str, Any]
    timestamp: datetime
    used_count: int = 0
    max_usage: int = 3  # How many times this pool can be reused
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'candidates': [candidate.to_dict() for candidate in self.candidates],
            'generated_for_intent': self.generated_for_intent,
            'generated_for_entities': self.generated_for_entities,
            'timestamp': self.timestamp.isoformat(),
            'used_count': self.used_count,
            'max_usage': self.max_usage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandidatePool':
        """Create from dictionary."""
        candidates = [UnifiedTrackMetadata.from_dict(c) for c in data['candidates']]
        return cls(
            candidates=candidates,
            generated_for_intent=data['generated_for_intent'],
            generated_for_entities=data['generated_for_entities'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            used_count=data.get('used_count', 0),
            max_usage=data.get('max_usage', 3)
        )
    
    def is_expired(self, max_age_minutes: int = 60) -> bool:
        """Check if the candidate pool is too old to be useful."""
        age = datetime.now() - self.timestamp
        return age > timedelta(minutes=max_age_minutes)
    
    def is_exhausted(self) -> bool:
        """Check if the candidate pool has been used too many times."""
        return self.used_count >= self.max_usage
    
    def can_be_reused(self, max_age_minutes: int = 60) -> bool:
        """Check if the candidate pool can still be reused."""
        return not self.is_expired(max_age_minutes) and not self.is_exhausted()


class SessionManagerService:
    """
    Enhanced session manager that consolidates conversation context management
    with intelligent intent resolution and candidate pool persistence.
    
    Key Features:
    - Original intent and entity storage for accurate follow-up interpretation
    - Candidate pool persistence for efficient "load more" functionality
    - Smart context decision making
    - User preference evolution tracking
    - Session state management
    """
    
    def __init__(self, cache_manager=None):
        """Initialize session manager service."""
        self.session_store = {}
        self.cache_manager = cache_manager
        self.logger = logger.bind(component="SessionManager")
        
        # Configuration
        self.context_decay_minutes = 30
        self.candidate_pool_max_age_minutes = 60
        self.max_interactions_per_session = 100
        
        # Reset triggers for context clearing
        self.reset_triggers = [
            "actually", "instead", "never mind", "different", "change of mind",
            "new request", "start over", "forget that", "something completely different"
        ]
        
        self.logger.info("Session Manager Service initialized")
    
    async def create_or_update_session(
        self, 
        session_id: str, 
        query: str,
        intent: str,
        entities: Dict[str, Any],
        recommendations: Optional[List[UnifiedTrackMetadata]] = None,
        user_feedback: Optional[Dict] = None,
        is_original_query: bool = True
    ) -> Dict[str, Any]:
        """
        Create new session or update existing session with interaction data.
        
        Args:
            session_id: Unique session identifier
            query: User query
            intent: Parsed intent from the query
            entities: Extracted entities
            recommendations: Generated recommendations
            user_feedback: User feedback on recommendations
            is_original_query: Whether this is an original query (not a follow-up)
            
        Returns:
            Updated session context
        """
        self.logger.info("Creating/updating session", session_id=session_id, is_original=is_original_query)
        
        if session_id not in self.session_store:
            self.session_store[session_id] = {
                "interaction_history": [],
                "original_query_context": None,  # NEW: Store original intent/entities
                "candidate_pools": {},  # NEW: Store candidate pools by intent
                "preference_profile": {
                    "preferred_genres": {},
                    "preferred_artists": {},
                    "preferred_moods": {},
                    "preferred_activities": {},
                    "discovery_openness": 0.5,
                    "quality_preferences": {}
                },
                "recommendation_history": [],
                "entity_evolution": {},
                "session_start": datetime.now(),
                "last_updated": datetime.now(),
                "context_state": ContextState.NEW_SESSION.value
            }
        
        session = self.session_store[session_id]
        
        # Add interaction to history
        interaction = {
            "timestamp": datetime.now(),
            "query": query,
            "intent": intent,
            "extracted_entities": entities,
            "recommendations": [rec.to_dict() for rec in recommendations] if recommendations else [],
            "user_feedback": user_feedback,
            "is_original_query": is_original_query
        }
        session["interaction_history"].append(interaction)
        
        # Store original query context for follow-up resolution
        if is_original_query:
            session["original_query_context"] = OriginalQueryContext(
                query=query,
                intent=intent,
                entities=entities,
                timestamp=datetime.now(),
                confidence=1.0
            ).to_dict()
            self.logger.info("Stored original query context", intent=intent, entities=list(entities.keys()))
        
        # Update recommendation history
        if recommendations:
            session["recommendation_history"].extend([rec.to_dict() for rec in recommendations])
        
        # Update preference profile
        await self._update_preference_profile(session, entities, user_feedback)
        
        # Track entity evolution
        await self._track_entity_evolution(session, entities)
        
        # Update timestamp and context state
        session["last_updated"] = datetime.now()
        session["context_state"] = self._determine_context_state(session, query)
        
        # Clean up old data if session is getting too large
        await self._cleanup_session_if_needed(session)
        
        self.logger.info(
            "Session updated", 
            session_id=session_id,
            interaction_count=len(session["interaction_history"]),
            context_state=session["context_state"]
        )
        
        return session
    
    async def store_candidate_pool(
        self,
        session_id: str,
        candidates: List[UnifiedTrackMetadata],
        intent: str,
        entities: Dict[str, Any],
        pool_key: Optional[str] = None
    ) -> str:
        """
        Store a candidate pool for efficient follow-up queries.
        
        Args:
            session_id: Session identifier
            candidates: List of candidate tracks
            intent: Intent this pool was generated for
            entities: Entities this pool was generated for
            pool_key: Optional custom key for the pool
            
        Returns:
            Key used to store the pool
        """
        if session_id not in self.session_store:
            await self.create_or_update_session(session_id, "", intent, entities, is_original_query=False)
        
        session = self.session_store[session_id]
        
        # Generate pool key if not provided
        if not pool_key:
            pool_key = f"{intent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create candidate pool
        candidate_pool = CandidatePool(
            candidates=candidates,
            generated_for_intent=intent,
            generated_for_entities=entities,
            timestamp=datetime.now()
        )
        
        # Store in session
        session["candidate_pools"][pool_key] = candidate_pool.to_dict()
        
        self.logger.info(
            "Stored candidate pool",
            session_id=session_id,
            pool_key=pool_key,
            candidate_count=len(candidates),
            intent=intent
        )
        
        return pool_key
    
    async def get_candidate_pool(
        self,
        session_id: str,
        intent: str,
        entities: Dict[str, Any],
        pool_key: Optional[str] = None
    ) -> Optional[CandidatePool]:
        """
        Retrieve a candidate pool for follow-up queries.
        
        Args:
            session_id: Session identifier
            intent: Intent to match
            entities: Entities to match
            pool_key: Specific pool key to retrieve
            
        Returns:
            CandidatePool if found and still valid, None otherwise
        """
        if session_id not in self.session_store:
            return None
        
        session = self.session_store[session_id]
        candidate_pools = session.get("candidate_pools", {})
        
        if not candidate_pools:
            return None
        
        # If specific pool key provided, try to get it
        if pool_key and pool_key in candidate_pools:
            pool_data = candidate_pools[pool_key]
            candidate_pool = CandidatePool.from_dict(pool_data)
            
            if candidate_pool.can_be_reused(self.candidate_pool_max_age_minutes):
                candidate_pool.used_count += 1
                candidate_pools[pool_key] = candidate_pool.to_dict()  # Update usage count
                return candidate_pool
            else:
                # Remove expired/exhausted pool
                del candidate_pools[pool_key]
                return None
        
        # Otherwise, find the most recent compatible pool
        compatible_pools = []
        for key, pool_data in candidate_pools.items():
            candidate_pool = CandidatePool.from_dict(pool_data)
            
            # Check if pool is compatible and still usable
            if (candidate_pool.generated_for_intent == intent and 
                candidate_pool.can_be_reused(self.candidate_pool_max_age_minutes)):
                compatible_pools.append((key, candidate_pool))
        
        if compatible_pools:
            # Sort by timestamp (most recent first)
            compatible_pools.sort(key=lambda x: x[1].timestamp, reverse=True)
            pool_key, candidate_pool = compatible_pools[0]
            
            # Update usage count
            candidate_pool.used_count += 1
            candidate_pools[pool_key] = candidate_pool.to_dict()
            
            self.logger.info(
                "Retrieved candidate pool",
                session_id=session_id,
                pool_key=pool_key,
                usage_count=candidate_pool.used_count
            )
            
            return candidate_pool
        
        return None
    
    async def get_original_query_context(self, session_id: str) -> Optional[OriginalQueryContext]:
        """
        Get the original query context for follow-up resolution.
        
        Args:
            session_id: Session identifier
            
        Returns:
            OriginalQueryContext if available, None otherwise
        """
        if session_id not in self.session_store:
            return None
        
        session = self.session_store[session_id]
        original_context_data = session.get("original_query_context")
        
        if original_context_data:
            return OriginalQueryContext.from_dict(original_context_data)
        
        return None
    
    async def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current session context.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session context or None if not found
        """
        return self.session_store.get(session_id)
    
    async def analyze_context_decision(
        self,
        current_query: str,
        session_id: str,
        current_intent: Optional[str] = None,
        current_entities: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze whether to maintain, modify, or reset conversation context.
        
        Args:
            current_query: User's current query
            session_id: Session identifier
            current_intent: Current query's intent (if available)
            current_entities: Current query's entities (if available)
            
        Returns:
            Context decision with recommendations
        """
        self.logger.info("Analyzing context decision", session_id=session_id)
        
        # Get current session context
        session_context = await self.get_session_context(session_id)
        
        if not session_context:
            return {
                "decision": ContextState.NEW_SESSION.value,
                "action": "create_new_context",
                "confidence": 1.0,
                "reasoning": "No existing session context found",
                "context_to_use": None,
                "reset_context": False,
                "is_followup": False
            }
        
        # Check for explicit reset triggers
        reset_trigger = self._check_reset_triggers(current_query)
        if reset_trigger:
            return {
                "decision": ContextState.RESET_NEEDED.value,
                "action": "reset_context",
                "confidence": 0.9,
                "reasoning": f"Explicit reset trigger detected: '{reset_trigger}'",
                "context_to_use": None,
                "reset_context": True,
                "is_followup": False
            }
        
        # Check temporal relevance
        temporal_analysis = self._analyze_temporal_relevance(session_context)
        if temporal_analysis["is_stale"]:
            return {
                "decision": ContextState.RESET_NEEDED.value,
                "action": "reset_context",
                "confidence": temporal_analysis["confidence"],
                "reasoning": temporal_analysis["reasoning"],
                "context_to_use": None,
                "reset_context": True,
                "is_followup": False
            }
        
        # Analyze for follow-up patterns
        followup_analysis = await self._analyze_followup_patterns(
            current_query, session_context, current_intent, current_entities
        )
        
        if followup_analysis["is_followup"]:
            return {
                "decision": ContextState.CONTINUING.value,
                "action": "use_context_with_followup",
                "confidence": followup_analysis["confidence"],
                "reasoning": followup_analysis["reasoning"],
                "context_to_use": session_context,
                "reset_context": False,
                "is_followup": True,
                "followup_type": followup_analysis["followup_type"],
                "original_context": followup_analysis.get("original_context")
            }
        
        # Check for intent switch
        if current_intent and session_context.get("original_query_context"):
            original_intent = session_context["original_query_context"]["intent"]
            if current_intent != original_intent:
                return {
                    "decision": ContextState.INTENT_SWITCH.value,
                    "action": "create_new_context",
                    "confidence": 0.8,
                    "reasoning": f"Intent switch detected: {original_intent} -> {current_intent}",
                    "context_to_use": None,
                    "reset_context": True,
                    "is_followup": False
                }
        
        # Default: continue with existing context
        return {
            "decision": ContextState.CONTINUING.value,
            "action": "use_existing_context",
            "confidence": 0.6,
            "reasoning": "Continuing with existing context",
            "context_to_use": session_context,
            "reset_context": False,
            "is_followup": False
        }
    
    async def get_recommendations_excluding_seen(
        self,
        candidates: List[UnifiedTrackMetadata],
        session_id: str,
        max_results: int = 10
    ) -> List[UnifiedTrackMetadata]:
        """
        Filter out tracks that have been recently shown to the user.
        
        Args:
            candidates: List of candidate tracks
            session_id: Session identifier
            max_results: Maximum number of results to return
            
        Returns:
            Filtered list of tracks excluding recently shown ones
        """
        session_context = await self.get_session_context(session_id)
        if not session_context:
            return candidates[:max_results]
        
        # Extract recently shown track IDs
        recently_shown_ids = set()
        recommendation_history = session_context.get("recommendation_history", [])
        
        for rec in recommendation_history:
            if isinstance(rec, dict) and "id" in rec:
                recently_shown_ids.add(rec["id"])
            elif hasattr(rec, "id"):
                recently_shown_ids.add(rec.id)
        
        # Filter out recently shown tracks
        filtered_candidates = []
        for candidate in candidates:
            if candidate.id not in recently_shown_ids:
                filtered_candidates.append(candidate)
                if len(filtered_candidates) >= max_results:
                    break
        
        self.logger.info(
            "Filtered recommendations",
            session_id=session_id,
            original_count=len(candidates),
            filtered_count=len(filtered_candidates),
            excluded_count=len(recently_shown_ids)
        )
        
        return filtered_candidates
    
    async def clear_session(self, session_id: str):
        """Clear session data."""
        if session_id in self.session_store:
            del self.session_store[session_id]
            self.logger.info("Session cleared", session_id=session_id)
    
    def _check_reset_triggers(self, query: str) -> Optional[str]:
        """Check if query contains explicit reset triggers."""
        query_lower = query.lower().strip()
        for trigger in self.reset_triggers:
            if trigger in query_lower:
                return trigger
        return None
    
    def _analyze_temporal_relevance(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if session context is temporally relevant."""
        last_updated = session_context.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        elif not isinstance(last_updated, datetime):
            last_updated = datetime.now() - timedelta(hours=1)  # Assume stale
        
        age_minutes = (datetime.now() - last_updated).total_seconds() / 60
        
        if age_minutes > self.context_decay_minutes:
            return {
                "is_stale": True,
                "confidence": min(0.9, age_minutes / self.context_decay_minutes),
                "reasoning": f"Context is {age_minutes:.1f} minutes old (threshold: {self.context_decay_minutes})"
            }
        
        return {
            "is_stale": False,
            "confidence": 0.8,
            "reasoning": f"Context is recent ({age_minutes:.1f} minutes old)"
        }
    
    async def _analyze_followup_patterns(
        self,
        current_query: str,
        session_context: Dict[str, Any],
        current_intent: Optional[str] = None,
        current_entities: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze if current query is a follow-up to previous queries."""
        query_lower = current_query.lower().strip()
        
        # Common follow-up patterns
        followup_patterns = [
            r"more\s+(like\s+)?(this|that|these|those)",
            r"more\s+tracks?",
            r"more\s+songs?",
            r"more\s+music",
            r"similar\s+(to\s+)?(this|that|these|those)",
            r"something\s+else",
            r"what\s+about",
            r"also",
            r"and\s+",
            r"continue",
            r"keep\s+going"
        ]
        
        import re
        is_followup_pattern = any(re.search(pattern, query_lower) for pattern in followup_patterns)
        
        if not is_followup_pattern:
            return {
                "is_followup": False,
                "confidence": 0.1,
                "reasoning": "No follow-up patterns detected"
            }
        
        # Get original query context
        original_context = session_context.get("original_query_context")
        if not original_context:
            return {
                "is_followup": False,
                "confidence": 0.2,
                "reasoning": "No original query context available"
            }
        
        # Determine follow-up type
        followup_type = "style_continuation"  # Default
        
        if "more" in query_lower and any(word in query_lower for word in ["tracks", "songs", "music"]):
            followup_type = "more_content"
        elif "similar" in query_lower:
            followup_type = "similarity_exploration"
        elif any(word in query_lower for word in ["else", "different", "other"]):
            followup_type = "variation_request"
        
        return {
            "is_followup": True,
            "confidence": 0.8,
            "reasoning": f"Follow-up pattern detected: {followup_type}",
            "followup_type": followup_type,
            "original_context": OriginalQueryContext.from_dict(original_context)
        }
    
    def _determine_context_state(self, session: Dict[str, Any], current_query: str) -> str:
        """Determine the current context state based on session and query."""
        interaction_count = len(session.get("interaction_history", []))
        
        if interaction_count == 1:
            return ContextState.NEW_SESSION.value
        elif self._check_reset_triggers(current_query):
            return ContextState.RESET_NEEDED.value
        else:
            return ContextState.CONTINUING.value
    
    async def _update_preference_profile(
        self, 
        session: Dict[str, Any], 
        entities: Dict[str, Any],
        user_feedback: Optional[Dict] = None
    ):
        """Update user preference profile based on entities and feedback."""
        preference_profile = session["preference_profile"]
        
        # Update genre preferences
        if "genres" in entities:
            genres = entities["genres"]
            if isinstance(genres, dict):
                for genre_list in [genres.get("primary", []), genres.get("secondary", [])]:
                    for genre in genre_list:
                        genre_name = genre if isinstance(genre, str) else genre.get("name", "")
                        if genre_name:
                            preference_profile["preferred_genres"][genre_name] = (
                                preference_profile["preferred_genres"].get(genre_name, 0) + 1
                            )
        
        # Update artist preferences
        if "artists" in entities:
            artists = entities["artists"]
            if isinstance(artists, list):
                for artist in artists:
                    artist_name = artist if isinstance(artist, str) else artist.get("name", "")
                    if artist_name:
                        preference_profile["preferred_artists"][artist_name] = (
                            preference_profile["preferred_artists"].get(artist_name, 0) + 1
                        )
        
        # Update mood preferences
        if "moods" in entities:
            moods = entities["moods"]
            if isinstance(moods, dict):
                for mood_list in [moods.get("primary", []), moods.get("secondary", [])]:
                    for mood in mood_list:
                        mood_name = mood if isinstance(mood, str) else mood.get("name", "")
                        if mood_name:
                            preference_profile["preferred_moods"][mood_name] = (
                                preference_profile["preferred_moods"].get(mood_name, 0) + 1
                            )
        
        # Apply user feedback if provided
        if user_feedback:
            # This could be enhanced based on specific feedback structure
            pass
    
    async def _track_entity_evolution(self, session: Dict[str, Any], entities: Dict[str, Any]):
        """Track how entities evolve across the session."""
        entity_evolution = session.get("entity_evolution", {})
        
        for entity_type, entity_data in entities.items():
            if entity_type not in entity_evolution:
                entity_evolution[entity_type] = []
            
            if isinstance(entity_data, list):
                for entity in entity_data:
                    if isinstance(entity, str):
                        entity_name = entity
                    elif isinstance(entity, dict):
                        entity_name = entity.get('name', str(entity))
                    else:
                        entity_name = str(entity)
                    
                    if entity_name not in [e['name'] for e in entity_evolution[entity_type]]:
                        entity_evolution[entity_type].append({
                            'name': entity_name,
                            'first_mentioned': datetime.now(),
                            'frequency': 1
                        })
                    else:
                        for e in entity_evolution[entity_type]:
                            if e['name'] == entity_name:
                                e['frequency'] += 1
                                break
        
        session["entity_evolution"] = entity_evolution
    
    async def save_recommendations(self, session_id: str, recommendations_data: Dict[str, Any]):
        """
        Save recommendations to session history.
        
        Args:
            session_id: Session identifier
            recommendations_data: Recommendation data to save
        """
        if session_id not in self.session_store:
            self.logger.warning(f"Session {session_id} not found for saving recommendations")
            return
        
        session = self.session_store[session_id]
        
        # Add to recommendation history
        recommendation_entry = {
            "timestamp": datetime.now(),
            "recommendations": recommendations_data,
            "session_id": session_id
        }
        
        if "recommendation_history" not in session:
            session["recommendation_history"] = []
        
        session["recommendation_history"].append(recommendation_entry)
        session["last_updated"] = datetime.now()
        
        self.logger.info(f"Saved recommendations to session {session_id}")
    
    async def _cleanup_session_if_needed(self, session: Dict[str, Any]):
        """Clean up session if it becomes too large or old."""
        interaction_count = len(session.get("interaction_history", []))
        
        # Clean up if too many interactions
        if interaction_count > self.max_interactions_per_session:
            # Keep only the most recent interactions
            keep_count = self.max_interactions_per_session // 2
            session["interaction_history"] = session["interaction_history"][-keep_count:]
            self.logger.info(f"Cleaned up session history, keeping {keep_count} recent interactions")
        
        # Clean up old candidate pools
        pools_to_remove = []
        for pool_key, pool_data in session.get("candidate_pools", {}).items():
            try:
                pool = CandidatePool.from_dict(pool_data)
                if pool.is_expired(self.candidate_pool_max_age_minutes):
                    pools_to_remove.append(pool_key)
            except Exception as e:
                self.logger.warning(f"Failed to parse candidate pool {pool_key}: {e}")
                pools_to_remove.append(pool_key)
        
        for pool_key in pools_to_remove:
            del session["candidate_pools"][pool_key]
            self.logger.debug(f"Removed expired candidate pool: {pool_key}") 