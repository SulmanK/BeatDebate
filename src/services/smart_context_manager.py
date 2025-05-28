"""
Smart Context Manager for BeatDebate

Handles intelligent conversation context management with:
- Intent change detection
- Conversation state transitions
- Context reset triggers
- User preference evolution tracking
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import re

from ..agents.conversation_context import ConversationContextManager


class IntentType(Enum):
    """Types of user intents in music discovery."""
    ARTIST_SIMILARITY = "artist_similarity"
    GENRE_EXPLORATION = "genre_exploration"
    MOOD_MATCHING = "mood_matching"
    ACTIVITY_CONTEXT = "activity_context"
    DISCOVERY = "discovery"
    PLAYLIST_BUILDING = "playlist_building"
    CONVERSATION_CONTINUATION = "conversation_continuation"
    FEEDBACK_RESPONSE = "feedback_response"


class ContextState(Enum):
    """States of conversation context."""
    NEW_SESSION = "new_session"
    CONTINUING = "continuing"
    INTENT_SWITCH = "intent_switch"
    PREFERENCE_REFINEMENT = "preference_refinement"
    RESET_NEEDED = "reset_needed"


class SmartContextManager:
    """
    Intelligent context manager that detects when to maintain vs reset context.
    
    Key Features:
    - Intent change detection
    - Context relevance scoring
    - Automatic context reset triggers
    - User preference evolution tracking
    - Conversation flow analysis
    """
    
    def __init__(self):
        """Initialize smart context manager."""
        self.logger = logging.getLogger(__name__)
        self.conversation_manager = ConversationContextManager()
        
        # Intent detection patterns
        self.intent_patterns = {
            IntentType.ARTIST_SIMILARITY: [
                r"music like\s+(.+)", r"similar to\s+(.+)", r"reminds me of\s+(.+)",
                r"sounds like\s+(.+)", r"artists like\s+(.+)", r"bands like\s+(.+)"
            ],
            IntentType.GENRE_EXPLORATION: [
                r"(?:some|more|new)\s+(.+)\s+music", r"explore\s+(.+)", 
                r"discover\s+(.+)", r"(.+)\s+genre", r"(.+)\s+style"
            ],
            IntentType.MOOD_MATCHING: [
                r"feeling\s+(.+)", r"mood for\s+(.+)", r"something\s+(.+)",
                r"(.+)\s+vibes?", r"make me feel\s+(.+)"
            ],
            IntentType.ACTIVITY_CONTEXT: [
                r"for\s+(working|studying|exercising|coding|driving|cooking)",
                r"while\s+(working|studying|exercising|coding|driving|cooking)",
                r"during\s+(work|study|exercise|coding|commute)"
            ],
            IntentType.CONVERSATION_CONTINUATION: [
                r"more like (?:that|this)", r"something else", r"what about",
                r"also", r"and", r"continue", r"keep going"
            ],
            IntentType.FEEDBACK_RESPONSE: [
                r"i (?:don't\s+)?like", r"not my thing", r"that's (?:not\s+)?good",
                r"(?:too|very)\s+(fast|slow|loud|quiet)", r"prefer something"
            ]
        }
        
        # Context reset triggers
        self.reset_triggers = [
            "actually", "instead", "never mind", "different", "change of mind",
            "new request", "start over", "forget that", "something completely different"
        ]
        
        # Time-based context decay (when to consider context stale)
        self.context_decay_minutes = 30
        
        self.logger.info("Smart Context Manager initialized")
    
    async def analyze_context_decision(
        self,
        current_query: str,
        session_id: str,
        llm_understanding: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze whether to maintain, modify, or reset conversation context.
        
        Args:
            current_query: User's current query
            session_id: Session identifier
            llm_understanding: Pure LLM understanding of the query
            
        Returns:
            Context decision with recommendations
        """
        self.logger.info(f"Analyzing context decision for session {session_id}")
        
        # Get current session context
        session_context = await self.conversation_manager.get_session_context(session_id)
        
        if not session_context:
            return {
                "decision": ContextState.NEW_SESSION.value,
                "action": "create_new_context",
                "confidence": 1.0,
                "reasoning": "No existing session context found",
                "context_to_use": None,
                "reset_context": False
            }
        
        # Analyze query for intent and context signals
        query_analysis = await self._analyze_query_intent(current_query, llm_understanding)
        
        # Check for explicit reset triggers
        reset_trigger = self._check_reset_triggers(current_query)
        if reset_trigger:
            return {
                "decision": ContextState.RESET_NEEDED.value,
                "action": "reset_context",
                "confidence": 0.9,
                "reasoning": f"Explicit reset trigger detected: '{reset_trigger}'",
                "context_to_use": None,
                "reset_context": True
            }
        
        # Analyze intent change
        intent_analysis = await self._analyze_intent_change(
            query_analysis, session_context
        )
        
        # Check temporal relevance
        temporal_analysis = self._analyze_temporal_relevance(session_context)
        
        # Analyze conversation continuity
        continuity_analysis = await self._analyze_conversation_continuity(
            current_query, query_analysis, session_context
        )
        
        # Make context decision
        context_decision = self._make_context_decision(
            query_analysis, intent_analysis, temporal_analysis, continuity_analysis
        )
        
        # Prepare context for use
        context_to_use = await self._prepare_context_for_use(
            context_decision, session_context, query_analysis
        )
        
        self.logger.info(
            f"Context decision: {context_decision['decision']} "
            f"(confidence: {context_decision['confidence']:.2f})"
        )
        
        return {
            **context_decision,
            "context_to_use": context_to_use,
            "session_analysis": {
                "query_analysis": query_analysis,
                "intent_analysis": intent_analysis,
                "temporal_analysis": temporal_analysis,
                "continuity_analysis": continuity_analysis
            }
        }
    
    async def _analyze_query_intent(
        self,
        query: str,
        llm_understanding: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Analyze the intent of the current query."""
        query_lower = query.lower().strip()
        
        # Use LLM understanding if available
        if llm_understanding:
            return {
                "primary_intent": llm_understanding.get("intent", {}).get("value", "unknown"),
                "artists": llm_understanding.get("artists", []),
                "genres": llm_understanding.get("genres", []),
                "moods": llm_understanding.get("moods", []),
                "activities": llm_understanding.get("activities", []),
                "confidence": llm_understanding.get("confidence", 0.5),
                "source": "llm_understanding"
            }
        
        # Fallback to pattern matching
        detected_intent = None
        intent_confidence = 0.0
        matched_entities = []
        
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    detected_intent = intent_type.value
                    intent_confidence = 0.7  # Pattern matching confidence
                    if match.groups():
                        matched_entities.extend(match.groups())
                    break
            if detected_intent:
                break
        
        return {
            "primary_intent": detected_intent or "discovery",
            "matched_entities": matched_entities,
            "confidence": intent_confidence,
            "source": "pattern_matching",
            "raw_query": query
        }
    
    def _check_reset_triggers(self, query: str) -> Optional[str]:
        """Check for explicit context reset triggers."""
        query_lower = query.lower()
        
        for trigger in self.reset_triggers:
            if trigger in query_lower:
                return trigger
        
        return None
    
    async def _analyze_intent_change(
        self,
        current_analysis: Dict[str, Any],
        session_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze if there's a significant intent change."""
        interaction_history = session_context.get("interaction_history", [])
        
        if not interaction_history:
            return {
                "intent_changed": False,
                "change_type": None,
                "confidence": 0.0,
                "previous_intent": None
            }
        
        # Get last interaction's intent
        last_interaction = interaction_history[-1]
        last_entities = last_interaction.get("extracted_entities", {})
        
        # Simple intent comparison (could be enhanced with entity analysis)
        current_intent = current_analysis.get("primary_intent")
        
        # Try to extract previous intent from entities or guess from query
        previous_query = last_interaction.get("query", "")
        previous_analysis = await self._analyze_query_intent(previous_query)
        previous_intent = previous_analysis.get("primary_intent")
        
        # Extract artists from current and previous queries for comparison
        current_artists = current_analysis.get("artists", [])
        previous_artists = previous_analysis.get("artists", [])
        
        # Determine if intent changed - either different intent types OR different artists in artist similarity
        basic_intent_changed = current_intent != previous_intent
        artist_switch_detected = False
        
        # Special case: Both are artist similarity but different artists
        if (current_intent == "artist_similarity" and previous_intent == "artist_similarity"):
            # Check if artists are different
            if current_artists and previous_artists:
                current_artist_set = set(a.lower().strip() for a in current_artists)
                previous_artist_set = set(a.lower().strip() for a in previous_artists)
                
                # If no overlap in artists, it's an artist switch
                if not current_artist_set.intersection(previous_artist_set):
                    artist_switch_detected = True
        
        # Determine overall intent change
        intent_changed = basic_intent_changed or artist_switch_detected
        change_type = None
        confidence = 0.0
        
        if intent_changed:
            # Classify type of change
            if artist_switch_detected:
                # Same intent type but different artists
                change_type = "artist_switch"
                confidence = 0.9  # High confidence for clear artist switch
            elif previous_intent == "artist_similarity" and current_intent == "artist_similarity":
                # Shouldn't reach here given the logic above, but safety fallback
                change_type = "artist_switch"
                confidence = 0.8
            elif previous_intent in ["mood_matching", "activity_context"] and current_intent in ["mood_matching", "activity_context"]:
                change_type = "context_shift"
                confidence = 0.7
            else:
                change_type = "major_intent_change"
                confidence = 0.9
        
        return {
            "intent_changed": intent_changed,
            "change_type": change_type,
            "confidence": confidence,
            "previous_intent": previous_intent,
            "current_intent": current_intent,
            "artist_switch_detected": artist_switch_detected,
            "current_artists": current_artists,
            "previous_artists": previous_artists
        }
    
    def _analyze_temporal_relevance(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal relevance of existing context."""
        interaction_history = session_context.get("interaction_history", [])
        
        if not interaction_history:
            return {
                "is_stale": False,
                "minutes_since_last": 0,
                "relevance_score": 1.0
            }
        
        last_interaction = interaction_history[-1]
        last_timestamp = last_interaction.get("timestamp")
        
        if not last_timestamp:
            return {
                "is_stale": True,
                "minutes_since_last": 999,
                "relevance_score": 0.0
            }
        
        time_diff = datetime.now() - last_timestamp
        minutes_since_last = time_diff.total_seconds() / 60
        
        # Calculate relevance score based on time decay
        relevance_score = max(0.0, 1.0 - (minutes_since_last / self.context_decay_minutes))
        is_stale = minutes_since_last > self.context_decay_minutes
        
        return {
            "is_stale": is_stale,
            "minutes_since_last": minutes_since_last,
            "relevance_score": relevance_score
        }
    
    async def _analyze_conversation_continuity(
        self,
        current_query: str,
        query_analysis: Dict[str, Any],
        session_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze conversation continuity indicators."""
        continuity_indicators = {
            "explicit_continuation": False,
            "entity_overlap": False,
            "semantic_similarity": False,
            "reference_words": False,
            "continuity_score": 0.0
        }
        
        query_lower = current_query.lower()
        
        # Check for explicit continuation words
        continuation_words = [
            "more", "also", "another", "similar", "like that", "what about",
            "continue", "keep", "again", "still", "too"
        ]
        
        for word in continuation_words:
            if word in query_lower:
                continuity_indicators["explicit_continuation"] = True
                break
        
        # Check for reference words
        reference_words = ["that", "this", "those", "these", "it", "them"]
        for word in reference_words:
            if word in query_lower:
                continuity_indicators["reference_words"] = True
                break
        
        # Check entity overlap with previous interactions
        interaction_history = session_context.get("interaction_history", [])
        if interaction_history:
            last_entities = interaction_history[-1].get("extracted_entities", {})
            current_entities = query_analysis.get("matched_entities", [])
            
            # Simple entity overlap check
            if current_entities and last_entities:
                continuity_indicators["entity_overlap"] = True
        
        # Calculate overall continuity score
        score = 0.0
        if continuity_indicators["explicit_continuation"]:
            score += 0.4
        if continuity_indicators["entity_overlap"]:
            score += 0.3
        if continuity_indicators["reference_words"]:
            score += 0.2
        if continuity_indicators["semantic_similarity"]:
            score += 0.1
        
        continuity_indicators["continuity_score"] = min(1.0, score)
        
        return continuity_indicators
    
    def _make_context_decision(
        self,
        query_analysis: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        temporal_analysis: Dict[str, Any],
        continuity_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make the final context decision based on all analyses."""
        
        # Start with base decision
        decision = ContextState.CONTINUING.value
        action = "maintain_context"
        confidence = 0.5
        reasoning_parts = []
        
        # Temporal analysis influence
        if temporal_analysis["is_stale"]:
            decision = ContextState.RESET_NEEDED.value
            action = "reset_context"
            confidence = 0.9
            reasoning_parts.append(
                f"Context is stale ({temporal_analysis['minutes_since_last']:.1f} minutes old)"
            )
            return {
                "decision": decision,
                "action": action,
                "confidence": confidence,
                "reasoning": "; ".join(reasoning_parts)
            }
        
        # Intent change analysis - PRIORITY CHECK
        if intent_analysis["intent_changed"]:
            change_type = intent_analysis["change_type"]
            
            if change_type == "major_intent_change":
                decision = ContextState.INTENT_SWITCH.value
                action = "reset_context"
                confidence = 0.85
                reasoning_parts.append(
                    f"Major intent change: {intent_analysis['previous_intent']} → {intent_analysis['current_intent']}"
                )
            elif change_type == "artist_switch":
                # Artist switch should reset context completely for different artists
                decision = ContextState.INTENT_SWITCH.value
                action = "reset_context"  # Changed from partial_reset to full reset
                confidence = 0.8
                current_artists = intent_analysis.get("current_artists", [])
                previous_artists = intent_analysis.get("previous_artists", [])
                reasoning_parts.append(
                    f"Artist similarity target changed: {previous_artists} → {current_artists}"
                )
            else:
                decision = ContextState.PREFERENCE_REFINEMENT.value
                action = "modify_context"
                confidence = 0.6
                reasoning_parts.append("Contextual preference shift detected")
        
        # Continuity analysis influence - BUT NOT FOR ARTIST SWITCHES
        continuity_score = continuity_analysis["continuity_score"]
        
        # Only apply continuity override if it's NOT an artist switch
        if continuity_score > 0.6 and intent_analysis.get("change_type") != "artist_switch":
            # Strong continuity signals - favor maintaining context
            if decision == ContextState.INTENT_SWITCH.value:
                decision = ContextState.PREFERENCE_REFINEMENT.value
                action = "modify_context"
            confidence = max(confidence, 0.7)
            reasoning_parts.append(f"Strong continuation signals (score: {continuity_score:.2f})")
        elif continuity_score < 0.2 and not intent_analysis["intent_changed"]:
            # Weak continuity but same intent - might be new direction
            decision = ContextState.PREFERENCE_REFINEMENT.value
            action = "modify_context"
            confidence = 0.6
            reasoning_parts.append("Weak continuation signals suggest preference refinement")
        
        # Combine reasoning
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Standard context continuation"
        
        return {
            "decision": decision,
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning
        }
    
    async def _prepare_context_for_use(
        self,
        context_decision: Dict[str, Any],
        session_context: Dict[str, Any],
        query_analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Prepare context based on the decision made."""
        action = context_decision["action"]
        
        if action == "reset_context":
            return None  # No context to use
        
        if action == "maintain_context":
            # Use full context
            return session_context
        
        if action == "modify_context":
            # Create modified context focusing on relevant parts
            modified_context = {
                "preference_profile": session_context.get("preference_profile", {}),
                "interaction_history": session_context.get("interaction_history", [])[-2:],  # Last 2 interactions
                "conversation_context": {
                    "session_continuity": True,
                    "context_modification": "preference_refinement"
                }
            }
            return modified_context
        
        if action == "partial_reset":
            # Keep some context but reset specific parts
            modified_context = {
                "preference_profile": {
                    # Keep general preferences but reset artist-specific ones
                    key: value for key, value in session_context.get("preference_profile", {}).items()
                    if "artist" not in key.lower()
                },
                "conversation_context": {
                    "session_continuity": True,
                    "context_modification": "partial_reset"
                }
            }
            return modified_context
        
        return session_context  # Fallback
    
    async def update_context_after_recommendation(
        self,
        session_id: str,
        query: str,
        llm_understanding: Optional[Dict],
        recommendations: List[Dict],
        context_decision: Dict[str, Any]
    ):
        """Update context after providing recommendations."""
        
        # Create entities dict from LLM understanding or query analysis
        if llm_understanding:
            entities = {
                "musical_entities": {
                    "artists": {"primary": llm_understanding.get("artists", [])},
                    "genres": {"primary": llm_understanding.get("genres", [])},
                },
                "contextual_entities": {
                    "moods": {"energy": llm_understanding.get("moods", [])},
                    "activities": {"physical": llm_understanding.get("activities", [])},
                },
                "llm_understanding": llm_understanding
            }
        else:
            entities = {"query_analysis": await self._analyze_query_intent(query)}
        
        # Reset context if decision requires it
        if context_decision.get("reset_context", False):
            await self.conversation_manager.clear_session(session_id)
        
        # Update session context
        await self.conversation_manager.update_session_context(
            session_id=session_id,
            query=query,
            entities=entities,
            recommendations=recommendations
        )
        
        self.logger.info(f"Context updated for session {session_id}")
    
    async def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the current context state."""
        session_context = await self.conversation_manager.get_session_context(session_id)
        
        if not session_context:
            return {"status": "no_context", "summary": "No active context"}
        
        interaction_count = len(session_context.get("interaction_history", []))
        preference_profile = session_context.get("preference_profile", {})
        
        # Analyze context health
        temporal_analysis = self._analyze_temporal_relevance(session_context)
        
        return {
            "status": "active",
            "interaction_count": interaction_count,
            "context_health": {
                "temporal_relevance": temporal_analysis["relevance_score"],
                "is_stale": temporal_analysis["is_stale"],
                "minutes_active": temporal_analysis["minutes_since_last"]
            },
            "preference_summary": {
                "preferred_genres": len(preference_profile.get("preferred_genres", {})),
                "preferred_artists": len(preference_profile.get("preferred_artists", {})),
                "activity_patterns": len(preference_profile.get("preferred_activities", {}))
            },
            "summary": f"Active context with {interaction_count} interactions"
        } 