"""
Conversation Context Service for BeatDebate

Manages conversation history, session state, and user preference evolution
across multiple interactions within a session.

Moved from agents to services layer for proper architecture.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


class ConversationContextManager:
    """
    Manages conversation history and user preference evolution.
    
    Tracks:
    - Session interaction history
    - User preference evolution
    - Recommendation history
    - Entity evolution patterns
    """
    
    def __init__(self):
        """Initialize conversation context manager."""
        self.session_store = {}
        self.logger = logger.bind(component="ConversationContext")
        
        self.logger.info("Conversation Context Manager initialized")
    
    async def update_session_context(
        self, 
        session_id: str, 
        query: str,
        entities: Dict[str, Any],
        recommendations: Optional[List[Dict]] = None,
        user_feedback: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Update session with new interaction data.
        
        Args:
            session_id: Unique session identifier
            query: User query
            entities: Extracted entities
            recommendations: Generated recommendations
            user_feedback: User feedback on recommendations
            
        Returns:
            Updated session context
        """
        self.logger.info("Updating session context", session_id=session_id)
        
        if session_id not in self.session_store:
            self.session_store[session_id] = {
                "interaction_history": [],
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
                "last_updated": datetime.now()
            }
        
        session = self.session_store[session_id]
        
        # Add interaction
        interaction = {
            "timestamp": datetime.now(),
            "query": query,
            "extracted_entities": entities,
            "recommendations": recommendations or [],
            "user_feedback": user_feedback
        }
        session["interaction_history"].append(interaction)
        
        # Update recommendation history
        if recommendations:
            session["recommendation_history"].extend(recommendations)
        
        # Update preference profile
        await self._update_preference_profile(session, entities, user_feedback)
        
        # Track entity evolution
        await self._track_entity_evolution(session, entities)
        
        # Update timestamp
        session["last_updated"] = datetime.now()
        
        self.logger.info(
            "Session context updated", 
            session_id=session_id,
            interaction_count=len(session["interaction_history"])
        )
        
        return session
    
    async def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current session context.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session context or None if not found
        """
        return self.session_store.get(session_id)
    
    async def resolve_session_references(
        self, 
        session_id: str, 
        entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve session references in entities to actual track/artist IDs.
        
        Args:
            session_id: Session identifier
            entities: Entities with potential session references
            
        Returns:
            Entities with resolved references
        """
        if session_id not in self.session_store:
            return entities
        
        session = self.session_store[session_id]
        conversation_entities = entities.get("conversation_entities", {})
        session_references = conversation_entities.get("session_references", [])
        
        resolved_references = []
        
        for reference in session_references:
            resolved_ref = await self._resolve_single_reference(reference, session)
            if resolved_ref:
                resolved_references.append(resolved_ref)
        
        # Update entities with resolved references
        if resolved_references:
            entities["conversation_entities"]["resolved_references"] = resolved_references
            
        return entities

    async def _resolve_single_reference(
        self, 
        reference: Dict[str, Any], 
        session: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a single session reference to actual content.
        
        Args:
            reference: Reference to resolve
            session: Session data
            
        Returns:
            Resolved reference or None
        """
        ref_type = reference.get("type", "")
        recommendation_history = session.get("recommendation_history", [])
        
        if not recommendation_history:
            return None
        
        if ref_type == "track_reference" or "last song" in reference.get("indicator", ""):
            # Get the last recommended track
            last_recommendation = recommendation_history[-1] if recommendation_history else None
            if last_recommendation and "tracks" in last_recommendation:
                last_track = last_recommendation["tracks"][0] if last_recommendation["tracks"] else None
                if last_track:
                    return {
                        "type": "resolved_track",
                        "track_id": last_track.get("id"),
                        "track_name": last_track.get("name"),
                        "artist_name": last_track.get("artist"),
                        "reference_source": "last_recommendation"
                    }
        
        elif ref_type == "artist_reference" or "that artist" in reference.get("indicator", ""):
            # Get the last mentioned artist
            last_recommendation = recommendation_history[-1] if recommendation_history else None
            if last_recommendation and "tracks" in last_recommendation:
                last_track = last_recommendation["tracks"][0] if last_recommendation["tracks"] else None
                if last_track:
                    return {
                        "type": "resolved_artist",
                        "artist_name": last_track.get("artist"),
                        "reference_source": "last_recommendation_artist"
                    }
        
        return None
    
    async def _update_preference_profile(
        self, 
        session: Dict[str, Any], 
        entities: Dict[str, Any],
        user_feedback: Optional[Dict] = None
    ):
        """Update user preference profile based on entities and feedback."""
        if "preference_profile" not in session:
            session["preference_profile"] = {
                "preferred_genres": {},
                "preferred_artists": {},
                "preferred_moods": {},
                "preferred_activities": {},
                "discovery_openness": 0.5,
                "quality_preferences": {}
            }
        
        profile = session["preference_profile"]
        
        # Update genre preferences
        genres = entities.get("musical_entities", {}).get("genres", {}).get("primary", [])
        for genre in genres:
            profile["preferred_genres"][genre] = profile["preferred_genres"].get(genre, 0) + 1
        
        # Update artist preferences
        artists = entities.get("musical_entities", {}).get("artists", {}).get("primary", [])
        for artist in artists:
            profile["preferred_artists"][artist] = profile["preferred_artists"].get(artist, 0) + 1
        
        # Update mood preferences
        moods = entities.get("contextual_entities", {}).get("moods", {})
        for mood_type, mood_list in moods.items():
            if mood_type not in profile["preferred_moods"]:
                profile["preferred_moods"][mood_type] = {}
            for mood in mood_list:
                profile["preferred_moods"][mood_type][mood] = profile["preferred_moods"][mood_type].get(mood, 0) + 1
        
        # Update activity preferences
        activities = entities.get("contextual_entities", {}).get("activities", {})
        for activity_type, activity_list in activities.items():
            if activity_type not in profile["preferred_activities"]:
                profile["preferred_activities"][activity_type] = {}
            for activity in activity_list:
                profile["preferred_activities"][activity_type][activity] = profile["preferred_activities"][activity_type].get(activity, 0) + 1
        
        # Update discovery preferences based on feedback
        if user_feedback:
            discovery_prefs = entities.get("preference_entities", {}).get("discovery_preferences", [])
            for pref in discovery_prefs:
                if "underground" in pref or "hidden" in pref:
                    profile["discovery_openness"] = min(1.0, profile["discovery_openness"] + 0.1)
                elif "mainstream" in pref or "popular" in pref:
                    profile["discovery_openness"] = max(0.0, profile["discovery_openness"] - 0.1)
    
    async def _track_entity_evolution(
        self, 
        session: Dict[str, Any], 
        entities: Dict[str, Any]
    ):
        """Track how entities evolve throughout the session."""
        if "entity_evolution" not in session:
            session["entity_evolution"] = {}
        
        evolution = session["entity_evolution"]
        
        # Initialize progression lists if they don't exist
        if "genre_progression" not in evolution:
            evolution["genre_progression"] = []
        if "mood_progression" not in evolution:
            evolution["mood_progression"] = []
        if "activity_progression" not in evolution:
            evolution["activity_progression"] = []
        if "discovery_progression" not in evolution:
            evolution["discovery_progression"] = []
        
        # Track genre evolution
        current_genres = entities.get("musical_entities", {}).get("genres", {}).get("primary", [])
        if current_genres:
            evolution["genre_progression"].append({
                "timestamp": datetime.now(),
                "genres": current_genres
            })
        
        # Track mood evolution
        current_moods = entities.get("contextual_entities", {}).get("moods", {})
        if any(current_moods.values()):
            evolution["mood_progression"].append({
                "timestamp": datetime.now(),
                "moods": current_moods
            })
        
        # Track activity evolution
        current_activities = entities.get("contextual_entities", {}).get("activities", {})
        if any(current_activities.values()):
            evolution["activity_progression"].append({
                "timestamp": datetime.now(),
                "activities": current_activities
            })
        
        # Track discovery preferences evolution
        discovery_prefs = entities.get("preference_entities", {}).get("discovery_preferences", [])
        if discovery_prefs:
            evolution["discovery_progression"].append({
                "timestamp": datetime.now(),
                "preferences": discovery_prefs
            })
    
    async def analyze_preference_evolution(
        self, 
        session_id: str
    ) -> Dict[str, Any]:
        """
        Analyze how user preferences have evolved during the session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Preference evolution analysis
        """
        if session_id not in self.session_store:
            return {"evolution_patterns": {}, "confidence_scores": {}, "recommendations": []}
        
        session = self.session_store[session_id]
        interaction_history = session.get("interaction_history", [])
        
        if len(interaction_history) < 2:
            return {"evolution_patterns": {}, "confidence_scores": {}, "recommendations": []}
        
        evolution_patterns = {
            "genre_drift": [],      # How genre preferences changed
            "energy_adjustment": [], # Energy level modifications
            "discovery_tolerance": [], # Openness to new music
            "artist_affinity": [],   # Artist preference patterns
            "activity_correlation": [] # Activity-music correlations
        }
        
        # Analyze patterns across interactions
        for i, interaction in enumerate(interaction_history):
            if i == 0:
                continue  # Skip first interaction as baseline
            
            prev_interaction = interaction_history[i-1]
            
            # Analyze genre drift
            genre_drift = await self._analyze_genre_drift(prev_interaction, interaction)
            if genre_drift:
                evolution_patterns["genre_drift"].append(genre_drift)
            
            # Analyze energy adjustments
            energy_adjustment = await self._analyze_energy_adjustment(prev_interaction, interaction)
            if energy_adjustment:
                evolution_patterns["energy_adjustment"].append(energy_adjustment)
            
            # Analyze discovery tolerance changes
            discovery_change = await self._analyze_discovery_tolerance(prev_interaction, interaction)
            if discovery_change:
                evolution_patterns["discovery_tolerance"].append(discovery_change)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_evolution_confidence(evolution_patterns)
        
        # Generate recommendations based on evolution
        recommendations = self._generate_preference_recommendations(evolution_patterns)
        
        return {
            "evolution_patterns": evolution_patterns,
            "confidence_scores": confidence_scores,
            "recommendations": recommendations,
            "session_length": len(interaction_history),
            "evolution_detected": any(len(patterns) > 0 for patterns in evolution_patterns.values())
        }

    async def _analyze_genre_drift(
        self, 
        prev_interaction: Dict[str, Any], 
        current_interaction: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze how genre preferences have shifted between interactions."""
        prev_entities = prev_interaction.get("extracted_entities", {})
        current_entities = current_interaction.get("extracted_entities", {})
        
        prev_genres = prev_entities.get("musical_entities", {}).get("genres", {}).get("primary", [])
        current_genres = current_entities.get("musical_entities", {}).get("genres", {}).get("primary", [])
        
        if not prev_genres or not current_genres:
            return None
        
        # Detect genre shifts
        new_genres = [g for g in current_genres if g not in prev_genres]
        dropped_genres = [g for g in prev_genres if g not in current_genres]
        
        if new_genres or dropped_genres:
            return {
                "timestamp": current_interaction.get("timestamp"),
                "new_genres": new_genres,
                "dropped_genres": dropped_genres,
                "shift_type": "expansion" if len(new_genres) > len(dropped_genres) else "refinement"
            }
        
        return None

    async def _analyze_energy_adjustment(
        self, 
        prev_interaction: Dict[str, Any], 
        current_interaction: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze energy level adjustments between interactions."""
        prev_query = prev_interaction.get("query", "").lower()
        current_query = current_interaction.get("query", "").lower()
        
        energy_indicators = {
            "increase": ["more upbeat", "more energetic", "higher energy", "more intense"],
            "decrease": ["calmer", "mellower", "more chill", "less intense", "quieter"]
        }
        
        for direction, indicators in energy_indicators.items():
            if any(indicator in current_query for indicator in indicators):
                return {
                    "timestamp": current_interaction.get("timestamp"),
                    "direction": direction,
                    "indicators": [ind for ind in indicators if ind in current_query]
                }
        
        return None

    async def _analyze_discovery_tolerance(
        self, 
        prev_interaction: Dict[str, Any], 
        current_interaction: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze changes in openness to music discovery."""
        prev_entities = prev_interaction.get("extracted_entities", {})
        current_entities = current_interaction.get("extracted_entities", {})
        
        prev_discovery = prev_entities.get("preference_entities", {}).get("discovery_preferences", [])
        current_discovery = current_entities.get("preference_entities", {}).get("discovery_preferences", [])
        
        # Analyze discovery preference changes
        if "underground" in current_discovery and "underground" not in prev_discovery:
            return {
                "timestamp": current_interaction.get("timestamp"),
                "change": "increased_openness",
                "type": "underground_exploration"
            }
        elif "mainstream" in current_discovery and "underground" in prev_discovery:
            return {
                "timestamp": current_interaction.get("timestamp"),
                "change": "decreased_openness",
                "type": "mainstream_preference"
            }
        
        return None

    def _calculate_evolution_confidence(self, evolution_patterns: Dict[str, List]) -> Dict[str, float]:
        """Calculate confidence scores for evolution patterns."""
        confidence_scores = {}
        
        for pattern_type, patterns in evolution_patterns.items():
            if not patterns:
                confidence_scores[pattern_type] = 0.0
            else:
                # Base confidence on number of observations and consistency
                base_confidence = min(len(patterns) / 3.0, 1.0)  # Max confidence at 3+ observations
                confidence_scores[pattern_type] = base_confidence
        
        # Overall confidence is average of individual confidences
        if confidence_scores:
            confidence_scores["overall"] = sum(confidence_scores.values()) / len(confidence_scores)
        else:
            confidence_scores["overall"] = 0.0
        
        return confidence_scores

    def _generate_preference_recommendations(self, evolution_patterns: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate recommendations based on preference evolution patterns."""
        recommendations = []
        
        # Genre drift recommendations
        if evolution_patterns.get("genre_drift"):
            latest_drift = evolution_patterns["genre_drift"][-1]
            if latest_drift.get("shift_type") == "expansion":
                recommendations.append({
                    "type": "genre_exploration",
                    "suggestion": "Continue exploring new genres",
                    "confidence": 0.8,
                    "new_genres": latest_drift.get("new_genres", [])
                })
        
        # Energy adjustment recommendations
        if evolution_patterns.get("energy_adjustment"):
            latest_adjustment = evolution_patterns["energy_adjustment"][-1]
            recommendations.append({
                "type": "energy_preference",
                "suggestion": f"User prefers {latest_adjustment.get('direction')} energy music",
                "confidence": 0.7,
                "direction": latest_adjustment.get("direction")
            })
        
        # Discovery tolerance recommendations
        if evolution_patterns.get("discovery_tolerance"):
            latest_tolerance = evolution_patterns["discovery_tolerance"][-1]
            recommendations.append({
                "type": "discovery_strategy",
                "suggestion": f"Adjust discovery strategy for {latest_tolerance.get('change')}",
                "confidence": 0.6,
                "change_type": latest_tolerance.get("change")
            })
        
        return recommendations

    async def clear_session(self, session_id: str):
        """Clear session data."""
        if session_id in self.session_store:
            del self.session_store[session_id]
            self.logger.info("Session cleared", session_id=session_id)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the session."""
        session = self.session_store.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "interaction_count": len(session["interaction_history"]),
            "recommendation_count": len(session["recommendation_history"]),
            "session_duration": (session["last_updated"] - session["session_start"]).total_seconds(),
            "preference_profile": session["preference_profile"],
            "entity_evolution": session["entity_evolution"]
        }

    async def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation summary
        """
        if session_id not in self.session_store:
            return {"error": "Session not found"}
        
        session = self.session_store[session_id]
        interaction_history = session.get("interaction_history", [])
        preference_profile = session.get("preference_profile", {})
        
        # Analyze conversation flow
        conversation_flow = {
            "total_interactions": len(interaction_history),
            "session_duration": None,
            "query_complexity_progression": [],
            "entity_evolution": []
        }
        
        if interaction_history:
            start_time = interaction_history[0].get("timestamp")
            end_time = interaction_history[-1].get("timestamp")
            if start_time and end_time:
                conversation_flow["session_duration"] = (end_time - start_time).total_seconds()
        
        # Track query complexity over time
        for interaction in interaction_history:
            entities = interaction.get("extracted_entities", {})
            decomposition = entities.get("decomposition_metadata", {})
            conversation_flow["query_complexity_progression"].append({
                "timestamp": interaction.get("timestamp"),
                "complexity": decomposition.get("query_complexity", "simple"),
                "entity_count": len(entities.get("musical_entities", {}).get("artists", {}).get("primary", []))
            })
        
        return {
            "session_id": session_id,
            "conversation_flow": conversation_flow,
            "preference_profile": preference_profile,
            "evolution_analysis": await self.analyze_preference_evolution(session_id),
            "last_interaction": interaction_history[-1] if interaction_history else None
        } 