"""
Context Analyzer Component for Planner Agent

Handles context interpretation and transformation logic for the planner agent.
Extracted from PlannerAgent for better modularization and single responsibility.
"""

from typing import Dict, Any, List
import structlog

from ...models.agent_models import QueryUnderstanding, QueryIntent

logger = structlog.get_logger(__name__)


class ContextAnalyzer:
    """
    Handles context analysis and interpretation for the planner agent.
    
    Responsibilities:
    - Analyzing context overrides and preserved entities
    - Creating understanding from context data
    - Creating entities from context data
    - Handling effective intent from IntentOrchestrationService
    - Processing follow-up queries
    """
    
    def __init__(self):
        """Initialize the ContextAnalyzer."""
        self.logger = logger
        self.logger.info("ContextAnalyzer initialized")
    
    def is_followup_with_preserved_context(self, context_override: Dict) -> bool:
        """
        Check if context override contains preserved entities that should skip query understanding.
        
        Following clean architecture principles, this method acts as a domain rule
        to determine when we should use preserved context vs fresh query understanding.
        
        Returns True for follow-ups with preserved entities like:
        - Artist deep dives with preserved genres ('hybrid_artist_genre')
        - Style continuations with preserved context ('style_continuation')
        - Artist refinements with preserved filters ('artist_style_refinement')
        - Artist similarity follow-ups with target entity ('artist_similarity')
        """
        if not isinstance(context_override, dict):
            return False
        
        # Check for follow-up indicators
        is_followup = context_override.get('is_followup', False)
        has_preserved_entities = 'preserved_entities' in context_override
        has_intent_override = 'intent_override' in context_override
        has_target_entity = context_override.get('target_entity') is not None
        
        # Define which intent overrides should use preserved context
        followup_types_with_context = [
            'hybrid_artist_genre', 'artist_style_refinement', 'style_continuation', 'by_artist'
        ]
        
        intent_override = context_override.get('intent_override')
        
        # Two types of valid follow-ups:
        # 1. Complex follow-ups with preserved entities (hybrid scenarios)
        # 2. Simple artist follow-ups with target entity (artist_similarity scenarios)
        
        complex_followup = (is_followup and 
                           has_preserved_entities and 
                           has_intent_override and
                           intent_override in followup_types_with_context)
        
        simple_artist_followup = (is_followup and
                                 has_target_entity and
                                 has_intent_override and
                                 intent_override in ['artist_similarity', 'by_artist'])
        
        result = complex_followup or simple_artist_followup
        
        self.logger.debug(
            "🔍 Context override validation",
            is_followup=is_followup,
            has_preserved_entities=has_preserved_entities,
            has_intent_override=has_intent_override,
            has_target_entity=has_target_entity,
            intent_override=intent_override,
            complex_followup=complex_followup,
            simple_artist_followup=simple_artist_followup,
            should_use_context=result
        )
        
        return result

    def create_understanding_from_context(self, user_query: str, context_override: Dict) -> QueryUnderstanding:
        """
        Create QueryUnderstanding from preserved context override.
        
        This method implements the domain logic for converting preserved conversation
        context into a proper QueryUnderstanding object, ensuring consistency with
        the rest of the system.
        """
        preserved_entities = context_override.get('preserved_entities', {})
        intent_override = context_override.get('intent_override', 'discovery')
        confidence = context_override.get('confidence', 0.9)
        target_entity = context_override.get('target_entity')
        
        # For artist similarity follow-ups, create entities from target_entity
        if intent_override == 'artist_similarity' and target_entity and not preserved_entities:
            # Simple artist follow-up: "More tracks" after "Music by Mk.gee"
            artists = [target_entity]
            genres = []
            moods = []
            self.logger.info(
                f"🎯 Artist similarity follow-up: Creating entities from target_entity='{target_entity}'"
            )
        else:
            # Complex follow-up with preserved entities
            artists = self._extract_entity_names(
                preserved_entities.get('artists', {}).get('primary', [])
            )
            genres = self._extract_entity_names(
                preserved_entities.get('genres', {}).get('primary', [])
            )
            moods = self._extract_entity_names(
                preserved_entities.get('moods', {}).get('primary', [])
            )
        
        # Map intent override to QueryIntent enum - domain rule mapping
        intent_mapping = {
            'hybrid_artist_genre': QueryIntent.HYBRID_SIMILARITY_GENRE,
            'artist_style_refinement': QueryIntent.HYBRID_SIMILARITY_GENRE, 
            'style_continuation': QueryIntent.GENRE_MOOD,
            'artist_deep_dive': QueryIntent.ARTIST_SIMILARITY,
            'artist_similarity': QueryIntent.ARTIST_SIMILARITY,  # Similar artists
            'by_artist': QueryIntent.BY_ARTIST,  # ✅ NEW: More tracks by the same artist
            'artist_genre': QueryIntent.ARTIST_GENRE,  # ✅ NEW: Artist tracks filtered by genre
            'discovering_serendipity': QueryIntent.DISCOVERING_SERENDIPITY  # Serendipitous discovery
        }
        
        intent = intent_mapping.get(intent_override, QueryIntent.DISCOVERY)
        
        self.logger.info(
            f"🎯 Created understanding from context",
            intent=intent.value,
            artists=artists,
            genres=genres,
            confidence=confidence,
            override_type=intent_override
        )
        
        return QueryUnderstanding(
            intent=intent,
            confidence=confidence,
            artists=artists,
            genres=genres,
            moods=moods,
            activities=[],
            original_query=user_query,
            normalized_query=user_query.lower(),
            reasoning=f"Context override: {intent_override} follow-up with preserved entities"
        )

    def create_entities_from_context(self, context_override: Dict) -> Dict[str, Any]:
        """
        Create entities structure from context override.
        
        This method transforms preserved conversation context into the standardized
        entities structure expected by downstream agents, maintaining architectural
        consistency.
        """
        preserved_entities = context_override.get('preserved_entities', {})
        intent_override = context_override.get('intent_override', 'discovery')
        target_entity = context_override.get('target_entity')
        
        # For artist similarity follow-ups, create entities from target_entity
        if intent_override == 'artist_similarity' and target_entity and not preserved_entities:
            # Simple artist follow-up: "More tracks" after "Music by Mk.gee"
            artists_primary = [target_entity]
            genres_primary = []
            moods_primary = []
            self.logger.info(
                f"🎯 Artist similarity follow-up: Creating entities structure from target_entity='{target_entity}'"
            )
        else:
            # Complex follow-up with preserved entities
            # Extract preserved entity data with safe navigation
            artists_data = preserved_entities.get('artists', {})
            genres_data = preserved_entities.get('genres', {})
            moods_data = preserved_entities.get('moods', {})
            
            artists_primary = self._extract_entity_names(artists_data.get('primary', []))
            genres_primary = self._extract_entity_names(genres_data.get('primary', []))
            moods_primary = self._extract_entity_names(moods_data.get('primary', []))
        
        # Convert to proper entities structure following established schema
        entities = {
            "musical_entities": {
                "artists": {
                    "primary": artists_primary,
                    "similar_to": []
                },
                "genres": {
                    "primary": genres_primary,
                    "secondary": []
                },
                "tracks": {
                    "primary": [],
                    "referenced": []
                },
                "moods": {
                    "primary": moods_primary,
                    "energy": [],
                    "emotion": []
                }
            },
            "contextual_entities": {
                "activities": {
                    "physical": [],
                    "mental": [],
                    "social": []
                },
                "temporal": {
                    "decades": [],
                    "periods": []
                }
            },
            "confidence_scores": {
                "overall": context_override.get('confidence', 0.9)
            },
            "extraction_method": "context_override_preserved",
            "intent_analysis": {
                "intent": intent_override,
                "confidence": context_override.get('confidence', 0.9),
                "context_override_applied": True
            }
        }
        
        self.logger.info(
            f"🎯 Created entities from context",
            artists_count=len(entities['musical_entities']['artists']['primary']),
            genres_count=len(entities['musical_entities']['genres']['primary']),
            moods_count=len(entities['musical_entities']['moods']['primary']),
            extraction_method=entities['extraction_method']
        )
        
        return entities

    def create_understanding_from_effective_intent(
        self, user_query: str, effective_intent: Dict[str, Any]
    ) -> QueryUnderstanding:
        """
        Create QueryUnderstanding from effective intent provided by IntentOrchestrationService.
        
        Phase 2: Simplified approach that trusts the intent orchestrator's resolution.
        """
        intent_str = effective_intent.get('intent', 'discovery')
        entities = effective_intent.get('entities', {})
        confidence = effective_intent.get('confidence', 0.8)
        
        # Extract entities from effective intent
        artists = self._extract_artists_from_effective_intent(entities)
        genres = self._extract_genres_from_effective_intent(entities)
        moods = self._extract_moods_from_effective_intent(entities)
        activities = self._extract_activities_from_effective_intent(entities)
        
        # Map intent string to QueryIntent enum
        intent_mapping = {
            'artist_similarity': QueryIntent.ARTIST_SIMILARITY,
            'genre_exploration': QueryIntent.GENRE_MOOD,
            'mood_matching': QueryIntent.GENRE_MOOD,
            'activity_context': QueryIntent.CONTEXTUAL,  # Fix: Map activity_context to CONTEXTUAL intent
            'contextual': QueryIntent.CONTEXTUAL,  # Add direct contextual mapping
            'genre_mood': QueryIntent.GENRE_MOOD,  # Fix: Add missing genre_mood mapping
            'style_continuation': QueryIntent.GENRE_MOOD,  # Fix: Map style_continuation to preserve original genre_mood intent
            'discovery': QueryIntent.DISCOVERY,
            'discovering_serendipity': QueryIntent.DISCOVERING_SERENDIPITY,  # Serendipitous discovery
            'hybrid_similarity_genre': QueryIntent.HYBRID_SIMILARITY_GENRE,  # ✅ FIX: Add full intent string
            'by_artist': QueryIntent.BY_ARTIST,
            'by_artist_underground': QueryIntent.BY_ARTIST_UNDERGROUND,
            'artist_genre': QueryIntent.ARTIST_GENRE  # ✅ NEW: Artist tracks filtered by genre
        }
        
        intent = intent_mapping.get(intent_str, QueryIntent.DISCOVERY)
        
        reasoning = "Phase 2: Effective intent from IntentOrchestrationService"
        if effective_intent.get('is_followup'):
            reasoning += f" (follow-up: {effective_intent.get('followup_type', 'unknown')})"
        
        self.logger.info(
            "Phase 2: Created understanding from effective intent",
            intent=intent.value,
            artists=artists,
            genres=genres,
            confidence=confidence,
            is_followup=effective_intent.get('is_followup', False)
        )
        
        return QueryUnderstanding(
            intent=intent,
            confidence=confidence,
            artists=artists,
            genres=genres,
            moods=moods,
            activities=activities,
            original_query=user_query,
            normalized_query=user_query.lower(),
            reasoning=reasoning
        )
    
    def create_entities_from_effective_intent(self, effective_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create entities structure from effective intent.
        
        Phase 2: Simplified approach that uses the entities resolved by IntentOrchestrationService.
        """
        entities = effective_intent.get('entities', {})
        
        # Extract entities using helper methods
        artists = self._extract_artists_from_effective_intent(entities)
        genres = self._extract_genres_from_effective_intent(entities)
        moods = self._extract_moods_from_effective_intent(entities)
        activities = self._extract_activities_from_effective_intent(entities)
        
        # Create standardized entities structure
        result = {
            "musical_entities": {
                "artists": {
                    "primary": artists,
                    "similar_to": []
                },
                "genres": {
                    "primary": genres,
                    "secondary": []
                },
                "tracks": {
                    "primary": [],
                    "referenced": []
                },
                "moods": {
                    "primary": moods,
                    "energy": [],
                    "emotion": []
                }
            },
            "contextual_entities": {
                "activities": {
                    "physical": activities,
                    "mental": [],
                    "social": []
                },
                "temporal": {
                    "decades": [],
                    "periods": []
                }
            },
            "confidence_scores": {
                "overall": effective_intent.get('confidence', 0.8)
            },
            "extraction_method": "effective_intent_phase2",
            "intent_analysis": {
                "intent": effective_intent.get('intent', 'discovery'),
                "confidence": effective_intent.get('confidence', 0.8),
                "is_followup": effective_intent.get('is_followup', False),
                "followup_type": effective_intent.get('followup_type'),
                "preserves_original_context": effective_intent.get('preserves_original_context', False)
            }
        }
        
        self.logger.info(
            "Phase 2: Created entities from effective intent",
            artists_count=len(artists),
            genres_count=len(genres),
            moods_count=len(moods),
            is_followup=effective_intent.get('is_followup', False)
        )
        
        return result
    
    def _extract_entity_names(self, entity_list: List) -> List[str]:
        """
        Extract names from entity list that may contain dicts or strings.
        
        This utility method handles the data transformation needed to work with
        various entity formats from preserved context.
        """
        names = []
        for item in entity_list:
            if isinstance(item, dict):
                # Handle confidence score format: {'name': 'Artist', 'confidence': 0.8}
                names.append(item.get('name', str(item)))
            elif isinstance(item, str):
                names.append(item)
            else:
                names.append(str(item))
        return names
    
    def _extract_artists_from_effective_intent(self, entities: Dict[str, Any]) -> List[str]:
        """Extract artist names from effective intent entities."""
        artists = []
        
        # Handle different entity structures
        if 'artists' in entities:
            artist_data = entities['artists']
            if isinstance(artist_data, list):
                artists.extend(self._extract_entity_names(artist_data))
        
        # Also check musical_entities structure
        if 'musical_entities' in entities and 'artists' in entities['musical_entities']:
            artist_data = entities['musical_entities']['artists']
            if isinstance(artist_data, dict) and 'primary' in artist_data:
                artists.extend(self._extract_entity_names(artist_data['primary']))
            elif isinstance(artist_data, list):
                artists.extend(self._extract_entity_names(artist_data))
        
        return list(set(artists))  # Remove duplicates
    
    def _extract_genres_from_effective_intent(self, entities: Dict[str, Any]) -> List[str]:
        """Extract genre names from effective intent entities."""
        genres = []
        
        if 'genres' in entities:
            genre_data = entities['genres']
            if isinstance(genre_data, dict):
                for genre_list in [genre_data.get('primary', []), genre_data.get('secondary', [])]:
                    genres.extend(self._extract_entity_names(genre_list))
            elif isinstance(genre_data, list):
                genres.extend(self._extract_entity_names(genre_data))
        
        # Also check musical_entities structure
        if 'musical_entities' in entities and 'genres' in entities['musical_entities']:
            genre_data = entities['musical_entities']['genres']
            if isinstance(genre_data, dict):
                for genre_list in [genre_data.get('primary', []), genre_data.get('secondary', [])]:
                    genres.extend(self._extract_entity_names(genre_list))
        
        return list(set(genres))  # Remove duplicates
    
    def _extract_moods_from_effective_intent(self, entities: Dict[str, Any]) -> List[str]:
        """Extract mood names from effective intent entities."""
        moods = []
        
        if 'moods' in entities:
            mood_data = entities['moods']
            if isinstance(mood_data, dict):
                for mood_list in [mood_data.get('primary', []), mood_data.get('secondary', [])]:
                    moods.extend(self._extract_entity_names(mood_list))
            elif isinstance(mood_data, list):
                moods.extend(self._extract_entity_names(mood_data))
        
        # Also check musical_entities structures
        if 'musical_entities' in entities and 'moods' in entities['musical_entities']:
            mood_data = entities['musical_entities']['moods']
            if isinstance(mood_data, dict):
                for mood_list in [mood_data.get('primary', []), mood_data.get('energy', []), mood_data.get('emotion', [])]:
                    moods.extend(self._extract_entity_names(mood_list))
        
        return list(set(moods))  # Remove duplicates
    
    def _extract_activities_from_effective_intent(self, entities: Dict[str, Any]) -> List[str]:
        """Extract activity names from effective intent entities."""
        activities = []
        
        if 'activities' in entities:
            activity_data = entities['activities']
            if isinstance(activity_data, list):
                activities.extend(self._extract_entity_names(activity_data))
        
        # Also check contextual_entities structure
        if 'contextual_entities' in entities and 'activities' in entities['contextual_entities']:
            activity_data = entities['contextual_entities']['activities']
            if isinstance(activity_data, dict):
                for activity_list in [activity_data.get('physical', []), activity_data.get('mental', []), activity_data.get('social', [])]:
                    activities.extend(self._extract_entity_names(activity_list))
        
        return list(set(activities))  # Remove duplicates 