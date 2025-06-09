"""
Contextual Generation Strategies

Strategies that focus on generating candidates based on contextual needs
and activity-based requirements.
"""

from typing import Dict, List, Any
import structlog
from .base_strategy import BaseGenerationStrategy

logger = structlog.get_logger(__name__)


class ContextualActivityStrategy(BaseGenerationStrategy):
    """
    Strategy for contextual activity-based music discovery.
    
    Extracts implicit moods and genres from activities like "coding", "workout", "study".
    Maps activities to appropriate musical characteristics.
    """
    
    def __init__(self, api_service, llm_client=None):
        super().__init__(api_service, llm_client)
        
        # Activity to mood/genre mappings
        self.activity_mappings = {
            'coding': {
                'moods': ['focused', 'calm', 'ambient', 'instrumental'],
                'genres': ['ambient', 'post-rock', 'electronic', 'instrumental'],
                'energy_level': 'low',
                'characteristics': ['no_vocals', 'repetitive', 'non_distracting']
            },
            'study': {
                'moods': ['focused', 'calm', 'peaceful', 'concentration'],
                'genres': ['classical', 'ambient', 'lo-fi', 'instrumental'],
                'energy_level': 'low',
                'characteristics': ['no_vocals', 'steady_tempo', 'minimal_variation']
            },
            'workout': {
                'moods': ['energetic', 'motivational', 'pumping', 'intense'],
                'genres': ['electronic', 'hip-hop', 'rock', 'dance'],
                'energy_level': 'high',
                'characteristics': ['strong_beat', 'high_tempo', 'driving']
            },
            'work': {
                'moods': ['productive', 'focused', 'steady', 'professional'],
                'genres': ['instrumental', 'ambient', 'jazz', 'classical'],
                'energy_level': 'medium',
                'characteristics': ['background_friendly', 'consistent', 'non_intrusive']
            },
            'relax': {
                'moods': ['calm', 'peaceful', 'soothing', 'mellow'],
                'genres': ['ambient', 'chillout', 'acoustic', 'new age'],
                'energy_level': 'low',
                'characteristics': ['slow_tempo', 'gentle', 'harmonious']
            },
            'party': {
                'moods': ['upbeat', 'social', 'danceable', 'fun'],
                'genres': ['dance', 'pop', 'hip-hop', 'electronic'],
                'energy_level': 'high',
                'characteristics': ['danceable', 'catchy', 'social']
            },
            'driving': {
                'moods': ['steady', 'engaging', 'rhythmic', 'journey'],
                'genres': ['rock', 'alternative', 'electronic', 'indie'],
                'energy_level': 'medium',
                'characteristics': ['steady_rhythm', 'engaging', 'road_friendly']
            },
            'sleep': {
                'moods': ['peaceful', 'dreamy', 'soft', 'lullaby'],
                'genres': ['ambient', 'new age', 'classical', 'acoustic'],
                'energy_level': 'very_low',
                'characteristics': ['very_slow', 'gentle', 'sleep_inducing']
            }
        }
        
        # Fallback mappings for partial matches
        self.activity_keywords = {
            'focus': 'study',
            'concentration': 'study',
            'studying': 'study',  # Fix: Add studying -> study mapping
            'exercise': 'workout',
            'gym': 'workout',
            'fitness': 'workout',
            'working out': 'workout',
            'chill': 'relax',
            'unwind': 'relax',
            'relaxing': 'relax',
            'dance': 'party',
            'dancing': 'party',
            'celebration': 'party',
            'car': 'driving',
            'road': 'driving',
            'bedtime': 'sleep',
            'night': 'sleep',
            'sleeping': 'sleep',
            'working': 'work',
            'coding': 'coding'  # Add direct coding mapping
        }
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate tracks based on contextual activity requirements.
        
        Args:
            entities: Extracted entities containing activity information
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of contextually appropriate candidate tracks
        """
        candidates = []
        
        # Extract activities from entities and query
        activities = self._extract_activities(entities, intent_analysis)
        
        if not activities:
            self.logger.warning("No activities found for ContextualActivityStrategy")
            return candidates
        
        self.logger.info(f"🎯 CONTEXTUAL ACTIVITY: Processing activities {', '.join(activities)}")
        
        # Process each activity
        for activity in activities[:2]:  # Focus on top 2 activities
            activity_config = self._get_activity_config(activity)
            
            if not activity_config:
                continue
                
            # Generate candidates for this activity
            activity_candidates = await self._generate_for_activity(
                activity, activity_config, limit // len(activities)
            )
            
            candidates.extend(activity_candidates)
            
            if len(candidates) >= limit:
                break
        
        # Sort by contextual relevance
        candidates = self._rank_by_contextual_relevance(candidates, activities)
        
        self.logger.info(f"🎯 CONTEXTUAL ACTIVITY: {len(candidates)} contextually relevant tracks found")
        return candidates[:limit]
    
    def _extract_activities(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[str]:
        """Extract activities from entities and intent analysis."""
        activities = []
        
        # DEBUG: Log what we're receiving
        self.logger.info(f"🔍 DEBUG ContextualActivity - entities keys: {list(entities.keys())}")
        self.logger.info(f"🔍 DEBUG ContextualActivity - intent_analysis keys: {list(intent_analysis.keys())}")
        
        # From entities - multiple possible structures
        if 'activities' in entities:
            if isinstance(entities['activities'], list):
                activities.extend(entities['activities'])
                self.logger.info(f"🔍 DEBUG: Found activities in entities (list): {entities['activities']}")
            elif isinstance(entities['activities'], dict):
                for activity_list in entities['activities'].values():
                    if isinstance(activity_list, list):
                        activities.extend(activity_list)
                        self.logger.info(f"🔍 DEBUG: Found activities in entities (dict): {activity_list}")
        
        # From contextual entities
        contextual_entities = entities.get('contextual_entities', {})
        if 'activities' in contextual_entities:
            activity_data = contextual_entities['activities']
            if isinstance(activity_data, dict):
                for activity_list in [activity_data.get('physical', []), 
                                    activity_data.get('mental', []), 
                                    activity_data.get('social', [])]:
                    if activity_list:
                        activities.extend(activity_list)
                        self.logger.info(f"🔍 DEBUG: Found activities in contextual_entities: {activity_list}")
        
        # From query text analysis - Fix: Extract original_query from the correct location
        original_query = intent_analysis.get('original_query', '')
        if not original_query:
            # Try to get from query_understanding object
            query_understanding = intent_analysis.get('query_understanding')
            if query_understanding and hasattr(query_understanding, 'original_query'):
                original_query = query_understanding.original_query
        
        original_query = original_query.lower()
        self.logger.info(f"🔍 DEBUG: Extracted original_query: '{original_query}'")
        
        for keyword, mapped_activity in self.activity_keywords.items():
            if keyword in original_query:
                if mapped_activity not in activities:
                    activities.append(mapped_activity)
                    self.logger.info(f"🔍 DEBUG: Found activity '{mapped_activity}' from keyword '{keyword}'")
        
        # Direct activity detection in query
        for activity in self.activity_mappings.keys():
            if activity in original_query:
                if activity not in activities:
                    activities.append(activity)
                    self.logger.info(f"🔍 DEBUG: Found direct activity '{activity}' in query")
        
        self.logger.info(f"🔍 DEBUG: Final extracted activities: {activities}")
        return list(set(activities))  # Remove duplicates
    
    def _get_activity_config(self, activity: str) -> Dict[str, Any]:
        """Get configuration for a specific activity."""
        # Direct match
        if activity in self.activity_mappings:
            return self.activity_mappings[activity]
        
        # Keyword mapping
        mapped_activity = self.activity_keywords.get(activity.lower())
        if mapped_activity and mapped_activity in self.activity_mappings:
            return self.activity_mappings[mapped_activity]
        
        # Partial match
        for key, config in self.activity_mappings.items():
            if key in activity.lower() or activity.lower() in key:
                return config
        
        return None
    
    async def _generate_for_activity(
        self, 
        activity: str, 
        activity_config: Dict[str, Any], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Generate candidates for a specific activity."""
        candidates = []
        
        moods = activity_config['moods']
        genres = activity_config['genres']
        energy_level = activity_config['energy_level']
        characteristics = activity_config['characteristics']
        
        self.logger.info(
            f"🎯 Generating for activity '{activity}'",
            moods=moods[:2],
            genres=genres[:2],
            energy_level=energy_level
        )
        
        # Strategy 1: Mood-based search
        mood_candidates = await self._search_by_moods(moods, limit // 2)
        candidates.extend(mood_candidates)
        
        # Strategy 2: Genre-based search with activity context
        genre_candidates = await self._search_by_genres_with_context(
            genres, activity, characteristics, limit // 2
        )
        candidates.extend(genre_candidates)
        
        # Add activity metadata to all candidates
        for candidate in candidates:
            candidate.update({
                'activity_context': activity,
                'energy_level': energy_level,
                'activity_moods': moods,
                'activity_characteristics': characteristics,
                'contextual_relevance_score': self._calculate_activity_relevance(
                    candidate, activity_config
                )
            })
        
        return candidates
    
    async def _search_by_moods(self, moods: List[str], limit: int) -> List[Dict[str, Any]]:
        """Search for tracks by mood tags."""
        candidates = []
        
        for mood in moods[:3]:  # Top 3 moods
            try:
                mood_tracks = await self.api_service.search_tracks_by_tags(
                    tags=[mood],
                    limit=min(10, limit // len(moods))
                )
                
                for track_metadata in mood_tracks:
                    track = self._convert_metadata_to_dict(
                        track_metadata,
                        source='contextual_mood_search',
                        source_confidence=0.8,
                        mood_tag=mood,
                        search_strategy='mood_based'
                    )
                    candidates.append(track)
                    
            except Exception as e:
                self.logger.warning(f"Mood search failed for '{mood}': {e}")
                continue
        
        return candidates
    
    async def _search_by_genres_with_context(
        self, 
        genres: List[str], 
        activity: str, 
        characteristics: List[str], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search for tracks by genres with activity context."""
        candidates = []
        
        for genre in genres[:3]:  # Top 3 genres
            try:
                # Combine genre with activity-specific terms
                search_terms = [genre]
                if 'instrumental' in characteristics:
                    search_terms.append('instrumental')
                if 'ambient' in characteristics:
                    search_terms.append('ambient')
                
                genre_tracks = await self.api_service.search_tracks_by_tags(
                    tags=search_terms,
                    limit=min(10, limit // len(genres))
                )
                
                for track_metadata in genre_tracks:
                    track = self._convert_metadata_to_dict(
                        track_metadata,
                        source='contextual_genre_search',
                        source_confidence=0.7,
                        genre_tag=genre,
                        activity_context=activity,
                        search_strategy='genre_with_context'
                    )
                    candidates.append(track)
                    
            except Exception as e:
                self.logger.warning(f"Genre search failed for '{genre}': {e}")
                continue
        
        return candidates
    
    def _calculate_activity_relevance(
        self, 
        candidate: Dict[str, Any], 
        activity_config: Dict[str, Any]
    ) -> float:
        """Calculate how well a candidate matches the activity requirements."""
        score = 0.5  # Base score
        
        # Check for activity-specific characteristics
        characteristics = activity_config['characteristics']
        candidate_tags = candidate.get('tags', [])
        
        for char in characteristics:
            if any(char.replace('_', ' ') in tag.lower() for tag in candidate_tags):
                score += 0.1
        
        # Energy level matching (if available)
        energy_level = activity_config['energy_level']
        if energy_level == 'high' and any(word in str(candidate).lower() 
                                        for word in ['energetic', 'upbeat', 'intense']):
            score += 0.2
        elif energy_level == 'low' and any(word in str(candidate).lower() 
                                         for word in ['calm', 'peaceful', 'ambient']):
            score += 0.2
        
        return min(1.0, score)
    
    def _rank_by_contextual_relevance(
        self, 
        candidates: List[Dict[str, Any]], 
        activities: List[str]
    ) -> List[Dict[str, Any]]:
        """Rank candidates by their contextual relevance to activities."""
        return sorted(
            candidates, 
            key=lambda x: x.get('contextual_relevance_score', 0), 
            reverse=True
        )


class FunctionalMoodStrategy(BaseGenerationStrategy):
    """
    Strategy for functional mood-based discovery.
    
    Focuses on how music functions in specific contexts rather than
    just emotional content.
    """
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate tracks based on functional mood requirements.
        
        Args:
            entities: Extracted entities containing mood information
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of functionally appropriate candidate tracks
        """
        candidates = []
        
        # Extract functional requirements
        functional_moods = self._extract_functional_moods(entities, intent_analysis)
        
        if not functional_moods:
            self.logger.warning("No functional moods found for FunctionalMoodStrategy")
            return candidates
        
        self.logger.info(f"🎯 FUNCTIONAL MOOD: Processing functional requirements {', '.join(functional_moods)}")
        
        # Generate candidates for each functional mood
        for mood in functional_moods[:3]:  # Top 3 functional moods
            try:
                functional_tracks = await self._search_functional_mood(mood, limit // len(functional_moods))
                candidates.extend(functional_tracks)
                
            except Exception as e:
                self.logger.warning(f"Functional mood search failed for '{mood}': {e}")
                continue
        
        self.logger.info(f"🎯 FUNCTIONAL MOOD: {len(candidates)} functionally appropriate tracks found")
        return candidates[:limit]
    
    def _extract_functional_moods(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[str]:
        """Extract functional mood requirements."""
        functional_moods = []
        
        # From entities
        moods = self._extract_moods(entities)
        for mood in moods:
            if mood.lower() in ['focused', 'productive', 'concentration', 'background', 'ambient']:
                functional_moods.append(mood)
        
        # From query analysis - Enhanced detection - Fix: Extract original_query from the correct location
        query = intent_analysis.get('original_query', '')
        if not query:
            # Try to get from query_understanding object
            query_understanding = intent_analysis.get('query_understanding')
            if query_understanding and hasattr(query_understanding, 'original_query'):
                query = query_understanding.original_query
        
        query = query.lower()
        
        # Focus/concentration related
        if any(word in query for word in ['focus', 'concentration', 'background', 'ambient', 'studying', 'study']):
            functional_moods.append('focused')
            
        # Productivity related
        if any(word in query for word in ['productive', 'work', 'working', 'coding', 'office']):
            functional_moods.append('productive')
            
        # Study/concentration specific
        if any(word in query for word in ['study', 'studying', 'homework', 'reading', 'learning']):
            functional_moods.append('concentration')
            
        # Background/ambient
        if any(word in query for word in ['background', 'ambient', 'atmospheric', 'chill']):
            functional_moods.append('background')
        
        return list(set(functional_moods))
    
    async def _search_functional_mood(self, mood: str, limit: int) -> List[Dict[str, Any]]:
        """Search for tracks matching functional mood requirements."""
        candidates = []
        
        # Functional mood to search terms mapping
        functional_terms = {
            'focused': ['instrumental', 'ambient', 'concentration', 'study'],
            'productive': ['background', 'work', 'office', 'professional'],
            'concentration': ['focus', 'study', 'minimal', 'repetitive'],
            'background': ['ambient', 'unobtrusive', 'subtle', 'atmospheric'],
            'ambient': ['atmospheric', 'soundscape', 'environmental', 'texture']
        }
        
        search_terms = functional_terms.get(mood, [mood])
        
        try:
            tracks = await self.api_service.search_tracks_by_tags(
                tags=search_terms,
                limit=limit
            )
            
            for track_metadata in tracks:
                track = self._convert_metadata_to_dict(
                    track_metadata,
                    source='functional_mood_search',
                    source_confidence=0.8,
                    functional_mood=mood,
                    search_terms=search_terms
                )
                candidates.append(track)
                
        except Exception as e:
            self.logger.warning(f"Functional mood search failed for '{mood}': {e}")
        
        return candidates 