"""
Entity Processor Component for Planner Agent

Handles entity extraction and processing logic for the planner agent.
Extracted from PlannerAgent for better modularization and single responsibility.
"""

from typing import Dict, Any, List
import structlog

logger = structlog.get_logger(__name__)


class EntityProcessor:
    """
    Handles entity extraction and processing for the planner agent.
    
    Responsibilities:
    - Extracting entity names from various formats
    - Processing entities from different sources (context, effective intent)
    - Standardizing entity structures
    - Utility methods for entity manipulation
    """
    
    def __init__(self):
        """Initialize the EntityProcessor."""
        self.logger = logger
        self.logger.info("EntityProcessor initialized")
    
    def extract_entity_names(self, entity_list: List) -> List[str]:
        """
        Extract names from entity list that may contain dicts or strings.
        
        This utility method handles the data transformation needed to work with
        various entity formats from preserved context.
        
        Args:
            entity_list: List of entities in various formats
            
        Returns:
            List of entity name strings
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
    
    def extract_artists_from_effective_intent(self, entities: Dict[str, Any]) -> List[str]:
        """
        Extract artist names from effective intent entities.
        
        Args:
            entities: Entities dictionary from effective intent
            
        Returns:
            List of artist names
        """
        artists = []
        
        # Handle different entity structures
        if 'artists' in entities:
            artist_data = entities['artists']
            if isinstance(artist_data, list):
                artists.extend(self.extract_entity_names(artist_data))
        
        # Also check musical_entities structure
        if 'musical_entities' in entities and 'artists' in entities['musical_entities']:
            artist_data = entities['musical_entities']['artists']
            if isinstance(artist_data, dict) and 'primary' in artist_data:
                artists.extend(self.extract_entity_names(artist_data['primary']))
            elif isinstance(artist_data, list):
                artists.extend(self.extract_entity_names(artist_data))
        
        return list(set(artists))  # Remove duplicates
    
    def extract_genres_from_effective_intent(self, entities: Dict[str, Any]) -> List[str]:
        """
        Extract genre names from effective intent entities.
        
        Args:
            entities: Entities dictionary from effective intent
            
        Returns:
            List of genre names
        """
        genres = []
        
        if 'genres' in entities:
            genre_data = entities['genres']
            if isinstance(genre_data, dict):
                for genre_list in [genre_data.get('primary', []), genre_data.get('secondary', [])]:
                    genres.extend(self.extract_entity_names(genre_list))
            elif isinstance(genre_data, list):
                genres.extend(self.extract_entity_names(genre_data))
        
        # Also check musical_entities structure
        if 'musical_entities' in entities and 'genres' in entities['musical_entities']:
            genre_data = entities['musical_entities']['genres']
            if isinstance(genre_data, dict):
                for genre_list in [genre_data.get('primary', []), genre_data.get('secondary', [])]:
                    genres.extend(self.extract_entity_names(genre_list))
        
        return list(set(genres))  # Remove duplicates
    
    def extract_moods_from_effective_intent(self, entities: Dict[str, Any]) -> List[str]:
        """
        Extract mood names from effective intent entities.
        
        Args:
            entities: Entities dictionary from effective intent
            
        Returns:
            List of mood names
        """
        moods = []
        
        if 'moods' in entities:
            mood_data = entities['moods']
            if isinstance(mood_data, dict):
                for mood_list in [mood_data.get('primary', []), mood_data.get('secondary', [])]:
                    moods.extend(self.extract_entity_names(mood_list))
            elif isinstance(mood_data, list):
                moods.extend(self.extract_entity_names(mood_data))
        
        # Also check musical_entities structures
        if 'musical_entities' in entities and 'moods' in entities['musical_entities']:
            mood_data = entities['musical_entities']['moods']
            if isinstance(mood_data, dict):
                for mood_list in [mood_data.get('primary', []), mood_data.get('energy', []), mood_data.get('emotion', [])]:
                    moods.extend(self.extract_entity_names(mood_list))
        
        return list(set(moods))  # Remove duplicates
    
    def extract_activities_from_effective_intent(self, entities: Dict[str, Any]) -> List[str]:
        """
        Extract activity names from effective intent entities.
        
        Args:
            entities: Entities dictionary from effective intent
            
        Returns:
            List of activity names
        """
        activities = []
        
        if 'activities' in entities:
            activity_data = entities['activities']
            if isinstance(activity_data, list):
                activities.extend(self.extract_entity_names(activity_data))
        
        # Also check contextual_entities structure
        if 'contextual_entities' in entities and 'activities' in entities['contextual_entities']:
            activity_data = entities['contextual_entities']['activities']
            if isinstance(activity_data, dict):
                for activity_list in [activity_data.get('physical', []), activity_data.get('mental', []), activity_data.get('social', [])]:
                    activities.extend(self.extract_entity_names(activity_list))
        
        return list(set(activities))  # Remove duplicates
    
    def create_standardized_entities_structure(
        self, 
        artists: List[str] = None,
        genres: List[str] = None,
        moods: List[str] = None,
        activities: List[str] = None,
        confidence: float = 0.8,
        extraction_method: str = "unknown",
        intent_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized entities structure.
        
        Args:
            artists: List of artist names
            genres: List of genre names
            moods: List of mood names
            activities: List of activity names
            confidence: Overall confidence score
            extraction_method: Method used for extraction
            intent_info: Additional intent analysis information
            
        Returns:
            Standardized entities dictionary
        """
        artists = artists or []
        genres = genres or []
        moods = moods or []
        activities = activities or []
        intent_info = intent_info or {}
        
        entities = {
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
                "overall": confidence
            },
            "extraction_method": extraction_method,
            "intent_analysis": intent_info
        }
        
        self.logger.debug(
            "Created standardized entities structure",
            artists_count=len(artists),
            genres_count=len(genres),
            moods_count=len(moods),
            activities_count=len(activities),
            extraction_method=extraction_method
        )
        
        return entities
    
    def validate_entities_structure(self, entities: Dict[str, Any]) -> bool:
        """
        Validate that an entities structure conforms to the expected format.
        
        Args:
            entities: Entities dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required top-level keys
            required_keys = ['musical_entities', 'contextual_entities', 'confidence_scores']
            for key in required_keys:
                if key not in entities:
                    self.logger.warning(f"Missing required key: {key}")
                    return False
            
            # Check musical_entities structure
            musical = entities['musical_entities']
            musical_required = ['artists', 'genres', 'tracks', 'moods']
            for key in musical_required:
                if key not in musical:
                    self.logger.warning(f"Missing musical entity key: {key}")
                    return False
            
            # Check artist structure
            artists = musical['artists']
            if not isinstance(artists, dict) or 'primary' not in artists:
                self.logger.warning("Invalid artists structure")
                return False
            
            # Check confidence scores
            confidence = entities['confidence_scores']
            if not isinstance(confidence, dict) or 'overall' not in confidence:
                self.logger.warning("Invalid confidence scores structure")
                return False
            
            overall_confidence = confidence['overall']
            if not isinstance(overall_confidence, (int, float)) or not 0 <= overall_confidence <= 1:
                self.logger.warning("Invalid overall confidence value")
                return False
            
            self.logger.debug("Entities structure validation passed")
            return True
            
        except Exception as e:
            self.logger.error("Entities structure validation failed", error=str(e))
            return False
    
    def merge_entities(self, entities1: Dict[str, Any], entities2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two entities structures, combining their contents.
        
        Args:
            entities1: First entities structure
            entities2: Second entities structure
            
        Returns:
            Merged entities structure
        """
        try:
            # Start with a copy of the first structure
            merged = self._deep_copy_entities(entities1)
            
            # Merge musical entities
            if 'musical_entities' in entities2:
                musical2 = entities2['musical_entities']
                merged_musical = merged['musical_entities']
                
                # Merge artists
                if 'artists' in musical2 and 'primary' in musical2['artists']:
                    existing_artists = set(merged_musical['artists']['primary'])
                    new_artists = set(musical2['artists']['primary'])
                    merged_musical['artists']['primary'] = list(existing_artists | new_artists)
                
                # Merge genres
                if 'genres' in musical2 and 'primary' in musical2['genres']:
                    existing_genres = set(merged_musical['genres']['primary'])
                    new_genres = set(musical2['genres']['primary'])
                    merged_musical['genres']['primary'] = list(existing_genres | new_genres)
                
                # Merge moods
                if 'moods' in musical2 and 'primary' in musical2['moods']:
                    existing_moods = set(merged_musical['moods']['primary'])
                    new_moods = set(musical2['moods']['primary'])
                    merged_musical['moods']['primary'] = list(existing_moods | new_moods)
            
            # Update confidence to average
            if 'confidence_scores' in entities2 and 'overall' in entities2['confidence_scores']:
                conf1 = merged['confidence_scores']['overall']
                conf2 = entities2['confidence_scores']['overall']
                merged['confidence_scores']['overall'] = (conf1 + conf2) / 2
            
            # Update extraction method to indicate merge
            method1 = merged.get('extraction_method', 'unknown')
            method2 = entities2.get('extraction_method', 'unknown')
            merged['extraction_method'] = f"merged_{method1}_{method2}"
            
            self.logger.debug("Successfully merged entities structures")
            return merged
            
        except Exception as e:
            self.logger.error("Failed to merge entities", error=str(e))
            return entities1  # Return first as fallback
    
    def _deep_copy_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of entities structure."""
        import copy
        return copy.deepcopy(entities)
    
    def get_entity_summary(self, entities: Dict[str, Any]) -> Dict[str, int]:
        """
        Get a summary of entity counts in the structure.
        
        Args:
            entities: Entities structure
            
        Returns:
            Dictionary with entity counts
        """
        try:
            musical = entities.get('musical_entities', {})
            contextual = entities.get('contextual_entities', {})
            
            summary = {
                'artists': len(musical.get('artists', {}).get('primary', [])),
                'genres': len(musical.get('genres', {}).get('primary', [])),
                'moods': len(musical.get('moods', {}).get('primary', [])),
                'tracks': len(musical.get('tracks', {}).get('primary', [])),
                'activities': len(contextual.get('activities', {}).get('physical', [])),
                'total_entities': 0
            }
            
            summary['total_entities'] = sum([
                summary['artists'], summary['genres'], summary['moods'],
                summary['tracks'], summary['activities']
            ])
            
            return summary
            
        except Exception as e:
            self.logger.error("Failed to get entity summary", error=str(e))
            return {'total_entities': 0} 