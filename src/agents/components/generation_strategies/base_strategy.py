"""
Base Strategy Class for Candidate Generation

Defines the common interface and shared functionality for all generation strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import structlog

from ....services.api_service import APIService
from ....models.metadata_models import UnifiedTrackMetadata


class BaseGenerationStrategy(ABC):
    """
    Abstract base class for all candidate generation strategies.
    
    Each strategy implements a specific approach to generating candidate tracks
    based on entities and intent analysis.
    """
    
    def __init__(self, api_service: APIService, llm_client=None):
        """
        Initialize the generation strategy.
        
        Args:
            api_service: Unified API service for music data access
            llm_client: Optional LLM client for AI-powered features
        """
        self.api_service = api_service
        self.llm_client = llm_client
        self.logger = structlog.get_logger(__name__)
    
    @abstractmethod
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate tracks using this strategy.
        
        Args:
            entities: Extracted entities from query
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of candidate track dictionaries
        """
        pass
    
    def _convert_metadata_to_dict(
        self, 
        track_metadata: UnifiedTrackMetadata, 
        source: str,
        source_confidence: float,
        **extra_metadata
    ) -> Dict[str, Any]:
        """
        Convert UnifiedTrackMetadata to candidate dictionary format.
        
        Args:
            track_metadata: Track metadata object
            source: Source identifier for this candidate
            source_confidence: Confidence score for this source
            **extra_metadata: Additional metadata to include
            
        Returns:
            Candidate dictionary with all required fields
        """
        # Use getattr with defaults for robust attribute access
        return {
            'id': f"{getattr(track_metadata, 'artist', 'unknown')}_{getattr(track_metadata, 'name', 'unknown')}".replace(' ', '_').lower(),
            'name': getattr(track_metadata, 'name', 'Unknown Track'),
            'artist': getattr(track_metadata, 'artist', 'Unknown Artist'),
            'album': getattr(track_metadata, 'album', None) or 'Unknown',
            'duration': getattr(track_metadata, 'duration_ms', None) or 0,
            'popularity': getattr(track_metadata, 'popularity', None) or 0.0,
            'genres': getattr(track_metadata, 'genres', None) or [],
            'tags': getattr(track_metadata, 'tags', None) or [],
            'audio_features': getattr(track_metadata, 'audio_features', None) or {},
            'listeners': getattr(track_metadata, 'listeners', None) or 0,
            'playcount': getattr(track_metadata, 'playcount', None) or 0,
            'source': source,
            'source_confidence': source_confidence,
            'timestamp': self._get_timestamp(),
            **extra_metadata
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for candidate tracking."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _extract_seed_artists(self, entities: Dict[str, Any]) -> List[str]:
        """
        Extract seed artists from entities for strategy use.
        
        Args:
            entities: Entity dictionary from planner
            
        Returns:
            List of artist names
        """
        artists = []
        
        # DEBUG: Log the entities structure
        self.logger.debug(f"🔍 ENTITIES DEBUG: Full entities structure: {entities}")
        
        # FIXED: Handle BOTH entity formats (flat and nested)
        
        # 1. Try flat format first (from context handlers): {'artists': ['Mk.gee'], ...}
        if 'artists' in entities and isinstance(entities['artists'], list):
            self.logger.debug(f"🔍 ENTITIES DEBUG: Found flat format artists: {entities['artists']}")
            for artist_obj in entities['artists']:
                if isinstance(artist_obj, dict) and 'name' in artist_obj:
                    artists.append(artist_obj['name'])
                    self.logger.debug(f"🔍 ENTITIES DEBUG: Extracted artist from dict: {artist_obj['name']}")
                elif isinstance(artist_obj, str):
                    artists.append(artist_obj)
                    self.logger.debug(f"🔍 ENTITIES DEBUG: Extracted artist from string: {artist_obj}")
        
        # 2. Try nested format (from planners): {'musical_entities': {'artists': {'primary': [...]}}}
        elif 'musical_entities' in entities and 'artists' in entities['musical_entities']:
            musical_artists = entities['musical_entities']['artists']
            
            # Extract from musical_entities.artists.primary
            if 'primary' in musical_artists:
                self.logger.debug(f"🔍 ENTITIES DEBUG: Found musical_entities.artists.primary: {musical_artists['primary']}")
                for artist_obj in musical_artists['primary']:
                    if isinstance(artist_obj, dict) and 'name' in artist_obj:
                        artists.append(artist_obj['name'])
                        self.logger.debug(f"🔍 ENTITIES DEBUG: Extracted artist from dict: {artist_obj['name']}")
                    elif isinstance(artist_obj, str):
                        artists.append(artist_obj)
                        self.logger.debug(f"🔍 ENTITIES DEBUG: Extracted artist from string: {artist_obj}")
            
            # Extract from musical_entities.artists.similar_to
            if 'similar_to' in musical_artists:
                for artist_obj in musical_artists['similar_to']:
                    if isinstance(artist_obj, dict) and 'name' in artist_obj:
                        artists.append(artist_obj['name'])
                    elif isinstance(artist_obj, str):
                        artists.append(artist_obj)
        
        # 3. Fallback: Try legacy format (artists.primary) for backward compatibility
        elif 'artists' in entities and isinstance(entities['artists'], dict) and 'primary' in entities['artists']:
            self.logger.debug(f"🔍 ENTITIES DEBUG: Found legacy artists.primary: {entities['artists']['primary']}")
            for artist_obj in entities['artists']['primary']:
                if isinstance(artist_obj, dict) and 'name' in artist_obj:
                    artists.append(artist_obj['name'])
                    self.logger.debug(f"🔍 ENTITIES DEBUG: Extracted artist from dict: {artist_obj['name']}")
                elif isinstance(artist_obj, str):
                    artists.append(artist_obj)
                    self.logger.debug(f"🔍 ENTITIES DEBUG: Extracted artist from string: {artist_obj}")
        
        else:
            self.logger.debug(f"🔍 ENTITIES DEBUG: No artists found - keys: {entities.keys()}")
            if 'musical_entities' in entities:
                self.logger.debug(f"🔍 ENTITIES DEBUG: Musical entities keys: {entities['musical_entities'].keys()}")
                if 'artists' in entities['musical_entities']:
                    self.logger.debug(f"🔍 ENTITIES DEBUG: Musical artists keys: {entities['musical_entities']['artists'].keys()}")
        
        self.logger.debug(f"🔍 ENTITIES DEBUG: Final extracted artists: {artists}")
        return list(set(artists))  # Remove duplicates
    
    def _extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """
        Extract target genres from entities for strategy use.
        
        Args:
            entities: Entity dictionary from planner
            
        Returns:
            List of genre names
        """
        genres = []
        
        # FIXED: Handle BOTH entity formats (flat and nested)
        
        # 1. Try flat format first (from context handlers): {'genres': ['rock'], ...}
        if 'genres' in entities and isinstance(entities['genres'], list):
            self.logger.debug(f"🔍 ENTITIES DEBUG: Found flat format genres: {entities['genres']}")
            for genre_obj in entities['genres']:
                if isinstance(genre_obj, dict) and 'name' in genre_obj:
                    genres.append(genre_obj['name'])
                elif isinstance(genre_obj, str):
                    genres.append(genre_obj)
        
        # 2. Try nested format (from planners): {'musical_entities': {'genres': {'primary': [...]}}}
        elif 'musical_entities' in entities and 'genres' in entities['musical_entities']:
            musical_genres = entities['musical_entities']['genres']
            
            # Extract from musical_entities.genres.primary
            if 'primary' in musical_genres:
                for genre_obj in musical_genres['primary']:
                    if isinstance(genre_obj, dict) and 'name' in genre_obj:
                        genres.append(genre_obj['name'])
                    elif isinstance(genre_obj, str):
                        genres.append(genre_obj)
            
            # Extract from musical_entities.genres.secondary
            if 'secondary' in musical_genres:
                for genre_obj in musical_genres['secondary']:
                    if isinstance(genre_obj, dict) and 'name' in genre_obj:
                        genres.append(genre_obj['name'])
                    elif isinstance(genre_obj, str):
                        genres.append(genre_obj)
        
        # 3. Fallback: Try legacy format for backward compatibility
        elif 'genres' in entities and isinstance(entities['genres'], dict):
            if 'primary' in entities['genres']:
                for genre_obj in entities['genres']['primary']:
                    if isinstance(genre_obj, dict) and 'name' in genre_obj:
                        genres.append(genre_obj['name'])
                    elif isinstance(genre_obj, str):
                        genres.append(genre_obj)
            
            if 'secondary' in entities['genres']:
                for genre_obj in entities['genres']['secondary']:
                    if isinstance(genre_obj, dict) and 'name' in genre_obj:
                        genres.append(genre_obj['name'])
                    elif isinstance(genre_obj, str):
                        genres.append(genre_obj)
        
        return list(set(genres))  # Remove duplicates
    
    def _extract_moods(self, entities: Dict[str, Any]) -> List[str]:
        """
        Extract moods from entities for strategy use.
        
        Args:
            entities: Entity dictionary from planner
            
        Returns:
            List of mood descriptors
        """
        moods = []
        
        # FIXED: Handle BOTH entity formats (flat and nested)
        
        # 1. Try flat format first (from context handlers): {'moods': ['upbeat'], ...}
        if 'moods' in entities and isinstance(entities['moods'], list):
            self.logger.debug(f"🔍 ENTITIES DEBUG: Found flat format moods: {entities['moods']}")
            for mood_obj in entities['moods']:
                if isinstance(mood_obj, dict) and 'name' in mood_obj:
                    moods.append(mood_obj['name'])
                elif isinstance(mood_obj, str):
                    moods.append(mood_obj)
        
        # 2. Try nested format (from planners): {'musical_entities': {'moods': {'primary': [...]}}}
        elif 'musical_entities' in entities and 'moods' in entities['musical_entities']:
            musical_moods = entities['musical_entities']['moods']
            
            # Extract from musical_entities.moods.primary
            if 'primary' in musical_moods:
                for mood_obj in musical_moods['primary']:
                    if isinstance(mood_obj, dict) and 'name' in mood_obj:
                        moods.append(mood_obj['name'])
                    elif isinstance(mood_obj, str):
                        moods.append(mood_obj)
        
        # 3. Fallback: Try legacy format for backward compatibility
        elif 'moods' in entities and isinstance(entities['moods'], dict) and 'primary' in entities['moods']:
            for mood_obj in entities['moods']['primary']:
                if isinstance(mood_obj, dict) and 'name' in mood_obj:
                    moods.append(mood_obj['name'])
                elif isinstance(mood_obj, str):
                    moods.append(mood_obj)
        
        return list(set(moods))  # Remove duplicates 