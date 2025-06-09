"""
Genre and Mood Fit Scorer for BeatDebate

Scores how well tracks fit the requested genres and moods,
evaluating genre matching, mood alignment, and artist similarity.
"""

from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class GenreMoodFitScorer:
    """
    Scores how well tracks fit the requested genres and moods.
    """
    
    def __init__(self):
        """Initialize genre/mood fit scorer."""
        self.logger = logger.bind(component="GenreMoodFitScorer")
        self.logger.info("Genre/Mood Fit Scorer initialized")
    
    def calculate_genre_mood_fit(
        self, 
        track_data: Dict, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate how well track fits requested genres and moods.
        
        Args:
            track_data: Track metadata
            entities: Extracted entities including genres and moods
            intent_analysis: User intent analysis
            
        Returns:
            Genre/mood fit score (0.0 - 1.0)
        """
        try:
            # Calculate genre fit
            genre_fit = self._calculate_genre_fit(track_data, entities)
            
            # Calculate mood fit
            mood_fit = self._calculate_mood_fit(track_data, entities, intent_analysis)
            
            # Calculate artist fit (if artist was mentioned)
            artist_fit = self._calculate_artist_fit(track_data, entities)
            
            # Combine scores with weights
            total_fit = (
                genre_fit * 0.4 +
                mood_fit * 0.4 +
                artist_fit * 0.2
            )
            
            self.logger.debug(
                "Genre/mood fit calculated",
                genre_fit=genre_fit,
                mood_fit=mood_fit,
                artist_fit=artist_fit,
                total_fit=total_fit
            )
            
            return min(1.0, max(0.0, total_fit))
            
        except Exception as e:
            self.logger.warning("Genre/mood fit calculation failed", error=str(e))
            return 0.5  # Default neutral score
    
    def _calculate_genre_fit(self, track_data: Dict, entities: Dict[str, Any]) -> float:
        """Calculate genre fit score."""
        requested_genres = entities.get("musical_entities", {}).get("genres", {}).get("primary", [])
        
        if not requested_genres:
            return 0.7  # Neutral score if no specific genres requested
        
        # This would need genre classification of the track
        # For now, use source information as proxy
        # source = track_data.get('source', '')  # TODO: Use when implementing genre classification
        search_term = track_data.get('search_term', '').lower()
        
        # Check if search term matches requested genres
        genre_matches = 0
        for genre in requested_genres:
            if genre.lower() in search_term:
                genre_matches += 1
        
        if genre_matches > 0:
            return min(1.0, genre_matches / len(requested_genres))
        else:
            return 0.5  # Default if no clear match
    
    def _calculate_mood_fit(
        self, 
        track_data: Dict, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> float:
        """Calculate mood fit score."""
        requested_moods = entities.get("contextual_entities", {}).get("moods", {})
        
        if not requested_moods:
            return 0.7  # Neutral score if no specific moods requested
        
        # Extract all mood terms
        all_moods = []
        for mood_category in requested_moods.values():
            all_moods.extend(mood_category)
        
        # Check if search term or source matches moods
        search_term = track_data.get('search_term', '').lower()
        exploration_tag = track_data.get('exploration_tag', '').lower()
        
        mood_matches = 0
        for mood in all_moods:
            if mood.lower() in search_term or mood.lower() in exploration_tag:
                mood_matches += 1
        
        if mood_matches > 0:
            return min(1.0, mood_matches / len(all_moods))
        else:
            return 0.5  # Default if no clear match
    
    def _calculate_artist_fit(self, track_data: Dict, entities: Dict[str, Any]) -> float:
        """Calculate artist fit score."""
        requested_artists = entities.get("musical_entities", {}).get("artists", {})
        
        if not requested_artists:
            return 0.7  # Neutral score if no specific artists requested
        
        track_artist = track_data.get('artist', '').lower()
        source_artist = track_data.get('source_artist', '').lower()
        
        # Check for direct artist matches
        all_artists = []
        all_artists.extend(requested_artists.get("primary", []))
        all_artists.extend(requested_artists.get("similar_to", []))
        
        for artist in all_artists:
            if artist.lower() in track_artist or artist.lower() in source_artist:
                return 1.0  # Perfect match
        
        # Check if from similar artist source
        if track_data.get('source') == 'similar_artists':
            return 0.8  # High score for similar artist tracks
        
        return 0.5  # Default score 