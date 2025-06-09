"""
Popularity Balancer for BeatDebate

Balances mainstream vs underground preferences based on user intent,
adjusting scoring based on track popularity and user's exploration preferences.
"""

import math
from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class PopularityBalancer:
    """
    Balances mainstream vs underground preferences based on user intent.
    
    Adjusts scoring based on track popularity and user's exploration preferences.
    """
    
    def __init__(self):
        """Initialize popularity balancer."""
        self.logger = logger.bind(component="PopularityBalancer")
        self.logger.info("Popularity Balancer initialized")
    
    def calculate_popularity_score(
        self, 
        listeners: int, 
        playcount: int, 
        exploration_openness: float = 0.5,
        entities: Dict[str, Any] = None,
        intent_analysis: Dict[str, Any] = None
    ) -> float:
        """
        Calculate popularity-based score with configurable exploration preference.
        
        Args:
            listeners: Number of unique listeners
            playcount: Total play count
            exploration_openness: 0.0 = prefer popular, 1.0 = prefer underground
            entities: Musical entities from query understanding
            intent_analysis: Intent analysis for context-aware scoring
            
        Returns:
            Score from 0.0 to 1.0
        """
        # 🎯 NEW: Adjust exploration for genre-hybrid queries
        if entities and intent_analysis and self._is_genre_hybrid_query(entities, intent_analysis):
            exploration_openness = 0.75  # More tolerant of popular tracks for genre examples
            
        base_popularity = self._calculate_base_popularity(listeners, playcount)
        
        # Apply exploration preference
        if exploration_openness <= 0.5:
            # Prefer popular tracks
            preference_factor = (0.5 - exploration_openness) * 2
            score = base_popularity + (1 - base_popularity) * preference_factor
        else:
            # Prefer underground tracks  
            preference_factor = (exploration_openness - 0.5) * 2
            score = base_popularity * (1 - preference_factor)
        
        final_score = max(0.0, min(1.0, score))
        
        self.logger.debug(
            "Popularity score calculated",
            listeners=listeners,
            playcount=playcount,
            base_popularity=base_popularity,
            exploration_openness=exploration_openness,
            final_score=final_score
        )
        
        return final_score
    
    def _calculate_base_popularity(self, listeners: int, playcount: int) -> float:
        """Calculate base popularity score from play counts."""
        # Use log scale to handle wide range of play counts
        if playcount > 0:
            # Normalize to roughly 0-1 scale (10M plays = 1.0)
            playcount_score = min(1.0, math.log10(max(1, playcount)) / 7.0)
        else:
            playcount_score = 0.0
        
        if listeners > 0:
            # Normalize to roughly 0-1 scale (1M listeners = 1.0)
            listeners_score = min(1.0, math.log10(max(1, listeners)) / 6.0)
        else:
            listeners_score = 0.0
        
        # Combine play count and listener count
        return (playcount_score + listeners_score) / 2
    
    def _is_genre_hybrid_query(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> bool:
        """
        Detect if this is a genre-hybrid query that should be more tolerant of popular tracks.
        
        Genre-hybrid queries like "Music like Kendrick Lamar but jazzy" want good examples
        of genre fusion, not just underground tracks.
        """
        if not entities or not intent_analysis:
            return False
            
        musical_entities = entities.get('musical_entities', {})
        if not musical_entities:
            return False
        
        # Check for genre constraints
        genres = musical_entities.get('genres', {})
        has_genres = len(genres.get('primary', [])) > 0 or len(genres.get('secondary', [])) > 0
        
        # Check for artist similarity component 
        has_artist_similarity = len(musical_entities.get('artists', [])) > 0
        
        # Check if it's a hybrid intent
        intent_type = intent_analysis.get('primary_intent', '')
        is_hybrid_intent = intent_type == 'hybrid_similarity_genre' or 'hybrid' in str(intent_type).lower()
        
        # All conditions must be true for genre-hybrid query
        return has_genres and has_artist_similarity and is_hybrid_intent 