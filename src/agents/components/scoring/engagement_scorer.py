"""
Engagement Scorer for BeatDebate

Scores tracks based on user engagement signals and track characteristics,
including engagement rates, recency, and tag diversity.
"""

from typing import Dict
import structlog

logger = structlog.get_logger(__name__)


class EngagementScorer:
    """
    Scores tracks based on user engagement signals and track characteristics.
    """
    
    def __init__(self):
        """Initialize engagement scorer."""
        self.logger = logger.bind(component="EngagementScorer")
        self.logger.info("Engagement Scorer initialized")
    
    async def calculate_engagement_score(
        self, 
        track_data: Dict, 
        intent_analysis: Dict
    ) -> float:
        """
        Calculate engagement score based on various signals.
        
        Args:
            track_data: Track metadata and statistics
            intent_analysis: User intent for context
            
        Returns:
            Engagement score (0.0 - 1.0)
        """
        try:
            # Calculate engagement rate (plays per listener)
            engagement_rate = self._calculate_engagement_rate(track_data)
            
            # Calculate recency score (newer tracks get slight boost)
            recency_score = self._calculate_recency_score(track_data)
            
            # Calculate tag diversity score
            tag_diversity = self._calculate_tag_diversity(track_data)
            
            # Combine scores with weights
            total_engagement = (
                engagement_rate * 0.5 +
                recency_score * 0.3 +
                tag_diversity * 0.2
            )
            
            self.logger.debug(
                "Engagement score calculated",
                engagement_rate=engagement_rate,
                recency_score=recency_score,
                tag_diversity=tag_diversity,
                total_score=total_engagement
            )
            
            return min(1.0, max(0.0, total_engagement))
            
        except Exception as e:
            self.logger.warning("Engagement scoring failed", error=str(e))
            return 0.5  # Default neutral score
    
    def _calculate_engagement_rate(self, track_data: Dict) -> float:
        """Calculate engagement rate from play counts and listeners."""
        playcount = int(track_data.get('playcount') or 0)
        listeners = int(track_data.get('listeners') or 1)  # Avoid division by zero
        
        # Ensure we don't divide by zero
        if listeners == 0:
            listeners = 1
        
        # Calculate plays per listener
        engagement_rate = playcount / listeners
        
        # Normalize to 0-1 scale (50 plays per listener = 1.0)
        normalized_rate = min(1.0, engagement_rate / 50.0)
        
        return normalized_rate
    
    def _calculate_recency_score(self, track_data: Dict) -> float:
        """Calculate recency score - slight boost for newer tracks."""
        # This is a placeholder - would need release date data
        # For now, return neutral score
        return 0.5
    
    def _calculate_tag_diversity(self, track_data: Dict) -> float:
        """Calculate tag diversity score."""
        # This is a placeholder - would need tag data from Last.fm
        # For now, return neutral score
        return 0.5 