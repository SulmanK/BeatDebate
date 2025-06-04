"""
Comprehensive Quality Scorer for BeatDebate

Main quality scoring system that combines all scoring components.
This is the primary entry point for quality assessment, coordinating
all individual scoring components.
"""

from typing import Dict, Any
import structlog

from .audio_quality_scorer import AudioQualityScorer
from .popularity_balancer import PopularityBalancer
from .engagement_scorer import EngagementScorer
from .genre_mood_fit_scorer import GenreMoodFitScorer
from .intent_aware_scorer import IntentAwareScorer

logger = structlog.get_logger(__name__)


class ComprehensiveQualityScorer:
    """
    Main quality scoring system that combines all scoring components.
    """
    
    def __init__(self):
        """Initialize comprehensive quality scorer."""
        self.logger = logger.bind(component="QualityScorer")
        
        # Initialize component scorers
        self.audio_scorer = AudioQualityScorer()
        self.popularity_balancer = PopularityBalancer()
        self.engagement_scorer = EngagementScorer()
        self.genre_mood_scorer = GenreMoodFitScorer()
        self.intent_aware_scorer = IntentAwareScorer()
        
        # Component weights for final score
        self.component_weights = {
            'audio_quality': 0.40,       # Audio features quality
            'popularity_balance': 0.25,  # Popularity balance
            'engagement': 0.20,          # User engagement signals
            'genre_mood_fit': 0.15       # Genre/mood fit
        }
        
        self.logger.info("Comprehensive Quality Scorer initialized")
    
    async def calculate_track_quality(
        self, 
        track_data: Dict, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive quality score for a track.
        
        Args:
            track_data: Track metadata and features
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent
            
        Returns:
            Dictionary with quality score and breakdown
        """
        try:
            # Get audio features (placeholder - would integrate with Spotify API)
            audio_features = self._get_audio_features_placeholder(track_data)
            
            # Calculate component scores
            audio_quality = self.audio_scorer.calculate_audio_quality_score(
                audio_features, intent_analysis
            )
            
            popularity_score = self.popularity_balancer.calculate_popularity_score(
                listeners=int(track_data.get('listeners', 0)),
                playcount=int(track_data.get('playcount', 0)),
                exploration_openness=intent_analysis.get('exploration_openness', 0.5),
                entities=entities,
                intent_analysis=intent_analysis
            )
            
            engagement_score = await self.engagement_scorer.calculate_engagement_score(
                track_data, intent_analysis
            )
            
            genre_mood_fit = self.genre_mood_scorer.calculate_genre_mood_fit(
                track_data, entities, intent_analysis
            )
            
            # Calculate weighted total score
            total_quality = (
                audio_quality * self.component_weights['audio_quality'] +
                popularity_score * self.component_weights['popularity_balance'] +
                engagement_score * self.component_weights['engagement'] +
                genre_mood_fit * self.component_weights['genre_mood_fit']
            )
            
            quality_result = {
                'total_quality_score': total_quality,
                'quality_breakdown': {
                    'audio_quality': audio_quality,
                    'popularity_balance': popularity_score,
                    'engagement': engagement_score,
                    'genre_mood_fit': genre_mood_fit
                },
                'component_weights': self.component_weights,
                'quality_tier': self._determine_quality_tier(total_quality)
            }
            
            self.logger.debug(
                "Track quality calculated",
                track=f"{track_data.get('artist', 'Unknown')} - {track_data.get('name', 'Unknown')}",
                total_score=total_quality,
                tier=quality_result['quality_tier']
            )
            
            return quality_result
            
        except Exception as e:
            self.logger.error("Quality calculation failed", error=str(e))
            return {
                'total_quality_score': 0.5,
                'quality_breakdown': {},
                'component_weights': self.component_weights,
                'quality_tier': 'medium'
            }
    
    def _get_audio_features_placeholder(self, track_data: Dict) -> Dict:
        """
        Placeholder for audio features - would integrate with Spotify API.
        
        For now, generate reasonable defaults based on available data.
        """
        # Generate reasonable defaults based on source and metadata
        source = track_data.get('source', 'unknown')
        
        if source == 'underground_gems':
            # Underground tracks tend to be more experimental
            return {
                'energy': 0.6,
                'danceability': 0.5,
                'valence': 0.4,
                'acousticness': 0.6,
                'instrumentalness': 0.3,
                'liveness': 0.2,
                'speechiness': 0.1,
                'tempo': 110
            }
        elif source == 'primary_search':
            # Primary search results tend to be more mainstream
            return {
                'energy': 0.7,
                'danceability': 0.6,
                'valence': 0.6,
                'acousticness': 0.3,
                'instrumentalness': 0.2,
                'liveness': 0.1,
                'speechiness': 0.1,
                'tempo': 120
            }
        else:
            # Default balanced features
            return {
                'energy': 0.6,
                'danceability': 0.6,
                'valence': 0.5,
                'acousticness': 0.4,
                'instrumentalness': 0.2,
                'liveness': 0.1,
                'speechiness': 0.1,
                'tempo': 115
            }
    
    def _determine_quality_tier(self, quality_score: float) -> str:
        """Determine quality tier based on score."""
        if quality_score >= 0.8:
            return 'excellent'
        elif quality_score >= 0.7:
            return 'high'
        elif quality_score >= 0.6:
            return 'good'
        elif quality_score >= 0.4:
            return 'medium'
        else:
            return 'low'

    async def calculate_quality_score(
        self, 
        track_data: Dict, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate quality score for a track (simplified version).
        
        Args:
            track_data: Track metadata and features
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent
            
        Returns:
            Quality score (0.0 - 1.0)
        """
        try:
            # Use the comprehensive calculation and return just the score
            quality_result = await self.calculate_track_quality(
                track_data, entities, intent_analysis
            )
            return quality_result['total_quality_score']
            
        except Exception as e:
            self.logger.error("Quality score calculation failed", error=str(e))
            return 0.5  # Default neutral score 

    def calculate_intent_aware_scores(
        self, 
        track_data: Dict, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        intent: str
    ) -> Dict[str, float]:
        """
        🔧 NEW: Calculate all intent-aware scoring components.
        
        This provides all the scoring components needed by our ranking logic
        for different intent types and hybrid sub-types.
        
        Args:
            track_data: Track metadata
            entities: Musical entities
            intent_analysis: Intent analysis results
            intent: Intent type (including hybrid sub-types)
            
        Returns:
            Dictionary with all scoring components
        """
        try:
            scores = {}
            
            # ✅ Basic quality score (always needed)
            scores['quality'] = self.calculate_quality_score_sync(track_data, entities, intent_analysis)
            
            # ✅ Genre/mood match score (always needed)
            scores['genre_mood_match'] = self.genre_mood_scorer.calculate_genre_mood_fit(
                track_data, entities, intent_analysis
            )
            scores['contextual_relevance'] = scores['genre_mood_match']  # Alias for compatibility
            
            # 🔧 FIXED: Novelty score using intent-aware calculation
            scores['novelty'] = self.intent_aware_scorer.calculate_novelty_score(
                track_data, intent, entities
            )
            scores['novelty_score'] = scores['novelty']  # Alias for compatibility
            
            # ✅ Similarity score (for artist similarity intents)
            scores['similarity'] = self.intent_aware_scorer.calculate_similarity_score(
                track_data, entities, intent_analysis
            )
            
            # ✅ Target artist boost (for artist similarity intents)
            scores['target_artist_boost'] = self.intent_aware_scorer.calculate_target_artist_boost(
                track_data, entities
            )
            
            # ✅ Underground score (for discovery intents)
            scores['underground'] = self.intent_aware_scorer.calculate_underground_score(
                track_data, intent
            )
            
            # ✅ Context fit score (for contextual intents)
            scores['context_fit'] = self.intent_aware_scorer.calculate_context_fit_score(
                track_data, intent_analysis, entities
            )
            
            # ✅ Familiarity score (for contextual intents)
            scores['familiarity'] = self.intent_aware_scorer.calculate_familiarity_score(
                track_data, intent_analysis
            )
            
            # 🔧 Compatibility scores
            scores['quality_score'] = scores['quality']
            scores['relevance_score'] = scores['genre_mood_match']
            
            self.logger.debug(
                "Intent-aware scores calculated",
                intent=intent,
                novelty=scores['novelty'],
                similarity=scores['similarity'],
                quality=scores['quality'],
                track=f"{track_data.get('artist', 'Unknown')} - {track_data.get('name', 'Unknown')}"
            )
            
            return scores
            
        except Exception as e:
            self.logger.error("Intent-aware scoring failed", error=str(e))
            return {
                'quality': 0.5,
                'genre_mood_match': 0.5,
                'contextual_relevance': 0.5,
                'novelty': 0.5,
                'novelty_score': 0.5,
                'similarity': 0.0,
                'target_artist_boost': 0.0,
                'underground': 0.5,
                'context_fit': 0.5,
                'familiarity': 0.5,
                'quality_score': 0.5,
                'relevance_score': 0.5
            }
    
    def calculate_quality_score_sync(
        self, 
        track_data: Dict, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> float:
        """
        Synchronous version of quality score calculation.
        
        Args:
            track_data: Track metadata and features
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent
            
        Returns:
            Quality score (0.0 - 1.0)
        """
        try:
            # Get audio features (placeholder)
            audio_features = self._get_audio_features_placeholder(track_data)
            
            # Calculate component scores (simplified for sync version)
            audio_quality = self.audio_scorer.calculate_audio_quality_score(
                audio_features, intent_analysis
            )
            
            popularity_score = self.popularity_balancer.calculate_popularity_score(
                listeners=int(track_data.get('listeners', 0)),
                playcount=int(track_data.get('playcount', 0)),
                exploration_openness=intent_analysis.get('exploration_openness', 0.5),
                entities=entities,
                intent_analysis=intent_analysis
            )
            
            # Simplified engagement score (skip async calculation)
            engagement_score = 0.5
            
            genre_mood_fit = self.genre_mood_scorer.calculate_genre_mood_fit(
                track_data, entities, intent_analysis
            )
            
            # Calculate weighted total score
            total_quality = (
                audio_quality * self.component_weights['audio_quality'] +
                popularity_score * self.component_weights['popularity_balance'] +
                engagement_score * self.component_weights['engagement'] +
                genre_mood_fit * self.component_weights['genre_mood_fit']
            )
            
            return min(1.0, max(0.0, total_quality))
            
        except Exception as e:
            self.logger.error("Sync quality score calculation failed", error=str(e))
            return 0.5  # Default neutral score


# 🔧 Export the main scorer class for easy importing
QualityScorer = ComprehensiveQualityScorer 