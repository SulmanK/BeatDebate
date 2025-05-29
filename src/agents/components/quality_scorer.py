"""
Quality Scoring System for BeatDebate

Implements multi-dimensional quality scoring for track candidates:
1. Audio Features Quality (40%)
2. Popularity Balance (25%) 
3. User Engagement Signals (20%)
4. Genre/Mood Fit (15%)
"""

import math
from typing import Dict, List, Any, Optional
import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)


class AudioQualityScorer:
    """
    Scores tracks based on audio feature analysis.
    
    Evaluates energy, danceability, valence, and other audio characteristics
    to determine musical quality and appropriateness.
    """
    
    def __init__(self):
        """Initialize audio quality scorer with feature weights."""
        self.logger = logger.bind(component="AudioQualityScorer")
        
        # Feature weights for audio quality calculation
        self.feature_weights = {
            'energy': 0.20,           # Energy level appropriateness
            'danceability': 0.15,     # Rhythmic quality
            'valence': 0.15,          # Emotional positivity
            'acousticness': 0.10,     # Acoustic vs electronic balance
            'instrumentalness': 0.10, # Vocal vs instrumental balance
            'liveness': 0.10,         # Live recording quality
            'speechiness': 0.10,      # Speech content appropriateness
            'tempo_consistency': 0.10 # Tempo stability
        }
        
        self.logger.info("Audio Quality Scorer initialized")
    
    def calculate_audio_quality_score(self, track_features: Dict, intent_analysis: Dict) -> float:
        """
        Calculate quality score based on audio features.
        
        Args:
            track_features: Audio features from Last.fm or Spotify
            intent_analysis: User intent for context-aware scoring
            
        Returns:
            Audio quality score (0.0 - 1.0)
        """
        try:
            quality_score = 0.0
            
            # Energy optimization based on intent
            energy_score = self._score_energy(track_features, intent_analysis)
            quality_score += energy_score * self.feature_weights['energy']
            
            # Danceability scoring
            danceability_score = self._score_danceability(track_features, intent_analysis)
            quality_score += danceability_score * self.feature_weights['danceability']
            
            # Valence (mood) scoring
            valence_score = self._score_valence(track_features, intent_analysis)
            quality_score += valence_score * self.feature_weights['valence']
            
            # Acousticness balance
            acousticness_score = self._score_acousticness(track_features, intent_analysis)
            quality_score += acousticness_score * self.feature_weights['acousticness']
            
            # Instrumentalness balance
            instrumentalness_score = self._score_instrumentalness(track_features, intent_analysis)
            quality_score += instrumentalness_score * self.feature_weights['instrumentalness']
            
            # Liveness quality
            liveness_score = self._score_liveness(track_features)
            quality_score += liveness_score * self.feature_weights['liveness']
            
            # Speechiness appropriateness
            speechiness_score = self._score_speechiness(track_features, intent_analysis)
            quality_score += speechiness_score * self.feature_weights['speechiness']
            
            # Tempo consistency
            tempo_score = self._score_tempo(track_features, intent_analysis)
            quality_score += tempo_score * self.feature_weights['tempo_consistency']
            
            final_score = min(1.0, max(0.0, quality_score))
            
            self.logger.debug(
                "Audio quality calculated",
                score=final_score,
                energy=energy_score,
                danceability=danceability_score,
                valence=valence_score
            )
            
            return final_score
            
        except Exception as e:
            self.logger.warning("Audio quality calculation failed", error=str(e))
            return 0.5  # Default neutral score
    
    def _score_energy(self, features: Dict, intent: Dict) -> float:
        """Score energy level based on intent context."""
        energy = features.get('energy', 0.5)
        
        # Get activity context for energy preferences
        primary_intent = intent.get('primary_intent', 'discovery')
        
        # Energy preferences by intent
        energy_preferences = {
            'concentration': 0.3,  # Lower energy for focus
            'relaxation': 0.2,     # Very low energy for relaxation
            'energy': 0.8,         # High energy for energetic activities
            'discovery': 0.5,      # Balanced energy for discovery
            'workout': 0.9,        # Very high energy for workouts
            'study': 0.3           # Lower energy for studying
        }
        
        target_energy = energy_preferences.get(primary_intent, 0.5)
        
        # Score based on distance from target
        energy_distance = abs(energy - target_energy)
        energy_score = 1.0 - (energy_distance / 1.0)  # Normalize to 0-1
        
        return max(0.0, energy_score)
    
    def _score_danceability(self, features: Dict, intent: Dict) -> float:
        """Score danceability based on context."""
        danceability = features.get('danceability', 0.5)
        
        # Higher danceability is generally positive
        # But adjust based on intent
        primary_intent = intent.get('primary_intent', 'discovery')
        
        if primary_intent in ['workout', 'energy', 'party']:
            # High danceability preferred
            return danceability
        elif primary_intent in ['concentration', 'study', 'relaxation']:
            # Moderate danceability preferred
            return 1.0 - abs(danceability - 0.4) / 0.6
        else:
            # Balanced approach
            return danceability * 0.8 + 0.2
    
    def _score_valence(self, features: Dict, intent: Dict) -> float:
        """Score valence (musical positivity) based on intent."""
        valence = features.get('valence', 0.5)
        
        primary_intent = intent.get('primary_intent', 'discovery')
        
        # Valence preferences by intent
        if primary_intent in ['energy', 'workout', 'party']:
            # Higher valence preferred for energetic activities
            return valence
        elif primary_intent in ['relaxation', 'study']:
            # Moderate valence preferred
            return 1.0 - abs(valence - 0.4) / 0.6
        else:
            # Avoid extremes, prefer moderate positivity
            return 1.0 - abs(valence - 0.6) / 0.6
    
    def _score_acousticness(self, features: Dict, intent: Dict) -> float:
        """Score acousticness based on context preferences."""
        acousticness = features.get('acousticness', 0.5)
        
        primary_intent = intent.get('primary_intent', 'discovery')
        
        if primary_intent in ['relaxation', 'study', 'concentration']:
            # Higher acousticness preferred for calm activities
            return acousticness
        elif primary_intent in ['workout', 'energy', 'party']:
            # Lower acousticness (more electronic) preferred
            return 1.0 - acousticness
        else:
            # Balanced preference
            return 1.0 - abs(acousticness - 0.5) / 0.5
    
    def _score_instrumentalness(self, features: Dict, intent: Dict) -> float:
        """Score instrumentalness based on context."""
        instrumentalness = features.get('instrumentalness', 0.5)
        
        primary_intent = intent.get('primary_intent', 'discovery')
        
        if primary_intent in ['concentration', 'study']:
            # Higher instrumentalness preferred for focus
            return instrumentalness
        else:
            # Generally prefer some vocals
            return 1.0 - (instrumentalness * 0.6)
    
    def _score_liveness(self, features: Dict) -> float:
        """Score liveness - generally prefer studio recordings."""
        liveness = features.get('liveness', 0.1)
        
        # Prefer studio recordings (lower liveness)
        # But don't penalize too heavily
        return 1.0 - (liveness * 0.5)
    
    def _score_speechiness(self, features: Dict, intent: Dict) -> float:
        """Score speechiness based on context."""
        speechiness = features.get('speechiness', 0.1)
        
        primary_intent = intent.get('primary_intent', 'discovery')
        
        if primary_intent in ['concentration', 'study']:
            # Lower speechiness preferred for focus
            return 1.0 - speechiness
        else:
            # Moderate speechiness is fine
            return 1.0 - (speechiness * 0.3)
    
    def _score_tempo(self, features: Dict, intent: Dict) -> float:
        """Score tempo appropriateness."""
        tempo = features.get('tempo', 120)
        
        primary_intent = intent.get('primary_intent', 'discovery')
        
        # Tempo preferences by intent
        tempo_preferences = {
            'concentration': (80, 110),   # Slower tempo for focus
            'relaxation': (60, 100),      # Very slow tempo
            'energy': (120, 180),         # Fast tempo for energy
            'workout': (130, 170),        # High tempo for workouts
            'study': (80, 120),           # Moderate tempo for studying
            'discovery': (90, 140)        # Wide range for discovery
        }
        
        min_tempo, max_tempo = tempo_preferences.get(primary_intent, (90, 140))
        
        if min_tempo <= tempo <= max_tempo:
            return 1.0
        elif tempo < min_tempo:
            # Too slow
            distance = min_tempo - tempo
            return max(0.0, 1.0 - (distance / 50))
        else:
            # Too fast
            distance = tempo - max_tempo
            return max(0.0, 1.0 - (distance / 50))


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
        track_data: Dict, 
        intent_analysis: Dict
    ) -> float:
        """
        Calculate popularity-adjusted quality score.
        
        Args:
            track_data: Track metadata including play counts
            intent_analysis: User intent including exploration preferences
            
        Returns:
            Popularity balance score (0.0 - 1.0)
        """
        try:
            playcount = int(track_data.get('playcount') or 0)
            listeners = int(track_data.get('listeners') or 0)
            
            # Calculate base popularity score (log scale)
            popularity_score = self._calculate_base_popularity(playcount, listeners)
            
            # Get user's exploration preferences
            exploration_openness = intent_analysis.get('exploration_openness', 0.5)
            
            # Adjust score based on exploration preferences
            adjusted_score = self._adjust_for_exploration_preference(
                popularity_score, 
                exploration_openness
            )
            
            self.logger.debug(
                "Popularity score calculated",
                playcount=playcount,
                listeners=listeners,
                base_popularity=popularity_score,
                exploration_openness=exploration_openness,
                final_score=adjusted_score
            )
            
            return adjusted_score
            
        except Exception as e:
            self.logger.warning("Popularity scoring failed", error=str(e))
            return 0.5  # Default neutral score
    
    def _calculate_base_popularity(self, playcount: int, listeners: int) -> float:
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
    
    def _adjust_for_exploration_preference(
        self, 
        popularity_score: float, 
        exploration_openness: float
    ) -> float:
        """Adjust popularity score based on user's exploration preferences."""
        
        if exploration_openness > 0.7:
            # User likes underground music - reward lower popularity
            return 1.0 - (popularity_score * 0.6)
        elif exploration_openness < 0.3:
            # User likes mainstream music - reward higher popularity
            return popularity_score
        else:
            # Balanced preference - reward moderate popularity
            return 1.0 - abs(popularity_score - 0.5) / 0.5


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
        source = track_data.get('source', '')
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
        
        # Component weights for final score
        self.component_weights = {
            'audio_quality': 0.40,      # Audio features quality
            'popularity_balance': 0.25, # Popularity balance
            'engagement': 0.20,         # User engagement signals
            'genre_mood_fit': 0.15      # Genre/mood fit
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
                track_data, intent_analysis
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