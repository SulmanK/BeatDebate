"""
Quality Scoring System for BeatDebate

Implements multi-dimensional quality scoring for track candidates:
1. Audio Features Quality (40%)
2. Popularity Balance (25%) 
3. User Engagement Signals (20%)
4. Genre/Mood Fit (15%)

🔧 NEW: Intent-aware scoring components for dynamic recommendation system:
- similarity: Acoustic/stylistic similarity to target artists
- novelty: Discovery/underground scoring (fixed calculation)
- underground: Popularity penalty for discovery queries
- genre_mood_match: Style/vibe matching accuracy
- context_fit: Functional fit for activities
- target_artist_boost: Boost target artist tracks
- familiarity: Known track preference for contextual queries
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
        is_hybrid_intent = intent_type == 'hybrid' or 'hybrid' in str(intent_type).lower()
        
        # All conditions must be true for genre-hybrid query
        return has_genres and has_artist_similarity and is_hybrid_intent


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


class IntentAwareScorer:
    """
    🔧 NEW: Intent-aware scoring for different recommendation approaches.
    
    Provides specialized scoring methods for all intent types:
    - Artist Similarity
    - Discovery/Exploration  
    - Genre/Mood
    - Contextual
    - Hybrid (with sub-types)
    """
    
    def __init__(self):
        """Initialize intent-aware scorer."""
        self.logger = logger.bind(component="IntentAwareScorer")
        self.logger.info("Intent-Aware Scorer initialized")
    
    def calculate_novelty_score(
        self, 
        track_data: Dict, 
        intent: str,
        entities: Dict[str, Any] = None
    ) -> float:
        """
        🔧 FIXED: Calculate proper novelty score based on popularity data.
        
        Args:
            track_data: Track metadata including play counts
            intent: Intent type (discovery, hybrid_discovery_primary, etc.)
            entities: Musical entities for context
            
        Returns:
            Novelty score (0.0 - 1.0) where 1.0 = truly underground
        """
        try:
            playcount = int(track_data.get('playcount') or 0)
            listeners = int(track_data.get('listeners') or 0)
            
            # Calculate popularity-based novelty (INVERSE of popularity)
            novelty_score = self._calculate_underground_novelty(playcount, listeners, intent)
            
            # Apply intent-specific thresholds and bonuses
            if 'discovery' in intent or intent == 'discovery':
                # Discovery intents: Strict novelty requirements
                novelty_score = self._apply_discovery_novelty_boost(
                    novelty_score, track_data, entities
                )
            elif 'similarity' in intent or intent == 'artist_similarity':
                # Similarity intents: Relaxed novelty, focus on similarity
                novelty_score = max(0.3, novelty_score)  # Minimum threshold
            
            self.logger.debug(
                "Novelty score calculated",
                playcount=playcount,
                listeners=listeners,
                intent=intent,
                novelty_score=novelty_score
            )
            
            return min(1.0, max(0.0, novelty_score))
            
        except Exception as e:
            self.logger.warning("Novelty scoring failed", error=str(e))
            return 0.5
    
    def _calculate_underground_novelty(self, playcount: int, listeners: int, intent: str) -> float:
        """
        Calculate novelty score based on popularity metrics.
        
        🔧 FIXED: Higher popularity = LOWER novelty score
        """
        # Handle zero values
        if playcount == 0 and listeners == 0:
            return 1.0  # Completely unknown = maximum novelty
        
        # Intent-specific popularity thresholds
        if 'discovery' in intent or intent == 'discovery':
            # Discovery: Strict but realistic thresholds for truly underground focus
            max_listeners = 500000   # 500k listeners = underground limit (was 50k - too strict!)
            max_playcount = 5000000  # 5M plays = underground limit (was 500k - too strict!)
        elif 'similarity' in intent or intent == 'artist_similarity':
            # Similarity: Relaxed thresholds (allow moderate popularity)
            max_listeners = 1000000   # 1M listeners = acceptable
            max_playcount = 10000000  # 10M plays = acceptable
        else:
            # Default: Moderate thresholds
            max_listeners = 200000    # 200k listeners
            max_playcount = 2000000   # 2M plays
        
        # Calculate novelty based on inverse popularity
        if listeners > 0:
            listener_novelty = 1.0 - min(1.0, listeners / max_listeners)
        else:
            listener_novelty = 1.0
        
        if playcount > 0:
            playcount_novelty = 1.0 - min(1.0, playcount / max_playcount)
        else:
            playcount_novelty = 1.0
        
        # Combine listener and playcount novelty
        combined_novelty = (listener_novelty + playcount_novelty) / 2
        
        return combined_novelty
    
    def _apply_discovery_novelty_boost(
        self, 
        base_novelty: float, 
        track_data: Dict, 
        entities: Dict[str, Any] = None
    ) -> float:
        """Apply discovery-specific novelty bonuses."""
        boosted_novelty = base_novelty
        
        # Bonus for underground indicators in tags/genres
        underground_tags = ['underground', 'experimental', 'obscure', 'hidden', 'indie']
        track_tags = track_data.get('tags', [])
        if isinstance(track_tags, list):
            for tag in track_tags:
                if any(indicator in tag.lower() for indicator in underground_tags):
                    boosted_novelty += 0.1
                    break
        
        # Bonus for very low listener counts
        listeners = int(track_data.get('listeners') or 0)
        if listeners < 1000:
            boosted_novelty += 0.2  # Very underground bonus
        elif listeners < 10000:
            boosted_novelty += 0.1  # Underground bonus
        
        return min(1.0, boosted_novelty)
    
    def calculate_similarity_score(
        self, 
        candidate_data: Dict, 
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity score for artist similarity intents.
        
        Args:
            candidate_data: Candidate track data
            entities: Musical entities including target artists
            intent_analysis: Intent analysis results
            
        Returns:
            Similarity score (0.0 - 1.0)
        """
        try:
            similarity_score = 0.0
            
            # Extract target artists from entities
            target_artists = self._extract_target_artists(entities)
            if not target_artists:
                return 0.0
            
            candidate_artist = candidate_data.get('artist', '').lower()
            
            # 🔧 NEW: Artist alias mapping for better artist recognition
            artist_aliases = {
                'bon iver': ['justin vernon', 'bonnie bear'],
                'justin vernon': ['bon iver', 'bonnie bear'],
                'radiohead': ['thom yorke', 'jonny greenwood', 'ed o\'brien', 'colin greenwood', 'phil selway'],
                'thom yorke': ['radiohead', 'atoms for peace'],
                'taylor swift': ['taylor swift feat.', 'taylor swift featuring'],
                'kanye west': ['ye', 'kanye', 'yeezy'],
                'the weeknd': ['weeknd'],
                'frank ocean': ['christopher francis ocean'],
                'tyler the creator': ['tyler okonma', 'ace creator'],
                'childish gambino': ['donald glover'],
                'daniel caesar': ['daniel caesar feat.'],
                'sza': ['solána imani rowe'],
                'daft punk': ['thomas bangalter', 'guy-manuel de homem-christo'],
                'lcd soundsystem': ['james murphy'],
                'tame impala': ['kevin parker'],
                'vampire weekend': ['ezra koenig'],
                'arcade fire': ['win butler', 'régine chassagne'],
                'fleet foxes': ['robin pecknold'],
                'sufjan stevens': ['sufjan stevens feat.']
            }
            
            # Check if candidate is from target artist (highest similarity)
            target_artists_lower = [artist.lower() for artist in target_artists]
            
            # Direct match
            if any(candidate_artist == artist.lower() for artist in target_artists):
                similarity_score = 1.0
            else:
                # 🔧 NEW: Check for artist aliases
                alias_match = False
                for target_artist in target_artists_lower:
                    # Check if target has aliases and candidate matches one
                    if target_artist in artist_aliases:
                        aliases = artist_aliases[target_artist]
                        if candidate_artist in [alias.lower() for alias in aliases]:
                            similarity_score = 1.0
                            alias_match = True
                            self.logger.info(f"🎯 SIMILARITY ALIAS MATCH: '{candidate_artist}' matches target '{target_artist}' via alias")
                            break
                    
                    # Check reverse mapping
                    if candidate_artist in artist_aliases:
                        aliases = artist_aliases[candidate_artist]
                        if target_artist in [alias.lower() for alias in aliases]:
                            similarity_score = 1.0
                            alias_match = True
                            self.logger.info(f"🎯 REVERSE SIMILARITY ALIAS: '{candidate_artist}' is primary for target '{target_artist}'")
                            break
                
                # If no alias match, calculate style similarity
                if not alias_match:
                    similarity_score = self._calculate_style_similarity(
                        candidate_data, target_artists, entities
                    )
            
            self.logger.debug(
                "Similarity score calculated",
                candidate_artist=candidate_artist,
                target_artists=target_artists,
                similarity_score=similarity_score
            )
            
            return similarity_score
            
        except Exception as e:
            self.logger.warning("Similarity scoring failed", error=str(e))
            return 0.0
    
    def _extract_target_artists(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target artists from entities."""
        target_artists = []
        
        musical_entities = entities.get('musical_entities', {})
        artists = musical_entities.get('artists', {})
        
        # Get primary and similar_to artists
        target_artists.extend(artists.get('primary', []))
        target_artists.extend(artists.get('similar_to', []))
        
        return list(set(target_artists))  # Remove duplicates
    
    def _calculate_style_similarity(
        self, 
        candidate_data: Dict, 
        target_artists: List[str], 
        entities: Dict[str, Any]
    ) -> float:
        """Calculate style-based similarity score."""
        # This is a simplified implementation
        # In a real system, this would use audio features, genre matching, etc.
        
        similarity_score = 0.0
        
        # Genre matching
        candidate_genres = set(candidate_data.get('genres', []))
        target_genres = set()
        
        musical_entities = entities.get('musical_entities', {})
        genres = musical_entities.get('genres', {})
        target_genres.update(genres.get('primary', []))
        target_genres.update(genres.get('secondary', []))
        
        if candidate_genres and target_genres:
            genre_overlap = len(candidate_genres.intersection(target_genres))
            total_genres = len(candidate_genres.union(target_genres))
            similarity_score += (genre_overlap / total_genres) * 0.6
        
        # Tag matching (simplified)
        similarity_score += 0.4  # Base similarity for being in similar artist results
        
        return min(1.0, similarity_score)
    
    def calculate_target_artist_boost(
        self, 
        candidate_data: Dict, 
        entities: Dict[str, Any]
    ) -> float:
        """
        Calculate boost for target artist's own tracks.
        
        Args:
            candidate_data: Candidate track data
            entities: Musical entities including target artists
            
        Returns:
            Target artist boost score (0.0 - 1.0)
        """
        try:
            target_artists = self._extract_target_artists(entities)
            if not target_artists:
                return 0.0
            
            candidate_artist = candidate_data.get('artist', '').lower()
            
            # Check if candidate is from target artist
            if any(candidate_artist == artist.lower() for artist in target_artists):
                return 0.8  # High boost for target artist tracks
            
            return 0.0
            
        except Exception as e:
            self.logger.warning("Target artist boost calculation failed", error=str(e))
            return 0.0
    
    def calculate_underground_score(
        self, 
        track_data: Dict, 
        intent: str
    ) -> float:
        """
        Calculate underground score for discovery intents.
        
        Args:
            track_data: Track metadata
            intent: Intent type
            
        Returns:
            Underground score (0.0 - 1.0)
        """
        # Underground score is similar to novelty but with stricter thresholds
        return self.calculate_novelty_score(track_data, intent)
    
    def calculate_context_fit_score(
        self, 
        track_data: Dict, 
        intent_analysis: Dict[str, Any],
        entities: Dict[str, Any]
    ) -> float:
        """
        Calculate context fit for contextual intents.
        
        Args:
            track_data: Track metadata
            intent_analysis: Intent analysis results
            entities: Musical entities
            
        Returns:
            Context fit score (0.0 - 1.0)
        """
        try:
            # Extract context from intent analysis
            contexts = intent_analysis.get('context_factors', [])
            if not contexts:
                return 0.5  # Neutral if no context specified
            
            context_score = 0.0
            
            # Context-specific scoring logic
            for context in contexts:
                if context in ['study', 'concentration', 'focus']:
                    # Prefer instrumental, moderate energy
                    context_score += self._score_study_context(track_data)
                elif context in ['workout', 'exercise', 'gym']:
                    # Prefer high energy, upbeat
                    context_score += self._score_workout_context(track_data)
                elif context in ['relaxation', 'chill', 'calm']:
                    # Prefer low energy, calming
                    context_score += self._score_relaxation_context(track_data)
                else:
                    context_score += 0.5  # Neutral for unknown contexts
            
            return min(1.0, context_score / len(contexts))
            
        except Exception as e:
            self.logger.warning("Context fit calculation failed", error=str(e))
            return 0.5
    
    def _score_study_context(self, track_data: Dict) -> float:
        """Score track for studying context."""
        score = 0.5  # Base score
        
        # Prefer instrumental tracks
        genre = track_data.get('genre', '').lower()
        tags = [tag.lower() for tag in (track_data.get('tags') or [])]
        
        # Positive indicators for studying
        study_positive = [
            'instrumental', 'ambient', 'classical', 'piano', 'acoustic',
            'minimal', 'meditation', 'focus', 'concentration', 'calm',
            'peaceful', 'soft', 'gentle', 'atmospheric', 'soundtrack'
        ]
        
        # Negative indicators for studying  
        study_negative = [
            'vocal', 'lyrics', 'aggressive', 'loud', 'energetic', 'party',
            'heavy', 'metal', 'punk', 'rap', 'hip hop', 'dance', 'club'
        ]
        
        # Score based on positive indicators
        positive_matches = sum(1 for indicator in study_positive 
                              if indicator in genre or any(indicator in tag for tag in tags))
        score += positive_matches * 0.1
        
        # Penalize negative indicators
        negative_matches = sum(1 for indicator in study_negative 
                              if indicator in genre or any(indicator in tag for tag in tags))
        score -= negative_matches * 0.15
        
        # Tempo considerations (if available)
        try:
            tempo = float(track_data.get('tempo') or 0)
            if 60 <= tempo <= 120:  # Ideal study tempo range
                score += 0.2
            elif tempo > 140:  # Too fast for studying
                score -= 0.3
        except (ValueError, TypeError):
            pass
        
        return min(1.0, max(0.0, score))
    
    def _score_workout_context(self, track_data: Dict) -> float:
        """Score track for workout context."""
        score = 0.5  # Base score
        
        genre = track_data.get('genre', '').lower()
        tags = [tag.lower() for tag in (track_data.get('tags') or [])]
        
        # Positive indicators for workout
        workout_positive = [
            'energetic', 'upbeat', 'pump', 'motivation', 'electronic', 'dance',
            'rock', 'metal', 'hip hop', 'pop', 'aggressive', 'powerful',
            'driving', 'intense', 'bass', 'beat', 'rhythm', 'gym'
        ]
        
        # Negative indicators for workout
        workout_negative = [
            'slow', 'ballad', 'sad', 'melancholy', 'ambient', 'meditation',
            'acoustic', 'folk', 'classical', 'piano', 'soft', 'gentle',
            'peaceful', 'calm', 'sleepy', 'relaxing'
        ]
        
        # Score based on positive indicators
        positive_matches = sum(1 for indicator in workout_positive 
                              if indicator in genre or any(indicator in tag for tag in tags))
        score += positive_matches * 0.1
        
        # Penalize negative indicators
        negative_matches = sum(1 for indicator in workout_negative 
                              if indicator in genre or any(indicator in tag for tag in tags))
        score -= negative_matches * 0.1
        
        # Tempo considerations (if available)
        try:
            tempo = float(track_data.get('tempo') or 0)
            if tempo >= 120:  # High energy tempo for workout
                score += 0.3
            elif tempo < 90:  # Too slow for workout
                score -= 0.2
        except (ValueError, TypeError):
            pass
            
        # Energy level (if available)
        try:
            energy = float(track_data.get('energy') or 0.5)
            if energy >= 0.7:  # High energy
                score += 0.2
            elif energy < 0.3:  # Low energy
                score -= 0.2
        except (ValueError, TypeError):
            pass
        
        return min(1.0, max(0.0, score))
    
    def _score_relaxation_context(self, track_data: Dict) -> float:
        """Score track for relaxation context."""
        score = 0.5  # Base score
        
        genre = track_data.get('genre', '').lower()
        tags = [tag.lower() for tag in (track_data.get('tags') or [])]
        
        # Positive indicators for relaxation
        relaxation_positive = [
            'chill', 'calm', 'peaceful', 'relaxing', 'soft', 'gentle',
            'ambient', 'meditation', 'spa', 'nature', 'acoustic',
            'piano', 'instrumental', 'lounge', 'jazz', 'smooth',
            'mellow', 'soothing', 'tranquil', 'serene'
        ]
        
        # Negative indicators for relaxation
        relaxation_negative = [
            'aggressive', 'loud', 'metal', 'punk', 'screaming', 'harsh',
            'distorted', 'chaotic', 'frantic', 'intense', 'heavy',
            'club', 'party', 'rave', 'thrash'
        ]
        
        # Score based on positive indicators
        positive_matches = sum(1 for indicator in relaxation_positive 
                              if indicator in genre or any(indicator in tag for tag in tags))
        score += positive_matches * 0.1
        
        # Penalize negative indicators
        negative_matches = sum(1 for indicator in relaxation_negative 
                              if indicator in genre or any(indicator in tag for tag in tags))
        score -= negative_matches * 0.2
        
        # Tempo considerations (if available)
        try:
            tempo = float(track_data.get('tempo') or 0)
            if 60 <= tempo <= 100:  # Relaxed tempo range
                score += 0.2
            elif tempo > 130:  # Too fast for relaxation
                score -= 0.3
        except (ValueError, TypeError):
            pass
            
        # Valence considerations (if available) - prefer positive but not too excited
        try:
            valence = float(track_data.get('valence') or 0.5)
            if 0.4 <= valence <= 0.7:  # Moderately positive
                score += 0.1
            elif valence < 0.2:  # Too sad for relaxation
                score -= 0.1
        except (ValueError, TypeError):
            pass
        
        return min(1.0, max(0.0, score))
    
    def calculate_familiarity_score(
        self, 
        track_data: Dict, 
        intent_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate familiarity score for contextual intents.
        
        Args:
            track_data: Track metadata
            intent_analysis: Intent analysis results
            
        Returns:
            Familiarity score (0.0 - 1.0)
        """
        try:
            # For contextual queries, sometimes familiar tracks are preferred
            playcount = int(track_data.get('playcount') or 0)
            listeners = int(track_data.get('listeners') or 0)
            
            # Moderate popularity = more familiar
            if listeners > 10000 and listeners < 1000000:
                return 0.8  # Well-known but not mainstream
            elif listeners > 1000000:
                return 0.6  # Very mainstream
            else:
                return 0.3  # Too underground for familiarity
                
        except Exception as e:
            self.logger.warning("Familiarity calculation failed", error=str(e))
            return 0.5


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