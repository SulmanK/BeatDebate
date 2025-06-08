"""
Intent-Aware Scorer for BeatDebate

Provides specialized scoring methods for different recommendation approaches:
- Artist Similarity
- Discovery/Exploration  
- Genre/Mood
- Contextual
- Hybrid (with sub-types)
"""

from typing import Dict, Any, List
import structlog

logger = structlog.get_logger(__name__)


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
        entities: Dict[str, Any] = None,
        all_candidates: List[Dict[str, Any]] = None
    ) -> float:
        """
        🔧 FIXED: Calculate proper novelty score based on popularity data.
        🎯 NEW: Relative scoring for by_artist_underground intent
        
        Args:
            track_data: Track metadata including play counts
            intent: Intent type (discovery, hybrid_discovery_primary, etc.)
            entities: Musical entities for context
            all_candidates: All candidates for relative scoring (needed for by_artist_underground)
            
        Returns:
            Novelty score (0.0 - 1.0) where 1.0 = truly underground
        """
        try:
            playcount = int(track_data.get('playcount') or 0)
            listeners = int(track_data.get('listeners') or 0)
            
            # 🎯 NEW: For by_artist_underground, calculate relative stats
            artist_stats = None
            if 'by_artist_underground' in intent and all_candidates:
                artist_stats = self._calculate_artist_stats(track_data.get('artist', ''), all_candidates)
            
            # Calculate popularity-based novelty (INVERSE of popularity)
            novelty_score = self._calculate_underground_novelty(playcount, listeners, intent, artist_stats)
            
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
                novelty_score=novelty_score,
                used_relative_scoring=artist_stats is not None
            )
            
            return min(1.0, max(0.0, novelty_score))
            
        except Exception as e:
            self.logger.warning("Novelty scoring failed", error=str(e))
            return 0.5
    
    def _calculate_underground_novelty(self, playcount: int, listeners: int, intent: str, artist_stats: Dict[str, Any] = None) -> float:
        """
        Calculate novelty score based on popularity metrics.
        
        🔧 FIXED: Higher popularity = LOWER novelty score
        🎯 NEW: Relative scoring for by_artist_underground intent
        """
        # Handle zero values
        if playcount == 0 and listeners == 0:
            return 1.0  # Completely unknown = maximum novelty
        
        # 🎯 NEW: For by_artist_underground, use RELATIVE scoring within artist's discography
        if 'by_artist_underground' in intent and artist_stats:
            return self._calculate_relative_underground_novelty(playcount, listeners, artist_stats)
        
        # Intent-specific popularity thresholds (for other intents)
        if 'discovery' in intent or intent == 'discovery':
            # Discovery: Strict but realistic thresholds for truly underground focus
            max_listeners = 500000   # 500k listeners = underground limit (was 50k - too strict!)
            max_playcount = 5000000  # 5M plays = underground limit (was 500k - too strict!)
        elif 'by_artist_underground' in intent:
            # Underground by artist: Very strict thresholds for truly underground tracks (fallback if no artist_stats)
            max_listeners = 50000    # 50k listeners = underground limit for artist tracks
            max_playcount = 500000   # 500k plays = underground limit for artist tracks  
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
            # For by_artist_underground, missing playcount should use listener data
            if 'by_artist_underground' in intent and listeners > 0:
                playcount_novelty = listener_novelty  # Use listener novelty if playcount missing
            else:
                playcount_novelty = 1.0
        
        # Combine listener and playcount novelty
        combined_novelty = (listener_novelty + playcount_novelty) / 2
        
        return combined_novelty
    
    def _calculate_relative_underground_novelty(self, playcount: int, listeners: int, artist_stats: Dict[str, Any]) -> float:
        """
        Calculate relative underground novelty within an artist's discography.
        
        Args:
            playcount: Track's playcount
            listeners: Track's listener count  
            artist_stats: Dict with 'max_listeners', 'min_listeners', 'max_playcount', 'min_playcount'
        
        Returns:
            Relative novelty score (0.0 = most popular track, 1.0 = least popular track)
        """
        try:
            # Extract artist discography stats
            max_listeners = artist_stats.get('max_listeners', listeners)
            min_listeners = artist_stats.get('min_listeners', listeners)
            max_playcount = artist_stats.get('max_playcount', playcount)
            min_playcount = artist_stats.get('min_playcount', playcount)
            
            # Calculate relative position within artist's range
            listener_novelty = 1.0
            if max_listeners > min_listeners and listeners > 0:
                # Position within artist's listener range (inverted - lower listeners = higher novelty)
                listener_position = (listeners - min_listeners) / (max_listeners - min_listeners)
                listener_novelty = 1.0 - listener_position
            
            playcount_novelty = 1.0
            if max_playcount > min_playcount and playcount > 0:
                # Position within artist's playcount range (inverted - lower playcount = higher novelty)
                playcount_position = (playcount - min_playcount) / (max_playcount - min_playcount)
                playcount_novelty = 1.0 - playcount_position
            elif playcount == 0 and listeners > 0:
                # Use listener novelty if playcount is missing
                playcount_novelty = listener_novelty
            
            # Combine relative scores
            relative_novelty = (listener_novelty + playcount_novelty) / 2
            
            self.logger.debug(
                f"🎯 RELATIVE UNDERGROUND SCORING: listeners={listeners} ({min_listeners}-{max_listeners}), "
                f"playcount={playcount} ({min_playcount}-{max_playcount}), relative_novelty={relative_novelty:.3f}"
            )
            
            return relative_novelty
            
        except Exception as e:
            self.logger.warning(f"Relative underground calculation failed: {e}")
            # Fallback to absolute scoring
            return self._calculate_underground_novelty(playcount, listeners, 'by_artist_underground', None)
    
    def _calculate_artist_stats(self, target_artist: str, all_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate artist discography statistics for relative underground scoring.
        
        Args:
            target_artist: Artist name to calculate stats for
            all_candidates: All candidates in the current batch
            
        Returns:
            Dict with min/max listener and playcount stats for the artist
        """
        try:
            # Filter candidates for target artist (normalize names for comparison)
            artist_tracks = []
            target_artist_normalized = target_artist.lower().strip()
            
            for candidate in all_candidates:
                candidate_artist = candidate.get('artist', '').lower().strip()
                if candidate_artist == target_artist_normalized:
                    listeners = int(candidate.get('listeners') or 0)
                    playcount = int(candidate.get('playcount') or 0)
                    if listeners > 0 or playcount > 0:  # Only include tracks with data
                        artist_tracks.append({
                            'listeners': listeners,
                            'playcount': playcount
                        })
            
            if not artist_tracks:
                self.logger.warning(f"No tracks found for artist stats: {target_artist}")
                return {}
            
            # Calculate min/max stats
            all_listeners = [track['listeners'] for track in artist_tracks if track['listeners'] > 0]
            all_playcounts = [track['playcount'] for track in artist_tracks if track['playcount'] > 0]
            
            stats = {}
            if all_listeners:
                stats['max_listeners'] = max(all_listeners)
                stats['min_listeners'] = min(all_listeners)
            if all_playcounts:
                stats['max_playcount'] = max(all_playcounts)
                stats['min_playcount'] = min(all_playcounts)
            
            self.logger.debug(
                f"🎯 ARTIST STATS for {target_artist}: {len(artist_tracks)} tracks, "
                f"listeners: {stats.get('min_listeners', 0)}-{stats.get('max_listeners', 0)}, "
                f"playcount: {stats.get('min_playcount', 0)}-{stats.get('max_playcount', 0)}"
            )
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Artist stats calculation failed for {target_artist}: {e}")
            return {}
    
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
        
        # DEBUG: Log the entities structure
        self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Full entities structure: {entities}")
        
        # FIXED: Handle BOTH entity formats (flat and nested)
        
        # 1. Try flat format first (from context handlers): {'artists': ['Mk.gee'], ...}
        if 'artists' in entities and isinstance(entities['artists'], list):
            self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Found flat format artists: {entities['artists']}")
            for artist_obj in entities['artists']:
                if isinstance(artist_obj, dict) and 'name' in artist_obj:
                    target_artists.append(artist_obj['name'])
                    self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Extracted artist from dict: {artist_obj['name']}")
                elif isinstance(artist_obj, str):
                    target_artists.append(artist_obj)
                    self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Extracted artist from string: {artist_obj}")
        
        # 2. Try nested format (from planners): {'musical_entities': {'artists': {'primary': [...]}}}
        elif 'musical_entities' in entities and 'artists' in entities['musical_entities']:
            musical_artists = entities['musical_entities']['artists']
            
            # Extract from musical_entities.artists.primary
            if 'primary' in musical_artists:
                self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Found musical_entities.artists.primary: {musical_artists['primary']}")
                for artist_obj in musical_artists['primary']:
                    if isinstance(artist_obj, dict) and 'name' in artist_obj:
                        target_artists.append(artist_obj['name'])
                        self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Extracted artist from dict: {artist_obj['name']}")
                    elif isinstance(artist_obj, str):
                        target_artists.append(artist_obj)
                        self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Extracted artist from string: {artist_obj}")
            
            # Extract from musical_entities.artists.similar_to
            if 'similar_to' in musical_artists:
                for artist_obj in musical_artists['similar_to']:
                    if isinstance(artist_obj, dict) and 'name' in artist_obj:
                        target_artists.append(artist_obj['name'])
                    elif isinstance(artist_obj, str):
                        target_artists.append(artist_obj)
        
        # 3. Fallback: Try legacy format (artists.primary) for backward compatibility
        elif 'artists' in entities and isinstance(entities['artists'], dict) and 'primary' in entities['artists']:
            self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Found legacy artists.primary: {entities['artists']['primary']}")
            for artist_obj in entities['artists']['primary']:
                if isinstance(artist_obj, dict) and 'name' in artist_obj:
                    target_artists.append(artist_obj['name'])
                    self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Extracted artist from dict: {artist_obj['name']}")
                elif isinstance(artist_obj, str):
                    target_artists.append(artist_obj)
                    self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Extracted artist from string: {artist_obj}")
        
        else:
            self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: No artists found - keys: {entities.keys()}")
            if 'musical_entities' in entities:
                self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Musical entities keys: {entities['musical_entities'].keys()}")
                if 'artists' in entities['musical_entities']:
                    self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Musical artists keys: {entities['musical_entities']['artists'].keys()}")
        
        self.logger.debug(f"🔍 SIMILARITY SCORER DEBUG: Final extracted artists: {target_artists}")
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