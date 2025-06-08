"""
Ranking Engine for Judge Agent

Core ranking algorithms, scoring components, and intent-aware ranking
for the judge agent's final recommendation evaluation.

Extracted from judge/agent.py and judge/ranking_logic.py for better modularity.
"""

from typing import Dict, List, Any, Tuple, Optional
import structlog

from src.models.recommendation_models import TrackRecommendation

logger = structlog.get_logger(__name__)


class RankingEngine:
    """
    Core ranking engine for judge agent recommendations.
    
    Responsibilities:
    - Multi-criteria ranking algorithms
    - Intent-aware scoring
    - Weighted score calculation
    - Novelty threshold filtering
    """
    
    def __init__(self):
        self.ranking_weights = self._initialize_ranking_weights()
        self.context_modifiers = self._initialize_context_modifiers()
        
        self.logger = structlog.get_logger(__name__)
    
    async def rank_candidates(
        self,
        candidates: List[Tuple[TrackRecommendation, Dict[str, float]]],
        state,
        intent: str = None,
        entities: Dict[str, Any] = None,
        intent_analysis: Dict[str, Any] = None
    ) -> List[Tuple[TrackRecommendation, Dict[str, float]]]:
        """
        Rank track recommendations using intent-aware scoring.
        
        Args:
            candidates: List of (track, scores) tuples
            state: Current workflow state
            intent: Intent type (including hybrid sub-types)
            entities: Musical entities from query
            intent_analysis: Intent analysis results
            
        Returns:
            Ranked list of (track, scores) tuples
        """
        try:
            # Extract parameters from state if not provided
            if intent is None:
                # FIXED: Use proper intent extraction like DiversityOptimizer
                intent = 'discovery'  # Default fallback
                if state:
                    # Try multiple sources for intent in order of preference
                    if hasattr(state, 'intent_analysis') and isinstance(state.intent_analysis, dict):
                        # 🔧 FIX: Try both 'intent' and 'primary_intent' keys
                        intent = state.intent_analysis.get('intent') or state.intent_analysis.get('primary_intent', intent)
                    elif hasattr(state, 'query_understanding') and hasattr(state.query_understanding, 'intent'):
                        intent = state.query_understanding.intent.value if hasattr(state.query_understanding.intent, 'value') else str(state.query_understanding.intent)
                    elif hasattr(state, 'intent'):
                        intent = state.intent
                    
                    # 🔧 DEBUG: Log what we extracted
                    self.logger.info(f"🔧 RANKING ENGINE: Extracted intent='{intent}' from state")
                    if hasattr(state, 'intent_analysis'):
                        self.logger.debug(f"🔧 DEBUG: state.intent_analysis keys: {list(state.intent_analysis.keys()) if isinstance(state.intent_analysis, dict) else 'not dict'}")
            if entities is None:
                entities = getattr(state, 'entities', {})
            if intent_analysis is None:
                intent_analysis = getattr(state, 'intent_analysis', {})
            
            from src.agents.components import QualityScorer
            
            # Initialize quality scorer for intent-aware scoring
            quality_scorer = QualityScorer()
            
            # Get intent-specific parameters (base threshold, will be dynamically adjusted per track)
            scoring_weights = self.get_intent_weights(intent, entities, intent_analysis)
            
            # Check for genre-hybrid queries
            is_genre_hybrid = self._is_genre_hybrid_query(entities, intent_analysis)
            if is_genre_hybrid:
                self.logger.info(f"🎯 Detected genre-hybrid query - DISABLED novelty filtering")
            
            self.logger.info(f"🔧 Ranking with intent-aware scoring", 
                            candidate_count=len(candidates),
                            intent=intent,
                            scoring_weights=scoring_weights)
            
            # Extract all candidate track data for relative scoring (needed for by_artist_underground)
            all_candidates_data = []
            for track_rec, _ in candidates:
                track_data = self._extract_track_data(track_rec)
                all_candidates_data.append(track_data)
            
            ranked_candidates = []
            
            for track_rec, agent_scores in candidates:
                try:
                    # Extract track data for scoring
                    track_data = self._extract_track_data(track_rec)
                    
                    # Reject fake/fallback tracks with no popularity data
                    if not self._is_valid_track(track_data):
                        continue
                    
                    # Calculate intent-aware scoring components
                    intent_scores = quality_scorer.calculate_intent_aware_scores(
                        track_data, entities, intent_analysis, intent, all_candidates_data)
                    
                    # Merge agent scores with intent-aware scores
                    combined_scores = {**agent_scores, **intent_scores}
                    
                    # Calculate final score using intent-specific weights
                    final_score = self._calculate_weighted_score(
                        combined_scores, scoring_weights
                    )
                    
                    # Apply novelty threshold filtering (with dynamic adjustment)
                    novelty_score = combined_scores.get('novelty', 0.5)
                    
                    # Get dynamic novelty threshold based on track data
                    if is_genre_hybrid:
                        novelty_threshold = 0.0  # Disable novelty filtering for genre fusion
                    else:
                        novelty_threshold = self.get_novelty_threshold(intent, track_data)
                    
                    if novelty_score >= novelty_threshold:
                        combined_scores['final_score'] = final_score
                        ranked_candidates.append((track_rec, combined_scores))
                        
                        self.logger.debug(
                            "Track passed novelty threshold",
                            track=f"{track_rec.artist} - {getattr(track_rec, 'name', 'Unknown')}",
                            novelty=novelty_score,
                            threshold=novelty_threshold,
                            final_score=final_score
                        )
                    else:
                        self.logger.debug(
                            "Track filtered by novelty threshold",
                            track=f"{track_rec.artist} - {getattr(track_rec, 'name', 'Unknown')}",
                            novelty=novelty_score,
                            threshold=novelty_threshold
                        )
                
                except Exception as e:
                    self.logger.warning(
                        "Scoring failed for track",
                        track=f"{track_rec.artist} - {getattr(track_rec, 'name', 'Unknown')}",
                        error=str(e)
                    )
                    # Keep track with original scores as fallback
                    agent_scores['final_score'] = 0.5
                    ranked_candidates.append((track_rec, agent_scores))
            
            # Sort by final score (descending)
            ranked_candidates.sort(key=lambda x: x[1].get('final_score', 0), reverse=True)
            
            self.logger.info(
                f"🔧 Ranking completed: {len(ranked_candidates)}/{len(candidates)} tracks passed filters",
                intent=intent,
                top_score=ranked_candidates[0][1].get('final_score', 0) if ranked_candidates else 0
            )
            
            return ranked_candidates
            
        except Exception as e:
            self.logger.error(f"Ranking failed: {e}")
            # Return candidates with original scores as fallback
            return candidates
    
    def _extract_track_data(self, track_rec: TrackRecommendation) -> Dict[str, Any]:
        """Extract track data safely from various sources."""
        raw_data = getattr(track_rec, 'raw_source_data', None) or {}
        additional_scores = getattr(track_rec, 'additional_scores', None) or {}
        
        # Try to get popularity data from multiple sources
        playcount = (
            raw_data.get('playcount', 0) or 
            additional_scores.get('playcount', 0) or
            getattr(track_rec, 'playcount', 0) or
            0
        )
        listeners = (
            raw_data.get('listeners', 0) or 
            additional_scores.get('listeners', 0) or
            getattr(track_rec, 'listeners', 0) or
            0
        )
        
        # If still no data, try to estimate from popularity
        if playcount == 0 and listeners == 0:
            popularity = additional_scores.get('popularity', 0)
            if popularity > 0:
                listeners = int(popularity * 10000)
                playcount = int(listeners * 5)
        
        return {
            'artist': track_rec.artist,
            'name': getattr(track_rec, 'name', getattr(track_rec, 'title', 'Unknown')),
            'playcount': playcount,
            'listeners': listeners,
            'tags': getattr(track_rec, 'tags', []),
            'genres': getattr(track_rec, 'genres', []),
            'source': getattr(track_rec, 'source', 'unknown')
        }
    
    def _is_valid_track(self, track_data: Dict[str, Any]) -> bool:
        """Check if track has valid data."""
        track_name = track_data.get('name', 'Unknown')
        artist_name = track_data.get('artist', 'Unknown')
        
        # Only reject if track has obviously missing/invalid data
        if track_name == 'Unknown' or artist_name == 'Unknown' or not track_name or not artist_name:
            self.logger.warning(
                "🚨 Rejecting track with missing/unknown data",
                track=f"{artist_name} - {track_name}",
                reason="Missing artist/track name indicates invalid data"
            )
            return False
        
        # Allow all tracks with valid names, regardless of popularity data
        playcount = track_data.get('playcount', 0)
        listeners = track_data.get('listeners', 0)
        if playcount == 0 and listeners == 0:
            self.logger.debug(
                "✅ Allowing track with zero popularity (discovery mode)",
                track=f"{artist_name} - {track_name}"
            )
        
        return True
    
    def _calculate_weighted_score(
        self, 
        scores: Dict[str, float], 
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted final score from component scores."""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            for component, weight in weights.items():
                if component in scores and weight > 0:
                    component_score = scores[component]
                    # Normalize score to 0-1 range if needed
                    if component_score > 1.0:
                        component_score = min(component_score / 100.0, 1.0)
                    
                    weighted_score = component_score * weight
                    total_score += weighted_score
                    total_weight += weight
                    
                    self.logger.debug(f"Score component: {component}={component_score:.3f}, weight={weight:.3f}, contribution={weighted_score:.3f}")
            
            # Normalize by total weight
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 0.5  # Default fallback
            
            return min(max(final_score, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.warning(f"Score calculation failed: {e}")
            return 0.5
    
    def calculate_contextual_relevance(
        self,
        candidate: TrackRecommendation,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> float:
        """Calculate contextual relevance score."""
        try:
            relevance_score = 0.0
            factors = 0
            
            # Artist matching
            target_artists = self._extract_target_artists_from_entities(entities)
            if target_artists and candidate.artist.lower() in [a.lower() for a in target_artists]:
                relevance_score += 0.8
                factors += 1
            
            # Genre matching
            target_genres = self._extract_target_genres(entities)
            candidate_genres = []
            if hasattr(candidate, 'genres') and candidate.genres:
                candidate_genres = [g.lower() for g in candidate.genres]
            elif hasattr(candidate, 'additional_scores') and candidate.additional_scores:
                candidate_genres = [g.lower() for g in candidate.additional_scores.get('genres', [])]
            
            if target_genres:
                genre_match = any(tg.lower() in candidate_genres for tg in target_genres)
                if genre_match:
                    relevance_score += 0.6
                    factors += 1
            
            # Mood matching
            target_moods = self._extract_target_moods(entities, intent_analysis)
            candidate_tags = []
            if hasattr(candidate, 'tags') and candidate.tags:
                candidate_tags = [t.lower() for t in candidate.tags]
            elif hasattr(candidate, 'additional_scores') and candidate.additional_scores:
                candidate_tags = [t.lower() for t in candidate.additional_scores.get('tags', [])]
            
            if target_moods:
                mood_match = any(tm.lower() in candidate_tags for tm in target_moods)
                if mood_match:
                    relevance_score += 0.5
                    factors += 1
            
            # Intent alignment
            intent = intent_analysis.get('primary_intent', 'discovery')
            intent_score = self._calculate_intent_alignment(candidate, intent_analysis, entities, intent)
            relevance_score += intent_score * 0.4
            factors += 1
            
            # Average and normalize
            if factors > 0:
                return min(relevance_score / factors, 1.0)
            else:
                return 0.3  # Baseline relevance
                
        except Exception as e:
            self.logger.warning(f"Contextual relevance calculation failed: {e}")
            return 0.3
    
    def _calculate_intent_alignment(
        self,
        candidate: TrackRecommendation,
        intent_analysis: Dict[str, Any],
        entities: Dict[str, Any],
        intent: str
    ) -> float:
        """Calculate how well candidate aligns with user intent."""
        try:
            alignment_score = 0.5  # Base alignment
            
            if intent == 'discovery':
                # Discovery prefers lower popularity (more underground)
                popularity = getattr(candidate, 'popularity', 50)
                alignment_score += (100 - popularity) / 200  # 0.0 to 0.5 bonus
                
            elif intent in ['similarity', 'artist_similarity']:
                # Similarity prefers genre/tag matching
                target_genres = self._extract_target_genres(entities)
                # FIXED: Safely access genres from multiple possible locations
                candidate_genres = []
                if hasattr(candidate, 'genres') and candidate.genres:
                    candidate_genres = candidate.genres
                elif hasattr(candidate, 'additional_scores') and candidate.additional_scores:
                    candidate_genres = candidate.additional_scores.get('genres', [])
                    
                if target_genres and candidate_genres:
                    match_ratio = len(set(target_genres) & set(candidate_genres)) / len(target_genres)
                    alignment_score += match_ratio * 0.4
            
            elif intent == 'mood':
                # Mood intent prefers tag matching
                mood_descriptors = intent_analysis.get('mood_descriptors', [])
                # FIXED: Safely access tags from multiple possible locations
                candidate_tags = []
                if hasattr(candidate, 'tags') and candidate.tags:
                    candidate_tags = candidate.tags
                elif hasattr(candidate, 'additional_scores') and candidate.additional_scores:
                    candidate_tags = candidate.additional_scores.get('tags', [])
                    
                if mood_descriptors and candidate_tags:
                    tag_matches = sum(1 for mood in mood_descriptors 
                                    if any(mood.lower() in tag.lower() for tag in candidate_tags))
                    alignment_score += (tag_matches / len(mood_descriptors)) * 0.5
                    
            elif intent == 'genre':
                # Genre intent strongly prefers exact genre matches
                target_genres = self._extract_target_genres(entities)
                # FIXED: Safely access genres from multiple possible locations
                candidate_genres = []
                if hasattr(candidate, 'genres') and candidate.genres:
                    candidate_genres = [g.lower() for g in candidate.genres]
                elif hasattr(candidate, 'additional_scores') and candidate.additional_scores:
                    candidate_genres = [g.lower() for g in candidate.additional_scores.get('genres', [])]
                    
                if target_genres:
                    exact_matches = sum(1 for genre in target_genres 
                                      if genre.lower() in candidate_genres)
                    alignment_score += (exact_matches / len(target_genres)) * 0.6
            
            return min(alignment_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Intent alignment calculation failed: {e}")
            return 0.5
    
    def get_novelty_threshold(self, intent: str, candidate_data: Dict[str, Any] = None) -> float:
        """Get novelty threshold based on intent."""
        thresholds = {
            'discovery': 0.3,        # Lower threshold for discovery
            'by_artist': 0.1,        # Very low threshold to allow more tracks by target artist
            'by_artist_underground': 0.6,  # High threshold for underground tracks by artist
            'underground': 0.5,      # Medium threshold for underground
            'similarity': 0.1,       # Very low threshold for similarity
            'artist_similarity': 0.1, # Very low threshold for artist similarity
            'genre': 0.2,            # Low threshold for genre exploration
            'mood': 0.15,            # Low threshold for mood matching
            'artist_deep_dive': 0.0, # No filtering for artist exploration
            'default': 0.2
        }
        
        base_threshold = thresholds.get(intent, thresholds['default'])
        
        # DYNAMIC ADJUSTMENT: For by_artist_underground, use relative novelty scoring
        # and disable absolute threshold filtering
        if intent == 'by_artist_underground':
            # 🔧 FIX: For by_artist_underground, always use relative novelty scoring
            # within the artist's catalog rather than absolute thresholds.
            # The relative scoring in IntentAwareScorer handles underground detection properly.
            self.logger.debug(
                f"🎯 BY_ARTIST_UNDERGROUND: Using relative novelty scoring, disabling absolute threshold",
                candidate_listeners=candidate_data.get('listeners', 0) if candidate_data else 'N/A'
            )
            return 0.0  # Disable absolute threshold - rely on relative ranking within artist catalog
        
        return base_threshold
    
    def _is_genre_hybrid_query(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> bool:
        """Check if this is a genre-hybrid query that should disable novelty filtering."""
        try:
            # Check for multiple genres in entities
            genres = entities.get('genres', [])
            if len(genres) >= 2:
                return True
            
            # Check for hybrid indicators in mood descriptors
            mood_descriptors = intent_analysis.get('mood_descriptors', [])
            hybrid_indicators = ['fusion', 'mix', 'blend', 'crossover', 'hybrid']
            if any(indicator in ' '.join(mood_descriptors).lower() for indicator in hybrid_indicators):
                return True
            
            # Check for genre mixing in query text
            query_text = intent_analysis.get('original_query', '').lower()
            genre_mixing_patterns = ['meets', 'mixed with', 'fusion', 'combined', 'blended']
            if any(pattern in query_text for pattern in genre_mixing_patterns):
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Genre hybrid detection failed: {e}")
            return False
    
    def get_intent_weights(self, intent: str, entities: Dict[str, Any] = None, intent_analysis: Dict[str, Any] = None) -> Dict[str, float]:
        """Get scoring weights based on intent."""
        base_weights = {
            'discovery': {
                'novelty': 0.25,
                'quality': 0.25, 
                'contextual_relevance': 0.20,
                'diversity': 0.15,
                'source_priority': 0.15
            },
            'by_artist': {
                'contextual_relevance': 0.35,  # Prioritize artist matching
                'quality': 0.30,               # High quality tracks by the artist
                'popularity': 0.20,            # 🔧 NEW: Favor popular tracks by the artist (opposite of novelty)
                'diversity': 0.10,             # Lower diversity (focusing on one artist)
                'source_priority': 0.05        # Reduced to make room for popularity
            },
            'by_artist_underground': {
                'novelty': 0.40,               # Highest priority for underground tracks
                'contextual_relevance': 0.30,  # Artist matching second priority
                'quality': 0.20,               # Lower quality threshold for underground gems
                'diversity': 0.05,             # Low diversity (focusing on one artist)
                'source_priority': 0.05        # Source less important
            },
            'similarity': {
                'contextual_relevance': 0.35,
                'quality': 0.25,
                'novelty': 0.15,
                'diversity': 0.15,
                'source_priority': 0.10
            },
            'artist_similarity': {
                'contextual_relevance': 0.40,  # Highest priority for artist similarity matching
                'quality': 0.30,               # High quality similar tracks
                'novelty': 0.10,               # Lower novelty - focus on good matches
                'diversity': 0.15,             # Some diversity among similar artists
                'source_priority': 0.05        # Source less important for similarity
            },
            'artist_genre': {
                'contextual_relevance': 0.45,  # Highest priority for artist+genre matching
                'quality': 0.30,               # High quality tracks from target artist
                'novelty': 0.05,               # Very low novelty - focus on known artist
                'diversity': 0.15,             # Some diversity within genre
                'source_priority': 0.05        # Source less important for artist tracks
            },
            'genre': {
                'contextual_relevance': 0.40,
                'quality': 0.25,
                'diversity': 0.20,
                'novelty': 0.10,
                'source_priority': 0.05
            },
            'mood': {
                'contextual_relevance': 0.35,
                'quality': 0.25,
                'diversity': 0.20,
                'novelty': 0.15,
                'source_priority': 0.05
            },
            'underground': {
                'novelty': 0.35,
                'quality': 0.25,
                'contextual_relevance': 0.20,
                'diversity': 0.15,
                'source_priority': 0.05
            },
            'hybrid_similarity_genre': {
                'contextual_relevance': 0.45,  # Highest priority for artist similarity + genre matching
                'quality': 0.30,               # High quality tracks
                'novelty': 0.05,               # Very low novelty - focus on good examples
                'diversity': 0.15,             # Some diversity among similar artists
                'source_priority': 0.05        # Source less important
            }
        }
        
        return base_weights.get(intent, base_weights['discovery'])
    
    def _extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target genres from entities."""
        return entities.get('genres', [])
    
    def _extract_target_moods(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[str]:
        """Extract target moods from entities and intent analysis."""
        moods = entities.get('moods', [])
        mood_descriptors = intent_analysis.get('mood_descriptors', [])
        return list(set(moods + mood_descriptors))
    
    def _extract_target_artists_from_entities(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target artists from entities."""
        return entities.get('artists', [])
    
    def _initialize_ranking_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialize intent-specific ranking weights."""
        return {
            'balanced': {
                'quality': 0.25,
                'contextual_relevance': 0.25,
                'novelty': 0.20,
                'diversity': 0.15,
                'source_priority': 0.15
            },
            'quality_focused': {
                'quality': 0.40,
                'contextual_relevance': 0.25,
                'novelty': 0.15,
                'diversity': 0.10,
                'source_priority': 0.10
            },
            'discovery_focused': {
                'novelty': 0.35,
                'quality': 0.25,
                'contextual_relevance': 0.20,
                'diversity': 0.15,
                'source_priority': 0.05
            }
        }
    
    def _initialize_context_modifiers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize context-based score modifiers."""
        return {
            'time_of_day': {
                'morning': {'energy_boost': 0.1, 'mellow_penalty': -0.05},
                'evening': {'mellow_boost': 0.1, 'energy_penalty': -0.05},
                'night': {'ambient_boost': 0.15, 'loud_penalty': -0.1}
            },
            'listening_context': {
                'workout': {'energy_boost': 0.2, 'tempo_boost': 0.15},
                'study': {'instrumental_boost': 0.2, 'vocal_penalty': -0.1},
                'party': {'danceable_boost': 0.2, 'popularity_boost': 0.1}
            }
        } 