"""
Ranking Logic for Judge Agent

Provides specialized ranking algorithms, scoring components, and selection
strategies for the judge agent's final recommendation evaluation.
"""

from typing import Dict, List, Any, Tuple
import structlog

from ...models.recommendation_models import TrackRecommendation

logger = structlog.get_logger(__name__)


class RankingLogic:
    """
    Specialized ranking logic for judge agent recommendations.
    
    Provides:
    - Multi-criteria ranking algorithms
    - Contextual relevance scoring
    - Diversity optimization
    - Selection strategies
    """
    
    def __init__(self):
        self.ranking_weights = self._initialize_ranking_weights()
        self.diversity_targets = self._initialize_diversity_targets()
        self.context_modifiers = self._initialize_context_modifiers()
        
        self.logger = structlog.get_logger(__name__)
    
    def rank_recommendations(
        self,
        candidates: List[Tuple[TrackRecommendation, Dict[str, float]]], 
        intent: str,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        novelty_threshold: float = None,
        scoring_weights: Dict[str, float] = None
    ) -> List[Tuple[TrackRecommendation, Dict[str, float]]]:
        """
        Rank track recommendations using intent-aware scoring.
        
        Args:
            candidates: List of (track, scores) tuples
            intent: Intent type (including hybrid sub-types)
            entities: Musical entities from query
            intent_analysis: Intent analysis results
            novelty_threshold: Novelty threshold for filtering
            scoring_weights: Intent-specific scoring weights
            
        Returns:
            Ranked list of (track, scores) tuples
        """
        try:
            from ..components.quality_scorer import QualityScorer
            
            # Initialize quality scorer for intent-aware scoring
            quality_scorer = QualityScorer()
            
            # Use provided parameters or get defaults
            if novelty_threshold is None:
                novelty_threshold = self.get_novelty_threshold(intent)
            if scoring_weights is None:
                scoring_weights = self.get_intent_weights(intent)
            
            self.logger.info(
                "🔧 Ranking with intent-aware scoring",
                intent=intent,
                novelty_threshold=novelty_threshold,
                scoring_weights=scoring_weights,
                candidate_count=len(candidates)
            )
            
            ranked_candidates = []
            
            for track_rec, agent_scores in candidates:
                try:
                    # 🔧 FIXED: Extract popularity data safely from raw source data
                    # Handle None values properly
                    
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
                    
                    # If still no data, try to get from LastFM API or use placeholder
                    if playcount == 0 and listeners == 0:
                        # Use popularity indicators from additional_scores
                        popularity = additional_scores.get('popularity', 0)
                        if popularity > 0:
                            # Estimate listeners from popularity (rough conversion)
                            listeners = int(popularity * 10000)
                            playcount = int(listeners * 5)
                    
                    track_data = {
                        'artist': track_rec.artist,
                        'name': getattr(track_rec, 'name', 
                                       getattr(track_rec, 'title', 'Unknown')),
                        'playcount': playcount,
                        'listeners': listeners,
                        'tags': getattr(track_rec, 'tags', []),
                        'genres': getattr(track_rec, 'genres', []),
                        'source': getattr(track_rec, 'source', 'unknown')
                    }
                    
                    self.logger.debug(
                        "🔧 Extracted track data for scoring",
                        track=f"{track_rec.artist} - {getattr(track_rec, 'name', getattr(track_rec, 'title', 'Unknown'))}",
                        playcount=playcount,
                        listeners=listeners,
                        source_has_raw_data=bool(raw_data),
                        source_has_additional_scores=bool(additional_scores)
                    )
                    
                    # 🚨 CRITICAL FIX: Detect and reject fake/fallback tracks  
                    # Tracks with 0 listeners AND 0 playcount are clearly fake fallback data
                    if playcount == 0 and listeners == 0:
                        self.logger.warning(
                            "🚨 Rejecting fake track with no popularity data",
                            track=f"{track_rec.artist} - {getattr(track_rec, 'name', getattr(track_rec, 'title', 'Unknown'))}",
                            reason="Zero listeners and playcount indicates fallback/fake data"
                        )
                        continue  # Skip this fake track completely
                    
                    # Calculate all intent-aware scoring components
                    intent_scores = quality_scorer.calculate_intent_aware_scores(
                        track_data, entities, intent_analysis, intent)
                    
                    # Merge agent scores with intent-aware scores
                    combined_scores = {**agent_scores, **intent_scores}
                    
                    # Calculate final score using intent-specific weights
                    final_score = self._calculate_weighted_score(
                        combined_scores, scoring_weights
                    )
                    
                    # Apply novelty threshold filtering
                    novelty_score = combined_scores.get('novelty', 0.5)
                    if novelty_score >= novelty_threshold:
                        combined_scores['final_score'] = final_score
                        ranked_candidates.append((track_rec, combined_scores))
                        
                        track_name = getattr(track_rec, 'name', 
                                           getattr(track_rec, 'title', 'Unknown'))
                        self.logger.debug(
                            "Track passed novelty threshold",
                            track=f"{track_rec.artist} - {track_name}",
                            novelty=novelty_score,
                            threshold=novelty_threshold,
                            final_score=final_score
                        )
                    else:
                        track_name = getattr(track_rec, 'name', 
                                           getattr(track_rec, 'title', 'Unknown'))
                        self.logger.debug(
                            "Track filtered by novelty threshold",
                            track=f"{track_rec.artist} - {track_name}",
                            novelty=novelty_score,
                            threshold=novelty_threshold
                        )
                
                except Exception as e:
                    self.logger.warning(
                        "Scoring failed for track",
                        track=f"{track_rec.artist} - {getattr(track_rec, 'name', getattr(track_rec, 'title', 'Unknown'))}",
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
            self.logger.error("Ranking failed", error=str(e))
            # Return candidates with default scores as fallback
            fallback_candidates = []
            for track_rec, scores in candidates:
                scores['final_score'] = scores.get('quality_score', 0.5)
                fallback_candidates.append((track_rec, scores))
            return fallback_candidates
    
    def _calculate_weighted_score(
        self, 
        scores: Dict[str, float], 
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate weighted final score from component scores.
        
        Args:
            scores: All scoring components
            weights: Intent-specific weights
            
        Returns:
            Final weighted score (0.0 - 1.0)
        """
        final_score = 0.0
        total_weight = 0.0
        
        for component, weight in weights.items():
            if component in scores:
                final_score += scores[component] * weight
                total_weight += weight
                self.logger.debug(
                    f"Applied weight: {component}={scores[component]:.3f} * {weight:.3f} = {scores[component] * weight:.3f}"
                )
        
        # Normalize by total weight to handle missing components
        if total_weight > 0:
            final_score = final_score / total_weight
        
        return min(1.0, max(0.0, final_score))
    
    def _calculate_intent_aware_score(
        self, 
        candidate: TrackRecommendation, 
        scores: Dict[str, float], 
        intent: str,
        weights: Dict[str, float],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> float:
        """Calculate final score using intent-specific weights."""
        
        final_score = 0.0
        
        # Apply weighted scoring based on available score components
        for component, weight in weights.items():
            score_value = 0.0
            
            if component == 'similarity' and 'relevance_score' in scores:
                score_value = scores['relevance_score']
            elif component == 'target_artist_boost':
                score_value = self._apply_target_artist_boost(candidate, entities)
            elif component == 'quality' and 'quality_score' in scores:
                score_value = scores['quality_score']
            elif component == 'novelty' and 'novelty_score' in scores:
                score_value = scores['novelty_score']
            elif component == 'underground':
                score_value = self._apply_underground_bonus(candidate)
            elif component == 'genre_mood_match' and 'contextual_relevance' in scores:
                score_value = scores['contextual_relevance']
            elif component == 'context_fit' and 'contextual_relevance' in scores:
                score_value = scores['contextual_relevance']
            elif component == 'familiarity':
                score_value = self._apply_familiarity_bonus(candidate)
            elif component in scores:
                # Fallback for any other matching score components
                score_value = scores[component]
            
            final_score += score_value * weight
            
            # Debug logging for hybrid sub-types
            if intent.startswith('hybrid_') and score_value > 0:
                self.logger.debug(f"🔧 {component}: {score_value:.3f} * {weight:.3f} = {score_value * weight:.3f}")
        
        return min(final_score, 1.0)  # Cap at 1.0
    
    def _apply_target_artist_boost(self, candidate: TrackRecommendation, entities: Dict[str, Any]) -> float:
        """Apply boost for target artist's own tracks in similarity queries."""
        if not entities:
            return 0.0
        
        # Access artists from dictionary structure
        musical_entities = entities.get('musical_entities', {})
        artist_entities = musical_entities.get('artists', {})
        
        # Get all target artists from primary and similar_to lists
        target_artists = []
        target_artists.extend(artist_entities.get('primary', []))
        target_artists.extend(artist_entities.get('similar_to', []))
        
        # Convert to lowercase for comparison
        target_artists_lower = [artist.lower() for artist in target_artists]
        
        if candidate.artist.lower() in target_artists_lower:
            return 1.0  # Maximum boost for target artist's tracks
        return 0.0
    
    def _apply_underground_bonus(self, candidate: TrackRecommendation) -> float:
        """Apply bonus for underground/low-popularity tracks in discovery queries."""
        underground_indicators = ['underground', 'hidden_gem', 'experimental', 'indie']
        
        for indicator in underground_indicators:
            if (hasattr(candidate, 'genres') and candidate.genres and 
                any(indicator in genre.lower() for genre in candidate.genres)):
                return 0.8
            if (hasattr(candidate, 'tags') and candidate.tags and 
                any(indicator in tag.lower() for tag in candidate.tags)):
                return 0.8
        
        return 0.2  # Base score for non-underground tracks
    
    def _apply_familiarity_bonus(self, candidate: TrackRecommendation) -> float:
        """Apply bonus for familiar tracks in contextual queries."""
        # Simplified implementation - in real system would check user history/popularity
        return 0.5  # Neutral score - to be enhanced with actual familiarity data
    
    def _rank_balanced(
        self,
        candidates: List[Tuple[Any, Dict[str, float]]],
        context: Dict[str, Any]
    ) -> List[Tuple[Any, Dict[str, float]]]:
        """Balanced ranking considering all factors."""
        weights = self.ranking_weights['balanced']
        
        # Calculate weighted scores
        for recommendation, scores in candidates:
            weighted_score = (
                scores.get('quality_score', 0) * weights['quality'] +
                scores.get('contextual_relevance', 0) * weights['relevance'] +
                scores.get('intent_alignment', 0) * weights['intent'] +
                scores.get('diversity_value', 0) * weights['diversity'] +
                scores.get('source_priority', 0) * weights['source']
            )
            
            # Apply context modifiers
            weighted_score = self._apply_context_modifiers(
                weighted_score, recommendation, context
            )
            
            scores['final_score'] = weighted_score
        
        # Sort by final score
        return sorted(candidates, key=lambda x: x[1]['final_score'], reverse=True)
    
    def _rank_by_quality(
        self,
        candidates: List[Tuple[Any, Dict[str, float]]],
        context: Dict[str, Any]
    ) -> List[Tuple[Any, Dict[str, float]]]:
        """Quality-focused ranking."""
        weights = self.ranking_weights['quality_focused']
        
        for recommendation, scores in candidates:
            quality_focused_score = (
                scores.get('quality_score', 0) * weights['quality'] +
                scores.get('contextual_relevance', 0) * weights['relevance'] +
                scores.get('intent_alignment', 0) * weights['intent']
            )
            scores['final_score'] = quality_focused_score
        
        return sorted(candidates, key=lambda x: x[1]['final_score'], reverse=True)
    
    def _rank_by_diversity(
        self,
        candidates: List[Tuple[Any, Dict[str, float]]],
        context: Dict[str, Any]
    ) -> List[Tuple[Any, Dict[str, float]]]:
        """Diversity-focused ranking with selection optimization."""
        # First, calculate diversity-weighted scores
        weights = self.ranking_weights['diversity_focused']
        
        for recommendation, scores in candidates:
            diversity_score = (
                scores.get('diversity_value', 0) * weights['diversity'] +
                scores.get('quality_score', 0) * weights['quality'] +
                scores.get('contextual_relevance', 0) * weights['relevance']
            )
            scores['diversity_score'] = diversity_score
        
        # Apply diversity selection algorithm
        return self._select_diverse_recommendations(candidates, context)
    
    def _rank_by_intent(
        self,
        candidates: List[Tuple[Any, Dict[str, float]]],
        context: Dict[str, Any]
    ) -> List[Tuple[Any, Dict[str, float]]]:
        """Intent-focused ranking."""
        weights = self.ranking_weights['intent_focused']
        primary_intent = context.get('intent_analysis', {}).get('primary_intent', 'discovery')
        
        # Adjust weights based on intent
        if primary_intent == 'discovery':
            weights = self.ranking_weights['discovery_intent']
        elif primary_intent == 'genre_mood':
            weights = self.ranking_weights['genre_mood_intent']
        
        for recommendation, scores in candidates:
            intent_score = (
                scores.get('intent_alignment', 0) * weights['intent'] +
                scores.get('quality_score', 0) * weights['quality'] +
                scores.get('contextual_relevance', 0) * weights['relevance']
            )
            scores['final_score'] = intent_score
        
        return sorted(candidates, key=lambda x: x[1]['final_score'], reverse=True)
    
    def _select_diverse_recommendations(
        self,
        candidates: List[Tuple[Any, Dict[str, float]]],
        context: Dict[str, Any]
    ) -> List[Tuple[Any, Dict[str, float]]]:
        """Select recommendations optimizing for diversity."""
        selected = []
        remaining = candidates.copy()
        
        # Track diversity metrics
        selected_artists = set()
        selected_genres = set()
        selected_sources = set()
        
        while remaining and len(selected) < 20:  # Max recommendations
            best_candidate = None
            best_score = -1
            
            for candidate, scores in remaining:
                recommendation = candidate
                
                # Calculate diversity bonus
                diversity_bonus = self._calculate_diversity_bonus(
                    recommendation, selected_artists, selected_genres, selected_sources
                )
                
                # Combined score with diversity bonus
                combined_score = scores.get('diversity_score', 0) + diversity_bonus
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = (candidate, scores)
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                
                # Update diversity tracking
                recommendation = best_candidate[0]
                selected_artists.add(getattr(recommendation, 'artist', ''))
                selected_genres.update(getattr(recommendation, 'genres', []))
                selected_sources.add(getattr(recommendation, 'source', ''))
        
        return selected
    
    def _calculate_diversity_bonus(
        self,
        recommendation: Any,
        selected_artists: set,
        selected_genres: set,
        selected_sources: set
    ) -> float:
        """Calculate diversity bonus for a recommendation."""
        bonus = 0.0
        
        # Artist diversity bonus
        artist = getattr(recommendation, 'artist', '')
        if artist and artist not in selected_artists:
            bonus += 0.3
        elif artist in selected_artists:
            bonus -= 0.2  # Penalty for repetition
        
        # Genre diversity bonus
        genres = set(getattr(recommendation, 'genres', []))
        genre_overlap = len(genres & selected_genres)
        if genre_overlap == 0:
            bonus += 0.2  # New genres
        elif genre_overlap < len(genres):
            bonus += 0.1  # Partial overlap
        else:
            bonus -= 0.1  # All genres already selected
        
        # Source diversity bonus
        source = getattr(recommendation, 'source', '')
        if source and source not in selected_sources:
            bonus += 0.1
        
        return bonus
    
    def _apply_context_modifiers(
        self,
        base_score: float,
        recommendation: Any,
        context: Dict[str, Any]
    ) -> float:
        """Apply context-based score modifiers."""
        modified_score = base_score
        
        # Intent-based modifiers
        intent_analysis = context.get('intent_analysis', {})
        primary_intent = intent_analysis.get('primary_intent', '')
        
        if primary_intent in self.context_modifiers:
            modifiers = self.context_modifiers[primary_intent]
            
            # Apply source modifiers
            source = getattr(recommendation, 'source', '')
            if source in modifiers.get('source_boost', {}):
                modified_score *= modifiers['source_boost'][source]
            
            # Apply tag modifiers
            tags = getattr(recommendation, 'tags', [])
            for tag in tags:
                if tag in modifiers.get('tag_boost', {}):
                    modified_score *= modifiers['tag_boost'][tag]
                    break  # Apply only first matching tag boost
        
        # Context factor modifiers
        context_factors = context.get('entities', {}).get('context_factors', [])
        for factor in context_factors:
            factor_lower = factor.lower()
            
            if 'workout' in factor_lower or 'exercise' in factor_lower:
                # Boost energetic tracks
                tags = getattr(recommendation, 'tags', [])
                if any('energetic' in tag.lower() for tag in tags):
                    modified_score *= 1.2
            
            elif 'study' in factor_lower or 'work' in factor_lower:
                # Boost calm/instrumental tracks
                tags = getattr(recommendation, 'tags', [])
                if any(word in tag.lower() for tag in tags for word in ['calm', 'instrumental', 'ambient']):
                    modified_score *= 1.2
        
        return modified_score
    
    def calculate_recommendation_confidence(
        self,
        recommendation: Any,
        scores: Dict[str, float],
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for a recommendation."""
        confidence = 0.5  # Base confidence
        
        # Quality-based confidence
        quality_score = scores.get('quality_score', 0)
        confidence += quality_score * 0.3
        
        # Relevance-based confidence
        relevance_score = scores.get('contextual_relevance', 0)
        confidence += relevance_score * 0.2
        
        # Intent alignment confidence
        intent_score = scores.get('intent_alignment', 0)
        confidence += intent_score * 0.2
        
        # Source reliability
        source = getattr(recommendation, 'source', '')
        source_reliability = {
            'genre_mood_agent': 0.9,
            'discovery_agent': 0.8,
            'planner_agent': 0.7
        }
        confidence += source_reliability.get(source, 0.5) * 0.1
        
        # Normalize confidence
        return min(max(confidence, 0.0), 1.0)
    
    def analyze_ranking_distribution(
        self,
        ranked_recommendations: List[Tuple[Any, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Analyze the distribution of ranked recommendations."""
        analysis = {
            'total_recommendations': len(ranked_recommendations),
            'score_distribution': {},
            'source_distribution': {},
            'genre_distribution': {},
            'artist_distribution': {},
            'quality_metrics': {}
        }
        
        if not ranked_recommendations:
            return analysis
        
        scores = [scores.get('final_score', 0) for _, scores in ranked_recommendations]
        sources = [getattr(rec, 'source', 'unknown') for rec, _ in ranked_recommendations]
        artists = [getattr(rec, 'artist', 'unknown') for rec, _ in ranked_recommendations]
        
        # Score distribution
        analysis['score_distribution'] = {
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'range': max(scores) - min(scores)
        }
        
        # Source distribution
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        analysis['source_distribution'] = source_counts
        
        # Artist distribution
        artist_counts = {}
        for artist in artists:
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
        analysis['artist_distribution'] = dict(sorted(
            artist_counts.items(), key=lambda x: x[1], reverse=True
        )[:10])  # Top 10 artists
        
        # Quality metrics
        quality_scores = [scores.get('quality_score', 0) for _, scores in ranked_recommendations]
        analysis['quality_metrics'] = {
            'average_quality': sum(quality_scores) / len(quality_scores),
            'high_quality_count': sum(1 for q in quality_scores if q > 0.8),
            'low_quality_count': sum(1 for q in quality_scores if q < 0.4)
        }
        
        return analysis
    
    def _initialize_ranking_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialize intent-aware ranking weight configurations."""
        return {
            # Intent-specific scoring weights from design document
            'artist_similarity': {
                'similarity': 0.6,
                'target_artist_boost': 0.2,
                'quality': 0.15,
                'novelty': 0.05
            },
            'discovery': {
                'novelty': 0.5,
                'underground': 0.3,
                'quality': 0.15,
                'similarity': 0.05
            },
            'genre_mood': {
                'genre_mood_match': 0.6,
                'quality': 0.25,
                'novelty': 0.15
            },
            'contextual': {
                'context_fit': 0.6,
                'quality': 0.25,
                'familiarity': 0.15
            },
            'hybrid': {
                'similarity': 0.4,
                'genre_mood_match': 0.35,
                'quality': 0.15,
                'novelty': 0.1
            },
            
            # Legacy compatibility - map old intents to new weights
            'similarity': {  # Maps to artist_similarity
                'similarity': 0.6,
                'target_artist_boost': 0.2,
                'quality': 0.15,
                'novelty': 0.05
            },
            'genre_exploration': {  # Maps to genre_mood
                'genre_mood_match': 0.6,
                'quality': 0.25,
                'novelty': 0.15
            },
            'mood_matching': {  # Maps to genre_mood
                'genre_mood_match': 0.6,
                'quality': 0.25,
                'novelty': 0.15
            },
            'activity_context': {  # Maps to contextual
                'context_fit': 0.6,
                'quality': 0.25,
                'familiarity': 0.15
            },
            
            # Fallback balanced scoring
            'balanced': {
                'quality': 0.25,
                'relevance': 0.25,
                'intent': 0.25,
                'diversity': 0.15,
                'source': 0.10
            },
            'quality_focused': {
                'quality': 0.50,
                'relevance': 0.30,
                'intent': 0.20
            },
            'diversity_focused': {
                'diversity': 0.40,
                'quality': 0.30,
                'relevance': 0.30
            },
            'intent_focused': {
                'intent': 0.50,
                'quality': 0.25,
                'relevance': 0.25
            }
        }
    
    def _initialize_diversity_targets(self) -> Dict[str, Any]:
        """Initialize diversity targets and constraints."""
        return {
            'max_per_artist': 2,
            'min_genres': 3,
            'max_genre_dominance': 0.5,  # Max 50% from one genre
            'source_distribution': {
                'genre_mood_agent': 0.4,
                'discovery_agent': 0.4,
                'planner_agent': 0.2
            }
        }
    
    def _initialize_context_modifiers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize context-based score modifiers."""
        return {
            'discovery': {
                'source_boost': {
                    'discovery_agent': 1.3,
                    'genre_mood_agent': 1.0,
                    'planner_agent': 0.8
                },
                'tag_boost': {
                    'underground': 1.2,
                    'hidden_gem': 1.2,
                    'experimental': 1.1
                }
            },
            'genre_mood': {
                'source_boost': {
                    'genre_mood_agent': 1.3,
                    'discovery_agent': 1.0,
                    'planner_agent': 0.9
                },
                'tag_boost': {
                    'energetic': 1.1,
                    'calm': 1.1,
                    'happy': 1.1
                }
            },
            'similarity': {
                'source_boost': {
                    'discovery_agent': 1.2,
                    'genre_mood_agent': 1.1,
                    'planner_agent': 1.0
                },
                'tag_boost': {
                    'similar': 1.2,
                    'related': 1.1
                }
            }
        } 
    
    def get_novelty_threshold(self, intent: str) -> float:
        """Get novelty threshold based on intent with hybrid sub-type support."""
        
        thresholds = {
            'artist_similarity': 0.15,           # Very relaxed
            'discovery': 0.4,                    # Moderately strict (was 0.6 - too strict!)
            'genre_mood': 0.3,                   # Moderate
            'contextual': 0.2,                   # Relaxed
            'hybrid': 0.25,                      # Moderate (default hybrid)
            
            # 🔧 NEW: Hybrid sub-type specific thresholds
            'hybrid_discovery_primary': 0.4,     # Moderately strict (was 0.6 - too strict!)
            'hybrid_similarity_primary': 0.25,   # Relaxed - similarity focus  
            'hybrid_genre_primary': 0.35,        # Moderate - balanced approach
        }
        
        threshold = thresholds.get(intent, 0.4)  # Default fallback
        self.logger.debug(f"Novelty threshold for '{intent}': {threshold}")
        return threshold
    
    def get_diversity_limits(self, intent: str) -> Dict[str, Any]:
        """Get intent-specific diversity limits."""
        limits = {
            'artist_similarity': {
                'max_per_artist': 5,    # 🔧 INCREASED: Allow more tracks from target artist
                'max_per_genre': 8, 
                'min_genres': 2
            },
            'discovery': {
                'max_per_artist': 1,    # Strict diversity for discovery
                'max_per_genre': 3, 
                'min_genres': 4
            },
            'genre_mood': {
                'max_per_artist': 2, 
                'max_per_genre': 6, 
                'min_genres': 3
            },
            'contextual': {
                'max_per_artist': 3, 
                'max_per_genre': 6, 
                'min_genres': 2
            },
            'hybrid': {
                'max_per_artist': 2, 
                'max_per_genre': 4, 
                'min_genres': 3
            },
            
            # Legacy compatibility
            'similarity': {'max_per_artist': 5, 'max_per_genre': 8, 'min_genres': 2},
            'genre_exploration': {'max_per_artist': 2, 'max_per_genre': 6, 'min_genres': 3},
            'mood_matching': {'max_per_artist': 2, 'max_per_genre': 6, 'min_genres': 3},
            'activity_context': {'max_per_artist': 3, 'max_per_genre': 6, 'min_genres': 2},
            'mood_based': {'max_per_artist': 2, 'max_per_genre': 6, 'min_genres': 3},
            'activity_based': {'max_per_artist': 3, 'max_per_genre': 6, 'min_genres': 2},
            'genre_specific': {'max_per_artist': 2, 'max_per_genre': 6, 'min_genres': 3}
        }
        return limits.get(intent, {'max_per_artist': 2, 'max_per_genre': 4, 'min_genres': 3})
    
    def get_intent_weights(self, intent: str) -> Dict[str, float]:
        """Get scoring weights based on intent with hybrid sub-type support."""
        
        # 🔧 NEW: Handle hybrid sub-types with different scoring priorities
        if intent.startswith('hybrid_'):
            # Use specific hybrid sub-type weights
            hybrid_weights = {
                'hybrid_discovery_primary': {
                    'novelty': 0.5,           # Most important - truly underground
                    'genre_mood_match': 0.4,  # Important - genre accuracy  
                    'quality': 0.1           # Basic threshold
                },
                'hybrid_similarity_primary': {
                    'similarity': 0.5,        # Most important - artist similarity
                    'genre_mood_match': 0.3,  # Important - style modifier
                    'quality': 0.2           # Good quality baseline
                },
                'hybrid_genre_primary': {
                    'genre_mood_match': 0.6,  # Most important - style accuracy
                    'novelty': 0.25,         # Secondary - some discovery
                    'quality': 0.15          # Basic quality
                }
            }
            
            if intent in hybrid_weights:
                weights = hybrid_weights[intent]
                self.logger.debug(f"🔧 Using {intent} weights: {weights}")
                return weights
            else:
                self.logger.warning(f"Unknown hybrid sub-type: {intent}, using default hybrid weights")
                intent = 'hybrid'  # Fallback to default hybrid
        
        # Original intent-specific weights
        intent_weights = {
            'artist_similarity': {
                'similarity': 0.6,
                'target_artist_boost': 0.2,
                'quality': 0.15,
                'novelty': 0.05
            },
            'discovery': {
                'novelty': 0.5,
                'underground': 0.3,
                'quality': 0.15,
                'similarity': 0.05
            },
            'genre_mood': {
                'genre_mood_match': 0.6,
                'quality': 0.25,
                'novelty': 0.15
            },
            'contextual': {
                'context_fit': 0.6,
                'quality': 0.25,
                'familiarity': 0.15
            },
            'hybrid': {
                'similarity': 0.4,
                'genre_mood_match': 0.35,
                'quality': 0.15,
                'novelty': 0.1
            }
        }
        
        weights = intent_weights.get(intent, intent_weights['hybrid'])
        self.logger.debug(f"Intent weights for '{intent}': {weights}")
        return weights 