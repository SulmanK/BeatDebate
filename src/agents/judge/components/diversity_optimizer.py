"""
Diversity Optimizer for Judge Agent

Handles diversity optimization and final selection algorithms
for the judge agent's recommendation output.

Extracted from judge/agent.py for better modularity.
"""

from typing import Dict, List, Any, Tuple
import structlog

from src.models.recommendation_models import TrackRecommendation

logger = structlog.get_logger(__name__)


class DiversityOptimizer:
    """
    Diversity optimizer for judge agent recommendations.
    
    Responsibilities:
    - Apply diversity constraints to final selections
    - Balance artist, genre, and source distribution
    - Optimize for discovery and engagement
    - Ensure variety in final recommendations
    """
    
    def __init__(self):
        """Initialize diversity optimizer with default targets."""
        self.diversity_targets = self._initialize_diversity_targets()
        self.logger = structlog.get_logger(__name__)
    
    def select_with_diversity(
        self,
        ranked_candidates: List[Tuple[TrackRecommendation, Dict[str, float]]],
        state=None,
        final_count: int = 20,
        diversity_config: Dict[str, Any] = None
    ) -> List[TrackRecommendation]:
        """
        Select final recommendations with diversity optimization.
        
        Args:
            ranked_candidates: List of (track, scores) tuples, pre-ranked
            state: Current workflow state (optional, for context)
            final_count: Number of final recommendations to select
            diversity_config: Custom diversity configuration
            
        Returns:
            Optimized list of diverse recommendations
        """
        try:
            if not ranked_candidates:
                self.logger.warning("No candidates provided for diversity selection")
                return []
            
            # Use provided config or defaults
            if diversity_config is None:
                diversity_config = self.diversity_targets
            
            # Extract intent-specific diversity limits
            intent = 'discovery'  # default
            if state:
                # Try multiple ways to extract intent
                if hasattr(state, 'intent_analysis') and state.intent_analysis:
                    intent = state.intent_analysis.get('intent', 'discovery')
                elif hasattr(state, 'query_understanding') and state.query_understanding:
                    if hasattr(state.query_understanding.intent, 'value'):
                        intent = state.query_understanding.intent.value
                    else:
                        intent = str(state.query_understanding.intent)
                elif hasattr(state, 'intent'):
                    # Fallback to direct intent attribute if it exists
                    intent = state.intent
            
            diversity_limits = self._get_diversity_limits(intent)
            
            self.logger.info(
                f"Selecting {final_count} diverse recommendations from {len(ranked_candidates)} candidates",
                intent=intent,
                diversity_limits=diversity_limits
            )
            
            selected_recommendations = []
            tracking = {
                'artists': {},
                'genres': set(),
                'sources': {},
                'selected_artists': set(),
                'selected_genres': set(),
                'selected_sources': set()
            }
            
            # First pass: Select top candidates while respecting diversity constraints
            for track_rec, scores in ranked_candidates:
                if len(selected_recommendations) >= final_count:
                    break
                
                # Check diversity constraints
                if self._passes_diversity_constraints(
                    track_rec, tracking, diversity_limits, selected_recommendations
                ):
                    selected_recommendations.append(track_rec)
                    self._update_tracking(track_rec, tracking)
                    
                    self.logger.debug(
                        f"Selected track {len(selected_recommendations)}: {track_rec.artist} - {getattr(track_rec, 'title', 'Unknown')}",
                        final_score=scores.get('final_score', 0),
                        artists_count=len(tracking['selected_artists']),
                        genres_count=len(tracking['selected_genres'])
                    )
                else:
                    self.logger.debug(
                        f"Skipped track (diversity): {track_rec.artist} - {getattr(track_rec, 'title', 'Unknown')}"
                    )
            
            # Second pass: Fill remaining slots if needed
            if len(selected_recommendations) < final_count:
                self._fill_remaining_slots(
                    ranked_candidates, selected_recommendations, tracking, 
                    final_count, diversity_limits
                )
            
            # Apply final diversity optimization
            optimized_recommendations = self._apply_final_optimization(
                selected_recommendations, diversity_config
            )
            
            # Generate diversity report
            diversity_report = self._generate_diversity_report(optimized_recommendations)
            self.logger.info(
                f"Diversity selection completed: {len(optimized_recommendations)} tracks",
                **diversity_report
            )
            
            return optimized_recommendations
            
        except Exception as e:
            self.logger.error(f"Diversity selection failed: {e}")
            # Return top N as fallback
            return [track for track, scores in ranked_candidates[:final_count]]
    
    def _passes_diversity_constraints(
        self,
        track_rec: TrackRecommendation,
        tracking: Dict[str, Any],
        diversity_limits: Dict[str, Any],
        current_selections: List[TrackRecommendation]
    ) -> bool:
        """Check if track passes diversity constraints."""
        try:
            # Artist constraint
            max_per_artist = diversity_limits.get('max_per_artist', 2)
            artist_count = tracking['artists'].get(track_rec.artist, 0)
            if artist_count >= max_per_artist:
                return False
            
            # Source constraint
            max_per_source = diversity_limits.get('max_per_source', 15)
            source = getattr(track_rec, 'source', 'unknown')
            source_count = tracking['sources'].get(source, 0)
            if source_count >= max_per_source:
                return False
            
            # Genre diversity requirement
            min_genres = diversity_limits.get('min_genres', 3)
            track_genres = set(getattr(track_rec, 'genres', []))
            
            # Allow if we haven't met minimum genre requirement and this adds new genres
            if len(tracking['selected_genres']) < min_genres:
                new_genres = track_genres - tracking['selected_genres']
                if new_genres:
                    return True  # This track adds genre diversity
            
            # If we have enough genres, apply standard constraints
            return True
            
        except Exception as e:
            self.logger.warning(f"Diversity constraint check failed: {e}")
            return True  # Default to allowing the track
    
    def _update_tracking(self, track_rec: TrackRecommendation, tracking: Dict[str, Any]):
        """Update tracking counters for diversity management."""
        # Update artist count
        artist = track_rec.artist
        tracking['artists'][artist] = tracking['artists'].get(artist, 0) + 1
        tracking['selected_artists'].add(artist)
        
        # Update source count
        source = getattr(track_rec, 'source', 'unknown')
        tracking['sources'][source] = tracking['sources'].get(source, 0) + 1
        tracking['selected_sources'].add(source)
        
        # Update genres
        track_genres = getattr(track_rec, 'genres', [])
        for genre in track_genres:
            tracking['selected_genres'].add(genre)
    
    def _fill_remaining_slots(
        self,
        ranked_candidates: List[Tuple[TrackRecommendation, Dict[str, float]]],
        selected_recommendations: List[TrackRecommendation],
        tracking: Dict[str, Any],
        final_count: int,
        diversity_limits: Dict[str, Any]
    ):
        """Fill remaining slots with relaxed diversity constraints."""
        try:
            selected_tracks = {f"{rec.artist}||{getattr(rec, 'title', '')}" for rec in selected_recommendations}
            
            # Relax constraints for remaining slots
            relaxed_limits = diversity_limits.copy()
            relaxed_limits['max_per_artist'] = relaxed_limits.get('max_per_artist', 2) + 1
            relaxed_limits['max_per_source'] = relaxed_limits.get('max_per_source', 15) + 5
            
            for track_rec, scores in ranked_candidates:
                if len(selected_recommendations) >= final_count:
                    break
                
                # Skip already selected tracks
                track_key = f"{track_rec.artist}||{getattr(track_rec, 'title', '')}"
                if track_key in selected_tracks:
                    continue
                
                # Check relaxed constraints
                if self._passes_diversity_constraints(
                    track_rec, tracking, relaxed_limits, selected_recommendations
                ):
                    selected_recommendations.append(track_rec)
                    self._update_tracking(track_rec, tracking)
                    selected_tracks.add(track_key)
                    
                    self.logger.debug(
                        f"Filled slot {len(selected_recommendations)} (relaxed): {track_rec.artist} - {getattr(track_rec, 'title', 'Unknown')}"
                    )
            
            self.logger.info(f"Filled {len(selected_recommendations)} total slots")
            
        except Exception as e:
            self.logger.warning(f"Slot filling failed: {e}")
    
    def _apply_final_optimization(
        self,
        recommendations: List[TrackRecommendation],
        diversity_config: Dict[str, Any]
    ) -> List[TrackRecommendation]:
        """Apply final diversity optimization strategies."""
        try:
            # For now, return as-is, but this could include:
            # - Reordering for better flow
            # - Swapping tracks for better genre distribution
            # - Optimizing for discovery vs familiarity balance
            
            self.logger.debug(f"Applied final optimization to {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Final optimization failed: {e}")
            return recommendations
    
    def calculate_diversity_score(
        self,
        recommendations: List[TrackRecommendation]
    ) -> Dict[str, float]:
        """Calculate diversity metrics for a set of recommendations."""
        try:
            if not recommendations:
                return {'overall_diversity': 0.0}
            
            # Artist diversity
            unique_artists = len(set(rec.artist for rec in recommendations))
            artist_diversity = unique_artists / len(recommendations)
            
            # Genre diversity
            all_genres = set()
            for rec in recommendations:
                all_genres.update(getattr(rec, 'genres', []))
            genre_diversity = min(len(all_genres) / 10, 1.0)  # Normalize to max 10 genres
            
            # Source diversity
            unique_sources = len(set(getattr(rec, 'source', 'unknown') for rec in recommendations))
            source_diversity = min(unique_sources / 3, 1.0)  # Normalize to max 3 sources
            
            # Confidence diversity (prefer mix of high and medium confidence)
            confidences = [getattr(rec, 'confidence', 0.5) for rec in recommendations]
            confidence_std = np.std(confidences) if len(confidences) > 1 else 0
            confidence_diversity = min(confidence_std * 2, 1.0)  # Normalize
            
            # Overall diversity score
            overall_diversity = (
                artist_diversity * 0.4 +
                genre_diversity * 0.3 +
                source_diversity * 0.2 +
                confidence_diversity * 0.1
            )
            
            return {
                'overall_diversity': overall_diversity,
                'artist_diversity': artist_diversity,
                'genre_diversity': genre_diversity,
                'source_diversity': source_diversity,
                'confidence_diversity': confidence_diversity,
                'unique_artists': unique_artists,
                'unique_genres': len(all_genres),
                'unique_sources': unique_sources
            }
            
        except Exception as e:
            self.logger.warning(f"Diversity score calculation failed: {e}")
            return {'overall_diversity': 0.5}
    
    def _generate_diversity_report(
        self,
        recommendations: List[TrackRecommendation]
    ) -> Dict[str, Any]:
        """Generate diversity report for logging."""
        try:
            artists = {}
            genres = set()
            sources = {}
            
            for rec in recommendations:
                # Count artists
                artist = rec.artist
                artists[artist] = artists.get(artist, 0) + 1
                
                # Collect genres
                rec_genres = getattr(rec, 'genres', [])
                genres.update(rec_genres)
                
                # Count sources
                source = getattr(rec, 'source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            return {
                'unique_artists': len(artists),
                'max_tracks_per_artist': max(artists.values()) if artists else 0,
                'unique_genres': len(genres),
                'unique_sources': len(sources),
                'source_distribution': sources
            }
            
        except Exception as e:
            self.logger.warning(f"Diversity report generation failed: {e}")
            return {'total_tracks': len(recommendations)}
    
    def _get_diversity_limits(self, intent: str) -> Dict[str, Any]:
        """Get diversity limits based on intent."""
        limits = {
            'discovery': {
                'max_per_artist': 3,    # Increased from 1 to 3 for better discovery exploration
                'max_per_source': 15,   # Increased from 12 to 15 for more options
                'min_genres': 4,
                'prefer_variety': True
            },
            'by_artist': {
                'max_per_artist': 25,   # Allow many tracks from target artist for discography
                'max_per_source': 30,
                'min_genres': 1,        # Artist focus, not genre diversity
                'prefer_variety': False
            },
            'artist_similarity': {
                'max_per_artist': 5,    # Allow more tracks per similar artist for diverse pool
                'max_per_source': 20,
                'min_genres': 2,
                'prefer_variety': True
            },
            'similarity': {
                'max_per_artist': 2,    # Allow more from same artist
                'max_per_source': 15,
                'min_genres': 2,
                'prefer_variety': False
            },
            'genre': {
                'max_per_artist': 2,
                'max_per_source': 15,
                'min_genres': 1,        # Genre focus
                'prefer_variety': False
            },
            'mood': {
                'max_per_artist': 2,
                'max_per_source': 15,
                'min_genres': 3,
                'prefer_variety': True
            },
            'underground': {
                'max_per_artist': 1,    # Maximize diversity for underground
                'max_per_source': 10,
                'min_genres': 5,
                'prefer_variety': True
            },
            'by_artist_underground': {
                'max_per_artist': 25,   # NO LIMIT: User wants underground tracks from specific artist
                'max_per_source': 30,
                'min_genres': 1,        # Artist focus, not genre diversity
                'prefer_variety': False
            },
            'artist_genre': {
                'max_per_artist': 25,   # NO LIMIT: User wants tracks from specific artist in specific genre
                'max_per_source': 30,
                'min_genres': 1,        # Genre is already filtered, focus on artist
                'prefer_variety': False
            },
            'hybrid_similarity_genre': {
                'max_per_artist': 5,    # Allow multiple tracks per similar artist
                'max_per_source': 20,
                'min_genres': 2,        # Genre filtering is important
                'prefer_variety': True  # Want variety in similar artists
            }
        }
        
        return limits.get(intent, limits['discovery'])
    
    def _initialize_diversity_targets(self) -> Dict[str, Any]:
        """Initialize default diversity targets."""
        return {
            'max_per_artist': 2,
            'min_genres': 3,
            'max_per_source': 15,
            'source_distribution': {
                'genre_mood_agent': 0.4, 
                'discovery_agent': 0.4, 
                'planner_agent': 0.2
            },
            'genre_balance': {
                'min_primary_genre_ratio': 0.3,
                'max_primary_genre_ratio': 0.7
            },
            'confidence_balance': {
                'min_high_confidence': 0.6,
                'max_high_confidence': 0.9
            }
        }


# Import numpy safely
try:
    import numpy as np
except ImportError:
    # Fallback implementation for std calculation
    import math
    class np:
        @staticmethod
        def std(values):
            if len(values) <= 1:
                return 0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return math.sqrt(variance)