"""
Discovery Diversity Manager for BeatDebate

Handles discovery-specific diversity logic including:
- Artist diversity management
- Genre variety optimization
- Adaptive diversity filtering based on intent
- Track distribution balancing

This module extracts diversity logic from DiscoveryAgent for better modularity.
"""

from typing import Dict, List, Any
import structlog

logger = structlog.get_logger(__name__)


class DiscoveryDiversity:
    """
    Discovery-specific diversity management system.
    
    This manager focuses on ensuring appropriate variety and distribution
    of tracks in discovery recommendations.
    """
    
    def __init__(self):
        """Initialize discovery diversity manager."""
        self.logger = logger.bind(component="DiscoveryDiversity")
        self.logger.info("Discovery Diversity Manager initialized")
    
    def ensure_discovery_diversity(
        self, 
        candidates: List[Dict[str, Any]], 
        intent_analysis: Dict[str, Any], 
        context_override: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Ensure appropriate diversity in discovery candidates.
        
        Args:
            candidates: List of candidate tracks
            intent_analysis: Intent analysis from PlannerAgent
            context_override: Optional context override for diversity rules
            
        Returns:
            List of candidates with appropriate diversity
        """
        if not candidates:
            return []
        
        intent = intent_analysis.get('intent', '').lower()
        self.logger.info(f"Applying diversity filtering for intent: {intent}")
        
        # Get intent-specific diversity parameters
        diversity_params = self._get_diversity_parameters(intent, context_override)
        
        # Apply adaptive diversity filtering
        diversified = self._adaptive_diversity_filtering(
            candidates, 
            diversity_params['candidate_limit'],
            diversity_params.get('target_artist_from_override', ''),
            intent
        )
        
        self.logger.info(f"Diversity filtering: {len(candidates)} -> {len(diversified)} candidates")
        return diversified
    
    def _get_diversity_parameters(self, intent: str, context_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get diversity parameters based on intent and context."""
        # Default parameters
        params = {
            'candidate_limit': 30,
            'max_tracks_per_artist': 2,
            'target_artist_from_override': ''
        }
        
        # Intent-specific adjustments
        if intent == 'by_artist':
            params.update({
                'candidate_limit': 50,
                'max_tracks_per_artist': 20,  # Allow many tracks from target artist
            })
        elif intent == 'by_artist_underground':
            params.update({
                'candidate_limit': 50,
                'max_tracks_per_artist': 20,  # Allow many underground tracks from target artist
            })
        elif intent == 'artist_similarity':
            params.update({
                'candidate_limit': 40,
                'max_tracks_per_artist': 3,   # Allow more tracks per similar artist
            })
        elif intent == 'discovery':
            params.update({
                'candidate_limit': 50,        # Increased from 30 to 50 for more discovery options
                'max_tracks_per_artist': 5,   # Increased from 1 to 3 for better exploration
            })
        elif intent == 'discovering_serendipity':
            params.update({
                'candidate_limit': 60,        # Even larger pool for serendipitous discovery
                'max_tracks_per_artist': 4,   # Allow more variety per artist for serendipity
            })
        elif intent == 'genre_mood':
            params.update({
                'candidate_limit': 50,
                'max_tracks_per_artist': 5,   # Balanced diversity
            })
        elif intent == 'contextual':
            params.update({
                'candidate_limit': 50,
                'max_tracks_per_artist': 5,   # Balanced diversity
            })
        elif intent == 'hybrid':
            params.update({
                'candidate_limit': 50,
                'max_tracks_per_artist': 5,   # Balanced diversity
            })
        
        # Apply context override if provided
        if context_override:
            target_artist = context_override.get('target_artist', '')
            if target_artist:
                params['target_artist_from_override'] = target_artist
                # Increase limit for target artist tracks
                if intent in ['by_artist', 'by_artist_underground']:
                    params['max_tracks_per_artist'] = 25
        
        return params
    
    def _adaptive_diversity_filtering(
        self, 
        candidates: List[Dict[str, Any]], 
        candidate_limit: int, 
        target_artist_from_override: str, 
        intent: str
    ) -> List[Dict[str, Any]]:
        """
        Apply adaptive diversity filtering based on intent and candidate pool.
        
        Args:
            candidates: List of candidate tracks
            candidate_limit: Maximum number of candidates to return
            target_artist_from_override: Target artist from context override
            intent: Intent string for diversity rules
            
        Returns:
            List of diversified candidates
        """
        if not candidates:
            return []
        
        # Count valid candidates first
        valid_count = self._count_valid_candidates(candidates)
        
        if valid_count <= candidate_limit:
            # If we have fewer candidates than limit, return all
            return candidates[:candidate_limit]
        
        # Get diversity parameters
        diversity_params = self._get_diversity_parameters(intent)
        max_tracks_per_artist = diversity_params['max_tracks_per_artist']
        
        # Apply diversity filtering
        return self._apply_diversity_filtering(
            candidates, 
            max_tracks_per_artist, 
            candidate_limit, 
            target_artist_from_override, 
            intent
        )
    
    def _count_valid_candidates(self, candidates: List[Dict[str, Any]]) -> int:
        """Count valid candidates (non-None with required fields)."""
        valid_count = 0
        for candidate in candidates:
            if (candidate and 
                isinstance(candidate, dict) and 
                candidate.get('artist') and 
                candidate.get('name')):
                valid_count += 1
        return valid_count
    
    def _apply_diversity_filtering(
        self, 
        candidates: List[Dict[str, Any]], 
        max_tracks_per_artist: int, 
        candidate_limit: int, 
        target_artist_from_override: str, 
        intent: str
    ) -> List[Dict[str, Any]]:
        """
        Apply diversity filtering with artist limits.
        
        Args:
            candidates: List of candidate tracks
            max_tracks_per_artist: Maximum tracks per artist
            candidate_limit: Maximum total candidates
            target_artist_from_override: Target artist from context
            intent: Intent string for special handling
            
        Returns:
            List of filtered candidates with diversity applied
        """
        if not candidates:
            return []
        
        artist_track_count = {}
        filtered_candidates = []
        
        for candidate in candidates:
            if not candidate or not isinstance(candidate, dict):
                continue
            
            artist = candidate.get('artist', '').strip()
            if not artist:
                continue
            
            artist_lower = artist.lower()
            
            # Special handling for target artist from context override
            if target_artist_from_override:
                target_lower = target_artist_from_override.lower()
                is_target_artist = (target_lower in artist_lower or 
                                  artist_lower in target_lower)
                
                if is_target_artist:
                    # Allow more tracks from target artist
                    current_count = artist_track_count.get(artist_lower, 0)
                    if current_count < (max_tracks_per_artist * 2):  # Double limit for target
                        artist_track_count[artist_lower] = current_count + 1
                        filtered_candidates.append(candidate)
                        continue
            
            # Regular diversity filtering
            current_count = artist_track_count.get(artist_lower, 0)
            if current_count < max_tracks_per_artist:
                artist_track_count[artist_lower] = current_count + 1
                filtered_candidates.append(candidate)
            
            # Stop if we've reached the candidate limit
            if len(filtered_candidates) >= candidate_limit:
                break
        
        self.logger.info(
            f"Diversity filtering applied",
            original_count=len(candidates),
            filtered_count=len(filtered_candidates),
            unique_artists=len(artist_track_count),
            max_per_artist=max_tracks_per_artist
        )
        
        return filtered_candidates[:candidate_limit]
    
    def balance_genre_diversity(
        self, 
        candidates: List[Dict[str, Any]], 
        target_genres: List[str] = None,
        max_per_genre: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Balance genre diversity in candidates.
        
        Args:
            candidates: List of candidate tracks
            target_genres: Optional target genres to prioritize
            max_per_genre: Maximum tracks per genre
            
        Returns:
            List of candidates with balanced genre diversity
        """
        if not candidates:
            return []
        
        genre_track_count = {}
        balanced_candidates = []
        
        # Prioritize target genres if specified
        if target_genres:
            target_genre_lower = [g.lower() for g in target_genres]
            
            # First pass: collect tracks from target genres
            for candidate in candidates:
                candidate_genres = [g.lower() for g in candidate.get('genres', [])]
                
                # Check if candidate matches target genres
                matches_target = any(
                    any(tg in cg or cg in tg for cg in candidate_genres)
                    for tg in target_genre_lower
                )
                
                if matches_target:
                    primary_genre = candidate_genres[0] if candidate_genres else 'unknown'
                    current_count = genre_track_count.get(primary_genre, 0)
                    
                    if current_count < max_per_genre:
                        genre_track_count[primary_genre] = current_count + 1
                        balanced_candidates.append(candidate)
        
        # Second pass: fill remaining slots with other genres
        for candidate in candidates:
            if candidate in balanced_candidates:
                continue
            
            candidate_genres = candidate.get('genres', [])
            primary_genre = candidate_genres[0].lower() if candidate_genres else 'unknown'
            
            current_count = genre_track_count.get(primary_genre, 0)
            if current_count < max_per_genre:
                genre_track_count[primary_genre] = current_count + 1
                balanced_candidates.append(candidate)
        
        self.logger.info(
            f"Genre diversity applied",
            original_count=len(candidates),
            balanced_count=len(balanced_candidates),
            unique_genres=len(genre_track_count)
        )
        
        return balanced_candidates
    
    def optimize_discovery_variety(
        self, 
        candidates: List[Dict[str, Any]], 
        variety_factors: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimize variety in discovery candidates using multiple factors.
        
        Args:
            candidates: List of candidate tracks
            variety_factors: Optional weights for variety factors
            
        Returns:
            List of candidates optimized for variety
        """
        if not candidates:
            return []
        
        # Default variety factors
        if variety_factors is None:
            variety_factors = {
                'artist_diversity': 0.4,
                'genre_diversity': 0.3,
                'popularity_spread': 0.2,
                'temporal_diversity': 0.1
            }
        
        # Calculate variety scores for each candidate
        variety_scored = []
        
        for i, candidate in enumerate(candidates):
            variety_score = self._calculate_variety_score(
                candidate, candidates[:i], variety_factors
            )
            
            candidate_copy = candidate.copy()
            candidate_copy['variety_score'] = variety_score
            variety_scored.append(candidate_copy)
        
        # Sort by variety score (higher is better)
        variety_scored.sort(key=lambda x: x.get('variety_score', 0), reverse=True)
        
        self.logger.info(f"Variety optimization applied to {len(candidates)} candidates")
        return variety_scored
    
    def _calculate_variety_score(
        self, 
        candidate: Dict[str, Any], 
        previous_candidates: List[Dict[str, Any]], 
        variety_factors: Dict[str, float]
    ) -> float:
        """Calculate variety score for a candidate relative to previous selections."""
        if not previous_candidates:
            return 1.0  # First candidate gets full variety score
        
        score = 0.0
        
        # Artist diversity
        candidate_artist = candidate.get('artist', '').lower()
        previous_artists = [c.get('artist', '').lower() for c in previous_candidates]
        
        if candidate_artist not in previous_artists:
            score += variety_factors.get('artist_diversity', 0.4)
        
        # Genre diversity
        candidate_genres = [g.lower() for g in candidate.get('genres', [])]
        previous_genres = []
        for c in previous_candidates:
            previous_genres.extend([g.lower() for g in c.get('genres', [])])
        
        genre_overlap = len(set(candidate_genres) & set(previous_genres))
        genre_diversity = 1.0 - (genre_overlap / max(len(candidate_genres), 1))
        score += genre_diversity * variety_factors.get('genre_diversity', 0.3)
        
        # Popularity spread
        candidate_listeners = candidate.get('listeners', 0) or 0
        previous_listeners = [c.get('listeners', 0) or 0 for c in previous_candidates]
        
        if previous_listeners:
            avg_previous = sum(previous_listeners) / len(previous_listeners)
            popularity_diff = abs(candidate_listeners - avg_previous) / max(avg_previous, 1)
            popularity_variety = min(popularity_diff / 1000000, 1.0)  # Normalize
            score += popularity_variety * variety_factors.get('popularity_spread', 0.2)
        
        return min(score, 1.0) 