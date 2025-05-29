"""
Ranking Logic for Judge Agent

Provides specialized ranking algorithms, scoring components, and selection
strategies for the judge agent's final recommendation evaluation.
"""

from typing import Dict, List, Any, Tuple, Optional
import structlog

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
        
        logger.debug("RankingLogic initialized")
    
    def rank_recommendations(
        self,
        candidates: List[Tuple[Any, Dict[str, float]]],
        context: Dict[str, Any],
        ranking_strategy: str = 'balanced'
    ) -> List[Tuple[Any, Dict[str, float]]]:
        """
        Rank recommendations using specified strategy.
        
        Args:
            candidates: List of (recommendation, scores) tuples
            context: Ranking context (entities, intent, etc.)
            ranking_strategy: Ranking strategy to use
            
        Returns:
            Ranked list of recommendations
        """
        if ranking_strategy == 'quality_focused':
            return self._rank_by_quality(candidates, context)
        elif ranking_strategy == 'diversity_focused':
            return self._rank_by_diversity(candidates, context)
        elif ranking_strategy == 'intent_focused':
            return self._rank_by_intent(candidates, context)
        else:  # balanced
            return self._rank_balanced(candidates, context)
    
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
        """Initialize ranking weight configurations."""
        return {
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
            },
            'discovery_intent': {
                'intent': 0.40,
                'diversity': 0.30,
                'quality': 0.20,
                'relevance': 0.10
            },
            'genre_mood_intent': {
                'intent': 0.40,
                'relevance': 0.35,
                'quality': 0.25
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