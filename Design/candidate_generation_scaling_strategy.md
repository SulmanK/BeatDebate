# Candidate Generation Scaling Strategy
## Problem Statement

The current system generates insufficient candidates for follow-up queries, leading to poor user experiences where only 1-2 tracks are returned instead of the requested 10. This occurs because:

1. **Static Generation**: Fixed 20 candidates regardless of query type
2. **Heavy Filtering**: Duplicate filtering removes 50%+ of candidates
3. **Insufficient Diversity**: Single-source generation limits variety

## Solution: Dynamic Multi-Stage Generation

### Stage 1: Intelligent Candidate Pool Sizing

```python
def calculate_candidate_pool_size(query_type: str, history_length: int, target_count: int) -> int:
    """Calculate optimal candidate pool size based on query characteristics."""
    
    # Base multipliers by query type
    multipliers = {
        'new_query': 3,      # 3x for new queries
        'followup': 12,      # 12x for follow-ups (heavy duplicate filtering)
        'by_artist': 8,      # 8x for artist-specific queries
        'discovery': 5       # 5x for discovery queries
    }
    
    # Adjust for history length (more history = more duplicates to filter)
    history_multiplier = 1 + (history_length / 50)  # +1x per 50 tracks in history
    
    # Calculate final pool size
    base_multiplier = multipliers.get(query_type, 3)
    final_multiplier = base_multiplier * history_multiplier
    
    return max(target_count * final_multiplier, 100)  # Minimum 100 candidates
```

### Stage 2: Multi-Source Generation Strategy

```python
def generate_candidates_multi_source(
    query_intent: str, 
    entities: dict, 
    pool_size: int
) -> List[dict]:
    """Generate candidates from multiple sources for robustness."""
    
    # Distribute candidate generation across sources
    sources = {
        'primary_artist': 0.4,      # 40% from target artist
        'similar_artists': 0.25,    # 25% from similar artists  
        'collaborative': 0.2,       # 20% from collaborative filtering
        'content_based': 0.15       # 15% from content similarity
    }
    
    candidates = []
    for source, ratio in sources.items():
        source_count = int(pool_size * ratio)
        source_candidates = generate_from_source(source, entities, source_count)
        candidates.extend(source_candidates)
    
    return candidates[:pool_size]  # Cap at requested size
```

### Stage 3: Progressive Filtering Pipeline

```python
def progressive_filtering_pipeline(
    candidates: List[dict], 
    recently_shown: List[str],
    target_count: int
) -> List[dict]:
    """Apply filters progressively to preserve candidate pool."""
    
    # Stage 3a: Hard duplicate removal (exact matches)
    candidates = remove_exact_duplicates(candidates, recently_shown)
    
    # Stage 3b: Quality threshold (only remove very low quality)
    candidates = filter_by_quality_threshold(candidates, min_threshold=0.1)
    
    # Stage 3c: Soft duplicate removal (similar tracks, keep best)
    candidates = soft_duplicate_removal(candidates, threshold=0.85)
    
    # Stage 3d: Final diversity selection
    final_candidates = diversity_selection(candidates, target_count)
    
    return final_candidates
```

### Stage 4: Fallback and Recovery Mechanisms

```python
def candidate_generation_with_fallbacks(
    query_intent: str,
    entities: dict,
    target_count: int,
    recently_shown: List[str]
) -> List[dict]:
    """Robust candidate generation with multiple fallback strategies."""
    
    # Primary attempt
    pool_size = calculate_candidate_pool_size(query_intent, len(recently_shown), target_count)
    candidates = generate_candidates_multi_source(query_intent, entities, pool_size)
    
    # Apply filtering
    filtered = progressive_filtering_pipeline(candidates, recently_shown, target_count)
    
    # Fallback strategies if insufficient results
    if len(filtered) < target_count * 0.7:  # Less than 70% of target
        
        # Fallback 1: Expand to genre/mood similarity
        if query_intent == 'by_artist':
            expanded_candidates = generate_genre_expanded(entities, pool_size // 2)
            candidates.extend(expanded_candidates)
            filtered = progressive_filtering_pipeline(candidates, recently_shown, target_count)
        
        # Fallback 2: Relax duplicate filtering
        if len(filtered) < target_count * 0.5:  # Less than 50% of target
            relaxed_filtered = relaxed_duplicate_filtering(candidates, recently_shown, target_count)
            filtered = relaxed_filtered
        
        # Fallback 3: Include older tracks from history (if still insufficient)
        if len(filtered) < target_count * 0.3:  # Less than 30% of target
            older_tracks = include_older_history_tracks(entities, recently_shown, target_count)
            filtered.extend(older_tracks)
    
    return filtered[:target_count]
```

## Implementation Plan

### Phase 1: Dynamic Pool Sizing
- [ ] Implement intelligent pool size calculation
- [ ] Add query type detection for multipliers
- [ ] Test with current bottleneck scenarios

### Phase 2: Multi-Source Generation  
- [ ] Implement source-distributed generation
- [ ] Add similar artist and genre expansion
- [ ] Integrate collaborative filtering fallbacks

### Phase 3: Progressive Filtering
- [ ] Replace binary filtering with progressive approach
- [ ] Implement soft duplicate detection
- [ ] Add quality threshold adjustments

### Phase 4: Monitoring and Optimization
- [ ] Add candidate pool metrics to logging
- [ ] Monitor conversion rates by query type
- [ ] A/B test different multiplier values

## Success Metrics

- **Fulfillment Rate**: % of queries returning target recommendation count
- **Diversity Score**: Average artist/genre diversity in recommendations
- **User Satisfaction**: Skip rates and engagement metrics
- **Fallback Usage**: % of queries requiring fallback strategies

## Expected Impact

- **Immediate**: 90%+ fulfillment rate for follow-up queries
- **Short-term**: 25% improvement in recommendation diversity
- **Long-term**: Better user retention and session length 