# Universal Follow-up System Design

## Overview

This document outlines the systematic approach for implementing proper follow-up behavior across all query intents in the BeatDebate recommendation system. Based on the successful implementation for `discovering_serendipity`, this design provides a blueprint for extending follow-up functionality to all intents.

## Problem Statement

Currently, follow-up queries like "more tracks", "show me more", or "give me different ones" don't work consistently across all intents because:

1. **Intent Mapping Gaps**: Some intents are missing from context analyzer mappings
2. **Pool Persistence Issues**: Not all intents are configured for candidate pool storage
3. **Context Handler Blind Spots**: Follow-up detection doesn't handle all intent patterns
4. **Strategy Planning Inconsistencies**: Pool size multipliers and generation logic varies

## System Architecture

### Core Components Flow

```
User Query → Context Handler → Intent Orchestration → Strategy Planning → Candidate Generation → Pool Storage
     ↓
Follow-up Query → Context Handler → Intent Preservation → Pool Retrieval → Judge Agent → Results
```

## Implementation Checklist

### 1. Context Analyzer Intent Mapping (`src/agents/planner/context_analyzer.py`)

**File**: `src/agents/planner/context_analyzer.py`
**Method**: `create_understanding_from_effective_intent`
**Lines**: ~270-280

#### Current State
```python
intent_mapping = {
    'artist_similarity': QueryIntent.ARTIST_SIMILARITY,
    'genre_exploration': QueryIntent.GENRE_MOOD,
    'mood_matching': QueryIntent.GENRE_MOOD,
    'activity_context': QueryIntent.GENRE_MOOD,
    'discovery': QueryIntent.DISCOVERY,
    'discovering_serendipity': QueryIntent.DISCOVERING_SERENDIPITY,  # ✅ Added
    'hybrid': QueryIntent.HYBRID,
    'by_artist': QueryIntent.BY_ARTIST,
    'by_artist_underground': QueryIntent.BY_ARTIST_UNDERGROUND
}
```

#### Required Additions
For each new intent, add mapping:
```python
'your_new_intent': QueryIntent.YOUR_NEW_INTENT,
```

### 2. Pool Persistence Configuration (`src/agents/components/unified_candidate_generator.py`)

**File**: `src/agents/components/unified_candidate_generator.py`
**Method**: `__init__`
**Lines**: ~54-60

#### Current State
```python
self.pool_persistence_intents = [
    QueryIntent.BY_ARTIST, 
    QueryIntent.BY_ARTIST_UNDERGROUND,
    QueryIntent.ARTIST_SIMILARITY, 
    QueryIntent.DISCOVERY,
    QueryIntent.DISCOVERING_SERENDIPITY,  # ✅ Added
    QueryIntent.GENRE_MOOD
]
```

#### Required Additions
For each intent that should support follow-ups:
```python
QueryIntent.YOUR_NEW_INTENT,  # Enable pool persistence for your_new_intent
```

### 3. Strategy Planning Configuration (`src/agents/planner/strategy_planner.py`)

**File**: `src/agents/planner/strategy_planner.py`

#### A. Follow-up Prone Intents (Lines ~320-325)
```python
followup_prone_intents = [
    'discovery', 'discovering_serendipity', 'similarity', 'genre_mood', 
    'by_artist', 'by_artist_underground', 'artist_similarity',
    'your_new_intent'  # Add here
]
```

#### B. Pool Size Multipliers (Lines ~330-340)
```python
if understanding.intent.value in ['discovery', 'discovering_serendipity', 'similarity', 'your_new_intent']:
    multiplier = 4  # These benefit most from large pools
elif understanding.intent.value in ['by_artist', 'by_artist_underground', 'artist_similarity']:
    multiplier = 5  # Artist-focused queries often have many good candidates
```

### 4. Context Handler Follow-up Detection (`src/services/components/context_handler.py`)

**File**: `src/services/components/context_handler.py`

#### A. Pattern Detection in `_extract_original_intent_from_history` (Lines ~478-510)

Add pattern detection for your intent:
```python
elif any(your_pattern_word in query for your_pattern_word in ['keyword1', 'keyword2', 'pattern']):
    return 'your_new_intent'
```

#### B. LLM Context Override Creation (Lines ~184-230)

Add handling for your intent:
```python
elif followup_type == 'artist_deep_dive' and original_intent == 'your_new_intent':
    # Preserve your_new_intent for follow-ups
    intent_override = 'your_new_intent'
    target_entity = 'your target entity description'
    followup_type = 'more_content'
```

#### C. Regex Fallback Logic (Lines ~280-320)

Add for both "simple_more" and "show_more" patterns:
```python
if original_intent == 'your_new_intent':
    # Preserve your_new_intent for "more tracks" follow-ups
    return {
        'is_followup': True,
        'intent_override': 'your_new_intent',
        'target_entity': 'your target entity',
        'style_modifier': None,
        'confidence': 0.9,
        'constraint_overrides': None,
        'entities': self._extract_complete_entities_from_history(conversation_history),
        'followup_type': 'your_followup_type'
    }
```

### 5. Discovery Diversity Configuration (`src/agents/discovery/discovery_diversity.py`)

**File**: `src/agents/discovery/discovery_diversity.py`
**Lines**: ~90-120

Add configuration for your intent:
```python
'your_new_intent': {
    'candidate_limit': 60,  # Adjust based on intent needs
    'max_tracks_per_artist': 3,  # Adjust based on diversity needs
    'diversity_weight': 0.8,
    'quality_threshold': 0.4
}
```

### 6. Strategy Factory Configuration (`src/agents/components/generation_strategies/factory.py`)

**File**: `src/agents/components/generation_strategies/factory.py`
**Lines**: ~50-70

Add your intent to the mapping:
```python
QueryIntent.YOUR_NEW_INTENT: ['strategy1', 'strategy2'],  # List appropriate strategies
```

## Intent-Specific Implementation Patterns

### Pattern 1: Discovery-Type Intents
**Examples**: `discovery`, `discovering_serendipity`, `genre_exploration`

**Characteristics**:
- Large candidate pools (600+ tracks)
- High diversity requirements
- Multiple generation strategies
- Pool multiplier: 4

**Implementation**:
```python
# Pool persistence: ✅ Required
# Follow-up prone: ✅ Required  
# Pattern detection: Keywords like "discover", "explore", "find"
# Target entity: "discovery exploration" or similar
```

### Pattern 2: Artist-Focused Intents
**Examples**: `by_artist`, `by_artist_underground`, `artist_similarity`

**Characteristics**:
- Very large candidate pools (750+ tracks)
- Artist-specific diversity
- Single primary strategy
- Pool multiplier: 5

**Implementation**:
```python
# Pool persistence: ✅ Required
# Follow-up prone: ✅ Required
# Pattern detection: Keywords like "by", "from", "artist", "similar"
# Target entity: Specific artist name or "similar artists"
```

### Pattern 3: Mood/Genre Intents
**Examples**: `genre_mood`, `activity_context`

**Characteristics**:
- Medium candidate pools (400+ tracks)
- Genre/mood consistency
- Multiple strategies possible
- Pool multiplier: 3

**Implementation**:
```python
# Pool persistence: ✅ Required
# Follow-up prone: ✅ Required
# Pattern detection: Keywords like "genre", "mood", "vibe", "style"
# Target entity: "genre/mood exploration"
```

### Pattern 4: Hybrid/Complex Intents
**Examples**: `hybrid`, `complex_query`

**Characteristics**:
- Variable pool sizes
- Multiple entity types
- Complex strategy combinations
- Pool multiplier: 4

**Implementation**:
```python
# Pool persistence: ✅ Required
# Follow-up prone: ✅ Required
# Pattern detection: Multiple keywords or complex patterns
# Target entity: "hybrid exploration" or specific combination
```

## Testing Checklist

For each new intent implementation:

### 1. Initial Query Test
- [ ] Intent correctly detected
- [ ] Appropriate strategies selected
- [ ] Candidate pool generated and stored
- [ ] Results returned with expected diversity

### 2. Follow-up Query Test
- [ ] "More tracks" preserves original intent
- [ ] "Show me more" preserves original intent  
- [ ] "Give me different ones" preserves original intent
- [ ] Pool retrieval works correctly
- [ ] No duplicate tracks from previous session
- [ ] Results maintain intent characteristics

### 3. Context Preservation Test
- [ ] Entities preserved across follow-ups
- [ ] Quality thresholds maintained
- [ ] Diversity requirements respected
- [ ] Performance remains acceptable

## Implementation Priority

### Phase 1: High-Impact Intents
1. `genre_mood` - Most common user pattern
2. `artist_similarity` - High follow-up likelihood
3. `by_artist` - Clear follow-up use case

### Phase 2: Specialized Intents
1. `by_artist_underground` - Niche but important
2. `hybrid` - Complex but valuable
3. Custom intents as needed

### Phase 3: Edge Cases
1. Temporal intents (if any)
2. Complex multi-entity intents
3. Conditional logic intents

## Monitoring and Validation

### Key Metrics
- Follow-up success rate per intent
- Pool retrieval performance
- User satisfaction with follow-up results
- System performance impact

### Log Monitoring
Monitor these log patterns for each intent:
```
🎯 LLM Context Override Created: {'intent_override': 'your_intent'}
🔄 Follow-up query detected - preserving session
✅ Found compatible candidate pool for intent=your_intent
```

## Common Pitfalls to Avoid

1. **Missing Intent Mapping**: Always add to context analyzer mapping
2. **Forgetting Pool Persistence**: Add to unified candidate generator list
3. **Incomplete Pattern Detection**: Add to both LLM and regex fallback logic
4. **Wrong Pool Multiplier**: Choose appropriate multiplier for intent type
5. **Missing Diversity Config**: Configure appropriate limits for intent
6. **Strategy Mismatch**: Ensure strategies align with intent characteristics

## Future Enhancements

1. **Dynamic Pool Sizing**: Adjust pool size based on user behavior
2. **Intent-Specific Diversity**: Fine-tune diversity requirements per intent
3. **Cross-Intent Follow-ups**: Handle intent transitions in follow-ups
4. **Personalized Follow-ups**: Adapt follow-up behavior to user preferences

## Conclusion

This systematic approach ensures consistent follow-up behavior across all intents. By following this checklist for each new intent, we can guarantee that users will have a seamless experience when asking for "more tracks" regardless of their original query type.

The key is maintaining consistency across all five components:
1. Context Analyzer mapping
2. Pool persistence configuration  
3. Strategy planning setup
4. Context handler detection
5. Diversity configuration

Each intent should be treated as a complete system that supports both initial queries and follow-up interactions. 