# Context-Aware Intent Override System

## Problem Statement

The current recommendation system enforces diversity constraints (max 1-2 tracks per artist) that can conflict with explicit user intent in follow-up queries. When a user first asks for "Music like Mk.gee" and then follows up with "I want more Mk.gee tracks", the system's diversity limits prevent delivering what the user explicitly requested.

**Core Issue**: Static diversity constraints don't adapt to contextual user intent, leading to suboptimal recommendations when users want to explore a specific artist more deeply.

**Value Proposition**: Enable the system to intelligently override diversity constraints when user context clearly indicates they want more of the same artist/style, improving user satisfaction and recommendation relevance.

## Requirements

### Functional Requirements
- **FR1**: Detect follow-up queries asking for "more" of specific artists from previous recommendations
- **FR2**: Override diversity constraints when user intent explicitly requests artist deep-dive
- **FR3**: Maintain session context to track previously recommended artists and user preferences
- **FR4**: Avoid recommending duplicate tracks while allowing artist expansion
- **FR5**: Provide up to 8-10 tracks from target artist when explicitly requested
- **FR6**: Gracefully fall back to standard diversity constraints when context is unclear

### Non-Functional Requirements
- **NFR1**: Context analysis should add <100ms to query processing time
- **NFR2**: Session context storage should be memory-efficient (max 1MB per session)
- **NFR3**: System should maintain backward compatibility with existing intent detection
- **NFR4**: Override logic should be explainable and debuggable

## Architecture

### High-Level Design

```
Query → Context Analysis → Intent Detection → Constraint Override → Ranking → Results
         ↑                   ↓                    ↓
    Session Context    Enhanced Intent      Dynamic Constraints
```

### Core Components

#### 1. Enhanced Query Understanding Engine
- **Purpose**: Detect follow-up queries and extract target entities
- **Location**: `src/agents/planner/query_understanding_engine.py`
- **New Methods**:
  - `analyze_followup_context(query, session_context)`
  - `extract_more_requests(query)`
  - `validate_target_entity(entity, previous_context)`

#### 2. Session Context Manager Enhancement
- **Purpose**: Track query progression and user interest signals
- **Location**: `src/services/conversation_context_service.py`
- **New Features**:
  - Artist interest tracking
  - Query progression analysis
  - Previous recommendation history

#### 3. Dynamic Constraint System
- **Purpose**: Override diversity constraints based on context
- **Location**: `src/agents/judge/ranking_logic.py`
- **New Methods**:
  - `apply_context_aware_constraints(candidates, intent, context)`
  - `get_override_constraints(intent_override, target_entity)`
  - `boost_target_entity_tracks(candidates, target)`

#### 4. Intent Override Detection
- **Purpose**: Identify when to override standard intent classification
- **New Intent Types**:
  - `ARTIST_DEEP_DIVE`: "more Mk.gee tracks"
  - `STYLE_CONTINUATION`: "more like this"
  - `PLAYLIST_EXPANSION`: "add similar tracks"

## Detailed Design

### Context Analysis Flow

```python
class ContextAnalyzer:
    def analyze_followup_intent(self, query: str, session_context: Dict) -> Dict:
        """
        Analyze if current query is a follow-up requesting more of something.
        
        Returns:
        {
            'is_followup': bool,
            'intent_override': str,  # artist_deep_dive, style_continuation, etc.
            'target_entity': str,    # artist name, style, etc.
            'confidence': float,     # 0.0-1.0
            'constraint_overrides': Dict
        }
        """
```

### Pattern Recognition

#### More Request Patterns
```python
MORE_PATTERNS = {
    'artist_deep_dive': [
        r"more (.+?) tracks",
        r"other (.+?) songs", 
        r"different (.+?) tracks",
        r"give me more (.+)",
        r"i want more (.+)"
    ],
    'style_continuation': [
        r"more like (this|that)",
        r"similar to (these|those)",
        r"keep this style"
    ]
}
```

#### Context Validation
```python
def validate_target_in_context(target: str, previous_recs: List) -> bool:
    """Check if target artist was in previous recommendations."""
    previous_artists = {rec.artist.lower() for rec in previous_recs}
    return target.lower() in previous_artists
```

### Constraint Override Logic

#### Dynamic Constraint Matrix
```python
CONTEXT_CONSTRAINT_OVERRIDES = {
    'artist_deep_dive': {
        'max_per_artist': lambda target: {target: 8, 'others': 1},
        'min_genres': 1,  # Relax genre diversity
        'prioritize_target': True,
        'novelty_threshold': 0.2  # Lower threshold for known artist
    },
    'style_continuation': {
        'max_per_artist': 2,  # Keep some diversity
        'min_genres': 2,
        'style_consistency_weight': 0.7
    }
}
```

### Session Context Data Model

#### Enhanced Session Context
```python
@dataclass
class EnhancedSessionContext:
    session_id: str
    query_history: List[Dict]
    recommendation_history: List[Dict]
    artist_interest_signals: Dict[str, float]  # artist -> interest score
    style_preferences: Dict[str, float]        # style -> preference score
    last_interaction_time: datetime
    context_confidence: float = 0.0
    
    def add_recommendation_feedback(self, track_id: str, feedback: str):
        """Track user feedback to refine interest signals."""
        
    def detect_artist_interest_spike(self, artist: str) -> bool:
        """Detect if user is showing increased interest in specific artist."""
```

## Implementation Plan

### Phase 1: Context Analysis Foundation (Week 1)
1. **Enhance Query Understanding Engine**
   - Add followup pattern detection
   - Implement target entity extraction
   - Add context validation logic

2. **Extend Session Context Service**
   - Add query progression tracking
   - Implement artist interest signals
   - Create context confidence scoring

### Phase 2: Intent Override System (Week 2)
1. **Add New Intent Types**
   - Implement `ARTIST_DEEP_DIVE` intent
   - Add `STYLE_CONTINUATION` intent
   - Create intent confidence scoring

2. **Build Dynamic Constraint System**
   - Implement constraint override logic
   - Add target entity prioritization
   - Create fallback mechanisms

### Phase 3: Integration & Testing (Week 3)
1. **Integrate with Existing Pipeline**
   - Update ranking logic to use dynamic constraints
   - Ensure backward compatibility
   - Add comprehensive logging

2. **Testing & Validation**
   - Unit tests for context analysis
   - Integration tests for full pipeline
   - Performance benchmarking

### Phase 4: Monitoring & Refinement (Week 4)
1. **Add Monitoring**
   - Context override success rates
   - User satisfaction metrics
   - Performance impact analysis

2. **Fine-tuning**
   - Adjust pattern recognition thresholds
   - Optimize constraint override parameters
   - Refine fallback logic

## Testing Strategy

### Unit Tests
- Context pattern recognition accuracy
- Intent override detection precision
- Constraint override logic validation
- Session context management

### Integration Tests
- End-to-end follow-up query processing
- Context-aware recommendation flow
- Constraint override impact on results
- Session continuity across queries

### Performance Tests
- Context analysis latency impact
- Memory usage for session storage
- Scalability with multiple sessions
- Database query optimization

### User Acceptance Tests
- **Scenario 1**: "Music like Mk.gee" → "I want more Mk.gee tracks"
- **Scenario 2**: "Indie rock recommendations" → "More like this style"
- **Scenario 3**: Complex multi-turn conversations
- **Scenario 4**: Edge cases and ambiguous requests

## Success Metrics

### Quantitative Metrics
- **Context Detection Accuracy**: >90% for clear follow-up patterns
- **User Satisfaction**: >85% for override scenarios
- **Response Time**: <150ms total latency increase
- **Override Success Rate**: >80% when confidence >0.7

### Qualitative Metrics
- **User Experience**: Seamless follow-up conversations
- **Recommendation Relevance**: Higher satisfaction for artist deep-dives
- **System Transparency**: Clear explanations for constraint overrides

## Risk Assessment

### Technical Risks
- **High**: Context analysis complexity could introduce bugs
- **Medium**: Performance impact from session context storage
- **Low**: Integration conflicts with existing intent system

### Mitigation Strategies
- Comprehensive unit test coverage
- Gradual rollout with feature flags
- Fallback to standard behavior on errors
- Performance monitoring and alerting

## Future Enhancements

### Phase 2 Extensions
- **Multi-entity Deep Dives**: "More tracks like The Strokes and Arctic Monkeys"
- **Temporal Context**: "Play something different from yesterday"
- **Mood Progression**: "Keep this energy but change the genre"

### Advanced Features
- **Learning User Patterns**: Predict when users want deep-dives
- **Cross-session Context**: Remember preferences across sessions
- **Social Context**: "More like what I shared with friends"

## Dependencies

### Internal Dependencies
- Enhanced Query Understanding Engine
- Session Context Service upgrades
- Dynamic Constraint System
- Intent classification improvements

### External Dependencies
- No new external dependencies required
- Potential database schema updates for session storage

## Conclusion

The Context-Aware Intent Override System addresses a critical gap in our recommendation engine by intelligently adapting constraints based on user context. This enhancement will significantly improve user satisfaction for follow-up queries while maintaining the quality and diversity of our recommendations.

The phased implementation approach ensures we can validate each component before full integration, minimizing risk while delivering incremental value.

---

**Document Version**: 1.0  
**Created**: 2025-05-31  
**Status**: Design Phase  
**Next Phase**: Implementation Planning 