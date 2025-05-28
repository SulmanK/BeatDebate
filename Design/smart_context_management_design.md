# Smart Context Management System Design

## Problem Statement

BeatDebate needs intelligent conversation context management to handle multi-turn music discovery conversations effectively. The challenge is knowing when to maintain context vs. when to reset it based on user intent changes.

**Key Issues Solved:**
- **Intent Switching Detection**: "Music like Mk.gee" → "Music like Chief Keef" should reset artist context
- **Conversation Continuity**: "More like that" should maintain context  
- **Context Staleness**: Old conversations should not pollute new requests
- **Preference Evolution**: Track how user preferences evolve within a session

## Solution Architecture

### Core Components

#### 1. SmartContextManager
**Purpose**: Central intelligence for context decisions
**Key Methods**:
- `analyze_context_decision()` - Main decision engine
- `update_context_after_recommendation()` - Update context post-recommendation
- `get_context_summary()` - Session state overview

#### 2. Context Decision States
```python
class ContextState(Enum):
    NEW_SESSION = "new_session"           # No existing context
    CONTINUING = "continuing"             # Maintain current context  
    INTENT_SWITCH = "intent_switch"       # Major intent change detected
    PREFERENCE_REFINEMENT = "preference_refinement"  # Refining within same context
    RESET_NEEDED = "reset_needed"         # Explicit reset or stale context
```

#### 3. Intent Detection Engine
**Pattern-Based Detection**:
- **Artist Similarity**: `"music like X"`, `"similar to X"`, `"artists like X"`
- **Genre Exploration**: `"some jazz music"`, `"explore electronic"`
- **Conversation Continuation**: `"more like that"`, `"also"`, `"what about"`
- **Feedback Response**: `"i like/don't like"`, `"too fast/slow"`

**LLM-Enhanced Detection**:
- Leverages Gemini for complex intent understanding
- Extracts artists, genres, moods, activities
- Provides confidence scores

#### 4. Context Decision Logic

```python
def _make_context_decision(
    query_analysis,      # Current query intent
    intent_analysis,     # Intent change detection  
    temporal_analysis,   # Time-based relevance
    continuity_analysis  # Conversation continuity signals
) -> Dict[str, Any]:
```

**Decision Flow**:
1. **Temporal Check**: Is context stale? (>30 min) → RESET
2. **Explicit Triggers**: "never mind", "actually" → RESET  
3. **Intent Analysis**: Major change → INTENT_SWITCH
4. **Continuity Signals**: Strong continuation → CONTINUING
5. **Default**: Preference refinement or maintain

### Context Actions

#### 1. Reset Context (`reset_context`)
**When**: Major intent changes, explicit triggers, stale context
**Action**: Clear all session context, start fresh
**Example**: "Music like Mk.gee" → "Actually, something for working out"

#### 2. Maintain Context (`maintain_context`) 
**When**: Strong continuity signals, same intent
**Action**: Use full existing context
**Example**: "Music like Mk.gee" → "More tracks like that"

#### 3. Modify Context (`modify_context`)
**When**: Preference refinement within same domain
**Action**: Keep general preferences, update specific parts
**Example**: "Upbeat indie music" → "Something more mellow but still indie"

#### 4. Partial Reset (`partial_reset`)
**When**: Artist switch within same intent type
**Action**: Keep genre/mood preferences, reset artist-specific context
**Example**: "Music like Mk.gee" → "Music like Chief Keef"

## Implementation Details

### 1. Integration with Recommendation Engine

```python
# Enhanced recommendation flow
async def get_recommendations(query, session_id):
    # 1. Get LLM understanding 
    llm_understanding = await self._get_llm_query_understanding(query)
    
    # 2. Analyze context decision
    context_decision = await self.smart_context_manager.analyze_context_decision(
        current_query=query,
        session_id=session_id, 
        llm_understanding=llm_understanding
    )
    
    # 3. Prepare context for agents
    conversation_context = context_decision.get("context_to_use")
    
    # 4. Process through agent workflow
    final_state = await self.process_query(
        query, session_id, llm_understanding, conversation_context
    )
    
    # 5. Update context after recommendations
    await self.smart_context_manager.update_context_after_recommendation(...)
```

### 2. Context Decision Examples

| Query Sequence | Context Decision | Reasoning |
|---|---|---|
| 1. "Music like Mk.gee"<br/>2. "More like that" | NEW_SESSION → CONTINUING | Explicit continuation signal |
| 1. "Music like Mk.gee"<br/>2. "Music like Chief Keef" | NEW_SESSION → INTENT_SWITCH | Artist target changed |
| 1. "Upbeat music"<br/>2. "Something more chill" | NEW_SESSION → PREFERENCE_REFINEMENT | Mood refinement within music domain |
| 1. "Music like Mk.gee"<br/>2. "Actually, workout music" | NEW_SESSION → RESET_NEEDED | Explicit reset trigger + major intent change |

### 3. Temporal Context Management

**Context Decay**: 
- **Fresh**: 0-5 minutes (relevance: 1.0)
- **Recent**: 5-15 minutes (relevance: 0.7-0.9) 
- **Aging**: 15-30 minutes (relevance: 0.3-0.7)
- **Stale**: >30 minutes (relevance: 0.0) → Auto-reset

**Context Health Monitoring**:
```python
{
    "temporal_relevance": 0.85,
    "is_stale": false,
    "minutes_active": 12.3
}
```

### 4. Continuity Signal Detection

**Strong Continuity** (score > 0.6):
- "more", "also", "another", "similar"
- "like that", "what about"
- Reference words: "that", "this", "those"

**Weak Continuity** (score < 0.2):
- No continuation words
- No entity overlap
- Different semantic domain

## Testing Strategy

### Test Scenarios

1. **Basic Continuity**
   - "Music like Mk.gee" → "More tracks like that"
   - Expected: CONTINUING

2. **Artist Switch** 
   - "Music like Mk.gee" → "Music like Chief Keef"
   - Expected: INTENT_SWITCH (partial_reset)

3. **Activity Switch**
   - "Music like Mk.gee" → "Music for working out"  
   - Expected: INTENT_SWITCH (reset_context)

4. **Explicit Reset**
   - "Music like Mk.gee" → "Actually, never mind. Something completely different"
   - Expected: RESET_NEEDED

5. **Temporal Decay**
   - "Music like Mk.gee" → [30+ minutes] → "More music"
   - Expected: RESET_NEEDED (stale context)

### Test Script

```bash
# Run context management tests
python test_smart_context.py
```

## API Integration

### New Endpoints

#### GET `/sessions/{session_id}/context`
**Purpose**: Get current context status
**Response**:
```json
{
    "session_id": "session_123",
    "context_summary": {
        "status": "active",
        "interaction_count": 3,
        "context_health": {
            "temporal_relevance": 0.85,
            "is_stale": false,
            "minutes_active": 12.3
        },
        "preference_summary": {
            "preferred_genres": 2,
            "preferred_artists": 1,
            "activity_patterns": 0
        }
    }
}
```

### Enhanced Recommendation Response
```json
{
    "recommendations": [...],
    "reasoning_log": [
        "Smart Context: Artist similarity target changed",
        "PlannerAgent: Strategic planning completed",
        ...
    ],
    "agent_coordination_log": [
        "SmartContext: partial_reset (confidence: 0.80)",
        "PlannerAgent: Strategic planning completed",
        ...
    ]
}
```

## Benefits

### For Users
- **Seamless Conversations**: Context maintained when appropriate
- **Smart Resets**: Fresh start when switching topics
- **Preference Learning**: System learns from conversation patterns
- **Natural Interactions**: No need to repeat context unnecessarily

### For Developers  
- **Predictable Behavior**: Clear decision logic
- **Debugging Support**: Full reasoning logs
- **Extensible Design**: Easy to add new intent types
- **Performance Optimized**: Context pruning prevents bloat

### For Recommendation Quality
- **Better Relevance**: Context-aware recommendations
- **Reduced Confusion**: Clean context boundaries  
- **Progressive Refinement**: Context builds intelligently
- **User Intent Alignment**: Matches user's conversational flow

## Future Enhancements

1. **Advanced Intent Detection**
   - Genre relationship understanding
   - Mood progression patterns
   - Activity transition logic

2. **User-Specific Learning**
   - Personal conversation patterns
   - Preferred context retention styles
   - Custom reset triggers

3. **Multi-Modal Context**
   - Audio preference analysis
   - Visual/UI interaction patterns
   - Cross-session learning

4. **Context Explanation**
   - Why context was maintained/reset
   - User feedback on context decisions
   - Confidence calibration 