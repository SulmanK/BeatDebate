# Context-Aware Planner Agent Fix

## Problem Statement

Follow-up queries like "More tracks" are being incorrectly classified as discovery queries instead of respecting conversation context, resulting in irrelevant recommendations.

**Root Cause**: The planner agent ignores context overrides and always runs fresh query understanding on raw queries, even when the conversation context service has correctly identified them as follow-ups with preserved entities and intent.

**Impact**: 
- Users get random discovery tracks (mostly Radiohead) instead of more Michael Jackson R&B tracks
- Broken conversation flow where "More tracks" doesn't build on previous queries
- Poor user experience with conversational music discovery

**Success Criteria**:
- "More tracks" after "Songs by Michael Jackson that are R&B" returns more Michael Jackson R&B tracks
- Fresh queries still work normally through query understanding
- No regression in existing functionality

## Current Architecture Analysis

### Current Flow (Broken)
1. **Context Analyzer** ✅ correctly identifies "More tracks" as follow-up, creates context override
2. **Planner Agent** ❌ ignores context override, runs fresh query understanding on "More tracks"
3. **Query Understanding** ❌ returns `intent: 'discovery'`, `entities: {artists: [], genres: []}`
4. **Discovery Agent** ✅ receives context override, tries to patch entities after wrong planning

### Log Evidence
```
Line 1363: 🔧 LLM RAW RESPONSE: {'intent': 'discovery', 'musical_entities': {'artists': [], 'genres': [], 'tracks': [], 'moods': []}
Line 1376: Converted QueryUnderstanding to entities [artists_count=0, genres_count=0, intent=discovery]
Line 1393: 🔧 DEBUG: context_override data: {'intent_override': 'hybrid_artist_genre', 'target_entity': 'Michael Jackson', 'required_genres': ['R&B']}
```

## Solution Design

### New Flow (Fixed)
1. **Context Analyzer** ✅ correctly identifies follow-up, creates context override
2. **Planner Agent** ✅ checks for context override before query understanding
3. **Context Override Path** ✅ uses preserved entities and intent from context
4. **Fresh Query Path** ✅ runs normal query understanding for new queries

### Architecture Changes

#### 1. Context-Aware Planner Agent

```python
class PlannerAgent(BaseAgent):
    async def process(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """Process user query with context-awareness."""
        try:
            self.logger.info("Starting planner agent processing")
            
            # 🚀 NEW: Check for context override before query understanding
            if hasattr(state, 'context_override') and state.context_override:
                context_override = state.context_override
                if self._is_followup_with_preserved_context(context_override):
                    self.logger.info("Context override detected, using preserved entities and intent")
                    
                    # Use preserved context instead of fresh query understanding
                    query_understanding = self._create_understanding_from_context(
                        state.user_query, context_override
                    )
                    entities = self._create_entities_from_context(context_override)
                    
                    state.query_understanding = query_understanding
                    state.entities = entities
                    
                    # Mark that we used context override
                    self._context_override_applied = True
                    
                else:
                    # Context override exists but not a preserved context case
                    query_understanding = await self._understand_user_query(state.user_query)
                    entities = self._convert_understanding_to_entities(query_understanding)
                    
                    state.query_understanding = query_understanding
                    state.entities = entities
            else:
                # No context override - fresh query understanding
                query_understanding = await self._understand_user_query(state.user_query)
                entities = self._convert_understanding_to_entities(query_understanding)
                
                state.query_understanding = query_understanding
                state.entities = entities
            
            # Continue with rest of planner logic (task analysis, strategy creation, etc.)
            # ... existing code ...
            
        except Exception as e:
            # ... existing error handling ...
```

#### 2. Context Override Detection

```python
def _is_followup_with_preserved_context(self, context_override: Dict) -> bool:
    """
    Check if context override contains preserved entities that should skip query understanding.
    
    Returns True for follow-ups with preserved entities like:
    - Artist deep dives with preserved genres
    - Style continuations with preserved context
    - Hybrid queries with preserved filters
    """
    if not isinstance(context_override, dict):
        return False
    
    # Check for follow-up indicators
    is_followup = context_override.get('is_followup', False)
    has_preserved_entities = 'preserved_entities' in context_override
    has_intent_override = 'intent_override' in context_override
    
    # Specific follow-up types that need preserved context
    followup_types_with_context = [
        'hybrid_artist_genre',  # "More tracks" after "Songs by X that are Y"
        'artist_style_refinement',  # "More jazzy tracks" after artist query
        'style_continuation'  # Continue with same style/mood
    ]
    
    intent_override = context_override.get('intent_override')
    
    return (is_followup and 
            has_preserved_entities and 
            has_intent_override and
            intent_override in followup_types_with_context)
```

#### 3. Context-to-Entities Conversion

```python
def _create_understanding_from_context(self, user_query: str, context_override: Dict) -> QueryUnderstanding:
    """Create QueryUnderstanding from preserved context override."""
    preserved_entities = context_override.get('preserved_entities', {})
    intent_override = context_override.get('intent_override', 'discovery')
    confidence = context_override.get('confidence', 0.9)
    
    # Extract preserved entities
    artists = self._extract_entity_names(preserved_entities.get('artists', {}).get('primary', []))
    genres = self._extract_entity_names(preserved_entities.get('genres', {}).get('primary', []))
    moods = self._extract_entity_names(preserved_entities.get('moods', {}).get('primary', []))
    
    # Map intent override to QueryIntent enum
    intent_mapping = {
        'hybrid_artist_genre': QueryIntent.HYBRID,
        'artist_style_refinement': QueryIntent.HYBRID, 
        'style_continuation': QueryIntent.GENRE_MOOD,
        'artist_deep_dive': QueryIntent.ARTIST_SIMILARITY
    }
    
    intent = intent_mapping.get(intent_override, QueryIntent.DISCOVERY)
    
    self.logger.info(
        f"🎯 Created understanding from context: intent={intent.value}, "
        f"artists={artists}, genres={genres}, confidence={confidence}"
    )
    
    return QueryUnderstanding(
        intent=intent,
        confidence=confidence,
        artists=artists,
        genres=genres,
        moods=moods,
        activities=[],
        original_query=user_query,
        normalized_query=user_query.lower(),
        reasoning=f"Context override: {intent_override} follow-up with preserved entities"
    )

def _create_entities_from_context(self, context_override: Dict) -> Dict[str, Any]:
    """Create entities structure from context override."""
    preserved_entities = context_override.get('preserved_entities', {})
    
    # Extract preserved entity data
    artists_data = preserved_entities.get('artists', {})
    genres_data = preserved_entities.get('genres', {})
    moods_data = preserved_entities.get('moods', {})
    
    # Convert to proper entities structure
    entities = {
        "musical_entities": {
            "artists": {
                "primary": self._extract_entity_names(artists_data.get('primary', [])),
                "similar_to": []
            },
            "genres": {
                "primary": self._extract_entity_names(genres_data.get('primary', [])),
                "secondary": []
            },
            "tracks": {
                "primary": [],
                "referenced": []
            },
            "moods": {
                "primary": self._extract_entity_names(moods_data.get('primary', [])),
                "energy": [],
                "emotion": []
            }
        },
        "contextual_entities": {
            "activities": {
                "physical": [],
                "mental": [],
                "social": []
            },
            "temporal": {
                "decades": [],
                "periods": []
            }
        },
        "confidence_scores": {
            "overall": context_override.get('confidence', 0.9)
        },
        "extraction_method": "context_override_preserved",
        "intent_analysis": {
            "intent": context_override.get('intent_override', 'hybrid'),
            "confidence": context_override.get('confidence', 0.9),
            "context_override_applied": True
        }
    }
    
    self.logger.info(
        f"🎯 Created entities from context: "
        f"artists={len(entities['musical_entities']['artists']['primary'])}, "
        f"genres={len(entities['musical_entities']['genres']['primary'])}"
    )
    
    return entities

def _extract_entity_names(self, entity_list: List) -> List[str]:
    """Extract names from entity list that may contain dicts or strings."""
    names = []
    for item in entity_list:
        if isinstance(item, dict):
            names.append(item.get('name', str(item)))
        elif isinstance(item, str):
            names.append(item)
        else:
            names.append(str(item))
    return names
```

## Implementation Plan

### Phase 1: Core Context Detection
1. Add context override detection to planner agent
2. Implement preserved entity extraction
3. Add logging for debugging context usage

### Phase 2: Context-to-Entities Conversion  
1. Implement understanding creation from context
2. Implement entities creation from context
3. Handle intent mapping from context override

### Phase 3: Integration & Testing
1. Test with "More tracks" scenario
2. Verify backward compatibility with fresh queries
3. Test edge cases (malformed context overrides)

### Phase 4: Monitoring & Validation
1. Add metrics for context override usage
2. Monitor recommendation quality
3. Validate conversation flow improvements

## Testing Strategy

### Test Cases

#### 1. Follow-up Query Success
```
Query 1: "Songs by Michael Jackson that are R&B"
Expected: Michael Jackson R&B tracks returned

Query 2: "More tracks" 
Expected: More Michael Jackson R&B tracks (not random discovery)
```

#### 2. Fresh Query Compatibility
```
Query: "Electronic music for studying"
Expected: Normal query understanding, genre/mood-based recommendations
```

#### 3. Edge Cases
```
- Context override with missing preserved_entities
- Malformed intent_override values
- Context override with empty entities
```

### Success Metrics
- Follow-up queries maintain conversation context (95%+ success rate)
- Fresh queries unaffected (no regression)
- User satisfaction with "More tracks" functionality

## Risk Mitigation

### Backward Compatibility
- Fresh queries (no context override) use existing query understanding path
- Malformed context overrides fall back to normal query understanding
- All existing planner functionality preserved

### Error Handling
- Invalid context overrides log warnings but don't break workflow
- Missing entities in context override fall back to empty entities
- Intent mapping failures default to DISCOVERY intent

### Performance
- Context override path is faster (skips LLM query understanding)
- No additional API calls or LLM overhead for follow-ups
- Minimal code complexity increase

## Metrics & Monitoring

### Key Metrics
1. **Context Override Usage Rate**: % of queries using context override
2. **Follow-up Success Rate**: % of follow-ups maintaining conversation context  
3. **Query Understanding Accuracy**: Fresh vs context-override query accuracy
4. **User Engagement**: Session length and follow-up query frequency

### Logging Enhancements
- Log when context override is detected and used
- Log preserved entities and intent mappings
- Log fallback cases for debugging

## Dependencies

### No External Dependencies
- Uses existing context override structure
- Uses existing QueryUnderstanding and entities formats
- No new API calls or services required

### Internal Dependencies
- Enhanced recommendation service context override creation
- Existing conversation context service functionality
- Existing planner agent query understanding flow

## Deployment Strategy

### Rollout Plan
1. **Dev Testing**: Implement and test in development environment
2. **A/B Testing**: Split traffic to compare context-aware vs original planner
3. **Gradual Rollout**: Start with 10% of follow-up queries, monitor metrics
4. **Full Deployment**: Roll out to 100% after validation

### Rollback Plan
- Feature flag to disable context override detection
- Fallback to original query understanding for all queries
- No database or API changes required for rollback

---

## Technical Implementation Notes

### Code Structure
- New methods added to existing `PlannerAgent` class
- No breaking changes to existing interfaces
- Context override detection is additive functionality

### Testing Requirements  
- Unit tests for context override detection
- Integration tests for end-to-end follow-up flow
- Performance tests to ensure no regression

### Documentation Updates
- Update planner agent documentation with context-aware behavior
- Add troubleshooting guide for context override issues
- Update system architecture diagrams 