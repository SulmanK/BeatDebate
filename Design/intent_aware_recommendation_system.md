# Intent-Aware Music Recommendation System Design

## Problem Statement

Our current music recommendation system uses one-size-fits-all scoring that creates counter-intuitive results. For "Music like Mk.gee" queries, the system favors underground tracks over actually similar artists (DIJON, Jai Paul) because novelty scoring penalizes moderate popularity. We need intent-aware scoring and workflows that adapt behavior based on user goals.

## Core Query Types & Intent Classification

### 1. **By Artist** (`"Music by Mk.gee"`)
**Goal:** Find tracks by the specified artist (their discography)
- Include ONLY the target artist's own tracks
- Focus on artist's discography depth
- Prioritize popular/quality tracks by that artist

### 2. **By Artist Underground** (`"Underground tracks by Kendrick Lamar"`) *(NEW)*
**Goal:** Discover lesser-known, underground tracks from a specific artist  
- Focus on artist's deep cuts and rare tracks
- Apply underground filtering with relaxed thresholds
- High novelty preference for obscure tracks

### 3. **Artist Similarity** (`"Music like Mk.gee"`)
**Goal:** Find artists that sound similar to the target artist
- Focus on acoustically similar artists
- Accept moderate popularity levels
- Target artist's own tracks should rank highest

### 4. **Discovery/Exploration** (`"Find me underground indie rock"`)
**Goal:** Discover truly new/unknown music
- Heavily prioritize novelty and underground tracks
- Strict popularity thresholds (<10k listeners)
- Reward experimental/uncommon tags

### 5. **Genre/Mood** (`"Upbeat electronic music"`)
**Goal:** Find tracks matching specific vibes
- Focus on genre/mood accuracy over popularity
- Energy level, tempo, mood tags critical
- Quality over novelty

### 6. **Contextual** (`"Music for studying"`)
**Goal:** Functional music for specific activities
- Optimize for activity-specific requirements
- May favor familiar, proven tracks
- Audio features (BPM, valence, energy) heavily weighted

### 7. **Hybrid Queries**
Complex queries combining multiple intents with different primary emphasis:

#### 7.1 **Discovery-Primary Hybrid** (`"Find me underground indie rock"`)
**Primary**: Discovery/Novelty | **Secondary**: Genre/Mood

#### 7.2 **Similarity-Primary Hybrid** (`"Music like Kendrick Lamar but jazzy"`)
**Primary**: Artist Similarity | **Secondary**: Genre/Mood

#### 7.3 **Genre-Primary Hybrid** (`"Upbeat indie rock with electronic elements"`)
**Primary**: Genre/Mood | **Secondary**: Discovery

### 8. **Follow-Up Queries**
Context-aware queries that reference previous recommendations:

#### 8.1 **Style Continuation** (`"More like that"`)
**Goal:** Continue with similar style/vibe to previous recommendations

#### 8.2 **Artist Deep Dive** (`"More from Mk.gee"`)
**Goal:** Explore more tracks from a specific artist mentioned in previous results

#### 8.3 **Genre/Style Refinement** (`"Similar but more upbeat"`)
**Goal:** Modify previous recommendations with style adjustments

#### 8.4 **Discovery Continuation** (`"More underground like these"`)
**Goal:** Continue discovering in the same underground/novel direction

#### 8.5 **Contextual Follow-ups** (`"More for studying like these"`)
**Goal:** Continue with same functional context but new tracks

## Complete Query Examples

### **Pure Intent Examples**

#### By Artist
- `"Music by Mk.gee"`
- `"Give me tracks by Radiohead"`
- `"Play some Beatles songs"`
- `"Mk.gee songs"`

#### By Artist Underground
- `"Discover underground tracks by Kendrick Lamar"`
- `"Find deep cuts by The Beatles"`
- `"Hidden gems by Radiohead"`

#### Artist Similarity
- `"Music like Mk.gee"`
- `"Similar artists to BROCKHAMPTON"`
- `"Songs that sound like Radiohead"`

#### Discovery
- `"Find me underground electronic music"`
- `"Something completely new and different"`
- `"Hidden gems in ambient music"`

#### Genre/Mood  
- `"Upbeat electronic music"`
- `"Sad indie songs"`
- `"Chill lo-fi hip hop"`

#### Contextual
- `"Music for studying"`
- `"Workout playlist songs"`
- `"Background music for coding"`

### **Hybrid Intent Examples**

#### Discovery-Primary Hybrid
- `"Find me underground indie rock"`
- `"New experimental jazz I haven't heard"`
- `"Hidden electronic gems"`

#### Similarity-Primary Hybrid  
- `"Music like Kendrick Lamar but jazzy"`
- `"Chill songs like Bon Iver"`
- `"Electronic music similar to Aphex Twin"`

#### Genre-Primary Hybrid
- `"Upbeat indie rock with electronic elements"`
- `"Dark ambient with jazz influences"`
- `"Aggressive punk with melodic vocals"`

#### Contextual Hybrid
- `"Study music like Max Richter"`
- `"Workout songs similar to Death Grips"`

### **Follow-Up Query Examples**

#### Style Continuation Follow-ups
- `"More like that"`
- `"More like this"`
- `"Similar to these"`
- `"Continue this style"`
- `"Keep going with this vibe"`

#### Artist Deep Dive Follow-ups
- `"More from Mk.gee"`
- `"More tracks by Mk.gee"`
- `"More by this artist"`
- `"Other songs by [Artist]"`
- `"Deep dive into [Artist]"`

#### Genre/Style Refinement Follow-ups
- `"More like [Artist] but [modifier]"`
- `"Similar but more upbeat"`
- `"More like this but heavier"`
- `"Continue but make it more chill"`
- `"Like these tracks but more electronic"`

#### Discovery Continuation Follow-ups
- `"More underground like these"`
- `"Keep discovering in this direction"`
- `"More hidden gems like this"`
- `"Continue the exploration"`
- `"Find more artists like these"`

#### Contextual Follow-ups
- `"More for studying like these"`
- `"Continue this workout vibe"`
- `"More background music like this"`
- `"Keep this energy for coding"`

## Intent-Aware Scoring Configuration

### Dynamic Scoring Weights
   ```python
   INTENT_SCORING_WEIGHTS = {
       'by_artist': {
        'quality': 0.5, 'popularity': 0.3, 'recency': 0.2
    },
    'by_artist_underground': {
        'novelty': 0.6, 'quality': 0.25, 'underground': 0.15
       },
       'artist_similarity': {
        'similarity': 0.6, 'target_artist_boost': 0.2, 'quality': 0.15, 'novelty': 0.05
       },
       'discovery': {
        'novelty': 0.5, 'underground': 0.3, 'quality': 0.15, 'similarity': 0.05
       },
       'genre_mood': {
        'genre_mood_match': 0.6, 'quality': 0.25, 'novelty': 0.15
       },
       'contextual': {
        'context_fit': 0.6, 'quality': 0.25, 'familiarity': 0.15
       },
    
    # Hybrid Sub-Types
       'hybrid_discovery_primary': {
        'novelty': 0.5, 'genre_mood_match': 0.4, 'quality': 0.1
       },
       'hybrid_similarity_primary': {
        'similarity': 0.5, 'genre_mood_match': 0.3, 'quality': 0.2
       },
       'hybrid_genre_primary': {
        'genre_mood_match': 0.6, 'novelty': 0.25, 'quality': 0.15
    },
    
    # Follow-Up Types
    'style_continuation': {
        'similarity': 0.6, 'quality': 0.25, 'novelty': 0.15
    },
    'artist_deep_dive': {
        'target_artist_boost': 0.5, 'similarity': 0.3, 'quality': 0.2
    },
    'artist_style_refinement': {
        'similarity': 0.4, 'genre_mood_modification': 0.4, 'quality': 0.2
    },
    'discovery_continuation': {
        'novelty': 0.5, 'similarity': 0.3, 'underground': 0.2
    },
    'contextual_continuation': {
        'context_fit': 0.5, 'similarity': 0.3, 'familiarity': 0.2
       }
   }
   ```

### Agent Workflow Configuration
   ```python
   INTENT_AGENT_SEQUENCES = {
       'by_artist': ['discovery', 'judge'],
    'by_artist_underground': ['discovery', 'judge'],
       'artist_similarity': ['discovery', 'judge'],
       'discovery': ['discovery', 'judge'],
       'genre_mood': ['genre_mood', 'discovery', 'judge'],
       'contextual': ['genre_mood', 'judge'],
       
    # Hybrid Sub-Types
       'hybrid_discovery_primary': ['discovery', 'genre_mood', 'judge'],
       'hybrid_similarity_primary': ['discovery', 'genre_mood', 'judge'], 
    'hybrid_genre_primary': ['genre_mood', 'discovery', 'judge'],
    
    # Follow-Up Types (inherit from base intent but with history context)
    'style_continuation': ['discovery', 'judge'],
    'artist_deep_dive': ['discovery', 'judge'],
    'discovery_continuation': ['discovery', 'judge']
}
```

## Follow-Up Query Processing

### Context Requirements
- **Conversation History:** Access to previous queries and recommendations
- **Session Context:** Maintained track history across conversation turns
- **Intent Analysis:** LLM-based follow-up detection with confidence scoring

### Duplicate Prevention Strategy
1. **History Extraction:** Extract track IDs from conversation context
2. **Intent-Specific Filtering:** Apply different filtering logic based on follow-up type
3. **Fallback Mechanisms:** Use session context when chat interface history incomplete
4. **Track ID Format:** Standardized `artist::title` format (lowercase, `::` separator)

### Technical Implementation
```python
def extract_recently_shown_tracks(conversation_history, intent_override, target_entity):
    """Extract tracks to filter based on follow-up intent type."""
    if intent_override == 'style_continuation':
        # Include ALL previous recommendations
        return extract_all_tracks(conversation_history)
    elif intent_override in ['artist_deep_dive', 'artist_similarity']:
        # Include only tracks by target artist
        return extract_artist_tracks(conversation_history, target_entity)
    elif intent_override == 'artist_style_refinement':
        # Include tracks by target artist from previous turns
        return extract_artist_tracks(conversation_history, target_entity)
    else:
        # Default: include all to avoid duplicates
        return extract_all_tracks(conversation_history)
```

## Implementation Strategy

### Phase 1: Intent-Aware Scoring Framework
1. **Enhanced Query Understanding**
   - Improve intent detection accuracy
   - Add hybrid sub-type detection for discovery/similarity/genre-primary hybrids
   - Add confidence scores for intent classification
   - Handle ambiguous queries (default to hybrid approach)

2. **Dynamic Scoring Weight Configuration**
   - Implement intent-specific scoring weights
   - Add hybrid sub-type detection logic
   - Create intent hierarchy for scoring priority

### Phase 2: Agent Workflow Optimization
1. **Dynamic Agent Sequencing with Hybrid Sub-Types**
   - Intent-specific agent sequences
   - Hybrid sub-type specific workflows
   - Follow-up query processing with context awareness

2. **Intent-Aware Agent Parameter Adaptation**
   - Discovery Agent: Variable novelty thresholds by intent
   - Genre/Mood Agent: Style matching intensity by intent
   - Judge Agent: Dynamic scoring weights based on detected intent

### Phase 3: Follow-Up Query Integration
1. **Context-Aware Processing**
   - Conversation history integration
   - Duplicate prevention mechanisms
   - Intent mapping for follow-up queries

2. **Session Context Management**
   - Track history preservation across workflow nodes
   - Fallback mechanisms for incomplete chat history
   - Intent-specific filtering logic

### Phase 4: Evaluation & Refinement
1. **Intent-Specific Success Metrics**
   - Artist similarity: Similarity score, target artist inclusion
   - Discovery: Novelty score, listener count distribution
   - Genre/mood: Genre accuracy, mood consistency
   - Contextual: Context relevance, audio feature alignment
   - Follow-ups: Duplicate prevention rate, style consistency

## Key Improvements Summary

### Problem Solved
**Before**: All queries used the same balanced scoring weights, causing poor results:
- `"Music like Mk.gee"` → Favored underground over similar artists
- `"Find me underground indie rock"` → Returned mainstream hits

**After**: Dynamic scoring based on detected intent:
- `"Music like Mk.gee"` → **Similarity-focused**: Similarity(0.6) + Artist Boost(0.2) + Quality(0.15) + Novelty(0.05)
- `"Find me underground indie rock"` → **Discovery-focused**: Novelty(0.5) + Genre(0.4) + Quality(0.1)
- `"More like that"` → **Context-aware**: Avoids duplicates while maintaining style consistency

### Impact
✅ **Artist similarity queries** now prioritize similar artists over underground tracks  
✅ **Discovery queries** maintain strict novelty requirements  
✅ **Follow-up queries** provide context-aware recommendations without duplicates  
✅ **Hybrid queries** receive appropriate sub-type specific scoring  
✅ **Dynamic workflows** ensure optimal agent coordination per intent type

This provides a clear path toward **matching system behavior to user intent** rather than forcing all queries through the same scoring pipeline.
