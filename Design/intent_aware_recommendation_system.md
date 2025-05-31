# Intent-Aware Music Recommendation System Design

## Problem Statement

Our current music recommendation system uses one-size-fits-all scoring that creates counter-intuitive results. For "Music like Mk.gee" queries, the system favors underground tracks over actually similar artists (DIJON, Jai Paul) because novelty scoring penalizes moderate popularity. We need intent-aware scoring and workflows that adapt behavior based on user goals.

## Core Query Types & Scoring Strategy

### 1. Artist Similarity (`"Music like Mk.gee"`)

**User Goal:** Find artists that sound similar to the target artist

**Current Issues:**
- Novelty scoring penalizes moderate popularity (DIJON, Jai Paul get low scores)
- System favors "underground" over "similar"
- Generic popular tracks (The Killers, Radiohead) score higher than similar artists

**Ideal Scoring Weights:**
```python
weights = {
    'similarity': 0.6,        # Most important - sound similarity
    'target_artist_boost': 0.2, # Target artist's own tracks
    'quality': 0.15,          # Decent audio quality
    'novelty': 0.05          # Minimal novelty requirement
}
```

**Scoring Modifications:**
- **Similarity >> Novelty** priority
- Accept moderate popularity (up to 10M listeners) 
- Boost tracks from acoustically similar artists
- Target artist's own tracks should rank highest
- Relaxed novelty thresholds for artist similarity

**Agent Workflow:**
- **Planner:** Sequence should be `['discovery', 'genre_mood', 'judge']` -> disocvery, judge
- **Discovery:** Focus on similar artists, allow 3+ tracks per similar artist
- **Genre/Mood:** Secondary validation of style consistency
- **Judge:** Heavy weighting toward similarity scores

### 2. Discovery/Exploration (`"Find me underground indie rock"`)

**User Goal:** Discover truly new/unknown music

**Current Logic:** ✅ Working well

**Ideal Scoring Weights:**
```python
weights = {
    'novelty': 0.5,          # Most important - truly unknown
    'underground': 0.3,       # Reward low popularity
    'quality': 0.15,         # Basic quality threshold
    'similarity': 0.05       # Minimal similarity to avoid chaos
}
```

**Scoring Modifications:**
- **Novelty >> Everything else** priority
- Heavily penalize popularity (strict <10k listener thresholds)
- Reward experimental/uncommon tags
- Underground bias appropriate here

**Agent Workflow:**
- **Planner:** Sequence should be `['discovery', 'judge']` (skip genre_mood for pure discovery)
- **Discovery:** Strict novelty filtering, favor serendipitous sources
- **Judge:** Heavy weighting toward novelty and underground scores

### 3. Genre/Mood (`"Upbeat electronic music"`)

**User Goal:** Find tracks matching specific vibes

**Ideal Scoring Weights:**
```python
weights = {
    'genre_mood_match': 0.6,  # Most important - style fit
    'quality': 0.25,         # Good audio quality
    'novelty': 0.15          # Some discovery element
}
```

**Scoring Modifications:**
- **Genre/Mood Match >> Everything else** priority
- Popularity somewhat irrelevant (focus on vibe)
- Quality over novelty
- Energy level, tempo, mood tags critical

**Agent Workflow:**
- **Planner:** Sequence should be `['genre_mood', 'discovery', 'judge']`
- **Genre/Mood:** Primary agent, extensive genre/mood filtering
- **Discovery:** Secondary validation, light novelty boost
- **Judge:** Heavy weighting toward genre/mood scores

### 4. Contextual (`"Music for studying"`, `"Workout playlist"`)

**User Goal:** Functional music for specific activities

**Ideal Scoring Weights:**
```python
weights = {
    'context_fit': 0.6,      # Most important - functional fit
    'quality': 0.25,         # Reliable quality
    'familiarity': 0.15      # May favor known tracks
}
```

**Scoring Modifications:**
- **Context Fit >> Novelty** priority
- May favor familiar, proven tracks (reverse novelty bias)
- Energy level, tempo, mood alignment critical
- Audio features (BPM, valence, energy) heavily weighted

**Agent Workflow:**
- **Planner:** Sequence should be `['genre_mood', 'judge']`
- **Genre/Mood:** Focus on audio features and contextual tags
- **Judge:** Context-aware scoring, familiarity bonus

### 5. Hybrid (`"Chill songs like Bon Iver"`, `"Underground indie rock"`, `"Jazzy music like Kendrick Lamar"`)

**User Goal:** Intersection of multiple intents with different primary emphasis

**Key Insight:** Hybrid queries are NOT one-size-fits-all. They need different scoring based on their primary intent:

#### 5.1 Discovery-Primary Hybrid (`"Find me underground indie rock"`, `"New experimental electronic music"`)
**Primary**: Discovery/Novelty | **Secondary**: Genre/Mood
```python
weights = {
    'novelty': 0.5,           # Most important - truly underground
    'genre_mood_match': 0.4,  # Important - genre accuracy  
    'quality': 0.1           # Basic threshold
}
```

#### 5.2 Similarity-Primary Hybrid (`"Music like Kendrick Lamar but jazzy"`, `"Chill songs like Bon Iver"`)
**Primary**: Artist Similarity | **Secondary**: Genre/Mood
```python
weights = {
    'similarity': 0.5,        # Most important - artist similarity
    'genre_mood_match': 0.3,  # Important - style modifier
    'quality': 0.2           # Good quality baseline
}
```

#### 5.3 Genre-Primary Hybrid (`"Upbeat indie rock with electronic elements"`, `"Dark ambient with jazz influences"`)
**Primary**: Genre/Mood | **Secondary**: Discovery
```python
weights = {
    'genre_mood_match': 0.6,  # Most important - style accuracy
    'novelty': 0.25,         # Secondary - some discovery
    'quality': 0.15          # Basic quality
}
```

**Scoring Modifications:**
- **Dynamic weighting** based on detected primary intent
- **Hybrid sub-type detection** from query analysis
- **Intent hierarchy** determines scoring priority

**Agent Workflow:**
- **Planner:** Sequence varies by sub-type:
  - Discovery-primary: `['discovery', 'genre_mood', 'judge']`
  - Similarity-primary: `['discovery', 'genre_mood', 'judge']` 
  - Genre-primary: `['genre_mood', 'discovery', 'judge']`
- **Agent weighting** varies by primary intent
- **Judge:** Uses sub-type specific scoring weights

## Query Examples & Intent Classification

### Pure Intent Examples

#### Artist Similarity
- `"Music like Mk.gee"` → Pure similarity focus
- `"Similar artists to BROCKHAMPTON"` → Artist similarity  
- `"Songs that sound like Radiohead"` → Stylistic similarity

#### Discovery
- `"Find me underground electronic music"` → Pure discovery
- `"Something completely new and different"` → Novelty focus
- `"Hidden gems in ambient music"` → Underground discovery

#### Genre/Mood  
- `"Upbeat electronic music"` → Genre + energy
- `"Sad indie songs"` → Genre + emotion
- `"Chill lo-fi hip hop"` → Style specificity

#### Contextual
- `"Music for studying"` → Functional context
- `"Workout playlist songs"` → Activity-based
- `"Background music for coding"` → Concentration context

### Hybrid Intent Examples

#### Discovery-Primary Hybrid
- `"Find me underground indie rock"` → Discovery(0.5) + Genre(0.4) + Quality(0.1)
- `"New experimental jazz I haven't heard"` → Discovery(0.5) + Genre(0.4) + Quality(0.1)
- `"Hidden electronic gems"` → Discovery(0.6) + Genre(0.3) + Quality(0.1)

#### Similarity-Primary Hybrid  
- `"Music like Kendrick Lamar but jazzy"` → Similarity(0.5) + Genre(0.3) + Quality(0.2)
- `"Chill songs like Bon Iver"` → Similarity(0.5) + Mood(0.3) + Quality(0.2)
- `"Electronic music similar to Aphex Twin"` → Similarity(0.5) + Genre(0.3) + Quality(0.2)

#### Genre-Primary Hybrid
- `"Upbeat indie rock with electronic elements"` → Genre(0.6) + Discovery(0.25) + Quality(0.15)
- `"Dark ambient with jazz influences"` → Genre(0.6) + Discovery(0.25) + Quality(0.15)
- `"Aggressive punk with melodic vocals"` → Genre(0.6) + Discovery(0.25) + Quality(0.15)

### Contextual Hybrid
- `"Study music like Max Richter"` → Context(0.4) + Similarity(0.4) + Quality(0.2)
- `"Workout songs similar to Death Grips"` → Context(0.4) + Similarity(0.4) + Quality(0.2)

## Implementation Strategy

### Phase 1: Intent-Aware Scoring Framework

1. **Enhanced Query Understanding**
   - Improve intent detection accuracy
   - **Add hybrid sub-type detection** for discovery/similarity/genre-primary hybrids
   - Add confidence scores for intent classification
   - Handle ambiguous queries (default to hybrid approach)

2. **Dynamic Scoring Weight Configuration**
   ```python
   INTENT_SCORING_WEIGHTS = {
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
       # 🔧 NEW: Hybrid sub-types with different priorities
       'hybrid_discovery_primary': {
           'novelty': 0.5,
           'genre_mood_match': 0.4,
           'quality': 0.1
       },
       'hybrid_similarity_primary': {
           'similarity': 0.5,
           'genre_mood_match': 0.3,
           'quality': 0.2
       },
       'hybrid_genre_primary': {
           'genre_mood_match': 0.6,
           'novelty': 0.25,
           'quality': 0.15
       }
   }
   ```

3. **Hybrid Sub-Type Detection Logic**
   ```python
   def detect_hybrid_subtype(query, entities):
       if has_discovery_indicators(query):  # "underground", "new", "hidden"
           return 'hybrid_discovery_primary'
       elif has_artist_references(entities):  # "like [Artist]"
           return 'hybrid_similarity_primary'  
       elif has_genre_emphasis(query):      # genre/mood words are primary
           return 'hybrid_genre_primary'
       else:
           return 'hybrid_balanced'  # fallback
   ```

### Phase 2: Agent Workflow Optimization

1. **Dynamic Agent Sequencing with Hybrid Sub-Types**
   ```python
   INTENT_AGENT_SEQUENCES = {
       'artist_similarity': ['discovery', 'judge'],
       'discovery': ['discovery', 'judge'],
       'genre_mood': ['genre_mood', 'discovery', 'judge'],
       'contextual': ['genre_mood', 'judge'],
       
       # 🔧 NEW: Hybrid sub-type specific sequences
       'hybrid_discovery_primary': ['discovery', 'genre_mood', 'judge'],
       'hybrid_similarity_primary': ['discovery', 'genre_mood', 'judge'], 
       'hybrid_genre_primary': ['genre_mood', 'discovery', 'judge']
   }
   ```

2. **Intent-Aware Agent Parameter Adaptation**
   - **Discovery Agent:** 
     - Discovery-primary hybrid: Strict novelty thresholds (0.6)
     - Similarity-primary hybrid: Relaxed novelty thresholds (0.3)
     - Genre-primary hybrid: Moderate novelty thresholds (0.4)
   - **Genre/Mood Agent:** 
     - Genre-primary hybrid: Strict genre matching
     - Discovery-primary hybrid: Broader genre interpretation
     - Similarity-primary hybrid: Style-focused filtering
   - **Judge Agent:** Uses dynamic scoring weights based on detected sub-type

3. **Adaptive Candidate Generation Strategy**
   - **Discovery-primary hybrid**: Underground sources + genre filtering
   - **Similarity-primary hybrid**: Similar artists + style modifiers  
   - **Genre-primary hybrid**: Genre-based search + novelty boost
   - **Contextual hybrid**: Audio features + functional requirements

### Phase 3: Evaluation & Refinement

1. **Intent-Specific Success Metrics**
   - Artist similarity: Similarity score, target artist inclusion
   - Discovery: Novelty score, listener count distribution
   - Genre/mood: Genre accuracy, mood consistency
   - Contextual: Context relevance, audio feature alignment

## Key Improvements: Hybrid Sub-Type System

### Problem Solved
**Before**: All hybrid queries used the same balanced scoring weights, causing poor results:
- `"Find me underground indie rock"` → Similarity(0.4) + Genre(0.35) + Quality(0.15) + Novelty(0.1) 
- Result: Mainstream hits like "Smells Like Teen Spirit" scored as "underground"

**After**: Dynamic scoring based on primary intent:
- `"Find me underground indie rock"` → **Discovery-Primary**: Novelty(0.5) + Genre(0.4) + Quality(0.1)
- `"Music like Kendrick Lamar but jazzy"` → **Similarity-Primary**: Similarity(0.5) + Genre(0.3) + Quality(0.2)

### Impact
✅ **Underground queries** now prioritize novelty over everything else  
✅ **Artist similarity queries** with style modifiers maintain similarity focus  
✅ **Genre-focused queries** balance style accuracy with discovery  
✅ **Dynamic thresholds** ensure appropriate filtering per sub-type

This provides a clear path toward fixing the core issue: **matching system behavior to user intent** rather than forcing all queries through the same scoring pipeline.

## Technical Implementation Details

### Modified Judge Agent Scoring with Hybrid Sub-Types
```python
def calculate_final_score(self, candidate, intent, agent_scores):
    # 🔧 NEW: Detect hybrid sub-type if intent is hybrid
    if intent == 'hybrid':
        intent = self.detect_hybrid_subtype(
            self.state.query_understanding.original_query,
            self.state.entities
        )
    
    weights = INTENT_SCORING_WEIGHTS[intent]
    
    final_score = 0.0
    for component, weight in weights.items():
        if component in agent_scores:
            final_score += agent_scores[component] * weight
    
    # Intent-specific bonuses
    if 'similarity' in intent or intent == 'artist_similarity':
        final_score += self._apply_target_artist_boost(candidate)
    elif 'discovery' in intent or intent == 'discovery':
        final_score += self._apply_underground_bonus(candidate)
    
    return min(final_score, 1.0)

def detect_hybrid_subtype(self, query, entities):
    """Detect primary intent within hybrid queries."""
    query_lower = query.lower()
    
    # Discovery indicators
    discovery_terms = ['underground', 'new', 'hidden', 'unknown', 'discover', 'find']
    if any(term in query_lower for term in discovery_terms):
        return 'hybrid_discovery_primary'
    
    # Artist similarity indicators  
    if entities.get('artists') and any(phrase in query_lower for phrase in ['like', 'similar', 'sounds like']):
        return 'hybrid_similarity_primary'
    
    # Genre/mood primary (default for style-focused queries)
    return 'hybrid_genre_primary'
```

### Dynamic Novelty Thresholds with Hybrid Sub-Types
```python
def get_novelty_threshold(self, intent):
    thresholds = {
        'artist_similarity': 0.15,           # Very relaxed
        'discovery': 0.6,                    # Strict
        'genre_mood': 0.3,                   # Moderate
        'contextual': 0.2,                   # Relaxed
        
        # 🔧 NEW: Hybrid sub-type specific thresholds
        'hybrid_discovery_primary': 0.6,     # Strict - underground focus
        'hybrid_similarity_primary': 0.25,   # Relaxed - similarity focus  
        'hybrid_genre_primary': 0.35,        # Moderate - balanced approach
        'hybrid': 0.4                        # Fallback
    }
    return thresholds.get(intent, 0.4)  # Default
```

### Intent-Aware Diversity Filtering
```python
def get_diversity_limits(self, intent):
    limits = {
        'artist_similarity': {'max_per_artist': 3, 'max_per_genre': 5},
        'discovery': {'max_per_artist': 1, 'max_per_genre': 3},
        'genre_mood': {'max_per_artist': 2, 'max_per_genre': 8},
        'contextual': {'max_per_artist': 2, 'max_per_genre': 6},
        'hybrid': {'max_per_artist': 2, 'max_per_genre': 4}
    }
    return limits.get(intent, {'max_per_artist': 1, 'max_per_genre': 3})
```

## Success Criteria

### Artist Similarity Queries
- ✅ Target artist's tracks appear in top 3 results
- ✅ Actually similar artists (DIJON, Jai Paul) rank higher than generic popular tracks
- ✅ Similarity score correlation > 0.8 with user ratings

### Discovery Queries  
- ✅ >80% of results have <100k listeners
- ✅ High novelty score distribution
- ✅ Diverse genre/tag representation

### Genre/Mood Queries
- ✅ >90% genre accuracy
- ✅ Consistent mood/energy levels
- ✅ Audio feature alignment within target ranges

### Contextual Queries
- ✅ Functional fit for intended activity
- ✅ Appropriate energy/tempo ranges
- ✅ User retention/completion rates for context

This design provides a clear path toward fixing the core issue: **matching system behavior to user intent** rather than forcing all queries through the same scoring pipeline. 