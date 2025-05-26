# Deep Dive: GenreMoodAgent & DiscoveryAgent Analysis

## Overview

This document provides a comprehensive analysis of how the GenreMoodAgent and DiscoveryAgent work internally, including their algorithms, data structures, and decision-making processes.

---

## 🎵 GenreMoodAgent Deep Dive

### Core Purpose
The **GenreMoodAgent** is the "mood and genre specialist" that finds music based on emotional states, energy levels, and musical styles using Last.fm's tag-based search system.

### 🧠 Internal Architecture

#### 1. **Initialization & Knowledge Base**

```python
# Mood Mappings (10 categories)
mood_tag_mappings = {
    'happy': ['happy', 'uplifting', 'cheerful', 'joyful', 'positive'],
    'sad': ['sad', 'melancholy', 'melancholic', 'emotional', 'depressing'],
    'energetic': ['energetic', 'upbeat', 'pump up', 'motivational', 'driving'],
    'chill': ['chill', 'relaxing', 'mellow', 'laid-back', 'easy listening'],
    'focus': ['instrumental', 'ambient', 'concentration', 'study', 'background'],
    'romantic': ['romantic', 'love', 'intimate', 'passionate', 'sensual'],
    'nostalgic': ['nostalgic', 'retro', 'vintage', 'classic', 'throwback'],
    'aggressive': ['aggressive', 'intense', 'heavy', 'hard', 'powerful'],
    'peaceful': ['peaceful', 'calm', 'serene', 'tranquil', 'soothing'],
    'party': ['party', 'dance', 'club', 'celebration', 'fun']
}

# Energy Level Mappings
energy_level_mappings = {
    'low': ['calm', 'peaceful', 'quiet', 'soft', 'gentle'],
    'medium': ['moderate', 'balanced', 'steady', 'comfortable'],
    'high': ['energetic', 'intense', 'powerful', 'driving', 'upbeat']
}

# Genre Mappings (12 major genres)
genre_tag_mappings = {
    'rock': ['rock', 'alternative rock', 'indie rock', 'classic rock'],
    'electronic': ['electronic', 'electronica', 'ambient', 'techno', 'house'],
    'indie': ['indie', 'independent', 'indie pop', 'indie rock'],
    'pop': ['pop', 'pop rock', 'electropop', 'synth-pop'],
    'hip-hop': ['hip hop', 'rap', 'hip-hop', 'urban'],
    'jazz': ['jazz', 'smooth jazz', 'contemporary jazz', 'jazz fusion'],
    # ... and more
}
```

#### 2. **5-Step Processing Workflow**

##### Step 1: Mood Analysis (`_analyze_mood_requirements()`)
```python
# Input: "I want upbeat music for working out"
# Output:
{
    'primary_mood': 'energetic',           # Detected from "upbeat"
    'secondary_moods': ['party'],          # Secondary matches
    'energy_level': 'high',               # From "working out" context
    'context_tags': ['motivational'],     # Activity-based tags
    'mood_confidence': 0.8                # Confidence in detection
}
```

**Algorithm**:
1. Extract strategy parameters from PlannerAgent
2. Scan query for mood keywords using mappings
3. Determine energy level from context clues
4. Calculate confidence based on keyword matches

##### Step 2: Search Tag Generation (`_generate_search_tags()`)
```python
# Input: mood_analysis + strategy
# Output: ['energetic', 'upbeat', 'high_energy', 'motivational']

def _generate_search_tags(strategy, mood_analysis):
    tags = []
    
    # Add primary mood tags (first 2)
    tags.extend(mood_tag_mappings['energetic'][:2])  # ['energetic', 'upbeat']
    
    # Add energy level tags (first 2)  
    tags.extend(energy_level_mappings['high'][:2])   # ['energetic', 'intense']
    
    # Add genre focus from strategy
    focus_areas = strategy.get('focus_areas', [])    # ['rock', 'electronic']
    for area in focus_areas[:2]:
        tags.extend(genre_tag_mappings[area][:2])
    
    return tags
```

##### Step 3: Track Search (`_search_tracks_by_tags()`)
```python
# Uses multiple tag combinations for comprehensive search
primary_combinations = [
    ['energetic', 'upbeat'],              # First 2 tags
    ['upbeat', 'high_energy'],            # Next 2 tags  
    ['energetic']                         # Single primary tag
]

# For each combination:
for tag_combo in primary_combinations:
    tag_query = ' '.join(tag_combo)       # "energetic upbeat"
    tracks = await lastfm_client.search_tracks(tag_query, limit=20)
    
    # Add search context to each track
    for track in tracks:
        track['search_tags'] = tag_combo
        track['search_type'] = 'tag_based'
```

**Search Strategy**:
- **Multiple combinations** to maximize coverage
- **Rate limiting** (0.1s delays) to respect Last.fm API
- **Deduplication** based on artist + title
- **Limit to 50 candidates** for processing efficiency

##### Step 4: Filtering & Ranking (`_filter_and_rank_tracks()`)
```python
def _calculate_track_score(track, mood_analysis, strategy):
    score = 0.0
    
    # Base metadata score (20%)
    if track.get('artist') and track.get('name'):
        score += 0.2
    
    # Search tag relevance (30%)
    search_tags = track.get('search_tags', [])
    primary_mood_tags = mood_tag_mappings[mood_analysis['primary_mood']]
    tag_overlap = len(set(search_tags) & set(primary_mood_tags))
    score += (tag_overlap / max(len(primary_mood_tags), 1)) * 0.3
    
    # Popularity score (20%) - log scale normalization
    listeners = int(track.get('listeners', 0))
    if listeners > 0:
        normalized_listeners = min(math.log10(listeners) / 6, 1.0)  # Max at 1M
        score += normalized_listeners * 0.2
    
    # URL availability (20%)
    if track.get('url'):
        score += 0.2
    
    # Quality indicators (10% each)
    if good_artist_name: score += 0.1
    if good_track_name: score += 0.1
    
    return min(score, 1.0)
```

**Scoring Breakdown**:
- **30%**: How well search tags match the primary mood
- **20%**: Popularity (more listeners = higher quality)
- **20%**: URL availability (playable tracks preferred)
- **20%**: Basic metadata quality
- **10%**: Artist/track name quality

##### Step 5: Recommendation Creation (`_create_recommendations()`)
```python
# Top 5 tracks become TrackRecommendation objects
for i, track in enumerate(top_tracks[:5]):
    recommendation = TrackRecommendation(
        title=track['name'],
        artist=track['artist'],
        source="lastfm",
        genres=extracted_genres,                    # From strategy focus areas
        moods=extracted_tags,                       # Primary mood + energy + search tags
        quality_score=track['genre_mood_score'],    # From scoring algorithm
        novelty_score=0.3,                         # Low (focuses on established genres)
        concentration_friendliness_score=calc_concentration_score(),
        confidence=track['genre_mood_score'],
        advocate_source_agent="GenreMoodAgent"
    )
```

### 🎯 GenreMoodAgent Strengths

1. **Comprehensive Mood Coverage**: 10 mood categories with 5 tags each
2. **Energy Level Matching**: High/medium/low energy optimization
3. **Activity Context**: Understands workout, study, party contexts
4. **Quality Filtering**: Prefers tracks with good metadata and popularity
5. **Tag-Based Discovery**: Leverages Last.fm's extensive tagging system

### 🔍 GenreMoodAgent Limitations

1. **Mainstream Bias**: Prefers popular tracks (high listener counts)
2. **Limited Novelty**: Low novelty scores (0.3) by design
3. **Tag Dependency**: Relies on Last.fm's tag quality
4. **No Artist Similarity**: Doesn't understand "like The Beatles" requests

---

## 🔍 DiscoveryAgent Deep Dive

### Core Purpose
The **DiscoveryAgent** is the "similarity and underground specialist" that finds music through artist similarity networks and underground exploration, prioritizing novelty and hidden gems.

### 🧠 Internal Architecture

#### 1. **Initialization & Knowledge Base**

```python
# Discovery Strategies (4 exploration types)
discovery_strategies = {
    'underground': {
        'max_listeners': 50000,      # Strict underground limit
        'novelty_weight': 0.8,       # High novelty priority
        'similarity_depth': 3        # Deep similarity exploration
    },
    'similar': {
        'max_listeners': 200000,     # More lenient for similarity
        'novelty_weight': 0.4,       # Lower novelty priority
        'similarity_depth': 2        # Focused similarity
    },
    'diverse': {
        'max_listeners': 500000,     # Broad listener range
        'novelty_weight': 0.6,       # Balanced novelty
        'similarity_depth': 4        # Wide exploration
    },
    'balanced': {
        'max_listeners': 100000,     # Default settings
        'novelty_weight': 0.6,
        'similarity_depth': 3
    }
}

# Seed Artists by Genre (curated for discovery)
seed_artists = {
    'rock': ['Radiohead', 'Arctic Monkeys', 'The Strokes', 'Queens of the Stone Age'],
    'electronic': ['Aphex Twin', 'Boards of Canada', 'Four Tet', 'Burial'],
    'indie': ['Bon Iver', 'Sufjan Stevens', 'Fleet Foxes', 'Grizzly Bear'],
    'underground': ['Death Grips', 'clipping.', 'JPEGMAFIA', 'Black Midi'],
    # ... more genres
}

# Underground Indicators (artist name patterns)
underground_indicators = [
    'collective', 'records', 'tape', 'bedroom', 'lo-fi', 'experimental',
    'ambient', 'drone', 'noise', 'post-', 'neo-', 'micro-', 'minimal'
]
```

#### 2. **5-Step Processing Workflow**

##### Step 1: Discovery Analysis (`_analyze_discovery_requirements()`)
```python
# Input: "Something like The Beatles but more underground"
# Output:
{
    'exploration_type': 'underground',     # Detected from "underground"
    'novelty_priority': 'high',           # From strategy
    'underground_bias': 0.8,              # Boosted from 0.6 due to "underground"
    'discovery_scope': 'medium',          # From strategy
    'similarity_base': 'artist_similarity', # From "like The Beatles"
    'novelty_threshold': 0.7,             # High threshold for underground
    'max_listeners_threshold': 20000      # Calculated from underground_bias
}
```

**Algorithm**:
```python
# Detect exploration type from keywords
exploration_indicators = {
    'underground': ['underground', 'hidden', 'unknown', 'discover', 'obscure'],
    'similar': ['similar', 'like', 'reminds', 'sounds like'],
    'diverse': ['diverse', 'variety', 'different', 'mix', 'eclectic'],
    'trending': ['new', 'latest', 'trending', 'popular', 'current']
}

# Adjust underground bias based on query
if 'underground' in query_lower:
    underground_bias = min(underground_bias + 0.2, 1.0)  # Boost bias
elif 'popular' in query_lower:
    underground_bias = max(underground_bias - 0.3, 0.0)  # Reduce bias

# Calculate listener threshold
max_listeners = int(100000 * (1.0 - underground_bias))  # Higher bias = lower threshold
```

##### Step 2: Seed Artist Discovery (`_find_seed_artists()`)
```python
# Method 1: Extract from query (THIS IS WHAT MOVES TO PLANNER!)
def _extract_artists_from_query(user_query):
    # Current regex patterns:
    like_match = re.search(r'like\s+([A-Z][a-zA-Z0-9]*(?:\s+[A-Za-z0-9]+)*)', query)
    similar_match = re.search(r'(?i)similar\s+to\s+([A-Z][a-zA-Z0-9]*(?:\s+[A-Za-z0-9]+)*)', query)
    
    # Returns: ["The Beatles"]

# Method 2: Genre-based seeds
focus_areas = strategy.get('focus_areas', [])  # From PlannerAgent
for genre in focus_areas:
    if genre in seed_artists:
        seed_artists.extend(seed_artists[genre][:2])  # Top 2 per genre

# Method 3: Fallback seeds
if not seed_artists:
    seed_artists = seed_artists['general']  # Default curated list
```

##### Step 3: Similarity Exploration (`_explore_similar_music()`)
```python
# For each seed artist: "The Beatles"
async def _explore_similar_music(seed_artists, discovery_analysis, strategy):
    all_tracks = []
    
    for seed_artist in seed_artists:  # "The Beatles"
        # Get similar artists
        similar_artists = await _get_similar_artists(seed_artist, client)
        # Returns: ["The Kinks", "The Rolling Stones", "The Who"]
        
        # Get tracks from similar artists
        for similar_artist in similar_artists[:3]:  # Top 3 similar
            artist_tracks = await client.get_artist_top_tracks(similar_artist, limit=10)
            
            for track in artist_tracks:
                track['seed_artist'] = 'The Beatles'
                track['similar_artist'] = similar_artist
                track['discovery_method'] = 'artist_similarity'
                all_tracks.append(track)
        
        # Also get tracks from seed artist itself
        seed_tracks = await client.get_artist_top_tracks(seed_artist, limit=5)
        for track in seed_tracks:
            track['discovery_method'] = 'seed_artist'
            all_tracks.append(track)
    
    return unique_tracks[:100]  # Deduplicated, limited
```

**Similarity Network Exploration**:
- **Seed Artist** → **Similar Artists** → **Their Top Tracks**
- **Discovery Chain**: Beatles → Kinks → "Waterloo Sunset"
- **Context Preservation**: Each track knows its discovery path
- **Rate Limiting**: 0.1-0.2s delays between API calls

##### Step 4: Underground Filtering (`_filter_for_underground()`)
```python
def _filter_for_underground(tracks, discovery_analysis, strategy):
    underground_tracks = []
    max_listeners = discovery_analysis['max_listeners_threshold']  # 20,000
    novelty_threshold = discovery_analysis['novelty_threshold']    # 0.7
    
    for track in tracks:
        # Calculate novelty score
        novelty_score = _calculate_novelty_score(track, discovery_analysis)
        
        # Apply underground bias filter
        listener_count = int(track.get('listeners', 0))
        underground_bias = discovery_analysis['underground_bias']  # 0.8
        
        if underground_bias > 0.7:  # High underground bias
            if listener_count > max_listeners:
                continue  # Strict exclusion of popular tracks
        elif underground_bias > 0.4:  # Medium underground bias
            if listener_count > max_listeners * 2:
                novelty_score *= 0.7  # Reduce score for popular tracks
        
        # Apply novelty threshold
        if novelty_score >= novelty_threshold:
            track['novelty_score'] = novelty_score
            underground_tracks.append(track)
    
    # Sort by novelty score (descending)
    return sorted(underground_tracks, key=lambda x: x['novelty_score'], reverse=True)[:20]
```

##### Step 5: Novelty Scoring (`_calculate_novelty_score()`)
```python
def _calculate_novelty_score(track, discovery_analysis):
    score = 0.0
    
    # Base metadata score (20%)
    if track.get('artist') and track.get('name'):
        score += 0.2
    
    # Underground bias scoring (40% - MOST IMPORTANT)
    listener_count = int(track.get('listeners', 0))
    max_threshold = discovery_analysis['max_listeners_threshold']  # 20,000
    
    if listener_count == 0:
        score += 0.4  # Maximum underground score
    elif listener_count <= max_threshold:
        # Scale from 0.4 to 0.2 based on listener count
        normalized = 1.0 - (listener_count / max_threshold)
        score += 0.2 + (normalized * 0.2)
    else:
        # Popular tracks get very low scores
        popularity_factor = min(listener_count / max_threshold, 5.0)
        if popularity_factor > 2.0:  # Very popular
            score += 0.05  # Very low score
        else:
            score += 0.1   # Minimum score
    
    # Discovery method bonus (20%)
    if track['discovery_method'] == 'artist_similarity':
        score += 0.2  # Bonus for similarity discovery
    elif track['discovery_method'] == 'seed_artist':
        score += 0.1  # Smaller bonus for direct seed
    
    # Artist diversity bonus (20%)
    artist = track.get('artist', '')
    if len(artist) > 10:  # Longer names often indicate indie artists
        score += 0.1
    if any(indicator in artist.lower() for indicator in underground_indicators):
        score += 0.1  # Bonus for underground name patterns
    
    # Metadata quality (10% each)
    if track.get('url'): score += 0.1
    if track.get('seed_artist') != track.get('similar_artist'): score += 0.1
    
    return min(score, 1.0)
```

**Novelty Scoring Breakdown**:
- **40%**: Underground bias (fewer listeners = higher score)
- **20%**: Discovery method (similarity > seed artist)
- **20%**: Artist diversity (indie indicators, name length)
- **20%**: Basic metadata and discovery chain quality

### 🎯 DiscoveryAgent Strengths

1. **Artist Similarity Networks**: Leverages Last.fm's similarity data
2. **Underground Discovery**: Sophisticated listener count filtering
3. **Novelty Optimization**: Complex scoring algorithm for hidden gems
4. **Discovery Chain Tracking**: Maintains provenance of recommendations
5. **Exploration Strategies**: 4 different exploration modes

### 🔍 DiscoveryAgent Limitations

1. **Artist Extraction**: Currently does basic regex parsing (MOVING TO PLANNER!)
2. **API Dependency**: Relies heavily on Last.fm similarity data
3. **Popularity Bias**: May miss quality popular tracks
4. **Limited Genre Expansion**: Focuses on similarity rather than genre exploration

---

## 🤝 How They Work Together

### Current Coordination
```python
# PlannerAgent creates strategies:
{
    "genre_mood_agent": {
        "focus_areas": ["rock", "pop"],
        "energy_level": "high",
        "mood_priority": "energetic"
    },
    "discovery_agent": {
        "novelty_priority": "medium",
        "similarity_base": "artist_similarity",
        "underground_bias": 0.6
    }
}
```

### Complementary Strengths
- **GenreMoodAgent**: Finds **established, quality tracks** that match **mood/energy**
- **DiscoveryAgent**: Finds **novel, underground tracks** through **artist similarity**
- **Together**: Provide both **familiar comfort** and **exciting discovery**

### Example Output Comparison

**Query**: "Upbeat music like The Beatles"

**GenreMoodAgent Results**:
- "Here Comes the Sun" - The Beatles (popular, upbeat, high quality)
- "Mr. Blue Sky" - ELO (similar mood, well-known)
- "Good as Hell" - Lizzo (modern upbeat, high energy)

**DiscoveryAgent Results**:
- "Waterloo Sunset" - The Kinks (Beatles-similar, underground classic)
- "She's Always a Woman" - Billy Joel (Beatles influence, moderate popularity)
- "Golden" - Harry Styles (Beatles-inspired, newer artist)

**Combined**: Perfect balance of **familiar favorites** and **exciting discoveries**!

---

## 🚀 Enhancement Opportunities

### 1. **Entity Recognition Migration**
- **Move** `DiscoveryAgent._extract_artists_from_query()` to **PlannerAgent**
- **Enhance** with LLM-based entity recognition
- **Provide** pre-extracted entities to both agents

### 2. **Cross-Agent Learning**
- **Share** successful discovery patterns between agents
- **Coordinate** to avoid duplicate recommendations
- **Balance** familiarity vs. novelty based on user feedback

### 3. **Advanced Filtering**
- **Activity-aware** filtering (workout vs. study music)
- **Time-based** preferences (morning vs. evening music)
- **Mood progression** (start calm, build energy)

This deep dive shows that both agents are sophisticated specialists with complementary strengths. The enhanced PlannerAgent will make them even more powerful by providing better entity understanding and coordination! 