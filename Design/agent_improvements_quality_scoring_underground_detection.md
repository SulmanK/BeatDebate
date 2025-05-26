# Agent Improvements: Quality Scoring & Underground Detection

## Problem Statement

We need to enhance our GenreMoodAgent and DiscoveryAgent with:
1. **Quality Scoring System** for the GenreMoodAgent to rank tracks by musical quality
2. **Multi-hop Similarity Exploration** and **Intelligent Underground Detection** for the DiscoveryAgent
3. **Enhanced Candidate Generation**: Pull 100 candidate tracks, then filter to top 10 using quality scoring

## Enhanced Candidate Generation Strategy

### Core Concept: 100 → 10 Filtering Pipeline

Instead of returning the first 10-20 tracks found, we now:
1. **Generate 100 candidate tracks** from multiple sources
2. **Apply comprehensive quality scoring** to all candidates
3. **Filter and rank** to return only the top 20 highest-quality tracks

### Benefits of Larger Candidate Pool

- **Higher Quality**: More options means better final selections
- **Better Diversity**: Can balance mainstream vs underground more effectively
- **Improved Discovery**: More chances to find hidden gems
- **Quality Consistency**: Every recommendation meets high standards

### Implementation Strategy

```python
class EnhancedCandidateGenerator:
    """Generates large pools of candidate tracks for quality filtering"""
    
    def __init__(self):
        self.target_candidates = 100
        self.final_recommendations = 20
        self.source_distribution = {
            'primary_search': 40,      # 40 tracks from main search
            'similar_artists': 30,     # 30 tracks from artist similarity
            'genre_exploration': 20,   # 20 tracks from genre/mood tags
            'underground_gems': 10     # 10 tracks from underground detection
        }
    
    async def generate_candidate_pool(self, query_analysis: Dict) -> List[Dict]:
        """
        Generate 100 candidate tracks from multiple sources
        """
        all_candidates = []
        
        # Source 1: Primary Search (40 tracks)
        primary_tracks = await self._get_primary_search_tracks(
            query_analysis, 
            limit=self.source_distribution['primary_search']
        )
        all_candidates.extend(primary_tracks)
        
        # Source 2: Similar Artists (30 tracks)
        if 'artists' in query_analysis.get('entities', {}):
            similar_tracks = await self._get_similar_artist_tracks(
                query_analysis['entities']['artists'],
                limit=self.source_distribution['similar_artists']
            )
            all_candidates.extend(similar_tracks)
        
        # Source 3: Genre/Mood Exploration (20 tracks)
        genre_tracks = await self._get_genre_exploration_tracks(
            query_analysis,
            limit=self.source_distribution['genre_exploration']
        )
        all_candidates.extend(genre_tracks)
        
        # Source 4: Underground Gems (10 tracks)
        underground_tracks = await self._get_underground_tracks(
            query_analysis,
            limit=self.source_distribution['underground_gems']
        )
        all_candidates.extend(underground_tracks)
        
        # Remove duplicates while preserving source information
        unique_candidates = self._deduplicate_preserve_sources(all_candidates)
        
        return unique_candidates[:self.target_candidates]
```

## Research Insights from Industry Leaders

### How Spotify Handles Quality Scoring
- **Audio Feature Analysis**: Uses 13+ audio features (danceability, energy, valence, acousticness, etc.)
- **Popularity Weighting**: Balances track popularity with user preferences
- **Collaborative Filtering**: User behavior patterns influence quality scores
- **Machine Learning Models**: Neural networks trained on user engagement data

### How Last.fm/Pandora Handle Underground Detection
- **Artist Network Analysis**: Multi-hop exploration through artist similarity graphs
- **Listener Count Thresholds**: Underground = <10K monthly listeners
- **Tag-based Discovery**: Uses semantic tags to find similar but lesser-known artists
- **Temporal Analysis**: Tracks rising artists before they become mainstream

---

## 🎵 GenreMoodAgent: Quality Scoring System

### Core Concept
Implement a **multi-dimensional quality scoring system** that evaluates tracks based on:
1. **Audio Features Quality** (40%)
2. **Popularity Balance** (25%)
3. **User Engagement Signals** (20%)
4. **Genre/Mood Fit** (15%)

### Implementation Strategy

#### 1. Audio Features Quality Scoring

```python
class AudioQualityScorer:
    """Scores tracks based on audio feature analysis"""
    
    def __init__(self):
        self.feature_weights = {
            'energy': 0.2,
            'danceability': 0.15,
            'valence': 0.15,
            'acousticness': 0.1,
            'instrumentalness': 0.1,
            'liveness': 0.1,
            'speechiness': 0.1,
            'tempo_consistency': 0.1
        }
    
    def calculate_audio_quality_score(self, track_features: Dict) -> float:
        """
        Calculate quality score based on audio features
        Returns: 0.0 - 1.0 quality score
        """
        quality_score = 0.0
        
        # Energy optimization (sweet spot: 0.4-0.8)
        energy = track_features.get('energy', 0.5)
        energy_score = 1.0 - abs(energy - 0.6) / 0.6
        
        # Danceability (higher is generally better)
        danceability = track_features.get('danceability', 0.5)
        dance_score = danceability
        
        # Valence (mood-dependent, but avoid extremes)
        valence = track_features.get('valence', 0.5)
        valence_score = 1.0 - abs(valence - 0.5) / 0.5
        
        # Combine weighted scores
        for feature, weight in self.feature_weights.items():
            if feature in track_features:
                quality_score += track_features[feature] * weight
        
        return min(1.0, quality_score)
```

#### 2. Popularity Balance Scoring

```python
class PopularityBalancer:
    """Balances mainstream vs underground preferences"""
    
    def calculate_popularity_score(self, track_data: Dict, user_preferences: Dict) -> float:
        """
        Calculate popularity-adjusted quality score
        """
        play_count = track_data.get('playcount', 0)
        listener_count = track_data.get('listeners', 0)
        
        # Normalize popularity (log scale)
        popularity_score = min(1.0, math.log10(max(1, play_count)) / 7.0)
        
        # User preference adjustment
        underground_preference = user_preferences.get('underground_factor', 0.5)
        
        # Balance formula: reward moderate popularity
        if underground_preference > 0.7:  # User likes underground
            return 1.0 - (popularity_score * 0.6)
        elif underground_preference < 0.3:  # User likes mainstream
            return popularity_score
        else:  # Balanced preference
            return 1.0 - abs(popularity_score - 0.5) / 0.5
```

#### 3. Enhanced GenreMoodAgent with Quality Scoring

```python
class EnhancedGenreMoodAgent(GenreMoodAgent):
    """GenreMoodAgent with integrated quality scoring"""
    
    def __init__(self):
        super().__init__()
        self.audio_scorer = AudioQualityScorer()
        self.popularity_balancer = PopularityBalancer()
        self.quality_threshold = 0.6  # Minimum quality score
    
    async def get_recommendations_with_quality(self, query_analysis: Dict) -> List[Dict]:
        """
        Get recommendations with enhanced 100→10 quality filtering
        """
        # Step 1: Generate 100 candidate tracks from multiple sources
        candidate_generator = EnhancedCandidateGenerator()
        candidate_pool = await candidate_generator.generate_candidate_pool(query_analysis)
        
        # Step 2: Score all 100 candidates
        scored_tracks = []
        for track in candidate_pool:
            quality_score = await self._calculate_track_quality(track, query_analysis)
            
            if quality_score >= self.quality_threshold:
                track['quality_score'] = quality_score
                track['quality_breakdown'] = self._get_quality_breakdown(track)
                track['candidate_source'] = track.get('source', 'unknown')
                scored_tracks.append(track)
        
        # Step 3: Apply advanced filtering and ranking
        final_recommendations = await self._apply_advanced_filtering(scored_tracks, query_analysis)
        
        # Step 4: Sort by quality score and return top 10
        final_recommendations.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return final_recommendations[:20]  # Return top 20 highest quality tracks
    
    async def _apply_advanced_filtering(self, scored_tracks: List[Dict], query_analysis: Dict) -> List[Dict]:
        """
        Apply advanced filtering to ensure diversity and quality
        """
        # Ensure diversity across sources
        source_balanced = self._balance_recommendation_sources(scored_tracks)
        
        # Apply underground/mainstream balance based on user preferences
        preference_balanced = self._apply_preference_balancing(source_balanced, query_analysis)
        
        # Remove tracks that are too similar to each other
        diversity_filtered = self._ensure_track_diversity(preference_balanced)
        
        return diversity_filtered
    
    async def _calculate_track_quality(self, track: Dict, query_analysis: Dict) -> float:
        """Calculate comprehensive quality score for a track"""
        
        # 1. Audio Features Quality (40%)
        audio_features = await self._get_audio_features(track)
        audio_quality = self.audio_scorer.calculate_audio_quality_score(audio_features)
        
        # 2. Popularity Balance (25%)
        popularity_score = self.popularity_balancer.calculate_popularity_score(
            track, query_analysis.get('user_preferences', {})
        )
        
        # 3. Genre/Mood Fit (15%)
        genre_fit = self._calculate_genre_mood_fit(track, query_analysis)
        
        # 4. User Engagement Signals (20%)
        engagement_score = await self._calculate_engagement_score(track)
        
        # Weighted combination
        total_quality = (
            audio_quality * 0.40 +
            popularity_score * 0.25 +
            genre_fit * 0.15 +
            engagement_score * 0.20
        )
        
        return total_quality
```

---

## 🔍 DiscoveryAgent: Multi-hop Similarity & Underground Detection

### Core Concept
Implement **intelligent graph exploration** that:
1. **Multi-hop Similarity**: Explores artist networks 2-3 degrees of separation
2. **Underground Detection**: Identifies quality artists with <50K monthly listeners
3. **Rising Artist Detection**: Finds artists with growing momentum

### Implementation Strategy

#### 1. Multi-hop Similarity Explorer

```python
class MultiHopSimilarityExplorer:
    """Explores artist similarity networks with multiple hops"""
    
    def __init__(self, lastfm_client):
        self.lastfm = lastfm_client
        self.similarity_cache = {}
        self.max_hops = 3
        self.similarity_threshold = 0.3
    
    async def explore_similarity_network(self, seed_artist: str, target_underground_ratio: float = 0.7) -> List[Dict]:
        """
        Explore artist network using multi-hop similarity
        
        Args:
            seed_artist: Starting artist for exploration
            target_underground_ratio: Desired ratio of underground artists (0.0-1.0)
        
        Returns:
            List of discovered artists with similarity scores and underground status
        """
        discovered_artists = []
        visited_artists = set()
        exploration_queue = [(seed_artist, 0, 1.0)]  # (artist, hop_count, similarity_score)
        
        while exploration_queue and len(discovered_artists) < 50:
            current_artist, hop_count, similarity_score = exploration_queue.pop(0)
            
            if current_artist in visited_artists or hop_count >= self.max_hops:
                continue
                
            visited_artists.add(current_artist)
            
            # Get similar artists for current hop
            similar_artists = await self._get_similar_artists_with_scores(current_artist)
            
            for similar_artist, sim_score in similar_artists:
                if similar_artist not in visited_artists:
                    # Calculate compound similarity score
                    compound_score = similarity_score * sim_score
                    
                    if compound_score >= self.similarity_threshold:
                        # Check if artist is underground
                        underground_status = await self._check_underground_status(similar_artist)
                        
                        artist_data = {
                            'name': similar_artist,
                            'similarity_score': compound_score,
                            'hop_distance': hop_count + 1,
                            'is_underground': underground_status['is_underground'],
                            'listener_count': underground_status['listener_count'],
                            'discovery_path': f"{seed_artist} → {current_artist} → {similar_artist}"
                        }
                        
                        discovered_artists.append(artist_data)
                        
                        # Add to exploration queue for next hop
                        if hop_count + 1 < self.max_hops:
                            exploration_queue.append((similar_artist, hop_count + 1, compound_score))
        
        # Filter and balance underground vs mainstream
        return self._balance_underground_mainstream(discovered_artists, target_underground_ratio)
```

#### 2. Intelligent Underground Detection

```python
class UndergroundDetector:
    """Detects and evaluates underground artists"""
    
    def __init__(self, lastfm_client):
        self.lastfm = lastfm_client
        self.underground_thresholds = {
            'max_listeners': 50000,      # Monthly listeners
            'max_playcount': 1000000,    # Total plays
            'min_quality_score': 0.6     # Minimum quality threshold
        }
    
    async def detect_underground_artists(self, genre_tags: List[str], mood_tags: List[str]) -> List[Dict]:
        """
        Detect high-quality underground artists in specified genres/moods
        """
        underground_candidates = []
        
        # Search by genre/mood tags with low popularity filters
        for tag in genre_tags + mood_tags:
            artists = await self._search_artists_by_tag(
                tag, 
                max_listeners=self.underground_thresholds['max_listeners']
            )
            
            for artist in artists:
                underground_score = await self._calculate_underground_score(artist)
                
                if underground_score >= 0.7:  # High underground potential
                    underground_candidates.append({
                        'artist': artist,
                        'underground_score': underground_score,
                        'discovery_tag': tag,
                        'estimated_quality': await self._estimate_artist_quality(artist)
                    })
        
        # Sort by underground score and quality
        underground_candidates.sort(
            key=lambda x: (x['underground_score'], x['estimated_quality']), 
            reverse=True
        )
        
        return underground_candidates[:15]  # Return top 15 underground artists
    
    async def _calculate_underground_score(self, artist_data: Dict) -> float:
        """
        Calculate how 'underground' an artist is
        
        Factors:
        - Low listener count (40%)
        - Recent activity but low mainstream presence (30%)
        - High engagement rate relative to size (20%)
        - Genre/tag diversity (10%)
        """
        listener_count = artist_data.get('listeners', 0)
        playcount = artist_data.get('playcount', 0)
        
        # Listener count score (lower = more underground)
        listener_score = max(0, 1.0 - (listener_count / self.underground_thresholds['max_listeners']))
        
        # Engagement rate (plays per listener)
        engagement_rate = playcount / max(1, listener_count)
        engagement_score = min(1.0, engagement_rate / 50.0)  # Normalize to 50 plays/listener
        
        # Recent activity score
        recent_activity = await self._check_recent_activity(artist_data['name'])
        
        # Tag diversity (more diverse = more interesting)
        tag_diversity = await self._calculate_tag_diversity(artist_data['name'])
        
        underground_score = (
            listener_score * 0.40 +
            recent_activity * 0.30 +
            engagement_score * 0.20 +
            tag_diversity * 0.10
        )
        
        return underground_score
```

#### 3. Enhanced DiscoveryAgent Integration

```python
class EnhancedDiscoveryAgent(DiscoveryAgent):
    """DiscoveryAgent with multi-hop similarity and underground detection"""
    
    def __init__(self):
        super().__init__()
        self.similarity_explorer = MultiHopSimilarityExplorer(self.lastfm)
        self.underground_detector = UndergroundDetector(self.lastfm)
        self.discovery_strategies = {
            'multi_hop_similarity': 0.4,
            'underground_detection': 0.3,
            'genre_exploration': 0.2,
            'rising_artists': 0.1
        }
    
    async def discover_with_intelligence(self, query_analysis: Dict) -> List[Dict]:
        """
        Intelligent discovery using enhanced 100→20 candidate generation
        """
        # Step 1: Generate 100 discovery candidates using multiple strategies
        discovered_tracks = []
        
        # Strategy 1: Multi-hop Similarity Exploration (40 tracks)
        if 'artists' in query_analysis.get('entities', {}):
            for artist in query_analysis['entities']['artists']:
                similar_tracks = await self._discover_via_multi_hop(artist, query_analysis, limit=40)
                discovered_tracks.extend(similar_tracks)
        
        # Strategy 2: Underground Detection (30 tracks)
        underground_tracks = await self._discover_underground_gems(query_analysis, limit=30)
        discovered_tracks.extend(underground_tracks)
        
        # Strategy 3: Genre Exploration (20 tracks)
        genre_tracks = await self._discover_via_genre_exploration(query_analysis, limit=20)
        discovered_tracks.extend(genre_tracks)
        
        # Strategy 4: Rising Artists (10 tracks)
        rising_tracks = await self._discover_rising_artists(query_analysis, limit=10)
        discovered_tracks.extend(rising_tracks)
        
        # Step 2: Apply discovery scoring to all 100 candidates
        scored_discoveries = []
        for track in discovered_tracks:
            discovery_score = await self._calculate_discovery_score(track, query_analysis)
            track['discovery_score'] = discovery_score
            track['discovery_breakdown'] = self._get_discovery_breakdown(track)
            scored_discoveries.append(track)
        
        # Step 3: Remove duplicates and apply intelligent ranking
        unique_tracks = self._deduplicate_and_rank_discoveries(scored_discoveries, query_analysis)
        
        # Step 4: Return top 20 discoveries
        return unique_tracks[:20]
    
    async def _calculate_discovery_score(self, track: Dict, query_analysis: Dict) -> float:
        """
        Calculate comprehensive discovery score for a track
        
        Factors:
        - Novelty/Underground factor (30%)
        - Similarity to query intent (25%)
        - Audio quality (25%)
        - Discovery potential (20%)
        """
        # Novelty score (how underground/unique is this track)
        novelty_score = await self._calculate_novelty_score(track)
        
        # Query similarity (how well does it match the request)
        similarity_score = await self._calculate_query_similarity(track, query_analysis)
        
        # Audio quality (basic quality metrics)
        quality_score = await self._calculate_basic_quality(track)
        
        # Discovery potential (likelihood user will enjoy and explore further)
        discovery_potential = await self._calculate_discovery_potential(track, query_analysis)
        
        total_discovery_score = (
            novelty_score * 0.30 +
            similarity_score * 0.25 +
            quality_score * 0.25 +
            discovery_potential * 0.20
        )
        
        return total_discovery_score
    
    async def _discover_via_multi_hop(self, seed_artist: str, query_analysis: Dict) -> List[Dict]:
        """Discover tracks using multi-hop similarity exploration"""
        
        underground_preference = query_analysis.get('preferences', {}).get('underground_factor', 0.5)
        
        # Explore artist network
        discovered_artists = await self.similarity_explorer.explore_similarity_network(
            seed_artist, 
            target_underground_ratio=underground_preference
        )
        
        # Get top tracks from discovered artists
        tracks = []
        for artist_data in discovered_artists[:10]:  # Top 10 discovered artists
            artist_tracks = await self._get_artist_top_tracks(artist_data['name'])
            
            for track in artist_tracks[:3]:  # Top 3 tracks per artist
                track['discovery_method'] = 'multi_hop_similarity'
                track['similarity_score'] = artist_data['similarity_score']
                track['hop_distance'] = artist_data['hop_distance']
                track['discovery_path'] = artist_data['discovery_path']
                tracks.append(track)
        
        return tracks
```

## Implementation Roadmap

### Phase 1: Enhanced Candidate Generation (Week 1-2)
1. Implement `EnhancedCandidateGenerator` class
2. Build multi-source track collection (100 candidates)
3. Add source tracking and deduplication
4. Create candidate pool caching system

### Phase 2: Quality Scoring Foundation (Week 2-3)
1. Implement `AudioQualityScorer` class
2. Add `PopularityBalancer` functionality
3. Integrate quality scoring into GenreMoodAgent
4. Build advanced filtering pipeline (100→20)

### Phase 3: Multi-hop Similarity & Underground Detection (Week 4-5)
1. Build `MultiHopSimilarityExplorer` class
2. Develop `UndergroundDetector` class
3. Implement discovery scoring algorithms
4. Add intelligent ranking and diversity filtering

### Phase 4: Enhanced DiscoveryAgent (Week 6-7)
1. Integrate multi-strategy discovery (100 candidates)
2. Implement discovery scoring system
3. Add novelty and discovery potential calculations
4. Build comprehensive ranking pipeline

### Phase 5: Integration & Optimization (Week 8)
1. Integrate all components into enhanced agents
2. Performance optimization and caching
3. A/B testing with current system
4. User feedback collection and iteration

## Enhanced Performance Expectations

### Candidate Generation Benefits
- **10x More Options**: 100 candidates vs 10-20 current
- **Higher Quality Floor**: Every recommendation meets quality threshold
- **Better Diversity**: Balanced sources ensure variety
- **Improved Discovery**: More underground gems surface to top 20

### Expected Quality Improvements
- **Recommendation Accuracy**: 40% improvement in user satisfaction
- **Discovery Rate**: 60% more unknown artists discovered
- **Quality Consistency**: 90% of tracks meet high quality standards
- **User Engagement**: 35% increase in full track completion

## Success Metrics

### Quality Scoring Success
- **User Satisfaction**: 15% increase in positive feedback
- **Skip Rate**: 20% reduction in track skips
- **Engagement**: 25% increase in full track plays

### Underground Detection Success
- **Discovery Rate**: 30% of recommendations are underground artists
- **User Retention**: Users discover 5+ new underground artists per session
- **Diversity Score**: 40% increase in artist diversity across recommendations

This comprehensive approach will significantly enhance both agents' capabilities while maintaining the existing architecture! 