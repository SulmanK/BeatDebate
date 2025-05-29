"""
Underground Detector for Discovery Agent

Provides underground and hidden gem detection capabilities for the discovery
agent, using shared components for API access and analysis.
"""

from typing import Dict, List, Any, Tuple
import structlog

logger = structlog.get_logger(__name__)


class UndergroundDetector:
    """
    Underground and hidden gem detector for discovery recommendations.
    
    Provides:
    - Underground music detection
    - Hidden gem identification
    - Obscurity scoring
    - Cult status analysis
    """
    
    def __init__(self, api_service):
        """
        Initialize underground detector with injected API service.
        
        Args:
            api_service: Unified API service for Last.fm access
        """
        self.api_service = api_service
        self.underground_thresholds = self._initialize_underground_thresholds()
        self.underground_indicators = self._initialize_underground_indicators()
        
        logger.debug("UndergroundDetector initialized")
    
    async def detect_underground_tracks(
        self,
        candidates: List[Dict[str, Any]],
        underground_bias: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Detect and score underground tracks from candidates.
        
        Args:
            candidates: List of candidate tracks
            underground_bias: Bias towards underground content (0.0-1.0)
            
        Returns:
            List of tracks with underground scores
        """
        underground_tracks = []
        
        for candidate in candidates:
            underground_score = await self._calculate_underground_score(candidate)
            
            # Apply underground bias threshold
            if underground_score >= (underground_bias * 0.5):
                candidate['underground_score'] = underground_score
                candidate['underground_category'] = self._categorize_underground_level(underground_score)
                underground_tracks.append(candidate)
        
        # Sort by underground score
        underground_tracks.sort(key=lambda x: x.get('underground_score', 0), reverse=True)
        
        logger.debug(f"Detected {len(underground_tracks)} underground tracks")
        return underground_tracks
    
    async def _calculate_underground_score(self, track: Dict[str, Any]) -> float:
        """Calculate comprehensive underground score for a track."""
        score = 0.0
        
        # Listener count analysis
        listeners = track.get('listeners', 0)
        score += self._score_listener_count(listeners)
        
        # Play count analysis
        playcount = track.get('playcount', 0)
        score += self._score_play_count(playcount)
        
        # Tag-based underground indicators
        tags = track.get('tags', [])
        score += self._score_underground_tags(tags)
        
        # Artist obscurity analysis
        artist = track.get('artist', '')
        artist_score = await self._score_artist_obscurity(artist)
        score += artist_score
        
        # Source-based scoring
        source = track.get('source', '')
        score += self._score_source_underground(source)
        
        # Normalize score
        return min(score, 1.0)
    
    def _score_listener_count(self, listeners: int) -> float:
        """Score based on listener count (lower = more underground)."""
        thresholds = self.underground_thresholds['listeners']
        
        if listeners == 0:
            return 0.5  # Unknown, moderate score
        elif listeners < thresholds['very_underground']:
            return 0.9  # Very underground
        elif listeners < thresholds['underground']:
            return 0.7  # Underground
        elif listeners < thresholds['niche']:
            return 0.5  # Niche
        elif listeners < thresholds['mainstream']:
            return 0.3  # Semi-mainstream
        else:
            return 0.1  # Mainstream
    
    def _score_play_count(self, playcount: int) -> float:
        """Score based on play count (lower = more underground)."""
        thresholds = self.underground_thresholds['playcount']
        
        if playcount == 0:
            return 0.3  # Unknown
        elif playcount < thresholds['very_underground']:
            return 0.8  # Very underground
        elif playcount < thresholds['underground']:
            return 0.6  # Underground
        elif playcount < thresholds['niche']:
            return 0.4  # Niche
        elif playcount < thresholds['mainstream']:
            return 0.2  # Semi-mainstream
        else:
            return 0.1  # Mainstream
    
    def _score_underground_tags(self, tags: List[str]) -> float:
        """Score based on underground indicator tags."""
        score = 0.0
        
        for tag in tags:
            tag_lower = tag.lower()
            
            # Direct underground indicators
            if any(indicator in tag_lower for indicator in self.underground_indicators['direct']):
                score += 0.3
            
            # Genre-based underground indicators
            elif any(indicator in tag_lower for indicator in self.underground_indicators['genre']):
                score += 0.2
            
            # Quality-based underground indicators
            elif any(indicator in tag_lower for indicator in self.underground_indicators['quality']):
                score += 0.1
        
        return min(score, 0.6)  # Cap tag contribution
    
    async def _score_artist_obscurity(self, artist: str) -> float:
        """Score artist obscurity level."""
        try:
            # Get artist info
            artist_info = await self.api_service.get_artist_info(artist)
            
            # Artist listener count
            artist_listeners = int(artist_info.get('stats', {}).get('listeners', 0))
            listener_score = self._score_listener_count(artist_listeners) * 0.5
            
            # Artist tag analysis
            artist_tags = [tag.get('name', '') for tag in artist_info.get('tags', {}).get('tag', [])]
            tag_score = self._score_underground_tags(artist_tags) * 0.3
            
            # Artist name length (longer names often indicate less mainstream)
            name_score = min(len(artist) / 20, 0.2) if len(artist) > 10 else 0
            
            return listener_score + tag_score + name_score
            
        except Exception as e:
            logger.debug(f"Failed to get artist info for {artist}: {e}")
            # Fallback: simple name-based scoring
            return min(len(artist) / 15, 0.3) if len(artist) > 8 else 0.1
    
    def _score_source_underground(self, source: str) -> float:
        """Score based on discovery source."""
        source_scores = {
            'underground_search': 0.4,
            'serendipitous_discovery': 0.3,
            'multi_hop_similarity': 0.2,
            'tag_exploration': 0.2,
            'genre_mood_search': 0.1,
            'popular_search': 0.0
        }
        
        return source_scores.get(source, 0.1)
    
    def _categorize_underground_level(self, score: float) -> str:
        """Categorize underground level based on score."""
        if score >= 0.8:
            return 'very_underground'
        elif score >= 0.6:
            return 'underground'
        elif score >= 0.4:
            return 'niche'
        elif score >= 0.2:
            return 'semi_mainstream'
        else:
            return 'mainstream'
    
    async def find_hidden_gems(
        self,
        genre: str = None,
        mood: str = None,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find hidden gems based on genre and mood criteria.
        
        Args:
            genre: Target genre (optional)
            mood: Target mood (optional)
            max_results: Maximum number of results
            
        Returns:
            List of hidden gem tracks
        """
        hidden_gems = []
        
        try:
            # Build search tags for hidden gems
            search_tags = ['underground', 'hidden gem', 'obscure']
            
            if genre:
                search_tags.append(genre)
            if mood:
                search_tags.append(mood)
            
            # Search for tracks with underground tags
            for tag in search_tags[:3]:  # Limit search scope
                tracks = await self.api_service.search_tracks_by_tag(tag, limit=50)
                
                for track_data in tracks.get('track', []):
                    track = {
                        'name': track_data.get('name', ''),
                        'artist': track_data.get('artist', ''),
                        'url': track_data.get('url', ''),
                        'listeners': int(track_data.get('listeners', 0)),
                        'playcount': int(track_data.get('playcount', 0)),
                        'source': 'underground_search',
                        'tags': [tag]
                    }
                    
                    # Calculate underground score
                    underground_score = await self._calculate_underground_score(track)
                    
                    if underground_score >= 0.6:  # Hidden gem threshold
                        track['underground_score'] = underground_score
                        track['underground_category'] = self._categorize_underground_level(underground_score)
                        hidden_gems.append(track)
            
            # Remove duplicates and sort
            unique_gems = self._deduplicate_tracks(hidden_gems)
            unique_gems.sort(key=lambda x: x.get('underground_score', 0), reverse=True)
            
            logger.debug(f"Found {len(unique_gems)} hidden gems")
            return unique_gems[:max_results]
            
        except Exception as e:
            logger.warning(f"Failed to find hidden gems: {e}")
            return []
    
    def _deduplicate_tracks(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tracks based on artist + name."""
        seen = set()
        unique_tracks = []
        
        for track in tracks:
            track_key = f"{track.get('artist', '').lower()}::{track.get('name', '').lower()}"
            if track_key not in seen:
                seen.add(track_key)
                unique_tracks.append(track)
        
        return unique_tracks
    
    def analyze_underground_trends(
        self,
        tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze underground trends in a collection of tracks.
        
        Args:
            tracks: List of tracks to analyze
            
        Returns:
            Underground trend analysis
        """
        analysis = {
            'total_tracks': len(tracks),
            'underground_distribution': {},
            'average_underground_score': 0.0,
            'top_underground_artists': [],
            'underground_genres': [],
            'trend_indicators': []
        }
        
        if not tracks:
            return analysis
        
        # Calculate distribution
        categories = {}
        total_score = 0.0
        artist_scores = {}
        genre_counts = {}
        
        for track in tracks:
            underground_score = track.get('underground_score', 0)
            total_score += underground_score
            
            # Category distribution
            category = track.get('underground_category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
            # Artist analysis
            artist = track.get('artist', '')
            if artist:
                if artist not in artist_scores:
                    artist_scores[artist] = []
                artist_scores[artist].append(underground_score)
            
            # Genre analysis
            tags = track.get('tags', [])
            for tag in tags[:3]:  # Top 3 tags as genres
                genre_counts[tag] = genre_counts.get(tag, 0) + 1
        
        # Finalize analysis
        analysis['underground_distribution'] = categories
        analysis['average_underground_score'] = total_score / len(tracks)
        
        # Top underground artists
        artist_avg_scores = {
            artist: sum(scores) / len(scores)
            for artist, scores in artist_scores.items()
        }
        analysis['top_underground_artists'] = sorted(
            artist_avg_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Underground genres
        analysis['underground_genres'] = sorted(
            genre_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Trend indicators
        if analysis['average_underground_score'] > 0.7:
            analysis['trend_indicators'].append('High underground content')
        if categories.get('very_underground', 0) > len(tracks) * 0.3:
            analysis['trend_indicators'].append('Strong hidden gem presence')
        if len(set(artist_scores.keys())) > len(tracks) * 0.8:
            analysis['trend_indicators'].append('High artist diversity')
        
        return analysis
    
    def _initialize_underground_thresholds(self) -> Dict[str, Dict[str, int]]:
        """Initialize thresholds for underground classification."""
        return {
            'listeners': {
                'very_underground': 1000,
                'underground': 10000,
                'niche': 100000,
                'mainstream': 1000000
            },
            'playcount': {
                'very_underground': 5000,
                'underground': 50000,
                'niche': 500000,
                'mainstream': 5000000
            }
        }
    
    def _initialize_underground_indicators(self) -> Dict[str, List[str]]:
        """Initialize underground indicator keywords."""
        return {
            'direct': [
                'underground', 'hidden gem', 'obscure', 'cult', 'rare',
                'unknown', 'undiscovered', 'secret', 'buried treasure'
            ],
            'genre': [
                'experimental', 'avant-garde', 'noise', 'drone', 'dark ambient',
                'black metal', 'doom', 'post-rock', 'math rock', 'krautrock',
                'minimal', 'microsound', 'lowercase', 'field recording'
            ],
            'quality': [
                'lo-fi', 'bedroom', 'diy', 'home recording', 'demo',
                'bootleg', 'unreleased', 'limited edition', 'small label'
            ]
        } 