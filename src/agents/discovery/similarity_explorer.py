"""
Similarity Explorer for Discovery Agent

Provides multi-hop similarity exploration and artist relationship mapping
for the discovery agent, using shared components for API access.
"""

from typing import Dict, List, Any, Set
import structlog

logger = structlog.get_logger(__name__)


class SimilarityExplorer:
    """
    Multi-hop similarity explorer for discovery recommendations.
    
    Provides:
    - Multi-hop artist similarity exploration
    - Artist relationship mapping
    - Similarity network traversal
    - Discovery path optimization
    """
    
    def __init__(self, api_service):
        """
        Initialize similarity explorer with injected API service.
        
        Args:
            api_service: Unified API service for Last.fm access
        """
        self.api_service = api_service
        self.max_hops = 3
        self.max_artists_per_hop = 20  # Increased from 10 to 20 for larger pools
        self.similarity_cache = {}
        
        logger.debug("SimilarityExplorer initialized")
    
    async def explore_multi_hop_similarity(
        self,
        seed_artists: List[str],
        target_tracks: int = 50,
        exploration_depth: int = 3,
        exclude_seed_artists: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Explore multi-hop artist similarity to find related tracks.
        
        Args:
            seed_artists: Starting artists for exploration
            target_tracks: Target number of tracks to find
            exploration_depth: Number of hops to explore
            exclude_seed_artists: If True, exclude tracks from seed artists (for artist_similarity)
            
        Returns:
            List of discovered tracks with similarity metadata
        """
        discovered_tracks = []
        explored_artists = set()
        current_artists = set(seed_artists)
        
        for hop in range(exploration_depth):
            logger.debug(f"Exploring hop {hop + 1} with {len(current_artists)} artists")
            
            next_hop_artists = set()
            
            for artist in current_artists:
                if artist in explored_artists:
                    continue
                
                # Get similar artists
                similar_artists = await self._get_similar_artists(artist)
                next_hop_artists.update(similar_artists[:self.max_artists_per_hop])
                
                # Get tracks from this artist (exclude seed artists if requested)
                should_get_tracks = True
                if exclude_seed_artists:
                    # Skip getting tracks from seed artists on ALL hops for artist_similarity
                    should_get_tracks = artist not in seed_artists
                    if not should_get_tracks:
                        logger.debug(f"Excluding seed artist tracks: {artist} (hop {hop + 1})")
                
                if should_get_tracks:
                    artist_tracks = await self._get_artist_tracks(artist, hop)
                    discovered_tracks.extend(artist_tracks)
                
                explored_artists.add(artist)
                
                # Stop if we have enough tracks
                if len(discovered_tracks) >= target_tracks:
                    break
            
            current_artists = next_hop_artists - explored_artists
            
            if not current_artists or len(discovered_tracks) >= target_tracks:
                break
        
        # Add similarity metadata
        for track in discovered_tracks:
            track['similarity_hop'] = self._calculate_similarity_hop(track, seed_artists)
            track['discovery_path'] = self._generate_discovery_path(track, seed_artists)
        
        logger.debug(f"Multi-hop exploration completed: {len(discovered_tracks)} tracks found")
        return discovered_tracks[:target_tracks]
    
    async def _get_similar_artists(self, artist: str) -> List[str]:
        """Get similar artists using cached results."""
        if artist in self.similarity_cache:
            return self.similarity_cache[artist]
        
        try:
            # Use API service to get similar artists
            similar_data = await self.api_service.get_similar_artists(artist)
            
            # FIXED: API service returns a list of ArtistMetadata objects, not a dict
            similar_artists = []
            if isinstance(similar_data, list):
                for artist_metadata in similar_data:
                    if hasattr(artist_metadata, 'name'):
                        similar_artists.append(artist_metadata.name)
                    elif isinstance(artist_metadata, dict):
                        similar_artists.append(artist_metadata.get('name', ''))
                    else:
                        similar_artists.append(str(artist_metadata))
            
            # Cache results
            self.similarity_cache[artist] = similar_artists
            return similar_artists
            
        except Exception as e:
            logger.warning(f"Failed to get similar artists for {artist}: {e}")
            return []
    
    async def _get_artist_tracks(self, artist: str, hop: int) -> List[Dict[str, Any]]:
        """Get tracks from an artist with hop metadata."""
        try:
            # Get top tracks for artist
            tracks_data = await self.api_service.get_artist_top_tracks(artist)
            tracks = []
            
            # FIXED: API service returns a list of UnifiedTrackMetadata objects, not a dict
            if isinstance(tracks_data, list):
                for track_metadata in tracks_data[:15]:  # Increased from 10 to 15 tracks per artist
                    track = {
                        'name': getattr(track_metadata, 'name', ''),
                        'artist': artist,
                        'url': getattr(track_metadata, 'url', ''),
                        'listeners': getattr(track_metadata, 'listeners', 0),
                        'playcount': getattr(track_metadata, 'playcount', 0),
                        'source': 'multi_hop_similarity',
                        'source_artist': artist,
                        'similarity_hop': hop,
                        'tags': getattr(track_metadata, 'tags', [])
                    }
                    tracks.append(track)
            
            return tracks
            
        except Exception as e:
            logger.warning(f"Failed to get tracks for {artist}: {e}")
            return []
    
    def _calculate_similarity_hop(self, track: Dict[str, Any], seed_artists: List[str]) -> int:
        """Calculate the similarity hop distance from seed artists."""
        track_artist = track.get('artist', '').lower()
        
        # Direct match with seed artists
        for seed in seed_artists:
            if seed.lower() == track_artist:
                return 0
        
        # Use stored hop information
        return track.get('similarity_hop', 1)
    
    def _generate_discovery_path(self, track: Dict[str, Any], seed_artists: List[str]) -> str:
        """Generate a discovery path description."""
        track_artist = track.get('artist', '')
        source_artist = track.get('source_artist', track_artist)
        hop = track.get('similarity_hop', 1)
        
        if hop == 0:
            return f"Direct from {track_artist}"
        elif hop == 1:
            # Find which seed artist led to this discovery
            for seed in seed_artists:
                if seed.lower() in source_artist.lower():
                    return f"{seed} → {track_artist}"
            return f"Similar to input → {track_artist}"
        else:
            return f"Multi-hop discovery → {track_artist}"
    
    async def find_artist_connections(
        self,
        artist1: str,
        artist2: str,
        max_depth: int = 3
    ) -> List[str]:
        """
        Find connection path between two artists.
        
        Args:
            artist1: First artist
            artist2: Second artist
            max_depth: Maximum search depth
            
        Returns:
            List of artists forming connection path
        """
        # Simple breadth-first search for artist connections
        queue = [(artist1, [artist1])]
        visited = {artist1.lower()}
        
        for _ in range(max_depth):
            if not queue:
                break
            
            current_artist, path = queue.pop(0)
            
            # Get similar artists
            similar_artists = await self._get_similar_artists(current_artist)
            
            for similar in similar_artists:
                similar_lower = similar.lower()
                
                if similar_lower == artist2.lower():
                    return path + [similar]
                
                if similar_lower not in visited:
                    visited.add(similar_lower)
                    queue.append((similar, path + [similar]))
        
        return []  # No connection found
    
    def calculate_artist_similarity_score(
        self,
        artist1: str,
        artist2: str,
        connection_path: List[str] = None
    ) -> float:
        """
        Calculate similarity score between two artists.
        
        Args:
            artist1: First artist
            artist2: Second artist
            connection_path: Optional connection path
            
        Returns:
            Similarity score from 0.0 to 1.0
        """
        if artist1.lower() == artist2.lower():
            return 1.0
        
        # Base similarity calculation
        score = 0.0
        
        # Name similarity (simple)
        name_similarity = self._calculate_name_similarity(artist1, artist2)
        score += name_similarity * 0.3
        
        # Connection path similarity
        if connection_path:
            path_length = len(connection_path)
            if path_length <= 2:
                score += 0.7  # Direct connection
            elif path_length <= 3:
                score += 0.5  # One degree separation
            elif path_length <= 4:
                score += 0.3  # Two degrees separation
            else:
                score += 0.1  # Distant connection
        
        return min(score, 1.0)
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate simple name similarity."""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Exact match
        if name1_lower == name2_lower:
            return 1.0
        
        # Substring match
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.7
        
        # Word overlap
        words1 = set(name1_lower.split())
        words2 = set(name2_lower.split())
        
        if words1 & words2:  # Common words
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            return overlap / total * 0.5
        
        return 0.0
    
    async def get_similarity_network(
        self,
        seed_artists: List[str],
        network_size: int = 50
    ) -> Dict[str, Any]:
        """
        Build a similarity network around seed artists.
        
        Args:
            seed_artists: Starting artists
            network_size: Target network size
            
        Returns:
            Network data with nodes and connections
        """
        network = {
            'nodes': {},
            'connections': [],
            'seed_artists': seed_artists
        }
        
        explored = set()
        to_explore = list(seed_artists)
        
        while to_explore and len(network['nodes']) < network_size:
            current_artist = to_explore.pop(0)
            
            if current_artist in explored:
                continue
            
            # Add artist as node
            network['nodes'][current_artist] = {
                'name': current_artist,
                'is_seed': current_artist in seed_artists,
                'connections': []
            }
            
            # Get similar artists
            similar_artists = await self._get_similar_artists(current_artist)
            
            for similar in similar_artists[:5]:  # Limit connections per node
                # Add connection
                network['connections'].append({
                    'from': current_artist,
                    'to': similar,
                    'similarity': 0.8  # Default similarity score
                })
                
                # Add to exploration queue
                if similar not in explored and similar not in to_explore:
                    to_explore.append(similar)
                
                # Update node connections
                network['nodes'][current_artist]['connections'].append(similar)
            
            explored.add(current_artist)
        
        logger.debug(f"Built similarity network: {len(network['nodes'])} nodes, {len(network['connections'])} connections")
        return network 