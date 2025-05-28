"""
Multi-hop Similarity Explorer for BeatDebate

Implements sophisticated artist similarity exploration using network analysis
to discover related artists through 2-3 degree connections.
"""

import asyncio
from typing import Dict, List, Any, Set, Tuple, Optional
import structlog
from collections import defaultdict, deque
import random

logger = structlog.get_logger(__name__)


class ArtistSimilarityNetwork:
    """
    Represents a network of artist similarities for multi-hop exploration.
    """
    
    def __init__(self):
        """Initialize the artist similarity network."""
        self.logger = logger.bind(component="ArtistSimilarityNetwork")
        
        # Network structure: artist -> [(similar_artist, similarity_score), ...]
        self.similarity_graph = defaultdict(list)
        
        # Cache for API calls to avoid redundant requests
        self.similarity_cache = {}
        
        # Track exploration paths for reasoning
        self.exploration_paths = []
        
        self.logger.info("Artist Similarity Network initialized")
    
    def add_similarity_edge(
        self, 
        artist1: str, 
        artist2: str, 
        similarity_score: float = 1.0
    ):
        """Add a bidirectional similarity edge between two artists."""
        self.similarity_graph[artist1].append((artist2, similarity_score))
        self.similarity_graph[artist2].append((artist1, similarity_score))
    
    def get_similar_artists(self, artist: str, max_results: int = 10) -> List[Tuple[str, float]]:
        """Get similar artists with their similarity scores."""
        similar = self.similarity_graph.get(artist, [])
        # Sort by similarity score (descending) and return top results
        return sorted(similar, key=lambda x: x[1], reverse=True)[:max_results]
    
    def get_network_size(self) -> int:
        """Get the total number of artists in the network."""
        return len(self.similarity_graph)
    
    def get_connection_strength(self, artist1: str, artist2: str) -> float:
        """Get direct connection strength between two artists."""
        for similar_artist, score in self.similarity_graph.get(artist1, []):
            if similar_artist == artist2:
                return score
        return 0.0


class MultiHopSimilarityExplorer:
    """
    Explores artist similarity networks using multi-hop traversal to discover
    related artists and tracks through 2-3 degree connections.
    """
    
    def __init__(self, lastfm_client):
        """
        Initialize multi-hop similarity explorer.
        
        Args:
            lastfm_client: Last.fm API client for similarity data
        """
        self.lastfm_api_key = lastfm_client.api_key
        self.lastfm_rate_limit = lastfm_client.rate_limiter.calls_per_second
        self.logger = logger.bind(component="MultiHopSimilarityExplorer")
        
        # Network for tracking artist relationships
        self.similarity_network = ArtistSimilarityNetwork()
        
        # Exploration parameters
        self.max_hops = 3  # Maximum degrees of separation
        self.max_artists_per_hop = 8  # Artists to explore per hop
        self.max_tracks_per_artist = 5  # Tracks to get per artist
        
        # Quality thresholds for underground detection
        self.underground_thresholds = {
            'max_listeners': 50000,    # Max listeners for underground
            'max_playcount': 500000,   # Max play count for underground
            'min_quality_score': 0.4   # Minimum quality for inclusion
        }
        
        self.logger.info("Multi-hop Similarity Explorer initialized")
    
    async def explore_similarity_network(
        self, 
        seed_artists: List[str], 
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        target_candidates: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Explore artist similarity network using multi-hop traversal.
        
        Args:
            seed_artists: Starting artists for exploration
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent
            target_candidates: Target number of candidate tracks
            
        Returns:
            List of candidate tracks with similarity path information
        """
        self.logger.info(
            "Starting multi-hop similarity exploration",
            seed_artists=seed_artists,
            target_candidates=target_candidates,
            max_hops=self.max_hops
        )
        
        all_candidates = []
        
        # Create Last.fm client for this exploration session
        from ..api.lastfm_client import LastFmClient
        
        async with LastFmClient(
            api_key=self.lastfm_api_key,
            rate_limit=self.lastfm_rate_limit
        ) as client:
            
            # Explore from each seed artist
            for seed_artist in seed_artists:
                try:
                    # Perform multi-hop exploration from this seed
                    artist_candidates = await self._explore_from_seed(
                        seed_artist, client, entities, intent_analysis
                    )
                    
                    # Add seed artist information to candidates
                    for candidate in artist_candidates:
                        candidate['seed_artist'] = seed_artist
                        candidate['source'] = 'multi_hop_similarity'
                    
                    all_candidates.extend(artist_candidates)
                    
                    # Rate limiting between seed artists
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    self.logger.warning(
                        "Multi-hop exploration failed for seed artist",
                        seed_artist=seed_artist,
                        error=str(e)
                    )
                    continue
        
        # Remove duplicates and limit to target
        unique_candidates = self._deduplicate_candidates(all_candidates)
        final_candidates = unique_candidates[:target_candidates]
        
        self.logger.info(
            "Multi-hop similarity exploration completed",
            total_candidates=len(all_candidates),
            unique_candidates=len(unique_candidates),
            final_candidates=len(final_candidates),
            network_size=self.similarity_network.get_network_size()
        )
        
        return final_candidates
    
    async def _explore_from_seed(
        self,
        seed_artist: str,
        client,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Explore similarity network from a single seed artist."""
        candidates = []
        visited_artists = set()
        
        # BFS-style exploration with hop tracking
        exploration_queue = deque([(seed_artist, 0, [])])  # (artist, hop_count, path)
        
        while exploration_queue and len(candidates) < 50:  # Limit per seed
            current_artist, hop_count, path = exploration_queue.popleft()
            
            if current_artist in visited_artists or hop_count > self.max_hops:
                continue
            
            visited_artists.add(current_artist)
            current_path = path + [current_artist]
            
            try:
                # Get similar artists for current artist
                similar_artists = await self._get_similar_artists_with_caching(
                    current_artist, client
                )
                
                # Add similarity edges to network
                for similar_artist in similar_artists:
                    self.similarity_network.add_similarity_edge(
                        current_artist, similar_artist, 1.0 - (hop_count * 0.2)
                    )
                
                # Get tracks from current artist if not seed (to avoid duplicates)
                if hop_count > 0:
                    artist_tracks = await self._get_artist_tracks_with_metadata(
                        current_artist, client, current_path, hop_count
                    )
                    candidates.extend(artist_tracks)
                
                # Add similar artists to exploration queue for next hop
                if hop_count < self.max_hops:
                    for similar_artist in similar_artists[:self.max_artists_per_hop]:
                        if similar_artist not in visited_artists:
                            exploration_queue.append((
                                similar_artist, 
                                hop_count + 1, 
                                current_path
                            ))
                
                # Rate limiting between API calls
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.warning(
                    "Failed to explore artist",
                    artist=current_artist,
                    hop_count=hop_count,
                    error=str(e)
                )
                continue
        
        return candidates
    
    async def _get_similar_artists_with_caching(
        self, 
        artist: str, 
        client
    ) -> List[str]:
        """Get similar artists with caching to avoid redundant API calls."""
        if artist in self.similarity_network.similarity_cache:
            return self.similarity_network.similarity_cache[artist]
        
        try:
            # Get artist info which includes similar artists
            artist_info = await client.get_artist_info(artist=artist)
            
            if not artist_info or not artist_info.similar_artists:
                return []
            
            # Extract similar artist names (limit to max_artists_per_hop)
            similar_artists = [
                similar_artist for similar_artist in artist_info.similar_artists[:self.max_artists_per_hop]
                if similar_artist and similar_artist.lower() != artist.lower()
            ]
            
            # Cache the result
            self.similarity_network.similarity_cache[artist] = similar_artists
            
            self.logger.debug(
                "Similar artists retrieved",
                artist=artist,
                similar_count=len(similar_artists)
            )
            
            return similar_artists
            
        except Exception as e:
            self.logger.warning(
                "Failed to get similar artists",
                artist=artist,
                error=str(e)
            )
            return []
    
    async def _get_artist_tracks_with_metadata(
        self,
        artist: str,
        client,
        similarity_path: List[str],
        hop_count: int
    ) -> List[Dict[str, Any]]:
        """Get tracks from an artist with similarity path metadata."""
        try:
            # Get top tracks for the artist
            tracks_data = await client.get_artist_top_tracks(
                artist=artist,
                limit=self.max_tracks_per_artist
            )
            
            tracks = []
            for track_metadata in tracks_data:
                # Convert to dictionary with similarity metadata
                track = {
                    'name': track_metadata.name,
                    'artist': track_metadata.artist,
                    'url': track_metadata.url,
                    'listeners': track_metadata.listeners,
                    'playcount': track_metadata.playcount,
                    'mbid': track_metadata.mbid,
                    'tags': track_metadata.tags,
                    
                    # Multi-hop specific metadata
                    'similarity_path': similarity_path.copy(),
                    'hop_count': hop_count,
                    'similarity_score': 1.0 - (hop_count * 0.2),  # Decay with distance
                    'discovery_method': 'multi_hop_similarity',
                    'path_length': len(similarity_path)
                }
                
                tracks.append(track)
            
            self.logger.debug(
                "Artist tracks retrieved with similarity metadata",
                artist=artist,
                track_count=len(tracks),
                hop_count=hop_count,
                path_length=len(similarity_path)
            )
            
            return tracks
            
        except Exception as e:
            self.logger.warning(
                "Failed to get artist tracks",
                artist=artist,
                error=str(e)
            )
            return []
    
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tracks while preserving the best similarity paths."""
        seen_tracks = {}
        unique_candidates = []
        
        for candidate in candidates:
            # Create unique key for track
            track_key = f"{candidate.get('artist', '').lower()}_{candidate.get('name', '').lower()}"
            
            if track_key not in seen_tracks:
                seen_tracks[track_key] = candidate
                unique_candidates.append(candidate)
            else:
                # Keep the candidate with better similarity score
                existing = seen_tracks[track_key]
                if candidate.get('similarity_score', 0) > existing.get('similarity_score', 0):
                    # Replace in unique_candidates list
                    for i, unique_candidate in enumerate(unique_candidates):
                        if unique_candidate is existing:
                            unique_candidates[i] = candidate
                            seen_tracks[track_key] = candidate
                            break
        
        # Sort by similarity score (descending)
        unique_candidates.sort(
            key=lambda x: x.get('similarity_score', 0), 
            reverse=True
        )
        
        return unique_candidates
    
    def get_exploration_summary(self) -> Dict[str, Any]:
        """Get summary of the exploration process."""
        return {
            'network_size': self.similarity_network.get_network_size(),
            'total_paths': len(self.similarity_network.exploration_paths),
            'cache_size': len(self.similarity_network.similarity_cache),
            'max_hops_used': self.max_hops,
            'artists_per_hop': self.max_artists_per_hop
        } 