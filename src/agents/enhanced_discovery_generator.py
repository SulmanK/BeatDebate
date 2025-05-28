"""
Enhanced Discovery Candidate Generator for BeatDebate

Combines multi-hop similarity exploration and underground detection
to generate large pools of discovery candidates for quality filtering.
"""

from typing import Dict, List, Any
import structlog
import hashlib

from .multi_hop_similarity import MultiHopSimilarityExplorer
from .underground_detector import UndergroundDetector

logger = structlog.get_logger(__name__)


class EnhancedDiscoveryGenerator:
    """
    Generates large pools of discovery candidates using multiple sophisticated
    exploration strategies.
    
    Strategy: Generate 100 candidates from multiple sources:
    - Multi-hop similarity exploration (50 candidates)
    - Underground detection (30 candidates) 
    - Serendipitous discovery (20 candidates)
    """
    
    def __init__(self, lastfm_client):
        """
        Initialize enhanced discovery generator.
        
        Args:
            lastfm_client: Last.fm API client for music data
        """
        self.lastfm_api_key = lastfm_client.api_key
        self.lastfm_rate_limit = lastfm_client.rate_limiter.calls_per_second
        self.logger = logger.bind(component="EnhancedDiscoveryGenerator")
        
        # Initialize exploration components
        self.similarity_explorer = MultiHopSimilarityExplorer(lastfm_client)
        self.underground_detector = UndergroundDetector(lastfm_client)
        
        # Generation parameters
        self.target_candidates = 100
        self.source_distribution = {
            'multi_hop_similarity': 50,
            'underground_detection': 30,
            'serendipitous_discovery': 20
        }
        
        self.logger.info("Enhanced Discovery Generator initialized")
    
    async def generate_candidate_pool(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        agent_type: str = "discovery"
    ) -> List[Dict[str, Any]]:
        """
        Generate large pool of discovery candidates using multiple strategies.
        
        Args:
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent
            agent_type: Type of agent requesting candidates
            
        Returns:
            List of candidate tracks with discovery metadata
        """
        self.logger.info(
            "Starting enhanced discovery candidate generation",
            agent_type=agent_type,
            target_candidates=self.target_candidates
        )
        
        all_candidates = []
        
        try:
            # Extract seed artists and genres from entities
            seed_artists = self._extract_seed_artists(entities)
            target_genres = self._extract_target_genres(entities)
            
            self.logger.debug(
                "Discovery context extracted",
                seed_artists=seed_artists,
                target_genres=target_genres
            )
            
            # Source 1: Multi-hop similarity exploration (50 candidates)
            similarity_candidates = await self._get_multi_hop_similarity_tracks(
                seed_artists, entities, intent_analysis, 
                self.source_distribution['multi_hop_similarity']
            )
            self.logger.debug(
                f"Multi-hop similarity: {len(similarity_candidates)} tracks"
            )
            all_candidates.extend(similarity_candidates)
            
            # Source 2: Underground detection (30 candidates)
            underground_candidates = await self._get_underground_tracks(
                target_genres, entities, intent_analysis,
                self.source_distribution['underground_detection']
            )
            self.logger.debug(
                f"Underground detection: {len(underground_candidates)} tracks"
            )
            all_candidates.extend(underground_candidates)
            
            # Source 3: Serendipitous discovery (20 candidates)
            serendipitous_candidates = await self._get_serendipitous_tracks(
                entities, intent_analysis,
                self.source_distribution['serendipitous_discovery']
            )
            self.logger.debug(
                f"Serendipitous discovery: {len(serendipitous_candidates)} tracks"
            )
            all_candidates.extend(serendipitous_candidates)
            
            # Deduplicate and finalize
            unique_candidates = self._deduplicate_candidates(all_candidates)
            final_candidates = unique_candidates[:self.target_candidates]
            
            # Add generation metadata
            for candidate in final_candidates:
                candidate['generation_agent'] = 'enhanced_discovery'
                candidate['generation_timestamp'] = self._get_timestamp()
                candidate['candidate_id'] = self._generate_candidate_id(candidate)
            
            self.logger.info(
                "Enhanced discovery candidate generation completed",
                sources_used=len(self.source_distribution),
                total_candidates=len(all_candidates),
                unique_from_total=len(unique_candidates),
                final_candidates=len(final_candidates)
            )
            
            return final_candidates
            
        except Exception as e:
            self.logger.error(
                "Enhanced discovery candidate generation failed",
                error=str(e)
            )
            return []
    
    def _extract_seed_artists(self, entities: Dict[str, Any]) -> List[str]:
        """Extract seed artists from entities for similarity exploration."""
        seed_artists = []
        
        # Get artists from musical entities
        musical_entities = entities.get("musical_entities", {})
        artists = musical_entities.get("artists", {})
        
        # Primary artists
        primary_artists = artists.get("primary", [])
        seed_artists.extend(primary_artists)
        
        # Similar-to artists
        similar_artists = artists.get("similar_to", [])
        seed_artists.extend(similar_artists)
        
        # Fallback artists if none found
        if not seed_artists:
            seed_artists = [
                "Radiohead", "Bon Iver", "Thom Yorke", "Fleet Foxes", 
                "Sufjan Stevens", "Arcade Fire"
            ]
        
        return seed_artists[:5]  # Limit to 5 seed artists
    
    def _extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target genres from entities for underground detection."""
        target_genres = []
        
        # Get genres from musical entities
        musical_entities = entities.get("musical_entities", {})
        genres = musical_entities.get("genres", {})
        
        # Primary genres
        primary_genres = genres.get("primary", [])
        target_genres.extend(primary_genres)
        
        # Secondary genres
        secondary_genres = genres.get("secondary", [])
        target_genres.extend(secondary_genres)
        
        # Fallback genres if none found
        if not target_genres:
            target_genres = ["indie", "experimental", "electronic", "folk"]
        
        return target_genres[:4]  # Limit to 4 genres
    
    async def _get_multi_hop_similarity_tracks(
        self,
        seed_artists: List[str],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get tracks using multi-hop similarity exploration."""
        try:
            candidates = await self.similarity_explorer.explore_similarity_network(
                seed_artists, entities, intent_analysis, limit
            )
            
            # Add source metadata
            for candidate in candidates:
                candidate['source'] = 'multi_hop_similarity'
                candidate['source_confidence'] = candidate.get('similarity_score', 0.7)
            
            return candidates
            
        except Exception as e:
            self.logger.warning(
                "Multi-hop similarity exploration failed",
                error=str(e)
            )
            return []
    
    async def _get_underground_tracks(
        self,
        target_genres: List[str],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get tracks using underground detection."""
        try:
            candidates = await self.underground_detector.detect_underground_artists(
                target_genres, entities, intent_analysis, limit
            )
            
            # Add source metadata
            for candidate in candidates:
                candidate['source'] = 'underground_detection'
                candidate['source_confidence'] = candidate.get('underground_score', 0.6)
            
            return candidates
            
        except Exception as e:
            self.logger.warning(
                "Underground detection failed",
                error=str(e)
            )
            return []
    
    async def _get_serendipitous_tracks(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get tracks using serendipitous discovery methods."""
        candidates = []
        
        try:
            # Create Last.fm client for serendipitous search
            from ..api.lastfm_client import LastFmClient
            
            async with LastFmClient(
                api_key=self.lastfm_api_key,
                rate_limit=self.lastfm_rate_limit
            ) as client:
                
                # Strategy 1: Random genre exploration
                random_candidates = await self._random_genre_exploration(
                    client, entities, limit // 2
                )
                candidates.extend(random_candidates)
                
                # Strategy 2: Mood-based serendipity
                mood_candidates = await self._mood_based_serendipity(
                    client, entities, intent_analysis, limit // 2
                )
                candidates.extend(mood_candidates)
            
            # Add source metadata
            for candidate in candidates:
                candidate['source'] = 'serendipitous_discovery'
                candidate['source_confidence'] = 0.5  # Lower confidence for serendipity
            
            return candidates[:limit]
            
        except Exception as e:
            self.logger.warning(
                "Serendipitous discovery failed",
                error=str(e)
            )
            return []
    
    async def _random_genre_exploration(
        self, client, entities: Dict[str, Any], limit: int
    ) -> List[Dict[str, Any]]:
        """Explore random genres for serendipitous discovery."""
        import random
        
        candidates = []
        
        # Random genre pool for exploration
        random_genres = [
            'ambient', 'post-rock', 'shoegaze', 'dream pop', 'krautrock',
            'minimal', 'drone', 'field recording', 'new age', 'world music',
            'contemporary classical', 'free jazz', 'dub', 'trip hop'
        ]
        
        # Select random genres
        selected_genres = random.sample(random_genres, min(5, len(random_genres)))
        
        for genre in selected_genres:
            try:
                tracks = await client.search_tracks(query=genre, limit=limit // 5)
                
                for track_metadata in tracks:
                    track = {
                        'name': track_metadata.name,
                        'artist': track_metadata.artist,
                        'url': track_metadata.url,
                        'listeners': track_metadata.listeners,
                        'playcount': track_metadata.playcount,
                        'mbid': track_metadata.mbid,
                        'tags': track_metadata.tags,
                        'serendipity_genre': genre,
                        'discovery_method': 'random_genre_exploration'
                    }
                    candidates.append(track)
                
            except Exception as e:
                self.logger.warning(
                    "Random genre exploration failed",
                    genre=genre,
                    error=str(e)
                )
                continue
        
        return candidates
    
    async def _mood_based_serendipity(
        self, 
        client, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Discover tracks based on mood combinations for serendipity."""
        candidates = []
        
        # Extract mood context
        contextual_entities = entities.get("contextual_entities", {})
        moods = contextual_entities.get("moods", {})
        
        # Combine different mood categories
        all_moods = []
        for mood_category in moods.values():
            all_moods.extend(mood_category)
        
        # Add intent-based mood modifiers
        primary_intent = intent_analysis.get('primary_intent', 'discovery')
        intent_moods = {
            'concentration': ['focus', 'calm', 'minimal'],
            'relaxation': ['chill', 'peaceful', 'ambient'],
            'energy': ['upbeat', 'energetic', 'driving'],
            'discovery': ['experimental', 'unique', 'interesting']
        }
        
        if primary_intent in intent_moods:
            all_moods.extend(intent_moods[primary_intent])
        
        # Create mood combinations for search
        import random
        if len(all_moods) >= 2:
            mood_combinations = [
                f"{mood1} {mood2}" 
                for mood1 in all_moods[:3] 
                for mood2 in all_moods[:3] 
                if mood1 != mood2
            ]
            selected_combinations = random.sample(
                mood_combinations, 
                min(5, len(mood_combinations))
            )
        else:
            selected_combinations = all_moods[:5]
        
        for combination in selected_combinations:
            try:
                tracks = await client.search_tracks(query=combination, limit=limit // 5)
                
                for track_metadata in tracks:
                    track = {
                        'name': track_metadata.name,
                        'artist': track_metadata.artist,
                        'url': track_metadata.url,
                        'listeners': track_metadata.listeners,
                        'playcount': track_metadata.playcount,
                        'mbid': track_metadata.mbid,
                        'tags': track_metadata.tags,
                        'mood_combination': combination,
                        'discovery_method': 'mood_based_serendipity'
                    }
                    candidates.append(track)
                
            except Exception as e:
                self.logger.warning(
                    "Mood-based serendipity failed",
                    combination=combination,
                    error=str(e)
                )
                continue
        
        return candidates
    
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tracks while preserving source diversity."""
        seen_tracks = {}
        unique_candidates = []
        
        for candidate in candidates:
            # Create unique key for track
            track_key = f"{candidate.get('artist', '').lower()}_{candidate.get('name', '').lower()}"
            
            if track_key not in seen_tracks:
                seen_tracks[track_key] = candidate
                unique_candidates.append(candidate)
            else:
                # Keep the candidate with higher source confidence
                existing = seen_tracks[track_key]
                if candidate.get('source_confidence', 0) > existing.get('source_confidence', 0):
                    # Replace in unique_candidates list
                    for i, unique_candidate in enumerate(unique_candidates):
                        if unique_candidate is existing:
                            unique_candidates[i] = candidate
                            seen_tracks[track_key] = candidate
                            break
        
        return unique_candidates
    
    def _generate_candidate_id(self, candidate: Dict[str, Any]) -> str:
        """Generate unique ID for candidate track."""
        # Use artist + track name + source for unique ID
        id_string = f"{candidate.get('artist', '')}_{candidate.get('name', '')}_{candidate.get('source', '')}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of the generation process."""
        return {
            'target_candidates': self.target_candidates,
            'source_distribution': self.source_distribution,
            'similarity_explorer': self.similarity_explorer.get_exploration_summary(),
            'components': ['multi_hop_similarity', 'underground_detection', 'serendipitous_discovery']
        } 