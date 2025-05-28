"""
Enhanced Candidate Generation Framework for BeatDebate

Generates large pools of candidate tracks (100) from multiple sources
for quality filtering and ranking by GenreMoodAgent and DiscoveryAgent.
"""

from typing import Dict, List, Any
import structlog
import hashlib

logger = structlog.get_logger(__name__)


class EnhancedCandidateGenerator:
    """
    Generates large pools of candidate tracks for quality filtering.
    
    Strategy: Generate 100 candidates from multiple sources, then filter to 
    top 20 using comprehensive quality scoring.
    """
    
    def __init__(self, lastfm_client):
        """
        Initialize candidate generator with Last.fm client.
        
        Args:
            lastfm_client: Last.fm API client for track search
        """
        self.lastfm = lastfm_client
        self.logger = logger.bind(component="CandidateGenerator")
        
        # Target numbers for candidate generation
        self.target_candidates = 100
        self.final_recommendations = 20
        
        # Source distribution for balanced candidate generation
        self.source_distribution = {
            'primary_search': 40,      # 40 tracks from main search
            'similar_artists': 30,     # 30 tracks from artist similarity
            'genre_exploration': 20,   # 20 tracks from genre/mood tags
            'underground_gems': 10     # 10 tracks from underground detection
        }
        
        self.logger.info("Enhanced Candidate Generator initialized")
    
    async def generate_candidate_pool(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        agent_type: str = "genre_mood"
    ) -> List[Dict]:
        """
        Generate 100 candidate tracks from multiple sources.
        
        Args:
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent
            agent_type: "genre_mood" or "discovery" for specialized generation
            
        Returns:
            List of up to 100 candidate tracks with source metadata
        """
        self.logger.info(
            "Starting candidate generation", 
            agent_type=agent_type,
            target_candidates=self.target_candidates
        )
        
        all_candidates = []
        
        try:
            # Source 1: Primary Search (40 tracks)
            primary_tracks = await self._get_primary_search_tracks(
                entities, intent_analysis, 
                limit=self.source_distribution['primary_search']
            )
            all_candidates.extend(primary_tracks)
            self.logger.debug(f"Primary search: {len(primary_tracks)} tracks")
            
            # Source 2: Similar Artists (30 tracks)
            similar_tracks = await self._get_similar_artist_tracks(
                entities, intent_analysis,
                limit=self.source_distribution['similar_artists']
            )
            all_candidates.extend(similar_tracks)
            self.logger.debug(f"Similar artists: {len(similar_tracks)} tracks")
            
            # Source 3: Genre/Mood Exploration (20 tracks)
            genre_tracks = await self._get_genre_exploration_tracks(
                entities, intent_analysis,
                limit=self.source_distribution['genre_exploration']
            )
            all_candidates.extend(genre_tracks)
            self.logger.debug(f"Genre exploration: {len(genre_tracks)} tracks")
            
            # Source 4: Underground Gems (10 tracks)
            underground_tracks = await self._get_underground_tracks(
                entities, intent_analysis,
                limit=self.source_distribution['underground_gems']
            )
            all_candidates.extend(underground_tracks)
            self.logger.debug(f"Underground gems: {len(underground_tracks)} tracks")
            
            # Remove duplicates while preserving source information
            unique_candidates = self._deduplicate_preserve_sources(all_candidates)
            
            # Ensure we have enough candidates
            final_candidates = unique_candidates[:self.target_candidates]
            
            self.logger.info(
                "Candidate generation completed",
                total_candidates=len(final_candidates),
                unique_from_total=len(unique_candidates),
                sources_used=len([s for s in self.source_distribution.keys() 
                                if any(c.get('source') == s for c in final_candidates)])
            )
            
            return final_candidates
            
        except Exception as e:
            self.logger.error("Candidate generation failed", error=str(e))
            # Return empty list on failure - calling agent will handle fallback
            return []
    
    async def _get_primary_search_tracks(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        limit: int = 40
    ) -> List[Dict]:
        """Get tracks from primary search based on entities and intent."""
        tracks = []
        
        try:
            # Extract search terms from entities
            search_terms = self._extract_primary_search_terms(entities, intent_analysis)
            
            # Create a new Last.fm client for this search session
            from ..api.lastfm_client import LastFmClient
            
            async with LastFmClient(
                api_key=self.lastfm.api_key,
                rate_limit=self.lastfm.rate_limiter.calls_per_second
            ) as client:
                # Search for tracks using multiple search terms
                for search_term in search_terms[:5]:  # Limit to 5 search terms
                    try:
                        search_results = await client.search_tracks(
                            query=search_term,
                            limit=min(15, limit // len(search_terms) + 5)
                        )
                        
                        for track_metadata in search_results:
                            # Convert TrackMetadata to dict
                            track = {
                                'name': track_metadata.name,
                                'artist': track_metadata.artist,
                                'url': track_metadata.url,
                                'listeners': track_metadata.listeners,
                                'playcount': track_metadata.playcount,
                                'mbid': track_metadata.mbid,
                                'tags': track_metadata.tags,
                                'source': 'primary_search',
                                'search_term': search_term,
                                'source_confidence': 0.8
                            }
                            tracks.append(track)
                            
                            if len(tracks) >= limit:
                                break
                                
                    except Exception as e:
                        self.logger.warning(
                            f"Primary search failed for term '{search_term}'", 
                            error=str(e)
                        )
                        continue
                    
                    if len(tracks) >= limit:
                        break
                        
        except Exception as e:
            self.logger.error("Primary search tracks failed", error=str(e))
        
        return tracks[:limit]
    
    async def _get_similar_artist_tracks(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        limit: int = 30
    ) -> List[Dict]:
        """Get tracks from artists similar to those mentioned in entities."""
        tracks = []
        
        try:
            # Extract artists from entities
            artists = self._extract_artists_from_entities(entities)
            
            if not artists:
                # Use fallback artists based on genres/moods
                artists = self._get_fallback_artists(entities, intent_analysis)
            
            # Create a new Last.fm client for this search session
            from ..api.lastfm_client import LastFmClient
            
            async with LastFmClient(
                api_key=self.lastfm.api_key,
                rate_limit=self.lastfm.rate_limiter.calls_per_second
            ) as client:
                # Get tracks from similar artists
                for artist in artists[:5]:  # Limit to 5 artists
                    try:
                        # Get artist's top tracks
                        artist_tracks = await client.get_artist_top_tracks(
                            artist=artist,
                            limit=min(10, limit // len(artists) + 3)
                        )
                        
                        for track_metadata in artist_tracks:
                            # Convert TrackMetadata to dict
                            track = {
                                'name': track_metadata.name,
                                'artist': track_metadata.artist,
                                'url': track_metadata.url,
                                'listeners': track_metadata.listeners,
                                'playcount': track_metadata.playcount,
                                'mbid': track_metadata.mbid,
                                'tags': track_metadata.tags,
                                'source': 'similar_artists',
                                'source_artist': artist,
                                'source_confidence': 0.7
                            }
                            tracks.append(track)
                            
                            if len(tracks) >= limit:
                                break
                                
                    except Exception as e:
                        self.logger.warning(
                            f"Similar artist search failed for '{artist}'", 
                            error=str(e)
                        )
                        continue
                    
                    if len(tracks) >= limit:
                        break
                        
        except Exception as e:
            self.logger.error("Similar artist tracks failed", error=str(e))
        
        return tracks[:limit]
    
    async def _get_genre_exploration_tracks(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        limit: int = 20
    ) -> List[Dict]:
        """Get tracks through genre and mood tag exploration."""
        tracks = []
        
        try:
            # Extract genres and moods for exploration
            exploration_tags = self._extract_exploration_tags(entities, intent_analysis)
            
            # Create a new Last.fm client for this search session
            from ..api.lastfm_client import LastFmClient
            
            async with LastFmClient(
                api_key=self.lastfm.api_key,
                rate_limit=self.lastfm.rate_limiter.calls_per_second
            ) as client:
                # Search by tags
                for tag in exploration_tags[:4]:  # Limit to 4 tags
                    try:
                        tag_tracks = await client.search_tracks(
                            query=tag,
                            limit=min(8, limit // len(exploration_tags) + 2)
                        )
                        
                        for track_metadata in tag_tracks:
                            # Convert TrackMetadata to dict
                            track = {
                                'name': track_metadata.name,
                                'artist': track_metadata.artist,
                                'url': track_metadata.url,
                                'listeners': track_metadata.listeners,
                                'playcount': track_metadata.playcount,
                                'mbid': track_metadata.mbid,
                                'tags': track_metadata.tags,
                                'source': 'genre_exploration',
                                'exploration_tag': tag,
                                'source_confidence': 0.6
                            }
                            tracks.append(track)
                            
                            if len(tracks) >= limit:
                                break
                                
                    except Exception as e:
                        self.logger.warning(
                            f"Genre exploration failed for tag '{tag}'", 
                            error=str(e)
                        )
                        continue
                    
                    if len(tracks) >= limit:
                        break
                        
        except Exception as e:
            self.logger.error("Genre exploration tracks failed", error=str(e))
        
        return tracks[:limit]
    
    async def _get_underground_tracks(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict]:
        """Get underground/lesser-known tracks based on entities."""
        tracks = []
        
        try:
            # Extract underground search terms
            underground_terms = self._extract_underground_terms(entities, intent_analysis)
            
            # Create a new Last.fm client for this search session
            from ..api.lastfm_client import LastFmClient
            
            async with LastFmClient(
                api_key=self.lastfm.api_key,
                rate_limit=self.lastfm.rate_limiter.calls_per_second
            ) as client:
                # Search for underground tracks
                for term in underground_terms[:3]:  # Limit to 3 terms
                    try:
                        underground_results = await client.search_tracks(
                            query=term,
                            limit=min(5, limit // len(underground_terms) + 2)
                        )
                        
                        # Filter for potentially underground tracks (lower play counts)
                        for track_metadata in underground_results:
                            playcount = int(track_metadata.playcount or 0)
                            listeners = int(track_metadata.listeners or 0)
                            
                            # Simple underground heuristic: lower play counts
                            if playcount < 100000 or listeners < 10000:
                                # Convert TrackMetadata to dict
                                track = {
                                    'name': track_metadata.name,
                                    'artist': track_metadata.artist,
                                    'url': track_metadata.url,
                                    'listeners': track_metadata.listeners,
                                    'playcount': track_metadata.playcount,
                                    'mbid': track_metadata.mbid,
                                    'tags': track_metadata.tags,
                                    'source': 'underground_gems',
                                    'underground_term': term,
                                    'source_confidence': 0.5,
                                    'underground_score': self._calculate_simple_underground_score(track_metadata)
                                }
                                tracks.append(track)
                                
                                if len(tracks) >= limit:
                                    break
                                    
                    except Exception as e:
                        self.logger.warning(
                            f"Underground search failed for term '{term}'", 
                            error=str(e)
                        )
                        continue
                    
                    if len(tracks) >= limit:
                        break
                        
        except Exception as e:
            self.logger.error("Underground tracks search failed", error=str(e))
        
        return tracks[:limit]
    
    def _extract_primary_search_terms(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract primary search terms from entities and intent."""
        search_terms = []
        
        # Add genres
        genres = entities.get("musical_entities", {}).get("genres", {}).get("primary", [])
        search_terms.extend(genres)
        
        # Add moods
        moods = entities.get("contextual_entities", {}).get("moods", {})
        for mood_category in moods.values():
            search_terms.extend(mood_category)
        
        # Add activities as context
        activities = entities.get("contextual_entities", {}).get("activities", {})
        for activity_category in activities.values():
            search_terms.extend(activity_category)
        
        # Add combined terms
        if genres and moods:
            combined_terms = [
                f"{genre} {mood}" 
                for genre in genres[:2] 
                for mood_list in moods.values() 
                for mood in mood_list[:2]
            ]
            search_terms.extend(combined_terms[:3])
        
        # Remove duplicates and empty terms
        unique_terms = list(set([
            term for term in search_terms 
            if term and len(term) > 2
        ]))
        
        return unique_terms[:10]  # Return top 10 search terms
    
    def _extract_artists_from_entities(self, entities: Dict[str, Any]) -> List[str]:
        """Extract artist names from entities."""
        artists = []
        
        musical_entities = entities.get("musical_entities", {})
        artist_entities = musical_entities.get("artists", {})
        
        # Add primary artists
        artists.extend(artist_entities.get("primary", []))
        
        # Add similar_to artists
        artists.extend(artist_entities.get("similar_to", []))
        
        return list(set(artists))  # Remove duplicates
    
    def _get_fallback_artists(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Get fallback artists when no artists are specified in entities."""
        fallback_artists = []
        
        # Genre-based fallback artists
        genres = entities.get("musical_entities", {}).get("genres", {}).get("primary", [])
        
        genre_artist_map = {
            "rock": ["Radiohead", "Arctic Monkeys", "The Strokes"],
            "electronic": ["Daft Punk", "Aphex Twin", "Burial"],
            "indie": ["Bon Iver", "Fleet Foxes", "Vampire Weekend"],
            "jazz": ["Miles Davis", "John Coltrane", "Bill Evans"],
            "pop": ["The Beatles", "Prince", "David Bowie"],
            "hip hop": ["Kendrick Lamar", "J Dilla", "MF DOOM"],
            "classical": ["Bach", "Mozart", "Beethoven"]
        }
        
        for genre in genres:
            if genre.lower() in genre_artist_map:
                fallback_artists.extend(genre_artist_map[genre.lower()])
        
        # If no genre matches, use general popular artists
        if not fallback_artists:
            fallback_artists = ["Radiohead", "Bon Iver", "Tame Impala", "Four Tet"]
        
        return fallback_artists[:5]
    
    def _extract_exploration_tags(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract tags for genre/mood exploration."""
        tags = []
        
        # Add genre tags
        genres = entities.get("musical_entities", {}).get("genres", {}).get("primary", [])
        tags.extend(genres)
        
        # Add mood tags
        moods = entities.get("contextual_entities", {}).get("moods", {})
        for mood_category in moods.values():
            tags.extend(mood_category)
        
        # Add activity-based tags
        activities = entities.get("contextual_entities", {}).get("activities", {})
        activity_tags = {
            "workout": ["energetic", "upbeat", "motivational"],
            "studying": ["ambient", "instrumental", "focus"],
            "relaxing": ["chill", "peaceful", "calm"],
            "party": ["dance", "electronic", "upbeat"]
        }
        
        for activity_category in activities.values():
            for activity in activity_category:
                if activity in activity_tags:
                    tags.extend(activity_tags[activity])
        
        return list(set(tags))[:8]  # Return unique tags, max 8
    
    def _extract_underground_terms(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract terms for underground track discovery."""
        underground_terms = []
        
        # Add underground-specific genre modifiers
        genres = entities.get("musical_entities", {}).get("genres", {}).get("primary", [])
        for genre in genres:
            underground_terms.extend([
                f"underground {genre}",
                f"indie {genre}",
                f"experimental {genre}"
            ])
        
        # Add general underground terms
        underground_terms.extend([
            "underground",
            "indie",
            "experimental",
            "lo-fi",
            "bedroom",
            "DIY"
        ])
        
        return underground_terms[:6]
    
    def _calculate_simple_underground_score(self, track_metadata) -> float:
        """Calculate a simple underground score for a track."""
        playcount = int(track_metadata.playcount or 0)
        listeners = int(track_metadata.listeners or 0)
        
        # Simple scoring: lower counts = higher underground score
        playcount_score = max(0, 1.0 - (playcount / 1000000))  # Normalize to 1M plays
        listeners_score = max(0, 1.0 - (listeners / 100000))   # Normalize to 100K listeners
        
        return (playcount_score + listeners_score) / 2
    
    def _deduplicate_preserve_sources(self, candidates: List[Dict]) -> List[Dict]:
        """Remove duplicate tracks while preserving source information."""
        seen_tracks = set()
        unique_candidates = []
        
        for track in candidates:
            # Create unique identifier for track
            track_id = self._create_track_identifier(track)
            
            if track_id not in seen_tracks:
                seen_tracks.add(track_id)
                unique_candidates.append(track)
        
        return unique_candidates
    
    def _create_track_identifier(self, track: Dict) -> str:
        """Create unique identifier for a track."""
        # Use artist + track name for identification
        artist = track.get('artist', '').lower().strip()
        name = track.get('name', '').lower().strip()
        
        # Create hash of artist + name for consistent identification
        identifier = f"{artist}::{name}"
        return hashlib.md5(identifier.encode()).hexdigest() 