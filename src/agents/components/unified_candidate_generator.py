"""
Unified Candidate Generation Framework for BeatDebate

Consolidates candidate generation logic from multiple agents, eliminating duplication
and providing a single, configurable source for generating candidate tracks.
"""

from typing import Dict, List, Any, Optional
import structlog
import hashlib
from datetime import datetime

from ...services.api_service import APIService
from ...models.metadata_models import UnifiedTrackMetadata

logger = structlog.get_logger(__name__)


class UnifiedCandidateGenerator:
    """
    Unified candidate generator that consolidates logic from both
    EnhancedCandidateGenerator and EnhancedDiscoveryGenerator.
    
    Supports multiple generation strategies:
    - Genre/Mood focused generation (for GenreMoodAgent)
    - Discovery focused generation (for DiscoveryAgent)
    - Configurable source distributions
    """
    
    def __init__(self, api_service: APIService):
        """
        Initialize unified candidate generator with API service.
        
        Args:
            api_service: Unified API service for music data access
        """
        self.api_service = api_service
        self.logger = logger.bind(component="UnifiedCandidateGenerator")
        
        # Default generation parameters - INCREASED for better coverage
        self.target_candidates = 100  # Increased from 60
        self.final_recommendations = 25  # Increased from 20
        
        # Strategy configurations - REDUCED for performance
        self.strategy_configs = {
            'genre_mood': {
                'primary_search': 25,      # Reduced from 40
                'similar_artists': 20,     # Reduced from 30
                'genre_exploration': 10,   # Reduced from 20
                'underground_gems': 5      # Reduced from 10
            },
            'discovery': {
                'multi_hop_similarity': 30,  # Reduced from 50
                'underground_detection': 15, # Reduced from 30
                'serendipitous_discovery': 10 # Reduced from 20
            }
        }
        
        self.logger.info("Unified Candidate Generator initialized")
    
    async def generate_candidate_pool(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        agent_type: str = "genre_mood",
        target_candidates: Optional[int] = None,
        detected_intent: str = None
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate pool using specified strategy.
        
        Args:
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent
            agent_type: "genre_mood" or "discovery" for strategy selection
            target_candidates: Override default target candidate count
            detected_intent: Specific intent for enhanced candidate generation
            
        Returns:
            List of candidate tracks with source metadata
        """
        if target_candidates:
            self.target_candidates = target_candidates
            
        self.logger.info(
            "Starting unified candidate generation",
            agent_type=agent_type,
            target_candidates=self.target_candidates,
            detected_intent=detected_intent
        )
        
        # 🚀 PHASE 2: Intent-aware candidate generation strategy
        if detected_intent:
            return await self._generate_intent_aware_candidates(
                entities, intent_analysis, detected_intent, agent_type
            )
        
        # Fallback to original strategy selection
        if agent_type == "discovery":
            return await self._generate_discovery_candidates(entities, intent_analysis)
        else:
            return await self._generate_genre_mood_candidates(entities, intent_analysis)
    
    async def _generate_genre_mood_candidates(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate candidates using genre/mood strategy."""
        all_candidates = []
        strategy = self.strategy_configs['genre_mood']
        
        try:
            # Source 1: Primary Search
            primary_tracks = await self._get_primary_search_tracks(
                entities, intent_analysis, 
                limit=strategy['primary_search']
            )
            all_candidates.extend(primary_tracks)
            self.logger.debug(f"Primary search: {len(primary_tracks)} tracks")
            
            # Early termination check
            if len(all_candidates) >= self.target_candidates:
                self.logger.info("Early termination: sufficient candidates from primary search")
                return self._finalize_candidates(all_candidates, "genre_mood")
            
            # Source 2: Similar Artists
            similar_tracks = await self._get_similar_artist_tracks(entities)
            all_candidates.extend(similar_tracks)
            self.logger.debug(f"Similar artists: {len(similar_tracks)} tracks")
            
            # Early termination check
            if len(all_candidates) >= self.target_candidates:
                self.logger.info("Early termination: sufficient candidates after similar artists")
                return self._finalize_candidates(all_candidates, "genre_mood")
            
            # Source 3: Genre Exploration
            genre_tracks = await self._get_genre_exploration_tracks(
                entities, intent_analysis,
                limit=strategy['genre_exploration']
            )
            all_candidates.extend(genre_tracks)
            self.logger.debug(f"Genre exploration: {len(genre_tracks)} tracks")
            
            # Early termination check
            if len(all_candidates) >= self.target_candidates:
                self.logger.info("Early termination: sufficient candidates after genre exploration")
                return self._finalize_candidates(all_candidates, "genre_mood")
            
            # Source 4: Underground Gems (only if we still need more)
            underground_tracks = await self._get_underground_tracks(
                entities, intent_analysis,
                limit=strategy['underground_gems']
            )
            all_candidates.extend(underground_tracks)
            self.logger.debug(f"Underground gems: {len(underground_tracks)} tracks")
            
            return self._finalize_candidates(all_candidates, "genre_mood")
            
        except Exception as e:
            self.logger.error("Genre/mood candidate generation failed", error=str(e))
            return []
    
    async def _generate_discovery_candidates(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate candidates using discovery strategy."""
        all_candidates = []
        strategy = self.strategy_configs['discovery']
        
        try:
            # Extract context for discovery
            seed_artists = self._extract_seed_artists(entities)
            target_genres = self._extract_target_genres(entities)
            
            self.logger.debug(
                "Discovery context extracted",
                seed_artists=seed_artists,
                target_genres=target_genres
            )
            
            # Source 1: Multi-hop Similarity
            similarity_candidates = await self._get_multi_hop_similarity_tracks(
                seed_artists, entities, intent_analysis,
                limit=strategy['multi_hop_similarity']
            )
            all_candidates.extend(similarity_candidates)
            self.logger.debug(f"Multi-hop similarity: {len(similarity_candidates)} tracks")
            
            # Source 2: Underground Detection
            underground_candidates = await self._get_underground_detection_tracks(
                target_genres, entities, intent_analysis,
                limit=strategy['underground_detection']
            )
            all_candidates.extend(underground_candidates)
            self.logger.debug(f"Underground detection: {len(underground_candidates)} tracks")
            
            # Source 3: Serendipitous Discovery
            serendipitous_candidates = await self._get_serendipitous_tracks(
                entities, intent_analysis,
                limit=strategy['serendipitous_discovery']
            )
            all_candidates.extend(serendipitous_candidates)
            self.logger.debug(f"Serendipitous discovery: {len(serendipitous_candidates)} tracks")
            
            return self._finalize_candidates(all_candidates, "discovery")
            
        except Exception as e:
            self.logger.error("Discovery candidate generation failed", error=str(e))
            return []
    
    async def _get_primary_search_tracks(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Get tracks from primary search based on entities and intent."""
        tracks = []
        
        try:
            search_terms = self._extract_primary_search_terms(entities, intent_analysis)
            
            # OPTIMIZATION: Limit to max 3 search terms and use smaller requests
            for search_term in search_terms[:3]:  # Reduced from 5 to 3
                if len(tracks) >= limit:
                    break
                    
                try:
                    # Use smaller per-search limit 
                    per_search_limit = min(10, (limit - len(tracks)))
                    search_results = await self.api_service.search_tracks(
                        query=search_term,
                        limit=per_search_limit
                    )
                    
                    for track_metadata in search_results:
                        if len(tracks) >= limit:
                            break
                            
                        track = self._convert_metadata_to_dict(
                            track_metadata, 
                            source='primary_search',
                            source_confidence=0.8,
                            search_term=search_term
                        )
                        tracks.append(track)
                            
                except Exception as e:
                    self.logger.warning(
                        f"Primary search failed for term '{search_term}'", 
                        error=str(e)
                    )
                    continue
                    
        except Exception as e:
            self.logger.error("Primary search tracks failed", error=str(e))
        
        return tracks[:limit]
    
    async def _get_similar_artist_tracks(
        self, 
        entities: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate tracks from artists similar to the target artist."""
        tracks = []
        
        try:
            target_artists = self._extract_seed_artists(entities)
            
            for artist in target_artists[:2]:  # Limit to avoid too many API calls
                try:
                    # Use the correct API service method to get similar artist tracks
                    similar_tracks = await self.api_service.get_similar_artist_tracks(
                        artist=artist,
                        limit=25  # Increased from 15 for more similar artist tracks
                    )
                    
                    for track_metadata in similar_tracks:
                        track = self._convert_metadata_to_dict(
                            track_metadata,
                            source='similar_artist_tracks',
                            source_confidence=0.8,  # High confidence for similar artists
                            similar_to=artist
                        )
                        tracks.append(track)
                            
                except Exception as e:
                    self.logger.warning(
                        f"Failed to get similar artist tracks for '{artist}'", 
                        error=str(e)
                    )
                    continue
                    
        except Exception as e:
            self.logger.error("Similar artist tracks generation failed", error=str(e))
        
        return tracks
    
    async def _get_genre_exploration_tracks(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get tracks through genre/mood tag exploration."""
        tracks = []
        
        try:
            exploration_tags = self._extract_exploration_tags(entities, intent_analysis)
            
            # OPTIMIZATION: Limit to max 2 tags and use smaller requests  
            for tag in exploration_tags[:2]:  # Reduced from 4 to 2
                if len(tracks) >= limit:
                    break
                    
                try:
                    # Use smaller per-tag limit
                    per_tag_limit = min(5, (limit - len(tracks)))
                    tag_tracks = await self.api_service.get_tracks_by_tag(
                        tag=tag,
                        limit=per_tag_limit
                    )
                    
                    for track_metadata in tag_tracks:
                        if len(tracks) >= limit:
                            break
                            
                        track = self._convert_metadata_to_dict(
                            track_metadata,
                            source='genre_exploration',
                            source_confidence=0.6,
                            exploration_tag=tag
                        )
                        tracks.append(track)
                            
                except Exception as e:
                    self.logger.warning(
                        f"Genre exploration failed for tag '{tag}'", 
                        error=str(e)
                    )
                    continue
                    
        except Exception as e:
            self.logger.error("Genre exploration tracks failed", error=str(e))
        
        return tracks[:limit]
    
    async def _get_underground_tracks(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get underground/lesser-known tracks."""
        tracks = []
        
        try:
            underground_terms = self._extract_underground_terms(entities, intent_analysis)
            
            # Check if we have any underground terms
            if not underground_terms:
                self.logger.debug("No underground terms found, skipping underground search")
                return tracks
            
            # OPTIMIZATION: Limit to max 2 terms and use smaller requests
            for term in underground_terms[:2]:  # Reduced from 3 to 2
                if len(tracks) >= limit:
                    break
                    
                try:
                    # Use smaller per-term limit
                    per_term_limit = min(3, (limit - len(tracks)))
                    search_results = await self.api_service.search_tracks(
                        query=term,
                        limit=per_term_limit
                    )
                    
                    # Filter for underground tracks (low play count)
                    for track_metadata in search_results:
                        if len(tracks) >= limit:
                            break
                            
                        underground_score = self._calculate_underground_score(track_metadata)
                        if underground_score > 0.3:  # Threshold for "underground"
                            track = self._convert_metadata_to_dict(
                                track_metadata,
                                source='underground_gems',
                                source_confidence=underground_score,
                                underground_term=term
                            )
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
            self.logger.error("Underground tracks failed", error=str(e))
        
        return tracks[:limit]
    
    async def _get_multi_hop_similarity_tracks(
        self,
        seed_artists: List[str],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get tracks through multi-hop similarity exploration."""
        tracks = []
        
        try:
            # Use API service for similarity exploration
            for artist in seed_artists[:3]:
                try:
                    similar_tracks = await self.api_service.get_similar_artist_tracks(
                        artist=artist,
                        limit=min(20, limit // len(seed_artists) + 5)
                    )
                    
                    for track_metadata in similar_tracks:
                        track = self._convert_metadata_to_dict(
                            track_metadata,
                            source='multi_hop_similarity',
                            source_confidence=0.8,
                            seed_artist=artist
                        )
                        tracks.append(track)
                        
                        if len(tracks) >= limit:
                            break
                            
                except Exception as e:
                    self.logger.warning(
                        f"Multi-hop similarity failed for artist '{artist}'", 
                        error=str(e)
                    )
                    continue
                
                if len(tracks) >= limit:
                    break
                    
        except Exception as e:
            self.logger.error("Multi-hop similarity tracks failed", error=str(e))
        
        return tracks[:limit]
    
    async def _get_underground_detection_tracks(
        self,
        target_genres: List[str],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Get tracks through underground detection."""
        tracks = []
        
        try:
            # Check if we have any target genres
            if not target_genres:
                self.logger.debug("No target genres found, skipping underground detection")
                return tracks
            
            for genre in target_genres[:3]:
                try:
                    genre_tracks = await self.api_service.get_tracks_by_tag(
                        tag=genre,
                        limit=min(15, limit // len(target_genres) + 5)
                    )
                    
                    # Filter for underground tracks
                    for track_metadata in genre_tracks:
                        underground_score = self._calculate_underground_score(track_metadata)
                        if underground_score > 0.4:  # Higher threshold for discovery
                            track = self._convert_metadata_to_dict(
                                track_metadata,
                                source='underground_detection',
                                source_confidence=underground_score,
                                target_genre=genre
                            )
                            tracks.append(track)
                            
                            if len(tracks) >= limit:
                                break
                                
                except Exception as e:
                    self.logger.warning(
                        f"Underground detection failed for genre '{genre}'", 
                        error=str(e)
                    )
                    continue
                
                if len(tracks) >= limit:
                    break
                    
        except Exception as e:
            self.logger.error("Underground detection tracks failed", error=str(e))
        
        return tracks[:limit]
    
    async def _get_serendipitous_tracks(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get serendipitous discovery tracks."""
        tracks = []
        
        try:
            # Random genre exploration
            random_tracks = await self._random_genre_exploration(entities, limit // 2)
            tracks.extend(random_tracks)
            
            # Mood-based serendipity
            mood_tracks = await self._mood_based_serendipity(
                entities, intent_analysis, limit - len(tracks)
            )
            tracks.extend(mood_tracks)
            
        except Exception as e:
            self.logger.error("Serendipitous tracks failed", error=str(e))
        
        return tracks[:limit]
    
    async def _random_genre_exploration(
        self, entities: Dict[str, Any], limit: int
    ) -> List[Dict[str, Any]]:
        """Random genre exploration for serendipity."""
        tracks = []
        
        # Define exploration genres
        exploration_genres = [
            "experimental", "ambient", "post-rock", "neo-soul", 
            "math rock", "shoegaze", "trip-hop", "downtempo"
        ]
        
        try:
            for genre in exploration_genres[:3]:
                try:
                    genre_tracks = await self.api_service.get_tracks_by_tag(
                        tag=genre,
                        limit=min(5, limit // 3 + 2)
                    )
                    
                    for track_metadata in genre_tracks:
                        track = self._convert_metadata_to_dict(
                            track_metadata,
                            source='serendipitous_discovery',
                            source_confidence=0.5,
                            exploration_type='random_genre',
                            exploration_genre=genre
                        )
                        tracks.append(track)
                        
                        if len(tracks) >= limit:
                            break
                            
                except Exception as e:
                    self.logger.warning(
                        f"Random genre exploration failed for '{genre}'", 
                        error=str(e)
                    )
                    continue
                
                if len(tracks) >= limit:
                    break
                    
        except Exception as e:
            self.logger.error("Random genre exploration failed", error=str(e))
        
        return tracks[:limit]
    
    async def _mood_based_serendipity(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Mood-based serendipitous discovery."""
        tracks = []
        
        # Extract mood indicators
        mood_indicators = intent_analysis.get('mood_indicators', [])
        
        # Map moods to discovery tags
        mood_tag_map = {
            'energetic': ['electronic', 'dance', 'punk'],
            'calm': ['ambient', 'folk', 'classical'],
            'melancholic': ['indie', 'alternative', 'post-rock'],
            'uplifting': ['pop', 'soul', 'funk']
        }
        
        try:
            for mood in mood_indicators[:2]:
                if mood.lower() in mood_tag_map:
                    tags = mood_tag_map[mood.lower()]
                    
                    for tag in tags[:2]:
                        try:
                            mood_tracks = await self.api_service.get_tracks_by_tag(
                                tag=tag,
                                limit=min(3, limit // len(mood_indicators) + 1)
                            )
                            
                            for track_metadata in mood_tracks:
                                track = self._convert_metadata_to_dict(
                                    track_metadata,
                                    source='serendipitous_discovery',
                                    source_confidence=0.6,
                                    exploration_type='mood_based',
                                    mood_indicator=mood,
                                    mood_tag=tag
                                )
                                tracks.append(track)
                                
                                if len(tracks) >= limit:
                                    break
                                    
                        except Exception as e:
                            self.logger.warning(
                                f"Mood-based serendipity failed for tag '{tag}'", 
                                error=str(e)
                            )
                            continue
                        
                        if len(tracks) >= limit:
                            break
                    
                    if len(tracks) >= limit:
                        break
                        
        except Exception as e:
            self.logger.error("Mood-based serendipity failed", error=str(e))
        
        return tracks[:limit]
    
    def _finalize_candidates(
        self, all_candidates: List[Dict[str, Any]], generation_type: str
    ) -> List[Dict[str, Any]]:
        """Finalize candidate list with deduplication and metadata."""
        # Remove duplicates while preserving source information
        unique_candidates = self._deduplicate_candidates(all_candidates)
        
        # Limit to target count
        final_candidates = unique_candidates[:self.target_candidates]
        
        # Add generation metadata
        for candidate in final_candidates:
            candidate['generation_agent'] = f'unified_{generation_type}'
            candidate['generation_timestamp'] = self._get_timestamp()
            candidate['candidate_id'] = self._generate_candidate_id(candidate)
        
        self.logger.info(
            f"Unified {generation_type} candidate generation completed",
            total_candidates=len(all_candidates),
            unique_from_total=len(unique_candidates),
            final_candidates=len(final_candidates)
        )
        
        return final_candidates
    
    def _convert_metadata_to_dict(
        self, 
        track_metadata: UnifiedTrackMetadata, 
        source: str,
        source_confidence: float,
        **extra_metadata
    ) -> Dict[str, Any]:
        """Convert UnifiedTrackMetadata to dict with source information."""
        track = {
            'name': track_metadata.name,
            'artist': track_metadata.artist,
            'album': track_metadata.album,
            'url': getattr(track_metadata, 'url', ''),
            'listeners': getattr(track_metadata, 'listeners', 0),
            'playcount': getattr(track_metadata, 'playcount', 0),
            'mbid': getattr(track_metadata, 'mbid', ''),
            'tags': getattr(track_metadata, 'tags', []),
            'source': source,
            'source_confidence': source_confidence
        }
        
        # Add any extra metadata
        track.update(extra_metadata)
        
        return track
    
    def _extract_seed_artists(self, entities: Dict[str, Any]) -> List[str]:
        """Extract seed artists from entities for similarity exploration."""
        seed_artists = []
        
        musical_entities = entities.get("musical_entities", {})
        artists = musical_entities.get("artists", {})
        
        # Primary and similar-to artists
        seed_artists.extend(artists.get("primary", []))
        seed_artists.extend(artists.get("similar_to", []))
        
        # Fallback artists if none found
        if not seed_artists:
            seed_artists = [
                "Radiohead", "Bon Iver", "Thom Yorke", "Fleet Foxes", 
                "Sufjan Stevens", "Arcade Fire"
            ]
        
        return seed_artists[:5]
    
    def _extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target genres from entities for underground detection."""
        target_genres = []
        
        musical_entities = entities.get("musical_entities", {})
        genres = musical_entities.get("genres", {})
        
        # Primary and secondary genres
        target_genres.extend(genres.get("primary", []))
        target_genres.extend(genres.get("secondary", []))
        
        # Fallback genres if none found
        if not target_genres:
            target_genres = ["indie", "experimental", "electronic", "folk"]
        
        return target_genres[:4]
    
    def _extract_primary_search_terms(
        self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract primary search terms from entities and intent."""
        search_terms = []
        
        # Extract from musical entities
        musical_entities = entities.get("musical_entities", {})
        
        # Add artists
        artists = musical_entities.get("artists", {})
        search_terms.extend(artists.get("primary", []))
        
        # Add genres with mood modifiers
        genres = musical_entities.get("genres", {})
        mood_indicators = intent_analysis.get("mood_indicators", [])
        
        for genre in genres.get("primary", []):
            search_terms.append(genre)
            # Combine with mood for more specific search
            for mood in mood_indicators[:2]:
                search_terms.append(f"{mood} {genre}")
        
        # Add tracks if specified
        tracks = musical_entities.get("tracks", {})
        search_terms.extend(tracks.get("primary", []))
        
        # Fallback search terms
        if not search_terms:
            search_terms = ["indie rock", "electronic", "alternative", "experimental"]
        
        return search_terms[:8]
    
    def _extract_artists_from_entities(self, entities: Dict[str, Any]) -> List[str]:
        """Extract artist names from entities."""
        artists = []
        
        musical_entities = entities.get("musical_entities", {})
        artist_entities = musical_entities.get("artists", {})
        
        artists.extend(artist_entities.get("primary", []))
        artists.extend(artist_entities.get("similar_to", []))
        
        return list(set(artists))  # Remove duplicates
    
    def _get_fallback_artists(
        self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Get fallback artists when no artists are specified."""
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
        
        # Default fallback
        if not fallback_artists:
            fallback_artists = ["Radiohead", "Bon Iver", "Tame Impala", "Four Tet"]
        
        return fallback_artists[:5]
    
    def _extract_exploration_tags(
        self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract tags for genre exploration."""
        tags = []
        
        # Extract from entities
        musical_entities = entities.get("musical_entities", {})
        genres = musical_entities.get("genres", {})
        moods = musical_entities.get("moods", {})
        
        tags.extend(genres.get("primary", []))
        tags.extend(genres.get("secondary", []))
        tags.extend(moods.get("primary", []))
        
        # Add mood indicators from intent analysis
        mood_indicators = intent_analysis.get("mood_indicators", [])
        tags.extend(mood_indicators)
        
        # Fallback tags
        if not tags:
            tags = ["indie", "alternative", "electronic", "experimental"]
        
        return list(set(tags))[:6]  # Remove duplicates, limit to 6
    
    def _extract_underground_terms(
        self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract terms for underground track search."""
        terms = []
        
        # Combine genres with underground modifiers
        musical_entities = entities.get("musical_entities", {})
        genres = musical_entities.get("genres", {})
        
        underground_modifiers = ["underground", "obscure", "hidden", "rare"]
        
        for genre in genres.get("primary", [])[:2]:
            for modifier in underground_modifiers[:2]:
                terms.append(f"{modifier} {genre}")
        
        # Add general underground terms
        terms.extend(["underground music", "hidden gems", "rare tracks"])
        
        return terms[:5]
    
    def _calculate_underground_score(self, track_metadata: UnifiedTrackMetadata) -> float:
        """Calculate underground score for a track."""
        listeners = getattr(track_metadata, 'listeners', 0) or 0
        playcount = getattr(track_metadata, 'playcount', 0) or 0
        
        # Simple underground scoring based on popularity
        if listeners == 0 and playcount == 0:
            return 0.8  # Unknown tracks get high underground score
        
        # Lower listener/playcount = higher underground score
        max_listeners = 1000000  # Threshold for "popular"
        max_playcount = 10000000
        
        listener_score = max(0, 1 - (listeners / max_listeners))
        playcount_score = max(0, 1 - (playcount / max_playcount))
        
        return (listener_score + playcount_score) / 2
    
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tracks while preserving source diversity."""
        seen_tracks = {}
        unique_candidates = []
        
        for candidate in candidates:
            track_key = self._create_track_identifier(candidate)
            
            if track_key not in seen_tracks:
                seen_tracks[track_key] = candidate
                unique_candidates.append(candidate)
            else:
                # Keep candidate with higher source confidence
                existing = seen_tracks[track_key]
                if candidate.get('source_confidence', 0) > existing.get('source_confidence', 0):
                    # Replace in unique_candidates list
                    for i, unique_candidate in enumerate(unique_candidates):
                        if unique_candidate is existing:
                            unique_candidates[i] = candidate
                            seen_tracks[track_key] = candidate
                            break
        
        return unique_candidates
    
    def _create_track_identifier(self, track: Dict[str, Any]) -> str:
        """Create unique identifier for a track."""
        artist = track.get('artist', '').lower().strip()
        name = track.get('name', '').lower().strip()
        identifier = f"{artist}::{name}"
        return hashlib.md5(identifier.encode()).hexdigest()
    
    def _generate_candidate_id(self, candidate: Dict[str, Any]) -> str:
        """Generate unique ID for candidate track."""
        id_string = f"{candidate.get('artist', '')}_{candidate.get('name', '')}_{candidate.get('source', '')}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        return datetime.now().isoformat() 

    async def _generate_intent_aware_candidates(
        self, 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        intent: str,
        agent_type: str
    ) -> List[Dict[str, Any]]:
        """
        Generate candidates using intent-specific strategies from design document.
        
        Args:
            entities: Extracted entities
            intent_analysis: Intent analysis
            intent: Detected intent (artist_similarity, discovery, etc.)
            agent_type: Which agent is requesting candidates
            
        Returns:
            List of intent-optimized candidate tracks
        """
        self.logger.info(f"Generating intent-aware candidates for: {intent}")
        
        all_candidates = []
        
        if intent == 'by_artist':
            # Strategy: Include ONLY target artist's own tracks
            if agent_type == "discovery":
                # Focus exclusively on the target artist's discography
                all_candidates.extend(
                    await self._generate_target_artist_tracks(entities)
                )
                # Add additional tracks by the same artist if needed
                target_artists = self._extract_seed_artists(entities)
                if target_artists:
                    for artist in target_artists[:2]:
                        try:
                            # Get more deep cuts from the artist
                            additional_tracks = await self.api_service.get_artist_top_tracks(
                                artist=artist,
                                limit=20  # Get more tracks for by_artist queries
                            )
                            for track_metadata in additional_tracks:
                                track = self._convert_metadata_to_dict(
                                    track_metadata,
                                    source='by_artist_deep_cuts',
                                    source_confidence=0.9,
                                    target_artist=artist
                                )
                                all_candidates.append(track)
                        except Exception as e:
                            self.logger.warning(f"Failed to get additional tracks by {artist}: {e}")
            elif agent_type == "genre_mood":
                # Support with artist-focused tracks organized by style
                all_candidates.extend(
                    await self._generate_target_artist_tracks(entities)
                )
        
        elif intent == 'artist_similarity':
            # Strategy: Focus on similar artists only (NOT target artist's own tracks)
            if agent_type == "discovery":
                # Focus ONLY on similar artists - users want "music LIKE artist", not "music BY artist"
                all_candidates.extend(
                    await self._get_similar_artist_tracks(entities)
                )
                # Add some genre exploration to broaden the similarity search
                similar_style_tracks = await self._get_genre_exploration_tracks(entities, {}, limit=15)
                all_candidates.extend(similar_style_tracks)
            elif agent_type == "genre_mood":
                # Support with style-consistent tracks
                all_candidates.extend(
                    await self._generate_style_consistent_tracks(entities)
                )
        
        elif intent == 'discovery':
            # Strategy: Focus on serendipitous and underground sources
            if agent_type == "discovery":
                all_candidates.extend(
                    await self._generate_underground_gems(entities)
                )
                all_candidates.extend(
                    await self._generate_serendipitous_discoveries(entities)
                )
            elif agent_type == "genre_mood":
                # Support with genre exploration
                all_candidates.extend(
                    await self._generate_genre_exploration_tracks(entities)
                )
        
        elif intent == 'genre_mood':
            # Strategy: Broad genre-based search with mood filtering
            if agent_type == "genre_mood":
                all_candidates.extend(
                    await self._generate_genre_focused_tracks(entities)
                )
                all_candidates.extend(
                    await self._generate_mood_filtered_tracks(entities)
                )
            elif agent_type == "discovery":
                # Support with genre diversity
                all_candidates.extend(
                    await self._generate_genre_diverse_tracks(entities)
                )
        
        elif intent == 'contextual':
            # Strategy: Audio feature-driven candidate generation
            if agent_type == "genre_mood":
                all_candidates.extend(
                    await self._generate_audio_feature_tracks(entities, intent_analysis)
                )
                all_candidates.extend(
                    await self._generate_functional_music_tracks(entities, intent_analysis)
                )
            elif agent_type == "discovery":
                # Support with activity-specific discovery
                all_candidates.extend(
                    await self._generate_activity_discovery_tracks(entities, intent_analysis)
                )
        
        elif intent == 'hybrid':
            # Strategy: Balanced approach combining similarity and mood
            if agent_type == "discovery":
                # For hybrid discovery, we want underground + genre focused
                all_candidates.extend(
                    await self._generate_underground_gems(entities)
                )
                all_candidates.extend(
                    await self._generate_genre_focused_discovery(entities)
                )
                # Also add similar artist tracks if available
                similar_tracks = await self._get_similar_artist_tracks(entities)
                if similar_tracks:
                    all_candidates.extend(similar_tracks)
            elif agent_type == "genre_mood":
                all_candidates.extend(
                    await self._generate_hybrid_style_tracks(entities)
                )
        
        # Fallback to default generation if no intent-specific candidates
        if not all_candidates:
            self.logger.warning(f"No intent-specific candidates for {intent}, falling back to default")
            if agent_type == "discovery":
                all_candidates = await self._generate_discovery_candidates(entities, intent_analysis)
            else:
                all_candidates = await self._generate_genre_mood_candidates(entities, intent_analysis)
        
        # Deduplicate and limit
        unique_candidates = self._deduplicate_candidates(all_candidates)
        return unique_candidates[:self.target_candidates]
    
    async def _generate_target_artist_tracks(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tracks by the target artist for artist similarity queries."""
        tracks = []
        
        try:
            target_artists = self._extract_seed_artists(entities)
            
            for artist in target_artists[:2]:  # Limit to avoid too many tracks
                try:
                    artist_tracks = await self.api_service.get_artist_top_tracks(
                        artist=artist,
                        limit=15  # Increased from 10 for more target artist tracks
                    )
                    
                    for track_metadata in artist_tracks:
                        track = self._convert_metadata_to_dict(
                            track_metadata,
                            source='target_artist_tracks',
                            source_confidence=0.95,
                            target_artist=artist
                        )
                        tracks.append(track)
                        
                except Exception as e:
                    self.logger.warning(
                        f"Failed to get tracks by target artist '{artist}'", 
                        error=str(e)
                    )
                    continue
                    
        except Exception as e:
            self.logger.error("Target artist tracks generation failed", error=str(e))
        
        return tracks[:20]  # Limit to 20 tracks
    
    async def _generate_style_consistent_tracks(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tracks that match the style of target artists."""
        return await self._get_genre_exploration_tracks(entities, {}, limit=20)
    
    async def _generate_underground_gems(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate underground/hidden gem tracks."""
        return await self._get_underground_tracks(entities, {}, limit=25)
    
    async def _generate_serendipitous_discoveries(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate serendipitous discovery tracks."""
        return await self._get_serendipitous_tracks(entities, {}, limit=20)
    
    async def _generate_genre_exploration_tracks(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate genre exploration tracks."""
        return await self._get_genre_exploration_tracks(entities, {}, limit=15)
    
    async def _generate_genre_diverse_tracks(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate genre diverse tracks for discovery support."""
        return await self._get_genre_exploration_tracks(entities, {}, limit=10)
    
    async def _generate_genre_focused_tracks(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate genre-focused tracks."""
        return await self._get_primary_search_tracks(entities, {}, limit=25)
    
    async def _generate_mood_filtered_tracks(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mood-filtered tracks."""
        return await self._mood_based_serendipity(entities, {}, limit=15)
    
    async def _generate_audio_feature_tracks(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tracks based on audio features for contextual queries."""
        return await self._get_primary_search_tracks(entities, intent_analysis, limit=20)
    
    async def _generate_functional_music_tracks(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate functional music tracks for specific activities."""
        return await self._get_genre_exploration_tracks(entities, intent_analysis, limit=15)
    
    async def _generate_activity_discovery_tracks(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate activity-specific discovery tracks."""
        return await self._get_underground_tracks(entities, intent_analysis, limit=10)
    
    async def _generate_mood_discovery_tracks(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mood-based discovery tracks."""
        return await self._mood_based_serendipity(entities, {}, limit=15)
    
    async def _generate_hybrid_style_tracks(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hybrid style tracks balancing similarity and mood."""
        # Combine similar artist tracks and genre exploration
        similar_tracks = await self._get_similar_artist_tracks(entities)
        genre_tracks = await self._get_genre_exploration_tracks(entities, {}, limit=15)
        
        all_tracks = similar_tracks + genre_tracks
        return self._deduplicate_candidates(all_tracks)[:20]

    async def _generate_genre_focused_discovery(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate genre-focused discovery tracks."""
        return await self._get_genre_exploration_tracks(entities, {}, limit=15) 