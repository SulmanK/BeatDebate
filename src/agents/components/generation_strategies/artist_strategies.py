"""
Artist-focused Generation Strategies

Strategies that focus on generating candidates based on specific artists
or similar artists.
"""

from typing import Dict, List, Any
from .base_strategy import BaseGenerationStrategy


class TargetArtistStrategy(BaseGenerationStrategy):
    """
    Strategy for generating tracks from specific target artists.
    
    Used for intents like 'by_artist' where users want tracks
    from a particular artist's discography.
    """
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate tracks from target artists' discographies.
        
        Args:
            entities: Extracted entities containing target artists
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of tracks from target artists
        """
        candidates = []
        target_artists = self._extract_seed_artists(entities)
        
        if not target_artists:
            self.logger.warning("No target artists found for TargetArtistStrategy")
            return candidates
        
        self.logger.info(f"🎯 ENHANCED ARTIST GENERATION: Generating diverse tracks for {', '.join(target_artists)}")
        
        # Strategy 1: Get top tracks from each target artist
        for artist in target_artists[:3]:  # Limit to top 3 artists
            try:
                # Get main tracks
                main_tracks = await self.api_service.get_artist_top_tracks(
                    artist=artist,
                    limit=min(100, limit // len(target_artists))
                )
                
                for track_metadata in main_tracks:
                    track = self._convert_metadata_to_dict(
                        track_metadata,
                        source='target_artist_main',
                        source_confidence=0.9,
                        target_artist=artist
                    )
                    candidates.append(track)
                
                self.logger.info(f"🎯 Strategy 1 (direct): {len(main_tracks)} tracks from {artist}")
                
            except Exception as e:
                self.logger.warning(f"Failed to get tracks from {artist}: {e}")
                continue
        
        self.logger.info(f"🎯 TOTAL ARTIST CANDIDATES: {len(candidates)} tracks for {', '.join(target_artists)}")
        return candidates[:limit]


class SimilarArtistStrategy(BaseGenerationStrategy):
    """
    Enhanced strategy for generating tracks from artists similar to the target artists.
    
    Uses multi-hop similarity exploration and style-aware filtering to find
    genuinely similar artists rather than falling back to mainstream options.
    """
    
    def __init__(self, api_service, llm_client=None, **kwargs):
        super().__init__(api_service, llm_client=llm_client, **kwargs)
        # Import here to avoid circular imports
        from ...discovery.similarity_explorer import SimilarityExplorer
        self.similarity_explorer = SimilarityExplorer(api_service)
        self.style_threshold = 0.3  # Minimum style similarity required
        self.max_popularity_threshold = 0.7  # Avoid overly mainstream artists
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate tracks from artists similar to target artists using enhanced similarity exploration.
        
        Args:
            entities: Extracted entities containing target artists
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of tracks from genuinely similar artists
        """
        candidates = []
        seed_artists = self._extract_seed_artists(entities)
        
        if not seed_artists:
            self.logger.warning("No seed artists found for SimilarArtistStrategy")
            return candidates
        
        self.logger.info(f"🔄 ENHANCED SIMILAR ARTIST GENERATION: Finding artists similar to {', '.join(seed_artists)}")
        
        # Strategy 1: Multi-hop similarity exploration (primary approach)
        try:
            similarity_tracks = await self.similarity_explorer.explore_multi_hop_similarity(
                seed_artists=seed_artists,
                target_tracks=min(120, limit),  # Increased from 40 to 120 for larger pools
                exploration_depth=3,
                exclude_seed_artists=True  # Exclude original artists for similarity search
            )
            
            # Filter and enhance similarity tracks
            for track in similarity_tracks:
                enhanced_track = await self._enhance_similarity_track(track, seed_artists)
                if enhanced_track:
                    candidates.append(enhanced_track)
            
            self.logger.info(f"🔄 Multi-hop similarity: {len(candidates)} high-quality tracks found")
            
        except Exception as e:
            self.logger.warning(f"Multi-hop similarity exploration failed: {e}")
        
        # Strategy 2: Direct API similar artists (backup with style filtering)
        if len(candidates) < limit * 0.6:  # More aggressive backup threshold
            api_candidates = await self._get_direct_similar_artists(seed_artists, limit - len(candidates))
            candidates.extend(api_candidates)
        
        # Strategy 3: Tag-based similarity discovery (tertiary)
        # 🚫 TEMPORARILY DISABLED: Testing without tag-based similarity to improve quality
        # if len(candidates) < limit * 0.9:  # More aggressive tag-based discovery
        #     tag_candidates = await self._get_tag_based_similar_tracks(seed_artists, limit - len(candidates))
        #     candidates.extend(tag_candidates)
        
        # Final filtering and ranking
        filtered_candidates = await self._apply_style_similarity_filter(candidates, seed_artists)
        
        self.logger.info(f"🔄 TOTAL ENHANCED SIMILARITY CANDIDATES: {len(filtered_candidates)} tracks")
        return filtered_candidates[:limit]
    
    async def _enhance_similarity_track(self, track: Dict[str, Any], seed_artists: List[str]) -> Dict[str, Any]:
        """Enhance a similarity track with additional metadata and filtering."""
        try:
            # Calculate style similarity score
            style_score = await self._calculate_style_similarity(track, seed_artists)
            
            # Filter out tracks that aren't similar enough
            if style_score < self.style_threshold:
                return None
            
            # Convert to our standard format
            enhanced_track = {
                'name': track.get('name', ''),
                'artist': track.get('artist', ''),
                'source': 'enhanced_similarity',
                'source_confidence': min(0.9, 0.6 + style_score * 0.3),
                'similarity_score': style_score,
                'similarity_hop': track.get('similarity_hop', 1),
                'discovery_path': track.get('discovery_path', ''),
                'listeners': track.get('listeners', 0),
                'playcount': track.get('playcount', 0),
                'seed_artists': seed_artists,
                'popularity_score': self._calculate_popularity_score(track)
            }
            
            return enhanced_track
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance similarity track {track.get('name', '')}: {e}")
            return None
    
    async def _get_direct_similar_artists(self, seed_artists: List[str], limit: int) -> List[Dict[str, Any]]:
        """Get similar artists using direct API calls with style filtering."""
        candidates = []
        
        for seed_artist in seed_artists[:2]:
            try:
                # Get more similar artists than before
                similar_artists = await self.api_service.get_similar_artists(
                    artist=seed_artist,
                    limit=15  # Increased from 10 to 15 for larger pools
                )
                
                if not similar_artists:
                    continue
                
                # Filter artists by style and popularity
                filtered_artists = []
                for similar_artist in similar_artists:
                    artist_name = getattr(similar_artist, 'name', str(similar_artist))
                    
                    # Check if artist is too mainstream
                    if await self._is_too_mainstream(artist_name):
                        continue
                    
                    # Check style compatibility
                    style_compatible = await self._check_style_compatibility(artist_name, seed_artist)
                    if style_compatible:
                        filtered_artists.append(artist_name)
                
                # Get tracks from filtered similar artists
                for artist_name in filtered_artists[:8]:  # Increased from 5 to 8 artists
                    try:
                        tracks = await self.api_service.get_artist_top_tracks(
                            artist=artist_name,
                            limit=min(12, limit // (len(seed_artists) * 2))  # More tracks per artist
                        )
                        
                        for track_metadata in tracks:
                            track = self._convert_metadata_to_dict(
                                track_metadata,
                                source='filtered_similar_artist',
                                source_confidence=0.75,
                                seed_artist=seed_artist,
                                similar_artist=artist_name
                            )
                            candidates.append(track)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to get tracks from filtered similar artist {artist_name}: {e}")
                        continue
                
            except Exception as e:
                self.logger.warning(f"Failed to get similar artists for {seed_artist}: {e}")
                continue
        
        self.logger.info(f"🔄 Direct API similarity (filtered): {len(candidates)} tracks")
        return candidates
    
    async def _get_tag_based_similar_tracks(self, seed_artists: List[str], limit: int) -> List[Dict[str, Any]]:
        """Get similar tracks using tag-based discovery."""
        candidates = []
        
        try:
            # Get tags from seed artists
            all_tags = []
            for artist in seed_artists:
                try:
                    artist_info = await self.api_service.get_artist_info(artist)
                    if artist_info and hasattr(artist_info, 'tags') and artist_info.tags:
                        # Filter out generic tags, keep specific style tags
                        style_tags = [tag for tag in artist_info.tags[:10] 
                                    if not self._is_generic_tag(tag)]
                        all_tags.extend(style_tags[:5])  # Top 5 style tags per artist
                except Exception as e:
                    self.logger.warning(f"Failed to get tags for {artist}: {e}")
            
            # Use unique tags for track discovery
            unique_tags = list(set(all_tags))[:8]  # Increased from 5 to 8 unique style tags
            
            if unique_tags:
                self.logger.info(f"🏷️ Using style tags: {', '.join(unique_tags)}")
                
                # Search by tags
                for tag in unique_tags:
                    try:
                        tag_tracks = await self.api_service.search_tracks_by_tags(
                            tags=[tag],
                            limit=min(15, limit // len(unique_tags))  # Increased from 10 to 15
                        )
                        
                        for track_metadata in tag_tracks:
                            # Additional filtering for tag-based results
                            if not await self._is_too_mainstream_track(track_metadata):
                                # 🔧 FIX: Exclude tracks from seed artists in tag-based search
                                track_artist = getattr(track_metadata, 'artist', '')
                                if any(seed_artist.lower() == track_artist.lower() for seed_artist in seed_artists):
                                    self.logger.debug(f"Excluding seed artist track from tag search: {track_artist} - {getattr(track_metadata, 'name', '')}")
                                    continue
                                
                                track = self._convert_metadata_to_dict(
                                    track_metadata,
                                    source='tag_based_similarity',
                                    source_confidence=0.6,
                                    discovery_tag=tag,
                                    seed_artists=seed_artists
                                )
                                candidates.append(track)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to search tracks by tag {tag}: {e}")
                        continue
        
        except Exception as e:
            self.logger.warning(f"Tag-based similarity search failed: {e}")
        
        self.logger.info(f"🏷️ Tag-based similarity: {len(candidates)} tracks")
        return candidates
    
    async def _calculate_style_similarity(self, track: Dict[str, Any], seed_artists: List[str]) -> float:
        """Calculate style similarity score between track and seed artists."""
        # This is a simplified style similarity calculation
        # In a production system, this would use audio features, genre analysis, etc.
        
        base_score = 0.5  # Base similarity for being in results
        
        # Factor in similarity hop (closer hops = higher similarity)
        hop = track.get('similarity_hop', 1)
        if hop == 0:
            base_score += 0.3  # Direct artist match
        elif hop == 1:
            base_score += 0.2  # One degree separation
        
        # Factor in popularity (less mainstream = potentially more similar to indie artists)
        popularity = self._calculate_popularity_score(track)
        if popularity < 0.5:  # Less mainstream
            base_score += 0.1
        
        # Factor in listener count (avoid completely unknown or overly popular)
        listeners = track.get('listeners', 0)
        if 1000 <= listeners <= 500000:  # Sweet spot for indie artists
            base_score += 0.1
        
        return min(1.0, base_score)
    
    async def _is_too_mainstream(self, artist_name: str) -> bool:
        """Check if an artist is too mainstream for similarity matching."""
        try:
            artist_info = await self.api_service.get_artist_info(artist_name)
            if not artist_info:
                return False
            
            # Check listener count threshold
            listeners = getattr(artist_info, 'listeners', 0)
            if listeners > 2000000:  # 2M+ listeners = too mainstream
                return True
            
            # Check for mainstream genre tags
            mainstream_indicators = [
                'pop', 'top 40', 'mainstream', 'radio', 'chart', 
                'commercial', 'hits', 'billboard'
            ]
            
            tags = getattr(artist_info, 'tags', [])
            for tag in tags[:5]:  # Check top 5 tags
                if any(indicator in str(tag).lower() for indicator in mainstream_indicators):
                    return True
            
            return False
            
        except Exception:
            return False  # If we can't check, assume it's okay
    
    async def _check_style_compatibility(self, artist1: str, artist2: str) -> bool:
        """Check if two artists have compatible musical styles."""
        try:
            # Simple style compatibility check using tags
            artist1_info = await self.api_service.get_artist_info(artist1)
            artist2_info = await self.api_service.get_artist_info(artist2)
            
            if not artist1_info or not artist2_info:
                return True  # Default to compatible if we can't check
            
            tags1 = set(getattr(artist1_info, 'tags', [])[:10])
            tags2 = set(getattr(artist2_info, 'tags', [])[:10])
            
            # Check for tag overlap
            common_tags = tags1.intersection(tags2)
            if len(common_tags) >= 2:  # At least 2 common tags
                return True
            
            # Check for compatible genres (not conflicting)
            conflicting_pairs = [
                ('metal', 'folk'), ('electronic', 'acoustic'), 
                ('punk', 'classical'), ('rap', 'jazz')
            ]
            
            for tag1 in tags1:
                for tag2 in tags2:
                    for conflict1, conflict2 in conflicting_pairs:
                        if (conflict1 in str(tag1).lower() and conflict2 in str(tag2).lower()) or \
                           (conflict2 in str(tag1).lower() and conflict1 in str(tag2).lower()):
                            return False
            
            return True
            
        except Exception:
            return True  # Default to compatible if check fails
    
    def _is_generic_tag(self, tag: str) -> bool:
        """Check if a tag is too generic to be useful for style matching."""
        generic_tags = [
            'music', 'alternative', 'rock', 'pop', 'indie', 'electronic',
            'seen live', 'favorites', 'love', 'awesome', 'great', 'good',
            'american', 'british', 'male', 'female', 'singer', 'band'
        ]
        return str(tag).lower() in generic_tags
    
    async def _is_too_mainstream_track(self, track_metadata) -> bool:
        """Check if a track is too mainstream."""
        try:
            listeners = getattr(track_metadata, 'listeners', 0)
            playcount = getattr(track_metadata, 'playcount', 0)
            
            # Threshold checks for mainstream status
            if listeners > 1000000 or playcount > 50000000:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _calculate_popularity_score(self, track: Dict[str, Any]) -> float:
        """Calculate popularity score (0.0 = underground, 1.0 = mainstream)."""
        listeners = track.get('listeners', 0)
        playcount = track.get('playcount', 0)
        
        # Normalize based on listener count
        if listeners == 0:
            return 0.0
        
        # Log scale for popularity
        import math
        listener_score = min(1.0, math.log10(listeners) / 7.0)  # 7 = log10(10M)
        playcount_score = min(1.0, math.log10(playcount + 1) / 8.0) if playcount > 0 else 0
        
        return (listener_score + playcount_score) / 2
    
    async def _apply_style_similarity_filter(self, candidates: List[Dict[str, Any]], seed_artists: List[str]) -> List[Dict[str, Any]]:
        """Apply final style-based filtering and ranking to candidates."""
        if not candidates:
            return candidates
        
        # Calculate final similarity scores
        for candidate in candidates:
            if 'similarity_score' not in candidate:
                candidate['similarity_score'] = await self._calculate_style_similarity(candidate, seed_artists)
        
        # Filter by minimum similarity threshold
        filtered = [c for c in candidates if c.get('similarity_score', 0) >= self.style_threshold]
        
        # Sort by similarity score (descending)
        filtered.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Apply diversity filtering (avoid too many tracks from same artist)
        diverse_candidates = []
        artist_counts = {}
        
        for candidate in filtered:
            artist = candidate.get('artist', '').lower()
            count = artist_counts.get(artist, 0)
            
            if count < 3:  # Max 3 tracks per artist
                diverse_candidates.append(candidate)
                artist_counts[artist] = count + 1
        
        self.logger.info(f"🎯 Style filtering: {len(diverse_candidates)} high-quality similar tracks")
        return diverse_candidates


class ArtistUndergroundStrategy(BaseGenerationStrategy):
    """
    Strategy for generating underground/lesser-known tracks from target artists.
    
    Used for discovering deep cuts and B-sides from specific artists.
    """
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate underground tracks from target artists.
        
        Args:
            entities: Extracted entities containing target artists
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of underground tracks from target artists
        """
        candidates = []
        target_artists = self._extract_seed_artists(entities)
        
        if not target_artists:
            self.logger.warning("No target artists found for ArtistUndergroundStrategy")
            return candidates
        
        self.logger.info(f"🕵️ UNDERGROUND ARTIST GENERATION: Finding deep cuts for {', '.join(target_artists)}")
        
        # Get full discography and filter for underground tracks
        for artist in target_artists[:2]:  # Limit to top 2 artists
            try:
                # Get larger set of tracks to find underground gems
                all_tracks = await self.api_service.get_artist_top_tracks(
                    artist=artist,
                    limit=50  # Get more tracks to filter
                )
                
                # Filter for underground tracks (lower popularity)
                underground_tracks = []
                for track_metadata in all_tracks:
                    underground_score = self._calculate_underground_score(track_metadata)
                    
                    if underground_score > 0.6:  # Threshold for "underground"
                        track = self._convert_metadata_to_dict(
                            track_metadata,
                            source='artist_underground',
                            source_confidence=0.8,
                            target_artist=artist,
                            underground_score=underground_score
                        )
                        underground_tracks.append(track)
                
                # Sort by underground score and take the best
                underground_tracks.sort(key=lambda x: x.get('underground_score', 0), reverse=True)
                candidates.extend(underground_tracks[:limit // len(target_artists)])
                
            except Exception as e:
                self.logger.warning(f"Failed to get underground tracks from {artist}: {e}")
                continue
        
        self.logger.info(f"🕵️ UNDERGROUND ARTIST: {len(candidates)} underground tracks found")
        return candidates[:limit]
    
    def _calculate_underground_score(self, track_metadata) -> float:
        """
        Calculate how "underground" a track is based on popularity metrics.
        
        Args:
            track_metadata: UnifiedTrackMetadata object
            
        Returns:
            Underground score (0.0 to 1.0, higher = more underground)
        """
        # Base score from low popularity
        popularity = getattr(track_metadata, 'popularity', 0.5)
        base_score = 1.0 - (popularity / 100.0) if popularity > 0 else 0.8
        
        # Boost for low listener counts
        listeners = getattr(track_metadata, 'listeners', 0)
        if listeners > 0:
            # Use log scale for listener count impact
            import math
            listener_factor = max(0, 1.0 - (math.log10(listeners) / 7.0))  # 7 = log10(10M)
            base_score = (base_score + listener_factor) / 2
        
        # Boost for certain genres that tend to be more underground
        genres = getattr(track_metadata, 'genres', [])
        underground_genres = ['experimental', 'ambient', 'drone', 'noise', 'industrial', 'avant-garde']
        if any(genre.lower() in underground_genres for genre in genres):
            base_score *= 1.2
        
        return min(1.0, base_score)  # Cap at 1.0 