"""
Discovery-focused Generation Strategies

Strategies that focus on serendipitous discovery and underground track finding.
"""

from typing import Dict, List, Any
from .base_strategy import BaseGenerationStrategy


class UndergroundGemsStrategy(BaseGenerationStrategy):
    """
    Strategy for finding underground and lesser-known tracks.
    
    Used for discovering hidden gems and deep cuts.
    Implements improved strategy based on design document:
    - Uses canonical API endpoints to get all artist tracks via pagination
    - Applies strict filtering for artist name matching and track validity
    - Ranks by inverse popularity (lowest listeners first)
    """
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 150,  # Increased from 50 to 150 for larger candidate pools
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate underground and lesser-known tracks.
        
        Args:
            entities: Extracted entities for context
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of underground candidate tracks
        """
        candidates = []
        target_genres = self._extract_target_genres(entities)
        
        # NEW: Check if user specified a specific artist for underground discovery
        target_artists = self._extract_seed_artists(entities)
        if target_artists:
            self.logger.info(f"🎯 Artist-specific underground discovery for: {target_artists}")
            
            # Use improved artist-specific underground discovery
            for artist in target_artists[:2]:  # Limit to top 2 artists
                try:
                    artist_tracks = await self._get_all_artist_tracks_paginated(artist)
                    
                    if artist_tracks:
                        # Apply strict filtering based on design document
                        filtered_tracks = self._apply_strict_filtering(artist_tracks, artist)
                        
                        # Sort by inverse popularity (lowest listeners first)
                        underground_tracks = self._rank_by_underground_score(filtered_tracks)
                        
                        # Convert to candidate format using base class method
                        for track in underground_tracks[:limit // len(target_artists)]:
                            if track is not None:
                                candidate = self._convert_metadata_to_dict(
                                    track, 
                                    source='underground_gems',
                                    source_confidence=0.8,
                                    discovery_type='underground_search',
                                    source_strategy='UndergroundGemsStrategy'
                                )
                                candidates.append(candidate)
                        
                        self.logger.info(f"🕵️ Found {len(underground_tracks)} underground tracks by {artist} (after filtering)")
                    
                except Exception as e:
                    self.logger.error(f"Error in underground discovery for {artist}: {e}")
            
            # If we found artist-specific underground tracks, return them
            if candidates:
                self.logger.info(f"✅ Returning {len(candidates)} artist-specific underground tracks")
                return candidates[:limit]
            else:
                self.logger.warning(f"❌ No underground tracks found for {target_artists}, falling back to genre search")
        
        # If no artists specified OR not enough artist-specific results, fall back to genre-based search
        if not target_genres:
            target_genres = ['experimental', 'ambient']
        
        # Get underground search terms from user query
        underground_terms = self._extract_underground_terms(entities, intent_analysis)
        if not underground_terms:
            underground_terms = ['underground', 'indie', 'alternative']
        
        self.logger.info(f"🕵️ UNDERGROUND GEMS: Searching for hidden tracks in {', '.join(target_genres[:3])}")
        
        for genre in target_genres[:2]:  # Focus on top 2 genres
            try:
                # Get underground-specific search terms for this genre
                genre_underground_terms = self._get_underground_terms_for_genre(genre)
                
                # Search for tracks using new underground search method
                search_tags = [genre] + underground_terms + genre_underground_terms[:2]
                
                self.logger.info(f"🕵️ Searching underground tracks with tags: {search_tags}")
                
                # Use the underground-focused search method
                genre_tracks = await self.api_service.search_underground_tracks_by_tags(
                    tags=search_tags,
                    limit=min(200, limit // len(target_genres)),  # Increased limit with cap at 200
                    max_listeners=25000  # Focus on very underground tracks
                )
                
                self.logger.info(f"🔍 Found {len(genre_tracks)} underground tracks for {genre}")
                
                # Sort by playcount (ascending = lowest playcount first) 
                genre_tracks.sort(key=lambda x: getattr(x, 'playcount', 0))
                
                # Convert to candidate format using base class method
                for track in genre_tracks:
                    if track is not None:
                        candidate = self._convert_metadata_to_dict(
                            track,
                            source='underground_gems',
                            source_confidence=0.7,
                            discovery_type='underground_search',
                            source_strategy='UndergroundGemsStrategy',
                            genre_context=genre,
                            search_tags=search_tags
                        )
                        candidates.append(candidate)
                
            except Exception as e:
                self.logger.error(f"Error searching underground tracks for genre {genre}: {e}")
        
        # Sort all candidates by playcount (lowest first) to prioritize most underground
        candidates.sort(key=lambda x: x.get('playcount', 0))
        
        self.logger.info(f"🔥 Generated {len(candidates)} underground candidates (sorted by lowest playcount)")
        return candidates[:limit]
    
    async def _get_all_artist_tracks_paginated(self, artist_name: str, max_pages: int = 15) -> List[Any]:  # Increased from 5 to 15 pages
        """
        Get all available tracks for an artist using pagination.
        
        Based on design document: use canonical API endpoint to fetch comprehensive track list.
        
        Args:
            artist_name: Artist name to fetch tracks for
            max_pages: Maximum pages to fetch (to prevent excessive API calls)
            
        Returns:
            List of all tracks by the artist
        """
        all_tracks = []
        page = 1
        
        while page <= max_pages:
            try:
                # Get tracks for this page (Last.fm supports page parameter)
                page_tracks = await self._get_artist_tracks_page(artist_name, page, limit=50)
                
                if not page_tracks:
                    # No more tracks available
                    break
                    
                all_tracks.extend(page_tracks)
                
                # If we got fewer than the limit, we've reached the end
                if len(page_tracks) < 50:  # Updated to match reduced per-page limit
                    break
                    
                page += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch page {page} for {artist_name}: {e}")
                break
        
        self.logger.info(f"📖 Fetched {len(all_tracks)} total tracks for {artist_name} across {page-1} pages")
        return all_tracks
    
    async def _get_artist_tracks_page(self, artist_name: str, page: int, limit: int = 50) -> List[Any]:  # Reduced back to 50 tracks per page for balance
        """
        Get a single page of tracks for an artist.
        
        Args:
            artist_name: Artist name
            page: Page number (1-indexed)
            limit: Number of tracks per page
            
        Returns:
            List of tracks for this page
        """
        try:
            # For now, we'll use a hybrid approach: combine top tracks with search results
            # This gives us more comprehensive coverage until we implement proper pagination
            
            if page <= 5:  # Increased from 3 to 5 pages for top tracks
                # First few pages: use proper pagination with LastFM top tracks
                tracks = await self.api_service.get_artist_top_tracks(artist_name, limit=limit, page=page)
                return tracks
            else:
                # Additional pages: use search-based discovery to find more obscure tracks
                search_queries = [
                    f'artist:"{artist_name}"',
                    f'"{artist_name}" rare',
                    f'"{artist_name}" deep cut',
                    f'"{artist_name}" album track'
                ]
                
                all_search_tracks = []
                for query in search_queries:
                    try:
                        search_tracks = await self.api_service.search_tracks(
                            query=query,
                            limit=limit // len(search_queries)
                        )
                        all_search_tracks.extend(search_tracks)
                    except Exception as e:
                        self.logger.warning(f"Search query '{query}' failed: {e}")
                
                # Remove duplicates and return
                seen_tracks = set()
                unique_tracks = []
                for track in all_search_tracks:
                    track_key = f"{getattr(track, 'artist', '')}:{getattr(track, 'name', '')}".lower()
                    if track_key not in seen_tracks:
                        seen_tracks.add(track_key)
                        unique_tracks.append(track)
                
                return unique_tracks[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get tracks page {page} for {artist_name}: {e}")
            return []
    
    def _apply_strict_filtering(self, tracks: List[Any], target_artist: str) -> List[Any]:
        """
        Apply strict filtering based on design document.
        
        Filters:
        1. Artist name normalization and exact matching
        2. Track validity (exclude generic/invalid titles)
        3. Remove features, vs., mixtape sources
        
        Args:
            tracks: List of track metadata objects
            target_artist: The target artist name for filtering
            
        Returns:
            List of filtered tracks
        """
        filtered_tracks = []
        target_artist_normalized = self._normalize_artist_name(target_artist)
        
        for track in tracks:
            try:
                track_artist = getattr(track, 'artist', '')
                track_name = getattr(track, 'name', '')
                
                # 1. Artist name normalization and matching
                track_artist_normalized = self._normalize_artist_name(track_artist)
                
                if not self._is_exact_artist_match(track_artist_normalized, target_artist_normalized):
                    self.logger.debug(f"Filtered out non-matching artist: {track_artist} (target: {target_artist})")
                    continue
                
                # 2. Track validity filter
                if not self._is_valid_track_name(track_name.lower(), track_artist.lower()):
                    self.logger.debug(f"Filtered out invalid track: {track_name} by {track_artist}")
                    continue
                
                # 3. Filter out features, vs., mixtape sources
                if self._is_feature_or_collab(track_artist, track_name):
                    self.logger.debug(f"Filtered out feature/collab: {track_name} by {track_artist}")
                    continue
                
                filtered_tracks.append(track)
                
            except Exception as e:
                self.logger.warning(f"Error filtering track: {e}")
                continue
        
        self.logger.info(f"🧹 Filtered {len(tracks)} -> {len(filtered_tracks)} tracks for {target_artist}")
        return filtered_tracks
    
    def _normalize_artist_name(self, artist_name: str) -> str:
        """Normalize artist name for comparison."""
        return artist_name.lower().strip().replace('  ', ' ')
    
    def _is_exact_artist_match(self, track_artist_normalized: str, target_artist_normalized: str) -> bool:
        """Check if track artist exactly matches target artist."""
        return track_artist_normalized == target_artist_normalized
    
    def _is_feature_or_collab(self, artist_name: str, track_name: str) -> bool:
        """Check if track is a feature, collaboration, or from mixtape source."""
        artist_lower = artist_name.lower()
        track_lower = track_name.lower()
        
        # Features and collaborations
        feature_indicators = ['feat.', 'ft.', 'featuring', 'vs.', 'vs', 'x ', ' x ', 'and ', '&']
        for indicator in feature_indicators:
            if indicator in artist_lower or indicator in track_lower:
                return True
        
        # Mixtape sources
        mixtape_sources = ['monstermixtapes', 'datpiff', 'livemixtapes', 'mixtape', 'dj ']
        for source in mixtape_sources:
            if source in artist_lower:
                return True
        
        return False
    
    def _rank_by_underground_score(self, tracks: List[Any]) -> List[Any]:
        """
        Rank tracks by underground score (inverse popularity).
        
        Args:
            tracks: List of filtered tracks
            
        Returns:
            List of tracks sorted by underground score (most underground first)
        """
        def get_underground_score(track):
            listeners = getattr(track, 'listeners', 0)
            
            # Lower listeners = higher underground score
            # Use a simple inverse ranking
            if listeners == 0:
                return 1000000  # Highest score for completely unknown tracks
            else:
                return 1000000 // (listeners + 1)  # Inverse popularity
        
        # Sort by underground score (highest first = most underground)
        tracks.sort(key=get_underground_score, reverse=True)
        return tracks
    
    def _is_valid_track_name(self, track_name: str, artist_name: str) -> bool:
        """Check if a track name appears to be a real song rather than metadata artifact."""
        # Filter out generic/missing names
        invalid_names = {
            'unknown', 'unknown track', 'untitled', '', 'n/a', 'na', 'null', 'none',
            'track', 'song', 'music', 'audio', 'file', 'mp3', 'wav', 'demo', 'test'
        }
        
        if track_name in invalid_names:
            return False
        
        # Filter out tracks that are clearly not songs
        exclusion_patterns = [
            'mixtape', 'compilation', 'radio', 'interview', 'freestyle', 'mix',
            'dj', 'mix tape', 'vs.', 'vs', 'monstermixtapes', 'datpiff',
            'wins', 'award', 'nominated', 'contest', 'battle', 'cipher',
            'live on', 'live at', 'performance', 'concert', 'show',
            'podcast', 'episode', 'part 1', 'part 2', 'part i', 'part ii',
            'skit', 'interlude', 'intro', 'outro'
        ]
        
        for pattern in exclusion_patterns:
            if pattern in track_name:
                return False
        
        # Filter out tracks that are mostly artist name (likely duplicates/metadata)
        if len(track_name) < 3:  # Very short names likely invalid
            return False
        
        # If track name is mostly just the artist name, probably not a real song
        artist_words = set(artist_name.split())
        track_words = set(track_name.split())
        if len(artist_words.intersection(track_words)) >= len(track_words) * 0.7:
            return False
        
        return True
    
    def _extract_underground_terms(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[str]:
        """Extract terms that indicate underground/experimental preferences."""
        underground_terms = []
        
        # Check for underground keywords in query
        query_text = intent_analysis.get('original_query', '').lower()
        underground_keywords = [
            'underground', 'experimental', 'indie', 'alternative', 'obscure',
            'hidden', 'deep', 'rare', 'unknown', 'emerging', 'lo-fi'
        ]
        
        for keyword in underground_keywords:
            if keyword in query_text:
                underground_terms.append(keyword)
        
        # Add general underground tags if none found
        if not underground_terms:
            underground_terms = ['indie', 'alternative']
        
        return underground_terms
    
    def _get_underground_terms_for_genre(self, genre: str) -> List[str]:
        """Get genre-specific underground search terms."""
        underground_by_genre = {
            'electronic': ['minimal', 'idm', 'microhouse', 'glitch', 'leftfield', 'experimental electronic'],
            'ambient': ['dark ambient', 'drone', 'field recording', 'lowercase', 'microsound'],
            'indie': ['lo-fi', 'bedroom', 'diy', 'cassette', 'home recording'],
            'experimental': ['avant-garde', 'noise', 'sound art', 'electroacoustic', 'concrete'],
            'techno': ['minimal techno', 'dub techno', 'ambient techno', 'experimental techno'],
            'house': ['minimal house', 'microhouse', 'deep house', 'experimental house']
        }
        
        return underground_by_genre.get(genre.lower(), ['underground', 'experimental'])


class SerendipitousDiscoveryStrategy(BaseGenerationStrategy):
    """
    Strategy for serendipitous music discovery.
    
    Focuses on finding recent, diverse, and genuinely surprising tracks
    from various genres and emerging artists for accessible discovery.
    """
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 150,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate serendipitous discovery tracks with focus on recency and diversity.
        
        Args:
            entities: Extracted entities for context
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of serendipitous candidate tracks
        """
        candidates = []
        
        self.logger.info("🎲 SERENDIPITOUS DISCOVERY: Finding recent, diverse, and surprising tracks")
        
        # Strategy 1: Recent diverse tracks (40% of candidates)
        recent_tracks = await self._find_recent_diverse_tracks(int(limit * 0.4))
        candidates.extend(recent_tracks)
        
        # Strategy 2: Cross-genre exploration (30% of candidates)
        cross_genre_tracks = await self._cross_genre_exploration(entities, int(limit * 0.3))
        candidates.extend(cross_genre_tracks)
        
        # Strategy 3: Emerging artist discovery (30% of candidates)
        emerging_tracks = await self._discover_emerging_artists(int(limit * 0.3))
        candidates.extend(emerging_tracks)
        
        # Remove duplicates and sort by discovery score (lowest play count first)
        unique_candidates = self._deduplicate_tracks(candidates)
        unique_candidates.sort(key=lambda x: x.get('playcount', 0))
        
        self.logger.info(f"🎲 SERENDIPITOUS DISCOVERY: {len(unique_candidates)} surprising tracks found")
        return unique_candidates[:limit]
    
    async def _find_recent_diverse_tracks(self, limit: int) -> List[Dict[str, Any]]:
        """Find recent tracks from diverse genres and artists."""
        tracks = []
        
        # Streamlined recent search terms - focus on diverse genres
        recent_search_terms = [
            "indie rock 2024", "indie pop 2023", "bedroom pop", "lo-fi hip hop",
            "indie folk", "dream pop", "new wave", "post-punk revival",
            "indie electronic", "chillwave", "synthpop", "art pop"
        ]
        
        # Limit to 6 terms for speed
        selected_terms = recent_search_terms[:6]
        
        for search_term in selected_terms:
            try:
                # Use regular search for more diverse and accessible recent tracks
                term_tracks = await self.api_service.search_tracks_by_tags(
                    tags=[search_term],
                    limit=max(5, limit // len(selected_terms))
                )
                
                for track_metadata in term_tracks:
                    track = self._convert_metadata_to_dict(
                        track_metadata,
                        source='serendipitous_discovery',
                        source_confidence=0.8,
                        exploration_type='recent_diverse',
                        search_term=search_term,
                        discovery_reason='recent_diverse_discovery'
                    )
                    tracks.append(track)
                    
            except Exception as e:
                self.logger.warning(f"Recent diverse search failed for '{search_term}'", error=str(e))
                continue
        
        self.logger.info(f"🆕 Found {len(tracks)} recent diverse tracks")
        return tracks
    
    async def _cross_genre_exploration(self, entities: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Explore unexpected genre combinations and fusion styles."""
        tracks = []
        
        # More diverse and mainstream-accessible fusion genres
        fusion_genres = [
            "indie jazz", "electronic folk", "neo-soul", "indie r&b",
            "psychedelic pop", "garage rock", "surf rock", "math rock",
            "indie country", "folk punk", "electro swing", "trip hop"
        ]
        
        # Limit to 4 terms for speed and focus
        selected_terms = fusion_genres[:4]
        
        for search_term in selected_terms:
            try:
                # Use regular search for genre fusion (allows more popular tracks)
                term_tracks = await self.api_service.search_tracks_by_tags(
                    tags=[search_term],
                    limit=max(3, limit // len(selected_terms))
                )
                
                # Use all tracks from regular search for better diversity
                for track_metadata in term_tracks:
                    track = self._convert_metadata_to_dict(
                        track_metadata,
                        source='serendipitous_discovery',
                        source_confidence=0.7,
                        exploration_type='cross_genre',
                        search_term=search_term,
                        discovery_reason='diverse_genre_fusion'
                    )
                    tracks.append(track)
                    
            except Exception as e:
                self.logger.warning(f"Cross-genre search failed for '{search_term}'", error=str(e))
                continue
        
        self.logger.info(f"🎭 Found {len(tracks)} cross-genre fusion tracks")
        return tracks
    
    async def _discover_emerging_artists(self, limit: int) -> List[Dict[str, Any]]:
        """Discover tracks from emerging and accessible artists."""
        tracks = []
        
        # Streamlined emerging artist search terms - focus on quality over quantity
        emerging_terms = [
            "debut album", "new artist 2024", "indie breakthrough",
            "emerging talent", "rising artist", "fresh sound"
        ]
        
        # Limit to 4 terms for speed
        selected_searches = emerging_terms[:4]
        
        for search_term in selected_searches:
            try:
                # Use regular search for emerging artists (more accessible discoveries)
                term_tracks = await self.api_service.search_tracks_by_tags(
                    tags=[search_term],
                    limit=max(3, limit // len(selected_searches))
                )
                
                for track_metadata in term_tracks:
                    track = self._convert_metadata_to_dict(
                        track_metadata,
                        source='serendipitous_discovery',
                        source_confidence=0.9,
                        exploration_type='emerging_artist',
                        search_term=search_term,
                        discovery_reason='emerging_artist_quality_discovery'
                    )
                    tracks.append(track)
            
            except Exception as e:
                self.logger.warning(f"Emerging artist search failed for '{search_term}'", error=str(e))
                continue
        
        self.logger.info(f"🌱 Found {len(tracks)} emerging artist tracks")
        return tracks
    
    def _deduplicate_tracks(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tracks based on artist and track name."""
        seen = set()
        unique_tracks = []
        
        for track in tracks:
            # Create a key based on normalized artist and track name
            artist = track.get('artist', '').lower().strip()
            name = track.get('name', '').lower().strip()
            key = f"{artist}::{name}"
            
            if key not in seen and artist and name:
                seen.add(key)
                unique_tracks.append(track)
        
        return unique_tracks
