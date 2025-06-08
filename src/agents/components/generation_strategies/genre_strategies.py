"""
Genre-focused Generation Strategies

Strategies that focus on generating candidates based on musical genres
and style exploration.
"""

from typing import Dict, List, Any
from .base_strategy import BaseGenerationStrategy


class GenreExplorationStrategy(BaseGenerationStrategy):
    """
    Strategy for exploring tracks within specific genres.
    
    Used for broad genre-based discovery and exploration.
    """
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate tracks for genre exploration.
        
        Args:
            entities: Extracted entities containing target genres
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of genre-based candidate tracks
        """
        candidates = []
        target_genres = self._extract_target_genres(entities)
        
        if not target_genres:
            # Fallback to extracting genres from artists
            artists = self._extract_seed_artists(entities)
            if artists:
                # Get genre information from artists
                for artist in artists[:2]:
                    try:
                        artist_info = await self.api_service.get_artist_info(artist)
                        if artist_info and hasattr(artist_info, 'tags') and artist_info.tags:
                            # Use artist tags as genres
                            target_genres.extend(artist_info.tags[:5])  # Top 5 tags as genres
                            self.logger.info(f"🎵 Found genres for {artist}: {', '.join(artist_info.tags[:3])}")
                    except Exception as e:
                        self.logger.warning(f"Failed to get genres for artist {artist}: {e}")
        
        if not target_genres:
            self.logger.warning("No target genres found for GenreExplorationStrategy")
            return candidates
        
        self.logger.info(f"🎵 GENRE EXPLORATION: Exploring {', '.join(target_genres[:3])}")
        
        # Strategy: Get tracks from each target genre
        for genre in target_genres[:3]:  # Limit to top 3 genres
            try:
                # Search for tracks by genre tags
                genre_tracks = await self.api_service.search_tracks_by_tags(
                    tags=[genre],
                    limit=min(20, limit // len(target_genres))
                )
                
                for track_metadata in genre_tracks:
                    track = self._convert_metadata_to_dict(
                        track_metadata,
                        source='genre_exploration',
                        source_confidence=0.8,
                        target_genre=genre
                    )
                    candidates.append(track)
                
                self.logger.info(f"🎵 Genre {genre}: {len(genre_tracks)} tracks found")
                
            except Exception as e:
                self.logger.warning(f"Failed to get tracks for genre {genre}: {e}")
                continue
        
        self.logger.info(f"🎵 TOTAL GENRE CANDIDATES: {len(candidates)} tracks")
        return candidates[:limit]


class GenreFocusedStrategy(BaseGenerationStrategy):
    """
    Strategy for focused genre-based candidate generation.
    
    Used when users have specific genre preferences.
    """
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate tracks focused on specific genres.
        
        Args:
            entities: Extracted entities containing target genres
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of genre-focused candidate tracks
        """
        candidates = []
        target_genres = self._extract_target_genres(entities)
        
        if not target_genres:
            self.logger.warning("No target genres found for GenreFocusedStrategy")
            return candidates
        
        self.logger.info(f"🎯 GENRE FOCUSED: Targeting {', '.join(target_genres)}")
        
        # Strategy: Deep dive into each genre with multiple approaches
        for genre in target_genres[:2]:  # Focus on top 2 genres for depth
            try:
                # Approach 1: Direct tag search
                tag_tracks = await self.api_service.search_tracks_by_tags(
                    tags=[genre],
                    limit=15
                )
                
                for track_metadata in tag_tracks:
                    track = self._convert_metadata_to_dict(
                        track_metadata,
                        source='genre_focused_tags',
                        source_confidence=0.9,
                        target_genre=genre
                    )
                    candidates.append(track)
                
                # Approach 2: Related genre exploration
                related_genres = self._get_related_genres(genre)
                for related_genre in related_genres[:2]:
                    try:
                        related_tracks = await self.api_service.search_tracks_by_tags(
                            tags=[related_genre],
                            limit=10
                        )
                        
                        for track_metadata in related_tracks:
                            track = self._convert_metadata_to_dict(
                                track_metadata,
                                source='genre_focused_related',
                                source_confidence=0.7,
                                target_genre=genre,
                                related_genre=related_genre
                            )
                            candidates.append(track)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to get related genre tracks for {related_genre}: {e}")
                        continue
                
            except Exception as e:
                self.logger.warning(f"Failed to get focused tracks for genre {genre}: {e}")
                continue
        
        return candidates[:limit]
    
    def _get_related_genres(self, genre: str) -> List[str]:
        """Get genres related to the target genre."""
        # Simple genre relationship mapping
        genre_relationships = {
            'rock': ['alternative rock', 'indie rock', 'classic rock'],
            'hip hop': ['rap', 'trap', 'conscious hip hop'],
            'electronic': ['techno', 'house', 'ambient'],
            'jazz': ['smooth jazz', 'bebop', 'fusion'],
            'pop': ['indie pop', 'synthpop', 'electropop'],
            'blues': ['electric blues', 'delta blues', 'chicago blues'],
            'country': ['folk', 'americana', 'bluegrass']
        }
        
        return genre_relationships.get(genre.lower(), [])


class RandomGenreStrategy(BaseGenerationStrategy):
    """
    Strategy for random genre exploration and serendipitous discovery.
    
    Used for broad musical discovery when users want variety.
    """
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate tracks from random genre exploration.
        
        Args:
            entities: Extracted entities (may contain genre hints)
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of randomly discovered candidate tracks
        """
        candidates = []
        
        # Define diverse genre pool for exploration
        exploration_genres = [
            'indie rock', 'electronic', 'jazz', 'folk', 'ambient',
            'world music', 'experimental', 'reggae', 'funk', 'soul',
            'blues', 'classical', 'post-rock', 'singer-songwriter'
        ]
        
        # Use existing genres as hints, but expand beyond them
        existing_genres = self._extract_target_genres(entities)
        if existing_genres:
            # Add related genres to existing ones
            for genre in existing_genres:
                related = self._get_expanded_genres(genre)
                exploration_genres.extend(related)
        
        # Remove duplicates and shuffle
        import random
        exploration_genres = list(set(exploration_genres))
        random.shuffle(exploration_genres)
        
        self.logger.info(f"🎲 RANDOM GENRE EXPLORATION: Exploring {len(exploration_genres)} genres")
        
        # Sample tracks from different genres
        genres_to_sample = exploration_genres[:min(6, len(exploration_genres))]
        tracks_per_genre = max(1, limit // len(genres_to_sample))
        
        for genre in genres_to_sample:
            try:
                genre_tracks = await self.api_service.search_tracks_by_tags(
                    tags=[genre],
                    limit=tracks_per_genre
                )
                
                for track_metadata in genre_tracks:
                    track = self._convert_metadata_to_dict(
                        track_metadata,
                        source='random_genre_exploration',
                        source_confidence=0.6,
                        exploration_genre=genre
                    )
                    candidates.append(track)
                
            except Exception as e:
                self.logger.warning(f"Failed to get random tracks for genre {genre}: {e}")
                continue
        
        self.logger.info(f"🎲 RANDOM GENRE: {len(candidates)} diverse tracks found")
        return candidates[:limit]
    
    def _get_expanded_genres(self, base_genre: str) -> List[str]:
        """Get an expanded list of genres related to the base genre."""
        expansion_map = {
            'rock': ['alternative', 'indie', 'post-rock', 'progressive rock'],
            'electronic': ['ambient', 'techno', 'house', 'experimental'],
            'hip hop': ['experimental hip hop', 'jazz rap', 'lo-fi hip hop'],
            'jazz': ['avant-garde jazz', 'world fusion', 'neo-soul'],
            'folk': ['indie folk', 'world music', 'singer-songwriter'],
            'pop': ['art pop', 'dream pop', 'electropop']
        }
        
        return expansion_map.get(base_genre.lower(), []) 