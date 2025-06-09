"""
GenreProcessor Component

Handles genre matching, filtering, and processing logic for the GenreMoodAgent.
Centralizes all genre-related operations including API-based matching and filtering.
"""

from typing import Dict, List, Any
import structlog

logger = structlog.get_logger(__name__)


class GenreProcessor:
    """
    Handles genre matching, filtering, and processing for GenreMoodAgent.
    
    Responsibilities:
    - Genre extraction from entities
    - Genre requirement validation
    - API-based genre matching
    - Batch genre filtering
    - Genre-based candidate scoring
    """
    
    def __init__(self, api_service):
        """
        Initialize GenreProcessor with API service.
        
        Args:
            api_service: APIService instance for genre matching
        """
        self.logger = logger.bind(component="GenreProcessor")
        self.api_service = api_service
        
        self.logger.info("GenreProcessor initialized with API service")
    
    def extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """
        Extract target genres from entities.
        
        Args:
            entities: Extracted entities from query
            
        Returns:
            List of target genre strings
        """
        musical_entities = entities.get('musical_entities', {})
        genres = musical_entities.get('genres', {})
        
        target_genres = []
        target_genres.extend(genres.get('primary', []))
        target_genres.extend(genres.get('secondary', []))
        
        unique_genres = list(set(target_genres))  # Remove duplicates
        
        self.logger.debug(f"Extracted target genres: {unique_genres}")
        return unique_genres
    
    def has_genre_requirements(self, entities: Dict[str, Any]) -> bool:
        """
        Check if the query has specific genre requirements.
        
        Args:
            entities: Extracted entities from query
            
        Returns:
            True if genre requirements exist, False otherwise
        """
        target_genres = self.extract_target_genres(entities)
        has_requirements = len(target_genres) > 0
        
        self.logger.debug(f"Genre requirements check: {has_requirements} (genres: {target_genres})")
        return has_requirements
    
    async def calculate_genre_score(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any]
    ) -> float:
        """
        Calculate genre-based relevance score for a candidate using enhanced API-based matching.
        
        Args:
            candidate: Track candidate to score
            entities: Extracted entities from query
            
        Returns:
            Genre relevance score (0.0 to 1.0)
        """
        score = 0.1  # Start with lower base score
        
        # Extract candidate information
        candidate_tags = candidate.get('tags', [])
        candidate_name = candidate.get('name', '').lower()
        candidate_artist = candidate.get('artist', '').lower()
        
        # Score based on genre matching using API service
        target_genres = self.extract_target_genres(entities)
        
        for genre in target_genres:
            try:
                # Check if candidate matches this genre using API method
                match_result = await self.check_genre_match(candidate, genre)
                if match_result.get('matches', False):
                    # Give high score for API-confirmed genre matches
                    score += 0.6  # Much higher than old 0.3
                    self.logger.debug(f"🎵 API-confirmed genre match: {candidate_artist} - {candidate_name} matches {genre}")
                    
            except Exception as e:
                self.logger.debug(f"API genre matching failed, falling back to tag matching for {genre}: {e}")
                # Fallback to simple tag matching if API fails
                if any(genre.lower() in tag.lower() for tag in candidate_tags):
                    score += 0.3
                if genre.lower() in candidate_name or genre.lower() in candidate_artist:
                    score += 0.2
        
        return min(score, 1.0)
    
    async def filter_by_genre_requirements(
        self,
        candidates: List[Dict[str, Any]], 
        entities: Dict[str, Any],
        llm_client=None
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates based on genre requirements using enhanced batch processing.
        
        Args:
            candidates: List of track candidates to filter
            entities: Extracted entities from query
            llm_client: LLM client for batch processing
            
        Returns:
            Filtered list of candidates that match genre requirements
        """
        # Extract required genres
        musical_entities = entities.get('musical_entities', {})
        genres = musical_entities.get('genres', {})
        required_genres = genres.get('primary', [])
        
        if not required_genres:
            self.logger.debug("No genre requirements found, keeping all candidates")
            return candidates
        
        self.logger.info(f"🎯 Filtering {len(candidates)} candidates for genres: {required_genres}")
        
        # For genre filtering, we require ALL primary genres to be matched
        filtered_candidates = candidates
        
        for genre in required_genres:
            genre_matches = []
            
            # Prepare track data for batch processing
            track_data = []
            for candidate in filtered_candidates:
                track_info = {
                    'artist': candidate.get('artist', ''),
                    'name': candidate.get('name', ''),
                    'tags': candidate.get('tags', [])
                }
                track_data.append(track_info)
            
            # Use batch processing for genre matching
            batch_results = await self.api_service.batch_check_tracks_genre_match(
                tracks=track_data,
                target_genre=genre,
                llm_client=llm_client,  # Pass the LLM client for batch processing
                include_related_genres=True
            )
            
            # Apply results to candidates
            for i, candidate in enumerate(filtered_candidates):
                track_key = f"{candidate.get('artist', 'Unknown')} - {candidate.get('name', 'Unknown')}"
                match_result = batch_results.get(track_key, {'matches': False})
                
                if match_result['matches']:
                    # Add match information to candidate
                    candidate['genre_match'] = {
                        'genre': genre,
                        'confidence': match_result['confidence'],
                        'matched_tags': match_result['matched_tags'],
                        'match_type': match_result['match_type'],
                        'explanation': match_result['explanation']
                    }
                    genre_matches.append(candidate)
                    
                    self.logger.debug(
                        f"✅ Genre match: {candidate.get('artist')} - {candidate.get('name')} matches {genre}",
                        match_type=match_result['match_type'],
                        confidence=match_result['confidence'],
                        matched_tags=match_result['matched_tags']
                    )
                else:
                    self.logger.debug(
                        f"❌ No genre match: {candidate.get('artist')} - {candidate.get('name')} doesn't match {genre}",
                        track_tags=candidate.get('tags', [])
                    )
            
            self.logger.info(f"🎯 Genre filtering for '{genre}': {len(filtered_candidates)} → {len(genre_matches)} candidates")
            
            # Update candidates for next genre (intersection)
            filtered_candidates = genre_matches
            
            if not filtered_candidates:
                break  # No candidates left
        
        self.logger.info(f"🎯 Final genre filtering: {len(candidates)} → {len(filtered_candidates)} candidates")
        return filtered_candidates
    
    async def check_genre_match(self, candidate: Dict[str, Any], genre: str) -> Dict[str, Any]:
        """
        Check if a candidate matches the target genre using API service.
        
        Args:
            candidate: Track candidate to check
            genre: Target genre to match against
            
        Returns:
            Dict with match information
        """
        try:
            # Check track-level genre match first
            track_match = await self.api_service.check_track_genre_match(
                artist=candidate.get('artist', ''),
                track=candidate.get('name', ''),
                target_genre=genre,
                include_related_genres=True
            )
            
            if track_match['matches']:
                self.logger.debug(
                    f"Genre match found for {candidate.get('artist')} - {candidate.get('name')}",
                    genre=genre,
                    match_type=track_match['match_type'],
                    confidence=track_match['confidence'],
                    matched_tags=track_match['matched_tags']
                )
                return track_match
            
            # Fall back to artist-level check if no track match
            artist_match = await self.api_service.check_artist_genre_match(
                artist=candidate.get('artist', ''),
                target_genre=genre,
                include_related_genres=True
            )
            
            if artist_match['matches']:
                self.logger.debug(
                    f"Artist genre match found for {candidate.get('artist')}",
                    genre=genre,
                    match_type=artist_match['match_type'],
                    confidence=artist_match['confidence'],
                    matched_tags=artist_match['matched_tags']
                )
                
                # Convert artist match to track match format
                return {
                    'matches': True,
                    'confidence': artist_match['confidence'] * 0.8,  # Slightly lower confidence
                    'matched_tags': artist_match['matched_tags'],
                    'track_tags': [],
                    'artist_match': artist_match,
                    'match_type': f"artist_{artist_match['match_type']}"
                }
            
            # No match found
            self.logger.debug(
                f"No genre match for {candidate.get('artist')} - {candidate.get('name')}",
                genre=genre,
                artist_tags=artist_match.get('artist_tags', [])
            )
            
            return {
                'matches': False,
                'confidence': 0.0,
                'matched_tags': [],
                'track_tags': track_match.get('track_tags', []),
                'artist_match': artist_match,
                'match_type': 'none'
            }
            
        except Exception as e:
            self.logger.error(f"Genre matching failed: {e}")
            return {
                'matches': False,
                'confidence': 0.0,
                'matched_tags': [],
                'track_tags': [],
                'artist_match': {'matches': False},
                'match_type': 'error'
            }
    
    def extract_genres_for_recommendation(self, candidate: Dict[str, Any], entities: Dict[str, Any]) -> List[str]:
        """
        Extract genres for recommendation display.
        
        Args:
            candidate: Track candidate
            entities: Extracted entities from query
            
        Returns:
            List of genre strings for the recommendation
        """
        # Use candidate tags as genres
        tags = candidate.get('tags', [])
        
        # Filter for genre-like tags
        genre_tags = []
        for tag in tags[:5]:  # Limit to first 5 tags
            if len(tag) > 2 and not tag.isdigit():
                genre_tags.append(tag)
        
        self.logger.debug(f"Extracted genres for {candidate.get('name')}: {genre_tags}")
        return genre_tags
    
    def validate_genre_requirements(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and analyze genre requirements from entities.
        
        Args:
            entities: Extracted entities from query
            
        Returns:
            Dictionary with genre requirement analysis
        """
        target_genres = self.extract_target_genres(entities)
        
        analysis = {
            'has_requirements': len(target_genres) > 0,
            'genre_count': len(target_genres),
            'primary_genres': entities.get('musical_entities', {}).get('genres', {}).get('primary', []),
            'secondary_genres': entities.get('musical_entities', {}).get('genres', {}).get('secondary', []),
            'all_genres': target_genres,
            'is_multi_genre': len(target_genres) > 1
        }
        
        self.logger.debug(f"Genre requirements analysis: {analysis}")
        return analysis 