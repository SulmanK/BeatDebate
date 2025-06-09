"""
Genre Analyzer Component

Handles genre analysis and matching operations for the BeatDebate system.
Extracted from APIService to follow single responsibility principle.
"""

from typing import Dict, Any, List

import structlog

from .client_manager import ClientManager
from .artist_operations import ArtistOperations

logger = structlog.get_logger(__name__)


class GenreAnalyzer:
    """
    Handles genre analysis and matching operations.
    
    Responsibilities:
    - Genre matching for artists and tracks
    - Genre relationship analysis
    - LLM-based genre reasoning
    - Batch genre checking
    """
    
    def __init__(self, client_manager: ClientManager, artist_operations: ArtistOperations):
        """
        Initialize genre analyzer.
        
        Args:
            client_manager: Client manager instance
            artist_operations: Artist operations instance
        """
        self.client_manager = client_manager
        self.artist_operations = artist_operations
        self.logger = logger.bind(component="GenreAnalyzer")
        
        self.logger.info("Genre Analyzer initialized")
    
    async def check_artist_genre_match(
        self,
        artist: str,
        target_genre: str,
        include_related_genres: bool = True
    ) -> Dict[str, Any]:
        """
        Check if an artist matches a specific genre by looking up their metadata.
        
        Args:
            artist: Artist name
            target_genre: Genre to check against (e.g., "r&b", "jazz", "rock")
            include_related_genres: Whether to include related/synonym genres
            
        Returns:
            Dict with match info: {
                'matches': bool,
                'confidence': float,
                'matched_tags': List[str],
                'artist_tags': List[str],
                'match_type': str  # 'direct', 'related', 'none'
            }
        """
        try:
            # Get artist information from Last.fm
            artist_info = await self.artist_operations.get_artist_info(artist, include_top_tracks=False)
            
            if not artist_info or not artist_info.tags:
                self.logger.debug(f"No tags found for artist: {artist}")
                return {
                    'matches': False,
                    'confidence': 0.0,
                    'matched_tags': [],
                    'artist_tags': [],
                    'match_type': 'none'
                }
            
            # Normalize target genre
            target_genre_lower = target_genre.lower().strip()
            artist_tags_lower = [tag.lower() for tag in artist_info.tags]
            
            # Create genre mappings for related genres
            genre_mappings = self._get_genre_mappings()
            
            # Check for direct match first
            if target_genre_lower in artist_tags_lower:
                return {
                    'matches': True,
                    'confidence': 1.0,
                    'matched_tags': [target_genre_lower],
                    'artist_tags': artist_info.tags,
                    'match_type': 'direct'
                }
            
            # Check for related genres if enabled
            if include_related_genres and target_genre_lower in genre_mappings:
                related_genres = genre_mappings[target_genre_lower]
                matched_related = []
                
                for related_genre in related_genres:
                    if related_genre in artist_tags_lower:
                        matched_related.append(related_genre)
                
                if matched_related:
                    # Calculate confidence based on how many related genres match
                    confidence = min(0.9, len(matched_related) * 0.3)
                    return {
                        'matches': True,
                        'confidence': confidence,
                        'matched_tags': matched_related,
                        'artist_tags': artist_info.tags,
                        'match_type': 'related'
                    }
            
            # Use dynamic genre relationship checking if static mapping fails
            if include_related_genres:
                for artist_tag in artist_tags_lower:
                    relationship = await self.check_genre_relationship_llm(
                        target_genre=target_genre,
                        candidate_genre=artist_tag,
                        llm_client=None  # For now, use static fallback only
                    )
                    
                    if relationship['related'] and relationship['confidence'] >= 0.7:
                        return {
                            'matches': True,
                            'confidence': relationship['confidence'] * 0.85,  # Slightly lower confidence
                            'matched_tags': [artist_tag],
                            'artist_tags': artist_info.tags,
                            'match_type': f"dynamic_{relationship['relationship_type']}"
                        }
            
            # No match found
            return {
                'matches': False,
                'confidence': 0.0,
                'matched_tags': [],
                'artist_tags': artist_info.tags,
                'match_type': 'none'
            }
            
        except Exception as e:
            self.logger.error(f"Artist genre check failed for {artist}: {e}")
            return {
                'matches': False,
                'confidence': 0.0,
                'matched_tags': [],
                'artist_tags': [],
                'match_type': 'error'
            }
    
    async def check_track_genre_match(
        self,
        artist: str,
        track: str,
        target_genre: str,
        include_related_genres: bool = True
    ) -> Dict[str, Any]:
        """
        Check if a track matches a specific genre by looking up track and artist metadata.
        
        Args:
            artist: Artist name
            track: Track name
            target_genre: Genre to check against
            include_related_genres: Whether to include related/synonym genres
            
        Returns:
            Dict with match info: {
                'matches': bool,
                'confidence': float,
                'matched_tags': List[str],
                'track_tags': List[str],
                'artist_match': Dict[str, Any],
                'match_type': str
            }
        """
        try:
            # Check both track-specific tags and artist genre
            # Note: We'll need to import track operations for this
            from .track_operations import TrackOperations
            track_ops = TrackOperations(self.client_manager)
            
            track_info = await track_ops.get_unified_track_info(artist, track, include_spotify=False)
            artist_match = await self.check_artist_genre_match(artist, target_genre, include_related_genres)
            
            track_tags = []
            track_match_result = {
                'matches': False,
                'confidence': 0.0,
                'matched_tags': [],
                'track_tags': track_tags,
                'artist_match': artist_match,
                'match_type': 'none'
            }
            
            # Get track tags if available
            if track_info and track_info.tags:
                track_tags = track_info.tags
                track_match_result['track_tags'] = track_tags
                
                # Check track tags for genre match
                target_genre_lower = target_genre.lower().strip()
                track_tags_lower = [tag.lower() for tag in track_tags]
                
                # Direct match in track tags
                if target_genre_lower in track_tags_lower:
                    track_match_result.update({
                        'matches': True,
                        'confidence': 1.0,
                        'matched_tags': [target_genre_lower],
                        'match_type': 'direct_track'
                    })
                    return track_match_result
                
                # Related genre match in track tags
                if include_related_genres:
                    genre_mappings = self._get_genre_mappings()
                    if target_genre_lower in genre_mappings:
                        related_genres = genre_mappings[target_genre_lower]
                        matched_related = []
                        
                        for related_genre in related_genres:
                            if related_genre in track_tags_lower:
                                matched_related.append(related_genre)
                        
                        if matched_related:
                            confidence = min(0.9, len(matched_related) * 0.3)
                            track_match_result.update({
                                'matches': True,
                                'confidence': confidence,
                                'matched_tags': matched_related,
                                'match_type': 'related_track'
                            })
                            return track_match_result
                    
                    # Use dynamic LLM-based genre relationship checking if static mapping fails
                    for track_tag in track_tags_lower:
                        relationship = await self.check_genre_relationship_llm(
                            target_genre=target_genre,
                            candidate_genre=track_tag,
                            llm_client=None  # For now, use static fallback only
                        )
                        
                        if relationship['related'] and relationship['confidence'] >= 0.7:
                            track_match_result.update({
                                'matches': True,
                                'confidence': relationship['confidence'] * 0.85,  # Slightly lower confidence
                                'matched_tags': [track_tag],
                                'match_type': f"dynamic_{relationship['relationship_type']}"
                            })
                            self.logger.debug(
                                f"✅ Dynamic genre match: {track_tag} relates to {target_genre}",
                                relationship_type=relationship['relationship_type'],
                                confidence=relationship['confidence'],
                                explanation=relationship['explanation']
                            )
                            return track_match_result
            
            # If track doesn't have genre info, fall back to artist genre
            if artist_match['matches']:
                track_match_result.update({
                    'matches': True,
                    'confidence': artist_match['confidence'] * 0.8,  # Slightly lower confidence
                    'matched_tags': artist_match['matched_tags'],
                    'match_type': f"artist_{artist_match['match_type']}"
                })
                return track_match_result
            
            # No match found
            return track_match_result
            
        except Exception as e:
            self.logger.error(f"Track genre check failed for {artist} - {track}: {e}")
            return {
                'matches': False,
                'confidence': 0.0,
                'matched_tags': [],
                'track_tags': [],
                'artist_match': {'matches': False, 'confidence': 0.0},
                'match_type': 'error'
            }
    
    def _get_genre_mappings(self) -> Dict[str, List[str]]:
        """
        Get genre mappings for related genre detection.
        
        Returns:
            Dict mapping primary genres to related genres
        """
        return {
            'r&b': [
                'rnb', 'rhythm and blues', 'soul', 'neo-soul', 'contemporary r&b',
                'motown', 'funk', 'urban', 'smooth r&b', 'r&b soul'
            ],
            'jazz': [
                'bebop', 'swing', 'cool jazz', 'fusion', 'smooth jazz',
                'jazz fusion', 'contemporary jazz', 'acid jazz', 'jazz-hop', 
                'jazz rap', 'nu jazz', 'neo-soul', 'soul'  # Added missing jazz-related genres
            ],
            'rock': [
                'classic rock', 'alternative rock', 'indie rock', 'hard rock',
                'soft rock', 'progressive rock', 'art rock', 'garage rock'
            ],
            'electronic': [
                'edm', 'techno', 'house', 'ambient', 'electronica',
                'synth', 'synthwave', 'electronic music', 'dance'
            ],
            'hip hop': [
                'hip-hop', 'rap', 'trap', 'underground hip hop', 'conscious hip hop',
                'old school hip hop', 'east coast hip hop', 'west coast hip hop',
                'jazz rap', 'jazz-hop'  # Added jazz-influenced hip-hop genres
            ],
            'pop': [
                'pop music', 'mainstream pop', 'indie pop', 'electro pop',
                'synth pop', 'dance pop', 'alternative pop'
            ],
            'indie': [
                'indie rock', 'indie pop', 'indie folk', 'alternative',
                'independent', 'lo-fi', 'bedroom pop'
            ],
            'folk': [
                'folk music', 'indie folk', 'contemporary folk', 'acoustic',
                'singer-songwriter', 'americana', 'country folk'
            ],
            'metal': [
                'heavy metal', 'death metal', 'black metal', 'thrash metal',
                'progressive metal', 'power metal', 'doom metal'
            ],
            'country': [
                'country music', 'modern country', 'classic country',
                'country rock', 'americana', 'bluegrass'
            ],
            'soul': [
                'neo-soul', 'classic soul', 'southern soul', 'motown',
                'rhythm and blues', 'r&b', 'funk', 'jazz'  # Added soul connections
            ]
        }
    
    async def check_genre_relationship_llm(
        self,
        target_genre: str,
        candidate_genre: str,
        llm_client=None
    ) -> Dict[str, Any]:
        """
        Use LLM to determine if two genres are related dynamically.
        
        Args:
            target_genre: The target genre (e.g., "jazz")
            candidate_genre: The candidate genre to check (e.g., "jazz-hop")
            llm_client: LLM client for reasoning (optional)
            
        Returns:
            Dict with relationship info: {
                'related': bool,
                'confidence': float,
                'relationship_type': str,  # 'direct', 'subgenre', 'fusion', 'influence', 'none'
                'explanation': str
            }
        """
        try:
            # Normalize inputs
            target_lower = target_genre.lower().strip()
            candidate_lower = candidate_genre.lower().strip()
            
            # Direct match
            if target_lower == candidate_lower:
                return {
                    'related': True,
                    'confidence': 1.0,
                    'relationship_type': 'direct',
                    'explanation': 'Exact genre match'
                }
            
            # Simple substring matching for obvious cases
            if target_lower in candidate_lower or candidate_lower in target_lower:
                confidence = 0.9 if target_lower in candidate_lower else 0.8
                return {
                    'related': True,
                    'confidence': confidence,
                    'relationship_type': 'subgenre',
                    'explanation': f'"{candidate_genre}" contains "{target_genre}"'
                }
            
            # Use LLM for more complex relationships if available
            if llm_client:
                return await self._llm_genre_relationship_check(
                    target_genre, candidate_genre, llm_client
                )
            
            # Fall back to static mappings if no LLM
            return self._static_genre_relationship_check(target_lower, candidate_lower)
            
        except Exception as e:
            self.logger.error(f"Genre relationship check failed: {e}")
            return {
                'related': False,
                'confidence': 0.0,
                'relationship_type': 'error',
                'explanation': f'Error checking relationship: {e}'
            }
    
    async def _llm_genre_relationship_check(
        self,
        target_genre: str,
        candidate_genre: str,
        llm_client
    ) -> Dict[str, Any]:
        """Use LLM to check genre relationships."""
        try:
            # Import and use LLMUtils - this is the correct pattern in our codebase
            from ...agents.components.llm_utils import LLMUtils
            llm_utils = LLMUtils(llm_client)
            
            prompt = f"""Analyze the relationship between these two music genres:
Target Genre: "{target_genre}"
Candidate Genre: "{candidate_genre}"

Determine if they are related and provide:
1. Are they related? (yes/no)
2. Confidence level (0.0-1.0)
3. Relationship type (direct, subgenre, fusion, influence, none)
4. Brief explanation

Respond in JSON format:
{{
    "related": true/false,
    "confidence": 0.0-1.0,
    "relationship_type": "type",
    "explanation": "explanation"
}}"""
            
            # Make the LLM call using the correct method
            response = await llm_utils.call_llm(prompt)
            
            # Parse the response
            import json
            try:
                # Strip potential markdown code blocks
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]  # Remove ```json
                if clean_response.startswith('```'):
                    clean_response = clean_response[3:]   # Remove ```
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]  # Remove trailing ```
                clean_response = clean_response.strip()
                
                llm_result = json.loads(clean_response)
                return llm_result
                
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Failed to parse LLM genre response: {e}")
                return self._static_genre_relationship_check(target_genre.lower(), candidate_genre.lower())
                
        except Exception as e:
            self.logger.error(f"LLM genre relationship check failed: {e}")
            return self._static_genre_relationship_check(target_genre.lower(), candidate_genre.lower())
    
    def _static_genre_relationship_check(
        self,
        target_lower: str,
        candidate_lower: str
    ) -> Dict[str, Any]:
        """Fall back to static genre relationship checking."""
        genre_mappings = self._get_genre_mappings()
        
        # Check if candidate is in target's related genres
        if target_lower in genre_mappings:
            related_genres = [g.lower() for g in genre_mappings[target_lower]]
            if candidate_lower in related_genres:
                return {
                    'related': True,
                    'confidence': 0.8,
                    'relationship_type': 'related',
                    'explanation': f'"{candidate_lower}" is in the known related genres for "{target_lower}"'
                }
        
        # Check reverse relationship
        if candidate_lower in genre_mappings:
            related_genres = [g.lower() for g in genre_mappings[candidate_lower]]
            if target_lower in related_genres:
                return {
                    'related': True,
                    'confidence': 0.8,
                    'relationship_type': 'related',
                    'explanation': f'"{target_lower}" is in the known related genres for "{candidate_lower}"'
                }
        
        # No relationship found
        return {
            'related': False,
            'confidence': 0.8,
            'relationship_type': 'none',
            'explanation': 'No known relationship between these genres'
        }
    
    async def batch_check_tracks_genre_match(
        self,
        tracks: List[Dict[str, Any]],
        target_genre: str,
        llm_client=None,
        include_related_genres: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check multiple tracks against a target genre in a single LLM call.
        
        Args:
            tracks: List of track dictionaries with 'artist', 'name', 'tags' keys
            target_genre: Genre to check against (e.g., "jazz")
            llm_client: LLM client for reasoning (optional)
            include_related_genres: Whether to include related/synonym genres
            
        Returns:
            Dict mapping track keys to match results
        """
        results = {}
        
        try:
            # First pass: Use static mappings for all tracks
            static_matches = []
            needs_llm_check = []
            
            for track in tracks:
                track_key = f"{track.get('artist', 'Unknown')} - {track.get('name', 'Unknown')}"
                track_tags = track.get('tags', [])
                
                # Try static matching first
                static_result = await self._static_track_genre_check(track, target_genre, include_related_genres)
                
                if static_result['matches']:
                    results[track_key] = static_result
                    static_matches.append(track_key)
                else:
                    # Needs LLM check
                    needs_llm_check.append({
                        'key': track_key,
                        'track': track,
                        'tags': track_tags
                    })
            
            self.logger.info(
                f"Static genre matching: {len(static_matches)} matches, {len(needs_llm_check)} need LLM check",
                target_genre=target_genre
            )
            
            # Second pass: Batch LLM check for remaining tracks
            if needs_llm_check and llm_client:
                llm_results = await self._batch_llm_genre_check(
                    needs_llm_check, target_genre, llm_client
                )
                results.update(llm_results)
            else:
                # No LLM available, mark remaining as no match
                for track_info in needs_llm_check:
                    results[track_info['key']] = {
                        'matches': False,
                        'confidence': 0.8,
                        'matched_tags': [],
                        'match_type': 'no_match',
                        'explanation': f'No static mapping found for {target_genre}'
                    }
            
            self.logger.info(
                f"Batch genre check completed: {len([r for r in results.values() if r['matches']])} total matches out of {len(tracks)} tracks"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch genre check failed: {e}")
            # Return no matches for all tracks on error
            return {
                f"{track.get('artist', 'Unknown')} - {track.get('name', 'Unknown')}": {
                    'matches': False,
                    'confidence': 0.0,
                    'matched_tags': [],
                    'match_type': 'error',
                    'explanation': f'Error during batch check: {e}'
                }
                for track in tracks
            }
    
    async def _static_track_genre_check(
        self,
        track: Dict[str, Any],
        target_genre: str,
        include_related_genres: bool
    ) -> Dict[str, Any]:
        """Check a single track using static mappings only."""
        track_tags = track.get('tags', [])
        target_genre_lower = target_genre.lower().strip()
        track_tags_lower = [tag.lower() for tag in track_tags]
        
        # Direct match in track tags
        if target_genre_lower in track_tags_lower:
            return {
                'matches': True,
                'confidence': 1.0,
                'matched_tags': [target_genre_lower],
                'match_type': 'direct_track',
                'explanation': f'Direct match: track tagged with {target_genre}'
            }
        
        # Related genre match in track tags
        if include_related_genres:
            genre_mappings = self._get_genre_mappings()
            if target_genre_lower in genre_mappings:
                related_genres = genre_mappings[target_genre_lower]
                matched_related = []
                
                for related_genre in related_genres:
                    if related_genre in track_tags_lower:
                        matched_related.append(related_genre)
                
                if matched_related:
                    confidence = min(0.9, len(matched_related) * 0.3)
                    return {
                        'matches': True,
                        'confidence': confidence,
                        'matched_tags': matched_related,
                        'match_type': 'related_track',
                        'explanation': f'Related genres match: {", ".join(matched_related)}'
                    }
        
        # No static match found
        return {
            'matches': False,
            'confidence': 0.8,
            'matched_tags': [],
            'match_type': 'no_static_match',
            'explanation': 'No static genre mapping found'
        }
    
    async def _batch_llm_genre_check(
        self,
        tracks_to_check: List[Dict[str, Any]],
        target_genre: str,
        llm_client
    ) -> Dict[str, Dict[str, Any]]:
        """Use LLM to check multiple tracks in a single call."""
        try:
            # Prepare track information for LLM
            track_info_list = []
            for i, track_info in enumerate(tracks_to_check):
                track = track_info['track']
                tags = track_info.get('tags', [])
                track_info_list.append(
                    f"{i+1}. \"{track.get('name', 'Unknown')}\" by {track.get('artist', 'Unknown')} [Tags: {', '.join(tags[:5]) if tags else 'No tags'}]"
                )
            
            tracks_text = '\n'.join(track_info_list)
            
            prompt = f"""I need to check which of these music tracks have connections to the genre "{target_genre}".

Please evaluate each track using a **FLEXIBLE and NUANCED** approach that considers:

DIRECT MATCHES:
- Explicit genre tags matching "{target_genre}"
- Primary artistic style is "{target_genre}"

INFLUENCED/FUSION MATCHES (IMPORTANT):
- **Jazz influences in hip-hop** (live instrumentation, complex chord progressions, improvisation elements)
- **Subgenre connections** (jazz-hop, jazz rap, neo-soul for jazz; boom-bap, conscious rap for hip-hop)
- **Musical characteristics** (jazz harmonies in rap, hip-hop rhythms in jazz)
- **Sampling and inspiration** (heavy use of {target_genre} samples or {target_genre}-influenced production)
- **Artist background** (known for blending {target_genre} with other genres)

EXAMPLES for Jazz + Hip-Hop:
- ✅ Track with live jazz instrumentation over hip-hop beats
- ✅ Hip-hop track heavily sampling jazz records
- ✅ Artist known for jazz-influenced rap (complex flows, sophisticated harmonies)
- ✅ Neo-soul or jazz-hop tagged tracks
- ❌ Pure pop/rock with no jazz connection
- ❌ Standard trap/drill with no jazz elements

Tracks to evaluate:
{tracks_text}

RESPOND ONLY with valid JSON. Be **GENEROUS** with matches that show clear {target_genre} influence or fusion elements:

{{
    "matches": [
        {{
            "track_number": 1,
            "matches": true,
            "confidence": 0.7,
            "matched_elements": ["jazz samples", "live instrumentation"],
            "relationship_type": "influenced",
            "explanation": "Hip-hop track with heavy jazz sampling and live horn sections"
        }}
    ]
}}"""

            # Import and use LLMUtils - this is the correct pattern in our codebase
            from ...agents.components.llm_utils import LLMUtils
            llm_utils = LLMUtils(llm_client)
            
            # Make the LLM call using the correct method
            response = await llm_utils.call_llm(prompt)
            
            # Parse the response
            import json
            try:
                # Strip potential markdown code blocks
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]  # Remove ```json
                if clean_response.startswith('```'):
                    clean_response = clean_response[3:]   # Remove ```
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]  # Remove trailing ```
                clean_response = clean_response.strip()
                
                llm_result = json.loads(clean_response)
                results = {}
                
                # Process each track result
                for match_info in llm_result.get('matches', []):
                    track_num = match_info.get('track_number', 0)
                    if 1 <= track_num <= len(tracks_to_check):
                        track_info = tracks_to_check[track_num - 1]
                        track_key = track_info['key']
                        
                        results[track_key] = {
                            'matches': match_info.get('matches', False),
                            'confidence': match_info.get('confidence', 0.0) * 0.9,  # Slightly lower confidence for LLM
                            'matched_tags': match_info.get('matched_elements', []),
                            'match_type': f"llm_{match_info.get('relationship_type', 'unknown')}",
                            'explanation': match_info.get('explanation', 'LLM-based genre match')
                        }
                
                self.logger.info(
                    f"LLM batch genre check completed: {len([r for r in results.values() if r['matches']])} matches out of {len(tracks_to_check)} tracks"
                )
                
                return results
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                self.logger.warning(f"Failed to parse LLM batch response: {e}")
                self.logger.debug(f"LLM raw response (first 500 chars): {response[:500]}")
                # Fall back to no matches
                return {
                    track_info['key']: {
                        'matches': False,
                        'confidence': 0.0,
                        'matched_tags': [],
                        'match_type': 'llm_parse_error',
                        'explanation': 'Failed to parse LLM response'
                    }
                    for track_info in tracks_to_check
                }
                
        except Exception as e:
            self.logger.error(f"LLM batch genre check failed: {e}")
            # Fall back to no matches
            return {
                track_info['key']: {
                    'matches': False,
                    'confidence': 0.0,
                    'matched_tags': [],
                    'match_type': 'llm_error',
                    'explanation': f'LLM check failed: {e}'
                }
                for track_info in tracks_to_check
            }
