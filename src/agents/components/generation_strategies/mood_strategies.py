"""
Mood-focused Generation Strategies

Strategies that focus on generating candidates based on emotional content
and mood preferences.
"""

from typing import Dict, List, Any
from .base_strategy import BaseGenerationStrategy


class MoodBasedSerendipityStrategy(BaseGenerationStrategy):
    """
    Strategy for mood-based serendipitous discovery.
    
    Used for finding tracks that match or explore emotional themes.
    """
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate tracks based on mood exploration.
        
        Args:
            entities: Extracted entities containing mood information
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of mood-based candidate tracks
        """
        candidates = []
        target_moods = self._extract_moods(entities)
        
        # If no explicit moods, extract from context
        if not target_moods:
            target_moods = self._extract_exploration_tags(entities, intent_analysis)
        
        # Fallback to default mood exploration
        if not target_moods:
            target_moods = ['energetic', 'mellow', 'uplifting']
        
        self.logger.info(f"🎭 MOOD-BASED SERENDIPITY: Exploring moods {', '.join(target_moods[:3])}")
        
        # Explore each mood with associated tags
        for mood in target_moods[:3]:
            mood_tags = self._get_mood_tags(mood)
            
            for tag in mood_tags[:2]:  # Top 2 tags per mood
                try:
                    # Search for tracks matching this mood tag
                    mood_tracks = await self.api_service.search_tracks_by_tags(
                        tags=[tag],
                        limit=min(10, limit // (len(target_moods) * 2))
                    )
                    
                    for track_metadata in mood_tracks:
                        track = self._convert_metadata_to_dict(
                            track_metadata,
                            source='mood_based_serendipity',
                            source_confidence=0.7,
                            exploration_type='mood_based',
                            mood_indicator=mood,
                            mood_tag=tag
                        )
                        candidates.append(track)
                        
                        if len(candidates) >= limit:
                            break
                            
                except Exception as e:
                    self.logger.warning(f"Mood-based serendipity failed for tag '{tag}'", error=str(e))
                    continue
                
                if len(candidates) >= limit:
                    break
            
            if len(candidates) >= limit:
                break
        
        self.logger.info(f"🎭 MOOD SERENDIPITY: {len(candidates)} mood-based tracks found")
        return candidates[:limit]
    
    def _extract_exploration_tags(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[str]:
        """Extract mood/exploration tags from context."""
        tags = []
        
        # Check query for mood keywords
        query_text = intent_analysis.get('original_query', '').lower()
        mood_keywords = [
            'happy', 'sad', 'energetic', 'mellow', 'upbeat', 'chill',
            'relaxing', 'intense', 'peaceful', 'aggressive', 'nostalgic',
            'dreamy', 'dark', 'bright', 'emotional', 'fun', 'serious'
        ]
        
        for keyword in mood_keywords:
            if keyword in query_text:
                tags.append(keyword)
        
        return tags
    
    def _get_mood_tags(self, mood: str) -> List[str]:
        """Get associated tags for a given mood."""
        mood_tag_map = {
            'energetic': ['upbeat', 'dance', 'high energy', 'pump up'],
            'mellow': ['chill', 'relaxing', 'smooth', 'laid back'],
            'uplifting': ['positive', 'inspiring', 'motivational', 'happy'],
            'melancholic': ['sad', 'emotional', 'introspective', 'moody'],
            'dreamy': ['ambient', 'ethereal', 'atmospheric', 'floating'],
            'nostalgic': ['retro', 'vintage', 'memories', 'throwback'],
            'dark': ['brooding', 'intense', 'mysterious', 'heavy'],
            'peaceful': ['serene', 'calm', 'meditative', 'tranquil']
        }
        
        return mood_tag_map.get(mood.lower(), [mood])


class MoodFilteredTracksStrategy(BaseGenerationStrategy):
    """
    Strategy for generating tracks filtered by specific mood criteria.
    
    Used when users have explicit mood preferences.
    """
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate tracks filtered by mood criteria.
        
        Args:
            entities: Extracted entities containing mood information
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of mood-filtered candidate tracks
        """
        candidates = []
        target_moods = self._extract_moods(entities)
        
        if not target_moods:
            self.logger.warning("No target moods found for MoodFilteredTracksStrategy")
            return candidates
        
        self.logger.info(f"🎯 MOOD FILTERED: Filtering for moods {', '.join(target_moods)}")
        
        # For each mood, get tracks and filter by audio features
        for mood in target_moods[:2]:  # Focus on top 2 moods
            try:
                # Get tracks matching mood tags
                mood_tracks = await self.api_service.search_tracks_by_tags(
                    tags=self._get_mood_tags(mood),
                    limit=min(30, limit // len(target_moods))
                )
                
                # Filter tracks by audio features that match the mood
                for track_metadata in mood_tracks:
                    if self._matches_mood_criteria(track_metadata, mood):
                        track = self._convert_metadata_to_dict(
                            track_metadata,
                            source='mood_filtered_tracks',
                            source_confidence=0.8,
                            target_mood=mood,
                            mood_match_score=self._calculate_mood_match_score(track_metadata, mood)
                        )
                        candidates.append(track)
                
            except Exception as e:
                self.logger.warning(f"Failed to get mood-filtered tracks for {mood}: {e}")
                continue
        
        # Sort by mood match score
        candidates.sort(key=lambda x: x.get('mood_match_score', 0), reverse=True)
        
        self.logger.info(f"🎯 MOOD FILTERED: {len(candidates)} mood-matched tracks found")
        return candidates[:limit]
    
    def _matches_mood_criteria(self, track_metadata, mood: str) -> bool:
        """Check if a track matches the mood criteria based on audio features."""
        audio_features = getattr(track_metadata, 'audio_features', {})
        
        if not audio_features:
            return True  # Accept if no audio features available
        
        # Define mood criteria based on audio features
        mood_criteria = {
            'energetic': {'energy': (0.6, 1.0), 'valence': (0.5, 1.0), 'danceability': (0.5, 1.0)},
            'mellow': {'energy': (0.0, 0.5), 'valence': (0.3, 0.8), 'danceability': (0.0, 0.6)},
            'uplifting': {'valence': (0.6, 1.0), 'energy': (0.4, 1.0)},
            'melancholic': {'valence': (0.0, 0.4), 'energy': (0.0, 0.6)},
            'peaceful': {'energy': (0.0, 0.4), 'valence': (0.4, 0.8)},
            'dark': {'valence': (0.0, 0.3), 'mode': (0, 0)}  # Minor key preference
        }
        
        criteria = mood_criteria.get(mood.lower(), {})
        
        # Check each criterion
        for feature, (min_val, max_val) in criteria.items():
            feature_value = audio_features.get(feature, 0.5)
            if not (min_val <= feature_value <= max_val):
                return False
        
        return True
    
    def _calculate_mood_match_score(self, track_metadata, mood: str) -> float:
        """Calculate how well a track matches the target mood."""
        audio_features = getattr(track_metadata, 'audio_features', {})
        
        if not audio_features:
            return 0.5  # Neutral score if no features
        
        # Scoring based on how well features match mood expectations
        mood_scoring = {
            'energetic': lambda af: (af.get('energy', 0.5) + af.get('danceability', 0.5)) / 2,
            'mellow': lambda af: 1.0 - af.get('energy', 0.5),
            'uplifting': lambda af: af.get('valence', 0.5),
            'melancholic': lambda af: 1.0 - af.get('valence', 0.5),
            'peaceful': lambda af: (1.0 - af.get('energy', 0.5) + af.get('valence', 0.5)) / 2
        }
        
        scoring_func = mood_scoring.get(mood.lower())
        if scoring_func:
            return scoring_func(audio_features)
        
        return 0.5  # Default neutral score

    def _get_mood_tags(self, mood: str) -> List[str]:
        """Get associated tags for a given mood."""
        mood_tag_map = {
            'energetic': ['upbeat', 'dance', 'high energy', 'pump up'],
            'mellow': ['chill', 'relaxing', 'smooth', 'laid back'],
            'uplifting': ['positive', 'inspiring', 'motivational', 'happy'],
            'upbeat': ['energetic', 'dance', 'high energy', 'positive'],
            'melancholic': ['sad', 'emotional', 'introspective', 'moody'],
            'dreamy': ['ambient', 'ethereal', 'atmospheric', 'floating'],
            'nostalgic': ['retro', 'vintage', 'memories', 'throwback'],
            'dark': ['brooding', 'intense', 'mysterious', 'heavy'],
            'peaceful': ['serene', 'calm', 'meditative', 'tranquil']
        }
        
        return mood_tag_map.get(mood.lower(), [mood])


class GenreMoodCombinedStrategy(BaseGenerationStrategy):
    """
    Strategy that combines genre and mood filtering for genre_mood queries.
    
    First gets tracks by genre, then filters by mood criteria to ensure
    both requirements are satisfied.
    """
    
    async def generate(
        self,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate tracks that match both genre and mood criteria.
        
        Uses combined genre+mood tag searches since Last.fm doesn't provide
        detailed tags in search results for filtering.
        
        Args:
            entities: Extracted entities containing genre and mood information
            intent_analysis: Intent analysis from planner
            limit: Maximum number of candidates to generate
            **kwargs: Additional parameters
            
        Returns:
            List of genre+mood filtered candidate tracks
        """
        candidates = []
        target_genres = self._extract_target_genres(entities)
        target_moods = self._extract_moods(entities)
        
        if not target_genres:
            self.logger.warning("No target genres found for GenreMoodCombinedStrategy")
            return candidates
            
        if not target_moods:
            self.logger.warning("No target moods found for GenreMoodCombinedStrategy")
            return candidates
        
        self.logger.info(f"🎯 GENRE+MOOD COMBINED: Targeting genres {', '.join(target_genres)} with moods {', '.join(target_moods)}")
        
        # Strategy: Get lots of mood candidates first, then filter by genre
        # This works better because mood tags are more reliable than combined searches
        
        for mood in target_moods[:2]:  # Focus on top 2 moods
            try:
                # Get mood-related search tags
                mood_search_tags = self._get_mood_search_tags(mood)
                
                self.logger.info(f"🎭 Searching for mood '{mood}' using tags: {mood_search_tags}")
                
                # Search for tracks by mood tags with much higher limit
                mood_tracks = await self.api_service.search_tracks_by_tags(
                    tags=mood_search_tags[:3],  # Use top 3 mood tags for more variety
                    limit=min(2000, limit * 20)  # Get MASSIVE pool of tracks to filter from
                )
                
                self.logger.info(f"🎵 Found {len(mood_tracks)} tracks for mood '{mood}'")
                
                # Use LLM client for genre classification if available
                llm_client = self.llm_client
                if llm_client:
                    self.logger.info(f"🤖 Using LLM for genre classification of {len(mood_tracks)} tracks")
                    
                    # Process tracks in batches for performance
                    batch_size = 100
                    classified_tracks = []
                    
                    for i in range(0, len(mood_tracks), batch_size):
                        batch = mood_tracks[i:i + batch_size]
                        batch_classified = await self._classify_tracks_by_genre_llm(
                            batch, target_genres, llm_client, mood
                        )
                        classified_tracks.extend(batch_classified)
                        
                        self.logger.debug(f"🤖 LLM classified batch {i//batch_size + 1}: {len(batch_classified)}/{len(batch)} tracks matched")
                    
                    self.logger.info(f"🤖 LLM classified {len(classified_tracks)} tracks as matching target genres")
                    candidates.extend(classified_tracks)
                    
                else:
                    self.logger.warning("No LLM client available for genre classification, falling back to lenient matching")
                    # Fallback: return all mood tracks (lenient matching)
                    candidates.extend(mood_tracks)
                
            except Exception as e:
                self.logger.warning(f"Failed to get mood+genre tracks for {mood}: {e}")
                continue
        
        # Remove duplicates based on track+artist
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            key = (candidate.get('track', '').lower(), candidate.get('artist', '').lower())
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
        
        self.logger.info(f"🎯 GENRE+MOOD COMBINED: {len(unique_candidates)} unique combined tracks found")
        return unique_candidates[:limit]
    
    def _create_combined_tags(self, genre: str, mood: str) -> List[List[str]]:
        """
        Create combinations of genre and mood tags for searching.
        
        Returns multiple tag combinations to try, ordered by specificity.
        """
        # Get mood-related tags
        mood_tags = self._get_mood_search_tags(mood)
        
        # Create combinations
        combinations = []
        
        # 1. Most specific: genre + specific mood tags
        for mood_tag in mood_tags[:2]:  # Top 2 mood tags
            combinations.append([genre, mood_tag])
        
        # 2. Medium specific: genre + general mood
        combinations.append([genre, mood])
        
        # 3. Broader: just mood tags (in case genre+mood is too restrictive)
        combinations.append(mood_tags[:1])  # Just the best mood tag
        
        return combinations
    
    def _get_mood_search_tags(self, mood: str) -> List[str]:
        """
        Get search tags that work well with Last.fm for a given mood.
        
        These are tags that are commonly used on Last.fm and likely to return results.
        """
        mood_search_tags = {
            'energetic': ['energetic', 'upbeat', 'dance', 'high energy'],
            'upbeat': ['upbeat', 'energetic', 'positive', 'dance'],
            'mellow': ['mellow', 'chill', 'relaxing', 'smooth'],
            'uplifting': ['uplifting', 'positive', 'inspiring', 'happy'],
            'melancholic': ['melancholic', 'sad', 'emotional', 'moody'],
            'peaceful': ['peaceful', 'calm', 'serene', 'tranquil'],
            'dark': ['dark', 'brooding', 'intense', 'heavy'],
            'dreamy': ['dreamy', 'ambient', 'ethereal', 'atmospheric'],
            'nostalgic': ['nostalgic', 'retro', 'vintage', 'throwback']
        }
        
        return mood_search_tags.get(mood.lower(), [mood])
    
    def _matches_genre_criteria(self, track_metadata, target_genres: List[str]) -> bool:
        """
        Check if a track matches any of the target genres.
        
        Uses multiple approaches since genre info might be in different places.
        """
        # Get available genre/tag information
        tags = getattr(track_metadata, 'tags', [])
        genres = getattr(track_metadata, 'genres', [])
        
        # Combine all available genre-related info
        all_genre_info = []
        if tags:
            all_genre_info.extend([tag.lower() for tag in tags])
        if genres:
            all_genre_info.extend([genre.lower() for genre in genres])
        
        # Also check artist name for genre hints (e.g., "DJ Something" suggests electronic)
        artist_name = getattr(track_metadata, 'artist', '').lower()
        
        # Check each target genre
        for target_genre in target_genres:
            target_lower = target_genre.lower()
            
            # Direct match in tags/genres
            if target_lower in all_genre_info:
                return True
            
            # Partial match (e.g., "electronic" matches "electronica", "electro")
            if any(target_lower in info or info in target_lower for info in all_genre_info):
                return True
            
            # Genre-specific heuristics
            if target_lower == 'electronic':
                electronic_indicators = [
                    'electro', 'techno', 'house', 'trance', 'dubstep', 'edm', 
                    'synth', 'dance', 'club', 'dj', 'remix'
                ]
                if any(indicator in all_genre_info for indicator in electronic_indicators):
                    return True
                # Check artist name for DJ/electronic indicators
                if any(indicator in artist_name for indicator in ['dj ', 'dj-', 'electronic']):
                    return True
            
            elif target_lower == 'rock':
                rock_indicators = [
                    'alternative', 'indie rock', 'hard rock', 'metal', 'punk', 
                    'grunge', 'classic rock', 'progressive'
                ]
                if any(indicator in all_genre_info for indicator in rock_indicators):
                    return True
            
            elif target_lower == 'pop':
                pop_indicators = [
                    'pop rock', 'indie pop', 'electropop', 'synthpop', 'dance pop'
                ]
                if any(indicator in all_genre_info for indicator in pop_indicators):
                    return True
        
        # If no genre info available, accept with low confidence
        if not all_genre_info:
            return True
        
        # If we reach here, no matches found - but be more lenient for discovery
        # Accept tracks that might be borderline matches
        return False
    
    def _calculate_genre_match_score(self, track_metadata, target_genres: List[str]) -> float:
        """
        Calculate how well a track matches the target genres.
        
        Returns a score from 0.0 to 1.0.
        """
        tags = getattr(track_metadata, 'tags', [])
        genres = getattr(track_metadata, 'genres', [])
        
        if not tags and not genres:
            return 0.5  # Neutral score if no genre info
        
        all_genre_info = []
        if tags:
            all_genre_info.extend([tag.lower() for tag in tags])
        if genres:
            all_genre_info.extend([genre.lower() for genre in genres])
        
        best_score = 0.0
        
        for target_genre in target_genres:
            target_lower = target_genre.lower()
            
            # Exact match = highest score
            if target_lower in all_genre_info:
                best_score = max(best_score, 1.0)
                continue
            
            # Partial matches = medium score
            partial_matches = [info for info in all_genre_info if target_lower in info or info in target_lower]
            if partial_matches:
                best_score = max(best_score, 0.8)
                continue
            
            # Genre-specific scoring
            if target_lower == 'electronic':
                electronic_score = 0.0
                electronic_indicators = {
                    'electro': 0.9, 'techno': 0.9, 'house': 0.9, 'trance': 0.9,
                    'edm': 0.9, 'dance': 0.7, 'synth': 0.7, 'club': 0.6
                }
                for indicator, score in electronic_indicators.items():
                    if indicator in all_genre_info:
                        electronic_score = max(electronic_score, score)
                best_score = max(best_score, electronic_score)
        
        return best_score
    
    async def _classify_tracks_by_genre_llm(self, mood_tracks: List, target_genres: List[str], llm_client, mood: str) -> List[Dict[str, Any]]:
        """
        Use LLM to classify track genres since Last.fm doesn't provide comprehensive genre tags.
        
                  Args:
              mood_tracks: List of track metadata from Last.fm
              target_genres: Target genres to filter for (e.g., ['electronic'])
              llm_client: LLM client for genre classification
              mood: The mood being searched for
            
        Returns:
            List of tracks that match the target genres according to LLM classification
        """
        if not mood_tracks or not target_genres:
            return []
        
        # Prepare tracks for LLM classification (batch process for efficiency)
        tracks_for_classification = []
        for track_metadata in mood_tracks[:100]:  # Limit to first 100 for performance
            track_info = {
                'artist': track_metadata.artist,
                'track': track_metadata.name,
                'metadata': track_metadata
            }
            tracks_for_classification.append(track_info)
        
        if not tracks_for_classification:
            return []
        
        try:
            # Create target genres string for logging
            target_genres_str = ", ".join(target_genres)
            
            # Debug: Log sample tracks being classified
            sample_tracks = tracks_for_classification[:3]
            sample_track_names = [f"{t['artist']} - {t['track']}" for t in sample_tracks]
            self.logger.debug(f"🤖 Sample tracks for LLM classification: {sample_track_names}")
            
            # Create batch classification prompt
            prompt = self._create_genre_classification_prompt(tracks_for_classification, target_genres)

            # Get LLM response - handle both sync and async clients
            response = llm_client.generate_content(prompt)
            
            # If it's a coroutine (async client), await it
            if hasattr(response, '__await__'):
                response = await response
            
            # Parse the response to get track indices
            response_text = response.text if hasattr(response, 'text') else str(response)
            self.logger.debug(f"🤖 LLM response for genre classification: {response_text[:200]}...")
            matching_indices = self._parse_llm_genre_response(response_text)
            self.logger.debug(f"🤖 Parsed {len(matching_indices)} matching track indices: {matching_indices[:10]}")
            
            # Convert matching tracks to our format
            genre_filtered_tracks = []
            for idx in matching_indices:
                if 0 <= idx < len(tracks_for_classification):
                    track_info = tracks_for_classification[idx]
                    track_metadata = track_info['metadata']
                    
                    # Calculate confidence based on LLM classification
                    genre_match_score = 0.9  # High confidence from LLM
                    
                    track = self._convert_metadata_to_dict(
                        track_metadata,
                        source='genre_mood_combined_llm',
                        source_confidence=0.85 + (genre_match_score * 0.15),  # 0.85-1.0 for LLM classification
                        target_mood=mood,
                        target_genres=target_genres,
                        genre_match_score=genre_match_score,
                        mood_first_approach=True,
                        llm_classified=True
                    )
                    genre_filtered_tracks.append(track)
            
            self.logger.info(f"🤖 LLM classified {len(genre_filtered_tracks)} tracks as {target_genres_str} from {len(tracks_for_classification)} candidates")
            return genre_filtered_tracks
            
        except Exception as e:
            self.logger.warning(f"LLM genre classification failed: {e}, falling back to lenient matching")
            return await self._fallback_genre_filtering(mood_tracks, target_genres, mood)
    
    def _create_genre_classification_prompt(self, tracks_for_classification: List[Dict], target_genres: List[str]) -> str:
        """
        Create a flexible, general-purpose genre classification prompt.
        
        Args:
            tracks_for_classification: List of track info dicts
            target_genres: Target genres to classify for
            
        Returns:
            Formatted prompt for LLM genre classification
        """
        target_genres_str = ", ".join(target_genres)
        tracks_list = "\n".join([f"{i+1}. \"{track['track']}\" by {track['artist']}" 
                                 for i, track in enumerate(tracks_for_classification)])
        
        # Create genre-specific guidance dynamically
        genre_guidance = self._get_genre_classification_guidance(target_genres)
        
        prompt = f"""You are an expert music genre classifier. I need you to identify tracks that match or have significant influences from these genres: {target_genres_str}

Please analyze each track and return ONLY the numbers (1, 2, 3, etc.) of tracks that match, separated by commas.

Tracks to classify:
{tracks_list}

Target genres: {target_genres_str}

Classification criteria:
{genre_guidance}

Important guidelines:
- Include tracks that are primarily in the target genre(s)
- Include tracks that have strong influences from the target genre(s), even if they're not purely that genre
- Consider production style, instrumentation, rhythm, and overall sonic characteristics
- Think about the artist's typical style and how this track fits their catalog
- Be inclusive rather than restrictive - cross-genre and fusion tracks are valuable discoveries

Return only the numbers of matching tracks (e.g., "1, 3, 7, 12"):</prompt>"""
        
        return prompt
    
    def _get_genre_classification_guidance(self, target_genres: List[str]) -> str:
        """
        Generate dynamic classification guidance based on target genres.
        
        Args:
            target_genres: List of target genres
            
        Returns:
            Formatted guidance text for genre classification
        """
        guidance_map = {
            'electronic': {
                'core': ['house', 'techno', 'EDM', 'electropop', 'synthpop', 'electronic dance', 'dubstep', 'trance', 'ambient electronic'],
                'influences': ['electronic production', 'synthesizers', 'drum machines', 'digital effects', 'electronic beats'],
                'examples': 'tracks with electronic production, synth-heavy arrangements, or dance-oriented electronic elements'
            },
            'rock': {
                'core': ['alternative rock', 'indie rock', 'hard rock', 'classic rock', 'punk rock', 'progressive rock', 'metal'],
                'influences': ['guitar-driven', 'rock instrumentation', 'rock rhythms', 'distorted guitars', 'rock vocals'],
                'examples': 'tracks with prominent guitars, rock drumming, or rock song structures'
            },
            'pop': {
                'core': ['mainstream pop', 'pop rock', 'dance pop', 'electropop', 'indie pop', 'synthpop'],
                'influences': ['catchy melodies', 'pop production', 'commercial appeal', 'pop song structures'],
                'examples': 'tracks with pop sensibilities, catchy hooks, or mainstream appeal'
            },
            'hip-hop': {
                'core': ['rap', 'hip hop', 'trap', 'conscious rap', 'gangsta rap', 'alternative hip hop'],
                'influences': ['rap vocals', 'hip-hop beats', 'sampling', 'urban production', 'rhythmic spoken word'],
                'examples': 'tracks with rap verses, hip-hop production, or urban music elements'
            },
            'r&b': {
                'core': ['contemporary R&B', 'neo-soul', 'classic R&B', 'soul', 'funk'],
                'influences': ['soulful vocals', 'R&B harmonies', 'groove-based rhythms', 'smooth production'],
                'examples': 'tracks with soulful singing, R&B vocal styles, or groove-oriented arrangements'
            },
            'jazz': {
                'core': ['traditional jazz', 'bebop', 'smooth jazz', 'fusion', 'contemporary jazz'],
                'influences': ['jazz harmonies', 'improvisation', 'jazz instrumentation', 'swing rhythms', 'complex chord progressions'],
                'examples': 'tracks with jazz chord progressions, improvised solos, or jazz-influenced arrangements'
            },
            'folk': {
                'core': ['traditional folk', 'contemporary folk', 'folk rock', 'indie folk', 'acoustic'],
                'influences': ['acoustic instruments', 'storytelling lyrics', 'organic production', 'traditional melodies'],
                'examples': 'tracks with acoustic guitars, folk storytelling, or organic, non-electronic production'
            },
            'classical': {
                'core': ['orchestral', 'chamber music', 'opera', 'contemporary classical', 'neoclassical'],
                'influences': ['orchestral arrangements', 'classical instruments', 'formal composition', 'classical harmonies'],
                'examples': 'tracks with orchestral elements, classical instruments, or formal compositional structures'
            }
        }
        
        guidance_parts = []
        for genre in target_genres:
            genre_lower = genre.lower()
            if genre_lower in guidance_map:
                info = guidance_map[genre_lower]
                core_genres = ", ".join(info['core'])
                influences = ", ".join(info['influences'])
                
                guidance_parts.append(f"""
• {genre.title()}: Include tracks that are {core_genres}, OR have influences like {influences}
  Example: {info['examples']}""")
            else:
                # Generic guidance for unknown genres
                guidance_parts.append(f"""
• {genre.title()}: Include tracks that are primarily this genre or have strong {genre.lower()} influences""")
        
        return "\n".join(guidance_parts)
    
    def _parse_llm_genre_response(self, response: str) -> List[int]:
        """Parse LLM response to extract track indices."""
        try:
            # Extract numbers from response
            import re
            numbers = re.findall(r'\b(\d+)\b', response)
            # Convert to 0-based indices
            indices = [int(num) - 1 for num in numbers if num.isdigit()]
            return indices
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []
    
    async def _fallback_genre_filtering(self, mood_tracks: List, target_genres: List[str], mood: str) -> List[Dict[str, Any]]:
        """Fallback to lenient genre matching if LLM is unavailable."""
        genre_filtered_tracks = []
        for track_metadata in mood_tracks:
            if self._matches_genre_criteria_lenient(track_metadata, target_genres):
                # Calculate genre match score
                genre_match_score = self._calculate_genre_match_score(track_metadata, target_genres)
                
                track = self._convert_metadata_to_dict(
                    track_metadata,
                    source='genre_mood_combined',
                    source_confidence=0.8 + (genre_match_score * 0.2),  # 0.8-1.0 based on genre match
                    target_mood=mood,
                    target_genres=target_genres,
                    genre_match_score=genre_match_score,
                    mood_first_approach=True
                )
                genre_filtered_tracks.append(track)
        return genre_filtered_tracks
    
    def _matches_genre_criteria_lenient(self, track_metadata, target_genres: List[str]) -> bool:
        """Check if track matches genre criteria with lenient matching."""
        if not target_genres:
            return True
            
        # Get track tags
        track_tags = []
        if hasattr(track_metadata, 'tags') and track_metadata.tags:
            track_tags = [tag.lower() for tag in track_metadata.tags]
        
        # Check for direct genre matches
        for genre in target_genres:
            genre_lower = genre.lower()
            if genre_lower in track_tags:
                return True
        
        # Check for related genre tags
        genre_mappings = {
            'electronic': ['electronic', 'electro', 'synth', 'techno', 'house', 'edm', 'dance', 'electronica'],
            'rock': ['rock', 'alternative', 'indie', 'punk', 'metal', 'grunge'],
            'pop': ['pop', 'mainstream', 'commercial', 'radio'],
            'hip-hop': ['hip-hop', 'rap', 'hip hop', 'hiphop'],
            'jazz': ['jazz', 'smooth jazz', 'bebop', 'swing'],
            'classical': ['classical', 'orchestral', 'symphony', 'opera'],
            'folk': ['folk', 'acoustic', 'singer-songwriter', 'country'],
            'r&b': ['r&b', 'rnb', 'soul', 'funk', 'motown']
        }
        
        for genre in target_genres:
            genre_lower = genre.lower()
            if genre_lower in genre_mappings:
                related_tags = genre_mappings[genre_lower]
                for tag in track_tags:
                    if any(related_tag in tag for related_tag in related_tags):
                        return True
        
        return False
