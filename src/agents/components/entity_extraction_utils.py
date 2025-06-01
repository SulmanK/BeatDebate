"""
Shared Entity Extraction Utilities for BeatDebate Agents

Consolidates entity extraction patterns that are duplicated across agents,
providing a unified approach to entity recognition and processing.
"""

import re
from typing import Dict, List, Any, Optional, Set
import structlog

logger = structlog.get_logger(__name__)


class EntityExtractionUtils:
    """
    Shared utilities for entity extraction across all agents.
    
    Consolidates:
    - Artist name extraction
    - Genre identification
    - Track/album extraction
    - Mood/context detection
    - Entity validation and cleaning
    """
    
    def __init__(self):
        """Initialize entity extraction utilities."""
        self.logger = logger.bind(component="EntityExtractionUtils")
        
        # Common genre patterns
        self.genre_patterns = {
            'rock': ['rock', 'alternative', 'indie rock', 'punk', 'grunge', 'metal'],
            'electronic': ['electronic', 'edm', 'techno', 'house', 'ambient', 'synth'],
            'pop': ['pop', 'mainstream', 'chart', 'commercial'],
            'hip hop': ['hip hop', 'rap', 'hip-hop', 'hiphop'],
            'jazz': ['jazz', 'bebop', 'swing', 'fusion'],
            'classical': ['classical', 'orchestra', 'symphony', 'baroque'],
            'folk': ['folk', 'acoustic', 'singer-songwriter'],
            'r&b': ['r&b', 'soul', 'funk', 'motown'],
            'country': ['country', 'bluegrass', 'americana'],
            'reggae': ['reggae', 'ska', 'dub']
        }
        
        # Common mood indicators
        self.mood_patterns = {
            'energetic': ['energetic', 'upbeat', 'high energy', 'pumped', 'intense'],
            'calm': ['calm', 'peaceful', 'relaxing', 'chill', 'mellow'],
            'melancholic': ['sad', 'melancholic', 'depressing', 'somber', 'moody'],
            'happy': ['happy', 'joyful', 'uplifting', 'cheerful', 'positive'],
            'aggressive': ['aggressive', 'angry', 'intense', 'heavy', 'brutal'],
            'romantic': ['romantic', 'love', 'intimate', 'sensual'],
            'nostalgic': ['nostalgic', 'vintage', 'retro', 'classic']
        }
        
        # Context/activity patterns
        self.context_patterns = {
            'work': ['work', 'coding', 'study', 'focus', 'concentration', 'productivity'],
            'exercise': ['workout', 'gym', 'running', 'exercise', 'fitness', 'training'],
            'party': ['party', 'dance', 'club', 'celebration', 'social'],
            'relax': ['relax', 'chill', 'unwind', 'rest', 'leisure'],
            'driving': ['driving', 'road trip', 'car', 'travel'],
            'sleep': ['sleep', 'bedtime', 'night', 'lullaby']
        }
        
        # Common artist indicators
        self.artist_indicators = [
            'by', 'from', 'artist', 'band', 'singer', 'musician',
            'like', 'similar to', 'sounds like', 'reminds me of'
        ]
    
    def extract_artists_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract artist names from text using pattern matching.
        
        Args:
            text: Input text to extract artists from
            
        Returns:
            Dictionary with 'primary' and 'similar_to' artist lists
        """
        text_lower = text.lower()
        artists = {'primary': [], 'similar_to': []}
        
        # Pattern 1: "by [artist]" or "from [artist]"
        by_pattern = r'\b(?:by|from)\s+([A-Za-z0-9\s&\-\'\.]+?)(?:\s*$|,|\?|!|\s+(?:music|songs?|tracks?|bands?|artists?))'
        by_matches = re.findall(by_pattern, text, re.IGNORECASE)
        for match in by_matches:
            artist = self._clean_artist_name(match)
            if artist and self._is_valid_artist_name(artist):
                artists['primary'].append(artist)
        
        # Pattern 2: "like [artist]" or "similar to [artist]"
        similar_pattern = (r'\b(?:like|similar to|sounds like|reminds me of)\s+'
                          r'([A-Za-z0-9\s&\-\'\.]+?)'
                          r'(?:\s*$|,|\?|!|\s+(?:but|music|songs?|tracks?|bands?|artists?|with|and|or|that|who|which))')
        similar_matches = re.findall(similar_pattern, text, re.IGNORECASE)
        for match in similar_matches:
            artist = self._clean_artist_name(match)
            if artist and self._is_valid_artist_name(artist):
                artists['similar_to'].append(artist)
        
        # Pattern 3: Quoted artist names
        quoted_pattern = r'["\']([A-Za-z0-9\s&\-\'\.]+?)["\']'
        quoted_matches = re.findall(quoted_pattern, text)
        for match in quoted_matches:
            artist = self._clean_artist_name(match)
            if artist and self._is_valid_artist_name(artist):
                # Check if it's in a similarity context
                match_pos = text_lower.find(match.lower())
                context_before = text_lower[max(0, match_pos-20):match_pos]
                if any(indicator in context_before for indicator in ['like', 'similar']):
                    artists['similar_to'].append(artist)
                else:
                    artists['primary'].append(artist)
        
        # Remove duplicates while preserving order
        artists['primary'] = list(dict.fromkeys(artists['primary']))
        artists['similar_to'] = list(dict.fromkeys(artists['similar_to']))
        
        self.logger.debug(
            "Artists extracted from text",
            primary_count=len(artists['primary']),
            similar_count=len(artists['similar_to']),
            text_length=len(text)
        )
        
        return artists
    
    def extract_genres_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract genres from text using pattern matching.
        
        Args:
            text: Input text to extract genres from
            
        Returns:
            Dictionary with 'primary' and 'secondary' genre lists
        """
        text_lower = text.lower()
        genres = {'primary': [], 'secondary': []}
        
        # Check for explicit genre mentions
        for main_genre, variations in self.genre_patterns.items():
            for variation in variations:
                if variation in text_lower:
                    # Determine if it's primary or secondary based on context
                    if self._is_primary_genre_mention(text_lower, variation):
                        if main_genre not in genres['primary']:
                            genres['primary'].append(main_genre)
                    else:
                        if main_genre not in genres['secondary'] and main_genre not in genres['primary']:
                            genres['secondary'].append(main_genre)
        
        # Pattern matching for genre-like words
        genre_pattern = r'\b([a-z]+(?:\s+[a-z]+)*)\s+(?:music|genre|style|sound)\b'
        genre_matches = re.findall(genre_pattern, text_lower)
        for match in genre_matches:
            cleaned_genre = match.strip()
            if cleaned_genre and len(cleaned_genre) > 2:
                if cleaned_genre not in genres['primary'] and cleaned_genre not in genres['secondary']:
                    genres['secondary'].append(cleaned_genre)
        
        self.logger.debug(
            "Genres extracted from text",
            primary_count=len(genres['primary']),
            secondary_count=len(genres['secondary'])
        )
        
        return genres
    
    def extract_moods_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract mood indicators from text.
        
        Args:
            text: Input text to extract moods from
            
        Returns:
            Dictionary with 'primary' mood list
        """
        text_lower = text.lower()
        moods = {'primary': []}
        
        # Check for explicit mood mentions
        for mood, indicators in self.mood_patterns.items():
            for indicator in indicators:
                if indicator in text_lower:
                    if mood not in moods['primary']:
                        moods['primary'].append(mood)
        
        # Pattern matching for mood adjectives
        mood_pattern = r'\b(feel|feeling|mood|vibe|atmosphere)\s+([a-z]+)\b'
        mood_matches = re.findall(mood_pattern, text_lower)
        for _, mood_word in mood_matches:
            if mood_word and len(mood_word) > 3:
                if mood_word not in moods['primary']:
                    moods['primary'].append(mood_word)
        
        self.logger.debug(
            "Moods extracted from text",
            mood_count=len(moods['primary'])
        )
        
        return moods
    
    def extract_context_from_text(self, text: str) -> List[str]:
        """
        Extract context/activity indicators from text.
        
        Args:
            text: Input text to extract context from
            
        Returns:
            List of context indicators
        """
        text_lower = text.lower()
        contexts = []
        
        # Check for explicit context mentions
        for context, indicators in self.context_patterns.items():
            for indicator in indicators:
                if indicator in text_lower:
                    if context not in contexts:
                        contexts.append(context)
        
        # Pattern matching for activity contexts
        activity_pattern = r'\b(?:for|while|during)\s+([a-z]+(?:\s+[a-z]+)*)\b'
        activity_matches = re.findall(activity_pattern, text_lower)
        for match in activity_matches:
            cleaned_activity = match.strip()
            if cleaned_activity and len(cleaned_activity) > 2:
                if cleaned_activity not in contexts:
                    contexts.append(cleaned_activity)
        
        self.logger.debug(
            "Contexts extracted from text",
            context_count=len(contexts)
        )
        
        return contexts
    
    def extract_tracks_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract track/song names from text.
        
        Args:
            text: Input text to extract tracks from
            
        Returns:
            Dictionary with 'primary' track list
        """
        tracks = {'primary': []}
        
        # Pattern 1: "song [title]" or "track [title]"
        song_pattern = r'\b(?:song|track)\s+["\']?([A-Za-z0-9\s&\-\'\.]+?)["\']?(?:\s|$|,|\.|\?|!)'
        song_matches = re.findall(song_pattern, text, re.IGNORECASE)
        for match in song_matches:
            track = self._clean_track_name(match)
            if track and self._is_valid_track_name(track):
                tracks['primary'].append(track)
        
        # Pattern 2: Quoted titles that might be songs
        quoted_pattern = r'["\']([A-Za-z0-9\s&\-\'\.]+?)["\']'
        quoted_matches = re.findall(quoted_pattern, text)
        for match in quoted_matches:
            track = self._clean_track_name(match)
            if track and self._is_valid_track_name(track):
                # Check if it's in a song context
                match_pos = text.lower().find(match.lower())
                context_before = text.lower()[max(0, match_pos-20):match_pos]
                context_after = text.lower()[match_pos:match_pos+len(match)+20]
                if any(indicator in context_before + context_after for indicator in ['song', 'track', 'play']):
                    tracks['primary'].append(track)
        
        # Remove duplicates
        tracks['primary'] = list(dict.fromkeys(tracks['primary']))
        
        self.logger.debug(
            "Tracks extracted from text",
            track_count=len(tracks['primary'])
        )
        
        return tracks
    
    def validate_and_enhance_entities(
        self, entities: Dict[str, Any], original_text: str
    ) -> Dict[str, Any]:
        """
        Validate and enhance extracted entities.
        
        Args:
            entities: Extracted entities dictionary
            original_text: Original text for context
            
        Returns:
            Enhanced entities dictionary
        """
        enhanced_entities = entities.copy()
        
        # Ensure musical_entities structure exists
        if 'musical_entities' not in enhanced_entities:
            enhanced_entities['musical_entities'] = {}
        
        musical_entities = enhanced_entities['musical_entities']
        
        # Helper to process entity categories (artists, genres, tracks, moods)
        def process_entity_category(entity_type_plural: str, cleaner_func, validator_func):
            if entity_type_plural not in musical_entities:
                # If not present at all, extract fresh from original_text
                if entity_type_plural == 'artists':
                    musical_entities[entity_type_plural] = self.extract_artists_from_text(original_text)
                elif entity_type_plural == 'genres':
                    musical_entities[entity_type_plural] = self.extract_genres_from_text(original_text)
                elif entity_type_plural == 'tracks':
                    musical_entities[entity_type_plural] = self.extract_tracks_from_text(original_text)
                elif entity_type_plural == 'moods':
                    musical_entities[entity_type_plural] = self.extract_moods_from_text(original_text)
            
            # Now process/validate the entity categories
            entity_categories_dict = musical_entities[entity_type_plural]
            if isinstance(entity_categories_dict, dict):
                for category_key in entity_categories_dict:
                    current_items = entity_categories_dict[category_key]
                    if isinstance(current_items, list):
                        processed_item_list = []
                        for item_entry in current_items:
                            if isinstance(item_entry, str):
                                # Simple string entry
                                cleaned_name_str = cleaner_func(item_entry)
                                if validator_func(cleaned_name_str):
                                    processed_item_list.append(cleaned_name_str)
                            elif isinstance(item_entry, dict):
                                # Dict entry with possible confidence scores
                                item_name_to_clean = item_entry.get('name')
                                original_confidence = item_entry.get('confidence', 0.5)
                                
                                if isinstance(item_name_to_clean, str):
                                    cleaned_name_str = cleaner_func(item_name_to_clean)
                                    if validator_func(cleaned_name_str):
                                        processed_item_list.append({
                                            'name': cleaned_name_str,
                                            'confidence': original_confidence 
                                        })
                                elif item_name_to_clean is None and isinstance(item_entry, dict):
                                    self.logger.warning(f"{entity_type_plural} entry object is a dict but missing 'name'", entity_entry=item_entry)
                                elif item_name_to_clean is not None:  # Should be string or None by now
                                    self.logger.warning(f"Unexpected type for {entity_type_plural} name_to_clean", type_found=type(item_name_to_clean))
                        entity_categories_dict[category_key] = processed_item_list
                    else:
                        self.logger.warning(f"Expected list for {entity_type_plural}.{category_key}, got {type(current_items)}")
        
        # Process each entity type
        process_entity_category('artists', self._clean_artist_name, self._is_valid_artist_name)
        process_entity_category('genres', lambda x: x.strip().lower(), lambda x: len(x) > 1)  # Simpler cleaning for genres
        process_entity_category('tracks', self._clean_track_name, self._is_valid_track_name)
        
        # Handle moods (they follow the same structure)
        if 'moods' not in musical_entities:
            musical_entities['moods'] = self.extract_moods_from_text(original_text)
        else:
            # Process moods with basic cleaning
            process_entity_category('moods', lambda x: x.strip().lower(), lambda x: len(x) > 1)
        
        # Add context information
        if 'context_factors' not in enhanced_entities:
            enhanced_entities['context_factors'] = self.extract_context_from_text(original_text)
        
        # Add confidence scores (this will convert remaining string items to dicts)
        enhanced_entities = self._add_confidence_scores(enhanced_entities, original_text)
        
        self.logger.debug(
            "Entities validated and enhanced",
            total_artists=len(musical_entities.get('artists', {}).get('primary', [])),
            total_genres=len(musical_entities.get('genres', {}).get('primary', [])),
            total_moods=len(musical_entities.get('moods', {}).get('primary', [])),
            total_tracks=len(musical_entities.get('tracks', {}).get('primary', []))
        )
        
        return enhanced_entities
    
    def _clean_artist_name(self, artist: str) -> str:
        """Clean and normalize artist name."""
        if not artist:
            return ""
        
        # Remove extra whitespace and common artifacts
        cleaned = re.sub(r'\s+', ' ', artist.strip())
        
        # Remove common prefixes/suffixes that aren't part of artist names
        cleaned = re.sub(r'^(the\s+)?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+(band|group|artist|singer)$', '', cleaned, flags=re.IGNORECASE)
        
        # Remove punctuation at the end
        cleaned = re.sub(r'[.,!?;]+$', '', cleaned)
        
        return cleaned.strip()
    
    def _clean_track_name(self, track: str) -> str:
        """Clean and normalize track name."""
        if not track:
            return ""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', track.strip())
        
        # Remove punctuation at the end
        cleaned = re.sub(r'[.,!?;]+$', '', cleaned)
        
        return cleaned.strip()
    
    def _is_valid_artist_name(self, artist: str) -> bool:
        """Check if artist name is valid."""
        if not artist or len(artist) < 2:
            return False
        
        # Check for common non-artist words
        invalid_words = {
            'music', 'song', 'track', 'album', 'genre', 'style', 'sound',
            'playlist', 'radio', 'station', 'channel', 'video', 'audio',
            'listen', 'play', 'hear', 'find', 'search', 'recommend',
            'good', 'great', 'best', 'new', 'old', 'popular', 'famous'
        }
        
        artist_lower = artist.lower()
        if artist_lower in invalid_words:
            return False
        
        # Check if it's mostly numbers or special characters
        if re.match(r'^[0-9\s\-_\.]+$', artist):
            return False
        
        return True
    
    def _is_valid_track_name(self, track: str) -> bool:
        """Check if track name is valid."""
        if not track or len(track) < 2:
            return False
        
        # Check for common non-track words
        invalid_words = {
            'music', 'artist', 'band', 'singer', 'musician', 'genre',
            'style', 'sound', 'playlist', 'radio', 'station', 'channel'
        }
        
        track_lower = track.lower()
        if track_lower in invalid_words:
            return False
        
        return True
    
    def _is_primary_genre_mention(self, text: str, genre: str) -> bool:
        """Determine if genre mention is primary based on context."""
        genre_pos = text.find(genre)
        if genre_pos == -1:
            return False
        
        # Check context around the genre mention
        context_before = text[max(0, genre_pos-30):genre_pos]
        context_after = text[genre_pos:genre_pos+len(genre)+30]
        
        # Primary indicators
        primary_indicators = ['want', 'need', 'looking for', 'find', 'recommend', 'love', 'like']
        
        # Secondary indicators
        secondary_indicators = ['also', 'maybe', 'sometimes', 'occasionally', 'similar']
        
        context = context_before + context_after
        
        if any(indicator in context for indicator in primary_indicators):
            return True
        elif any(indicator in context for indicator in secondary_indicators):
            return False
        
        # Default to primary if no clear indicators
        return True
    
    def _add_confidence_scores(
        self, entities: Dict[str, Any], original_text: str
    ) -> Dict[str, Any]:
        """Add confidence scores to extracted entities."""
        text_length = len(original_text)
        
        # Calculate confidence based on text length and entity specificity
        base_confidence = min(0.8, 0.3 + (text_length / 200))
        
        musical_entities = entities.get('musical_entities', {})
        
        # Helper function to process entity categories
        def add_confidence_to_category(category_name: str, confidence_modifier: float = 1.0):
            if category_name in musical_entities:
                for category_key in musical_entities[category_name]:
                    category_list = musical_entities[category_name][category_key]
                    if isinstance(category_list, list):
                        for i, item in enumerate(category_list):
                            if isinstance(item, dict):
                                # Item already has confidence, preserve it unless it's too low
                                if item.get('confidence', 0) < 0.3:
                                    item['confidence'] = base_confidence * confidence_modifier
                            elif isinstance(item, str):
                                # Convert string to dict with confidence
                                item_confidence = base_confidence * confidence_modifier
                                if category_name == 'artists':
                                    # Higher confidence for longer, more specific artist names
                                    item_confidence += (len(item) / 100)
                                
                                category_list[i] = {
                                    'name': item,
                                    'confidence': min(0.95, item_confidence)
                                }
                            else:
                                # Handle unexpected item types
                                self.logger.warning(
                                    f"Unexpected item type in {category_name}.{category_key}",
                                    item_type=type(item),
                                    item=str(item)
                                )
        
        # Add confidence to each entity type
        add_confidence_to_category('artists', 1.0)
        add_confidence_to_category('genres', 1.0)
        add_confidence_to_category('moods', 1.0)
        add_confidence_to_category('tracks', 1.0)
        
        return entities
    
    def extract_similarity_indicators(self, text: str) -> Dict[str, Any]:
        """
        Extract similarity indicators and comparison patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with similarity information
        """
        text_lower = text.lower()
        similarity_info = {
            'has_similarity_request': False,
            'similarity_type': None,
            'comparison_artists': [],
            'similarity_strength': 'medium'
        }
        
        # Check for similarity patterns
        similarity_patterns = [
            r'\b(?:like|similar to|sounds like|reminds me of)\s+([A-Za-z0-9\s&\-\'\.]+)',
            r'\b(?:in the style of|influenced by)\s+([A-Za-z0-9\s&\-\'\.]+)',
            r'\b(?:comparable to|along the lines of)\s+([A-Za-z0-9\s&\-\'\.]+)'
        ]
        
        for pattern in similarity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                similarity_info['has_similarity_request'] = True
                for match in matches:
                    artist = self._clean_artist_name(match)
                    if artist and self._is_valid_artist_name(artist):
                        similarity_info['comparison_artists'].append(artist)
        
        # Determine similarity type
        if 'exactly like' in text_lower or 'just like' in text_lower:
            similarity_info['similarity_type'] = 'exact'
            similarity_info['similarity_strength'] = 'high'
        elif 'somewhat like' in text_lower or 'kind of like' in text_lower:
            similarity_info['similarity_type'] = 'loose'
            similarity_info['similarity_strength'] = 'low'
        elif similarity_info['has_similarity_request']:
            similarity_info['similarity_type'] = 'moderate'
            similarity_info['similarity_strength'] = 'medium'
        
        return similarity_info 