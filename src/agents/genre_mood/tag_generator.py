"""
Tag Generator for Genre Mood Agent

Provides specialized tag generation for genre and mood-based music search,
including genre mapping, tag optimization, and search strategy generation.
"""

from typing import Dict, List, Any, Set
import structlog

logger = structlog.get_logger(__name__)


class TagGenerator:
    """
    Specialized tag generator for genre/mood-based music search.
    
    Provides:
    - Genre-specific tag generation
    - Mood-based tag optimization
    - Search tag prioritization
    - Tag combination strategies
    """
    
    def __init__(self):
        self.genre_mappings = self._initialize_genre_mappings()
        self.genre_synonyms = self._initialize_genre_synonyms()
        self.tag_priorities = self._initialize_tag_priorities()
        self.tag_combinations = self._initialize_tag_combinations()
        
        logger.debug("TagGenerator initialized with genre mappings")
    
    def generate_search_tags(
        self,
        entities: Dict[str, Any],
        mood_analysis: Dict[str, Any],
        max_tags: int = 10
    ) -> List[str]:
        """
        Generate optimized search tags for genre/mood-based discovery.
        
        Args:
            entities: Extracted entities from query
            mood_analysis: Mood analysis results
            max_tags: Maximum number of tags to return
            
        Returns:
            Prioritized list of search tags
        """
        all_tags = []
        
        # Generate genre-based tags
        genre_tags = self._generate_genre_tags(entities)
        all_tags.extend(genre_tags)
        
        # Generate mood-based tags
        mood_tags = self._generate_mood_tags(mood_analysis)
        all_tags.extend(mood_tags)
        
        # Generate combination tags
        combo_tags = self._generate_combination_tags(genre_tags, mood_tags)
        all_tags.extend(combo_tags)
        
        # Prioritize and deduplicate
        prioritized_tags = self._prioritize_tags(all_tags, entities, mood_analysis)
        
        # Return top tags
        return prioritized_tags[:max_tags]
    
    def _generate_genre_tags(self, entities: Dict[str, Any]) -> List[str]:
        """Generate genre-specific search tags."""
        genre_tags = []
        
        # Extract target genres
        musical_entities = entities.get('musical_entities', {})
        genres = musical_entities.get('genres', {})
        
        primary_genres = genres.get('primary', [])
        secondary_genres = genres.get('secondary', [])
        
        # Process primary genres
        for genre in primary_genres:
            genre_lower = genre.lower()
            
            # Add direct genre
            genre_tags.append(genre_lower)
            
            # Add genre mappings
            if genre_lower in self.genre_mappings:
                genre_tags.extend(self.genre_mappings[genre_lower][:3])
            
            # Add genre synonyms
            if genre_lower in self.genre_synonyms:
                genre_tags.extend(self.genre_synonyms[genre_lower][:2])
        
        # Process secondary genres (lower priority)
        for genre in secondary_genres:
            genre_lower = genre.lower()
            genre_tags.append(genre_lower)
            
            if genre_lower in self.genre_mappings:
                genre_tags.extend(self.genre_mappings[genre_lower][:2])
        
        return list(set(genre_tags))  # Remove duplicates
    
    def _generate_mood_tags(self, mood_analysis: Dict[str, Any]) -> List[str]:
        """Generate mood-specific search tags."""
        mood_tags = []
        
        # Primary mood tags
        primary_moods = mood_analysis.get('primary_moods', [])
        for mood in primary_moods:
            mood_tags.append(mood.lower())
        
        # Energy level tags
        energy_level = mood_analysis.get('energy_level', 'medium')
        energy_tags = {
            'high': ['energetic', 'upbeat', 'intense'],
            'medium': ['moderate', 'balanced', 'steady'],
            'low': ['calm', 'peaceful', 'mellow']
        }
        mood_tags.extend(energy_tags.get(energy_level, []))
        
        # Context modifier tags
        context_modifiers = mood_analysis.get('context_modifiers', [])
        mood_tags.extend(context_modifiers)
        
        # Mood-specific tags from analysis
        mood_specific_tags = mood_analysis.get('mood_tags', [])
        mood_tags.extend(mood_specific_tags)
        
        return list(set(mood_tags))  # Remove duplicates
    
    def _generate_combination_tags(
        self, 
        genre_tags: List[str], 
        mood_tags: List[str]
    ) -> List[str]:
        """Generate combination tags from genres and moods."""
        combo_tags = []
        
        # Genre + mood combinations
        for genre in genre_tags[:3]:  # Top 3 genres
            for mood in mood_tags[:3]:  # Top 3 moods
                if genre in self.tag_combinations:
                    if mood in self.tag_combinations[genre]:
                        combo_tags.extend(self.tag_combinations[genre][mood])
        
        # Common genre-mood combinations
        common_combinations = {
            ('rock', 'energetic'): ['hard rock', 'arena rock', 'driving rock'],
            ('electronic', 'energetic'): ['dance', 'edm', 'techno'],
            ('jazz', 'calm'): ['smooth jazz', 'cool jazz', 'ambient jazz'],
            ('classical', 'peaceful'): ['classical relaxation', 'chamber music'],
            ('indie', 'melancholic'): ['indie folk', 'dream pop', 'shoegaze'],
            ('hip hop', 'aggressive'): ['hardcore rap', 'gangsta rap'],
            ('pop', 'happy'): ['upbeat pop', 'dance pop', 'feel good']
        }
        
        for genre in genre_tags:
            for mood in mood_tags:
                combo_key = (genre, mood)
                if combo_key in common_combinations:
                    combo_tags.extend(common_combinations[combo_key])
        
        return list(set(combo_tags))  # Remove duplicates
    
    def _prioritize_tags(
        self,
        all_tags: List[str],
        entities: Dict[str, Any],
        mood_analysis: Dict[str, Any]
    ) -> List[str]:
        """Prioritize tags based on relevance and effectiveness."""
        tag_scores = {}
        
        for tag in set(all_tags):  # Remove duplicates
            score = 0.0
            
            # Base priority from tag priorities
            if tag in self.tag_priorities:
                score += self.tag_priorities[tag]
            else:
                score += 0.5  # Default score
            
            # Boost for primary genres
            musical_entities = entities.get('musical_entities', {})
            primary_genres = musical_entities.get('genres', {}).get('primary', [])
            if any(tag in genre.lower() for genre in primary_genres):
                score += 0.3
            
            # Boost for primary moods
            primary_moods = mood_analysis.get('primary_moods', [])
            if any(tag in mood.lower() for mood in primary_moods):
                score += 0.3
            
            # Boost for high energy level
            if mood_analysis.get('energy_level') == 'high' and tag in ['energetic', 'upbeat', 'intense']:
                score += 0.2
            
            # Boost for mood intensity
            mood_intensity = mood_analysis.get('mood_intensity', 0.5)
            if mood_intensity > 0.7 and tag in ['intense', 'powerful', 'strong']:
                score += 0.2
            elif mood_intensity < 0.3 and tag in ['gentle', 'soft', 'subtle']:
                score += 0.2
            
            tag_scores[tag] = score
        
        # Sort by score and return
        sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, score in sorted_tags]
    
    def get_genre_alternatives(self, genre: str) -> List[str]:
        """Get alternative genre tags for broader search."""
        genre_lower = genre.lower()
        alternatives = []
        
        if genre_lower in self.genre_synonyms:
            alternatives.extend(self.genre_synonyms[genre_lower])
        
        if genre_lower in self.genre_mappings:
            alternatives.extend(self.genre_mappings[genre_lower])
        
        return list(set(alternatives))
    
    def _initialize_genre_mappings(self) -> Dict[str, List[str]]:
        """Initialize genre to related tags mappings."""
        return {
            'rock': [
                'alternative rock', 'indie rock', 'classic rock', 'hard rock',
                'progressive rock', 'psychedelic rock', 'garage rock'
            ],
            'electronic': [
                'electronica', 'edm', 'techno', 'house', 'ambient',
                'downtempo', 'synthwave', 'drum and bass'
            ],
            'jazz': [
                'smooth jazz', 'bebop', 'cool jazz', 'fusion', 'swing',
                'contemporary jazz', 'acid jazz', 'nu jazz'
            ],
            'classical': [
                'baroque', 'romantic', 'contemporary classical', 'chamber music',
                'orchestral', 'piano', 'string quartet'
            ],
            'hip hop': [
                'rap', 'trap', 'old school hip hop', 'conscious rap',
                'gangsta rap', 'alternative hip hop', 'underground hip hop'
            ],
            'pop': [
                'dance pop', 'synth pop', 'indie pop', 'electropop',
                'dream pop', 'art pop', 'experimental pop'
            ],
            'indie': [
                'indie rock', 'indie pop', 'indie folk', 'indie electronic',
                'lo-fi', 'bedroom pop', 'indie alternative'
            ],
            'folk': [
                'folk rock', 'indie folk', 'contemporary folk', 'acoustic',
                'singer-songwriter', 'americana', 'alt-country'
            ],
            'metal': [
                'heavy metal', 'death metal', 'black metal', 'progressive metal',
                'doom metal', 'thrash metal', 'metalcore'
            ],
            'reggae': [
                'dub', 'ska', 'dancehall', 'roots reggae', 'reggae fusion'
            ]
        }
    
    def _initialize_genre_synonyms(self) -> Dict[str, List[str]]:
        """Initialize genre synonyms for broader matching."""
        return {
            'electronic': ['edm', 'electronica', 'dance', 'techno'],
            'hip hop': ['rap', 'hip-hop', 'hiphop'],
            'r&b': ['rnb', 'rhythm and blues', 'soul'],
            'alternative': ['alt', 'alternative rock', 'indie'],
            'classical': ['orchestral', 'symphony', 'chamber'],
            'country': ['americana', 'alt-country', 'folk'],
            'reggae': ['ska', 'dub', 'dancehall'],
            'punk': ['punk rock', 'hardcore', 'post-punk'],
            'blues': ['electric blues', 'delta blues', 'chicago blues']
        }
    
    def _initialize_tag_priorities(self) -> Dict[str, float]:
        """Initialize tag priority scores."""
        return {
            # High priority genre tags
            'rock': 0.9, 'pop': 0.9, 'electronic': 0.9, 'hip hop': 0.9,
            'indie': 0.8, 'alternative': 0.8, 'jazz': 0.8, 'classical': 0.8,
            
            # High priority mood tags
            'energetic': 0.9, 'calm': 0.9, 'happy': 0.9, 'melancholic': 0.8,
            'upbeat': 0.8, 'peaceful': 0.8, 'intense': 0.8, 'chill': 0.8,
            
            # Medium priority tags
            'ambient': 0.7, 'acoustic': 0.7, 'instrumental': 0.7,
            'experimental': 0.6, 'underground': 0.6, 'vintage': 0.6,
            
            # Context tags
            'workout': 0.8, 'study': 0.7, 'party': 0.7, 'driving': 0.7
        }
    
    def _initialize_tag_combinations(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize effective genre-mood tag combinations."""
        return {
            'rock': {
                'energetic': ['hard rock', 'arena rock', 'driving'],
                'aggressive': ['metal', 'hardcore', 'heavy'],
                'melancholic': ['alternative rock', 'grunge', 'post-rock']
            },
            'electronic': {
                'energetic': ['dance', 'edm', 'techno', 'house'],
                'calm': ['ambient', 'downtempo', 'chillout'],
                'mysterious': ['dark ambient', 'industrial', 'experimental']
            },
            'jazz': {
                'calm': ['smooth jazz', 'cool jazz', 'lounge'],
                'energetic': ['fusion', 'bebop', 'swing'],
                'romantic': ['vocal jazz', 'standards', 'ballads']
            },
            'indie': {
                'melancholic': ['indie folk', 'dream pop', 'shoegaze'],
                'happy': ['indie pop', 'jangle pop', 'twee'],
                'energetic': ['indie rock', 'garage rock', 'post-punk']
            }
        } 