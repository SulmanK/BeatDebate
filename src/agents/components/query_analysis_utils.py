"""
Shared Query Analysis Utilities for BeatDebate Agents

Consolidates query analysis patterns that are duplicated across agents,
providing a unified approach to query understanding and intent detection.
"""

import re
from typing import Dict, List, Any
import structlog

logger = structlog.get_logger(__name__)


class QueryAnalysisUtils:
    """
    Shared utilities for query analysis across all agents.
    
    Consolidates:
    - Intent detection
    - Complexity analysis
    - Mood/context extraction
    - Query classification
    - Pattern matching utilities
    """
    
    def __init__(self):
        """Initialize query analysis utilities."""
        self.logger = logger.bind(component="QueryAnalysisUtils")
        
        # Intent patterns
        self.intent_patterns = {
            'discovery': [
                'find', 'discover', 'explore', 'recommend', 'suggest',
                'new', 'different', 'unknown', 'underground', 'hidden',
                'fresh', 'novel', 'rare', 'obscure', 'gems'
            ],
            'similarity': [
                'like', 'similar', 'sounds like', 'reminds me of',
                'style of', 'same as', 'comparable to'
            ],
            'mood_based': [
                'feel', 'mood', 'vibe', 'atmosphere', 'energy',
                'upbeat', 'calm', 'relaxing', 'energetic', 'chill'
            ],
            'activity_based': [
                'work', 'study', 'studying', 'exercise', 'party', 'relax',
                'driving', 'cooking', 'sleeping', 'focus', 'workout',
                'concentration', 'background', 'ambient', 'for work',
                'for study', 'for studying', 'for exercise', 'for driving',
                'for cooking', 'for relaxing', 'for sleeping', 'for focus',
                'while working', 'while studying', 'while driving',
                'while cooking', 'while exercising', 'gym music',
                'office music', 'study music', 'workout music',
                'background music', 'productivity', 'homework'
            ],
            'genre_specific': [
                'rock', 'pop', 'electronic', 'jazz', 'classical',
                'hip hop', 'country', 'folk', 'metal', 'indie'
            ]
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            'simple': [
                'just', 'only', 'simple', 'basic', 'easy',
                'quick', 'fast', 'straightforward'
            ],
            'medium': [
                'some', 'few', 'several', 'maybe', 'perhaps',
                'could', 'might', 'possibly'
            ],
            'complex': [
                'detailed', 'comprehensive', 'thorough', 'deep',
                'extensive', 'elaborate', 'sophisticated', 'nuanced'
            ]
        }
        
        # Urgency indicators
        self.urgency_patterns = {
            'high': ['urgent', 'asap', 'immediately', 'right now', 'quickly'],
            'medium': ['soon', 'when possible', 'at your convenience'],
            'low': ['whenever', 'no rush', 'take your time']
        }
        
        # Quality preferences
        self.quality_patterns = {
            'high_quality': [
                'best', 'top', 'excellent', 'outstanding', 'premium',
                'high quality', 'masterpiece', 'classic'
            ],
            'popular': [
                'popular', 'mainstream', 'well-known', 'famous',
                'chart', 'hit', 'trending'
            ],
            'underground': [
                'underground', 'obscure', 'hidden', 'rare',
                'unknown', 'indie', 'alternative'
            ]
        }
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine primary intent and confidence.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with intent analysis
        """
        # Ensure query is a string
        if isinstance(query, dict):
            query = query.get('query', str(query))
        elif not isinstance(query, str):
            query = str(query)
            
        query_lower = query.lower()
        intent_scores = {}
        
        # Score each intent category
        for intent, patterns in self.intent_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if pattern in query_lower:
                    score += 1
                    matched_patterns.append(pattern)
            
            if score > 0:
                intent_scores[intent] = {
                    'score': score,
                    'confidence': min(0.9, score * 0.3),
                    'matched_patterns': matched_patterns
                }
        
        # 🔧 FIX: Detect hybrid intents for queries with multiple strong intent signals
        primary_intent = 'discovery'  # Default
        primary_confidence = 0.3
        
        if intent_scores:
            # Check for hybrid intent patterns
            has_genre = 'genre_specific' in intent_scores  
            has_similarity = 'similarity' in intent_scores
            has_mood = 'mood_based' in intent_scores
            
            # Multi-intent detection for hybrid queries
            if has_similarity and (has_mood or has_genre):
                # "chill songs like Bon Iver" = similarity + mood -> HYBRID
                primary_intent = 'hybrid'
                primary_confidence = 0.8
                self.logger.info(f"🔧 HYBRID DETECTED: similarity + mood/genre in query: '{query}'")
            else:
                # Single intent - pick the highest scoring
                primary_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]['score'])
                primary_confidence = intent_scores[primary_intent]['confidence']
        
        # Determine secondary intents
        secondary_intents = [
            intent for intent, data in intent_scores.items() 
            if intent != primary_intent and data['score'] > 0
        ]
        
        self.logger.debug(
            "Query intent analyzed",
            primary_intent=primary_intent,
            primary_confidence=primary_confidence,
            secondary_intents=secondary_intents,
            total_intent_scores=len(intent_scores)
        )
        
        return {
            'primary_intent': primary_intent,
            'primary_confidence': primary_confidence,
            'secondary_intents': secondary_intents,
            'intent_scores': intent_scores,
            'has_multiple_intents': len(intent_scores) > 1
        }
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity based on various factors.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with complexity analysis
        """
        # Ensure query is a string
        if isinstance(query, dict):
            query = query.get('query', str(query))
        elif not isinstance(query, str):
            query = str(query)
            
        query_lower = query.lower()
        
        # Basic metrics
        word_count = len(query.split())
        sentence_count = len(re.split(r'[.!?]+', query))
        
        # Complexity indicators
        complexity_scores = {}
        for level, indicators in self.complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            if score > 0:
                complexity_scores[level] = score
        
        # Determine complexity level
        if complexity_scores.get('complex', 0) > 0 or word_count > 20:
            complexity_level = 'complex'
            confidence = 0.8
        elif complexity_scores.get('simple', 0) > 0 or word_count < 5:
            complexity_level = 'simple'
            confidence = 0.7
        else:
            complexity_level = 'medium'
            confidence = 0.6
        
        # Additional complexity factors
        has_multiple_entities = self._count_entities_in_query(query) > 2
        has_conditional_logic = any(word in query_lower for word in ['if', 'when', 'unless', 'but'])
        has_comparisons = any(word in query_lower for word in ['better', 'worse', 'more', 'less', 'than'])
        
        # Adjust complexity based on additional factors
        if has_multiple_entities or has_conditional_logic or has_comparisons:
            if complexity_level == 'simple':
                complexity_level = 'medium'
            elif complexity_level == 'medium':
                complexity_level = 'complex'
        
        self.logger.debug(
            "Query complexity analyzed",
            complexity_level=complexity_level,
            word_count=word_count,
            sentence_count=sentence_count,
            has_multiple_entities=has_multiple_entities
        )
        
        return {
            'complexity_level': complexity_level,
            'confidence': confidence,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'complexity_scores': complexity_scores,
            'has_multiple_entities': has_multiple_entities,
            'has_conditional_logic': has_conditional_logic,
            'has_comparisons': has_comparisons
        }
    
    def extract_mood_indicators(self, query: str) -> List[str]:
        """
        Extract mood indicators from query text.
        
        Args:
            query: User query text
            
        Returns:
            List of detected mood indicators
        """
        query_lower = query.lower()
        mood_indicators = []
        
        # Direct mood words
        mood_words = [
            'happy', 'sad', 'energetic', 'calm', 'relaxed', 'excited',
            'melancholic', 'upbeat', 'chill', 'intense', 'peaceful',
            'aggressive', 'romantic', 'nostalgic', 'dreamy', 'dark'
        ]
        
        for mood in mood_words:
            if mood in query_lower:
                mood_indicators.append(mood)
        
        # Contextual mood patterns
        mood_patterns = [
            (r'\bfeel(?:ing)?\s+(\w+)', 'feeling'),
            (r'\bmood\s+(?:is\s+)?(\w+)', 'mood'),
            (r'\bvibe\s+(?:is\s+)?(\w+)', 'vibe'),
            (r'\b(\w+)\s+energy\b', 'energy'),
            (r'\bmake\s+me\s+feel\s+(\w+)', 'effect')
        ]
        
        for pattern, context in mood_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if len(match) > 2:  # Filter out very short words
                    mood_indicators.append(f"{match} ({context})")
        
        # Remove duplicates while preserving order
        mood_indicators = list(dict.fromkeys(mood_indicators))
        
        self.logger.debug(
            "Mood indicators extracted",
            mood_count=len(mood_indicators),
            moods=mood_indicators
        )
        
        return mood_indicators
    
    def extract_context_factors(self, query: str) -> List[str]:
        """
        Extract context/activity factors from query text.
        
        Args:
            query: User query text
            
        Returns:
            List of detected context factors
        """
        query_lower = query.lower()
        context_factors = []
        
        # Activity contexts
        activities = [
            'work', 'working', 'study', 'studying', 'exercise', 'workout',
            'party', 'partying', 'relax', 'relaxing', 'drive', 'driving',
            'cook', 'cooking', 'sleep', 'sleeping', 'focus', 'focusing',
            'read', 'reading', 'clean', 'cleaning', 'travel', 'traveling'
        ]
        
        for activity in activities:
            if activity in query_lower:
                # Normalize to base form
                base_activity = activity.rstrip('ing').rstrip('e') + ('e' if activity.endswith('ing') and not activity.endswith('eing') else '')
                if base_activity not in context_factors:
                    context_factors.append(base_activity)
        
        # Time contexts
        time_patterns = [
            (r'\b(morning|afternoon|evening|night)\b', 'time_of_day'),
            (r'\b(weekend|weekday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 'day_type'),
            (r'\b(summer|winter|spring|fall|autumn)\b', 'season')
        ]
        
        for pattern, context_type in time_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                context_factors.append(f"{match} ({context_type})")
        
        # Location contexts
        location_patterns = [
            r'\b(home|office|car|gym|outdoors|inside|outside)\b',
            r'\b(at\s+(?:the\s+)?(\w+))\b'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                if len(match) > 2:
                    context_factors.append(f"{match} (location)")
        
        # Remove duplicates
        context_factors = list(dict.fromkeys(context_factors))
        
        self.logger.debug(
            "Context factors extracted",
            context_count=len(context_factors),
            contexts=context_factors
        )
        
        return context_factors
    
    def detect_urgency_level(self, query: str) -> Dict[str, Any]:
        """
        Detect urgency level from query text.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with urgency analysis
        """
        query_lower = query.lower()
        urgency_scores = {}
        
        # Check for urgency patterns
        for level, patterns in self.urgency_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                urgency_scores[level] = score
        
        # Determine urgency level
        if urgency_scores:
            urgency_level = max(urgency_scores.keys(), key=lambda x: urgency_scores[x])
            confidence = min(0.9, urgency_scores[urgency_level] * 0.4)
        else:
            urgency_level = 'medium'  # Default
            confidence = 0.3
        
        # Check for time-sensitive language
        time_sensitive_patterns = [
            r'\bnow\b', r'\btoday\b', r'\btonight\b', r'\bimmediately\b',
            r'\bquickly\b', r'\basap\b', r'\burgent\b'
        ]
        
        has_time_pressure = any(
            re.search(pattern, query_lower) for pattern in time_sensitive_patterns
        )
        
        if has_time_pressure and urgency_level == 'medium':
            urgency_level = 'high'
            confidence = max(confidence, 0.7)
        
        self.logger.debug(
            "Urgency level detected",
            urgency_level=urgency_level,
            confidence=confidence,
            has_time_pressure=has_time_pressure
        )
        
        return {
            'urgency_level': urgency_level,
            'confidence': confidence,
            'urgency_scores': urgency_scores,
            'has_time_pressure': has_time_pressure
        }
    
    def detect_quality_preferences(self, query: str) -> Dict[str, Any]:
        """
        Detect quality preferences from query text.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with quality preference analysis
        """
        query_lower = query.lower()
        quality_scores = {}
        
        # Check for quality patterns
        for preference, patterns in self.quality_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                quality_scores[preference] = score
        
        # Determine primary quality preference
        if quality_scores:
            primary_preference = max(quality_scores.keys(), key=lambda x: quality_scores[x])
            confidence = min(0.9, quality_scores[primary_preference] * 0.3)
        else:
            primary_preference = 'balanced'  # Default
            confidence = 0.3
        
        # Check for specific quality indicators
        has_quality_focus = any(word in query_lower for word in [
            'quality', 'good', 'great', 'excellent', 'amazing', 'perfect'
        ])
        
        has_quantity_focus = any(word in query_lower for word in [
            'many', 'lots', 'bunch', 'several', 'multiple', 'various'
        ])
        
        self.logger.debug(
            "Quality preferences detected",
            primary_preference=primary_preference,
            confidence=confidence,
            has_quality_focus=has_quality_focus,
            has_quantity_focus=has_quantity_focus
        )
        
        return {
            'primary_preference': primary_preference,
            'confidence': confidence,
            'quality_scores': quality_scores,
            'has_quality_focus': has_quality_focus,
            'has_quantity_focus': has_quantity_focus
        }
    
    def classify_query_type(self, query: str) -> Dict[str, Any]:
        """
        Classify the overall type of query.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with query classification
        """
        # Ensure query is a string
        if isinstance(query, dict):
            query = query.get('query', str(query))
        elif not isinstance(query, str):
            query = str(query)
        
        query_lower = query.lower()
        
        # Query type patterns
        type_patterns = {
            'recommendation': [
                'recommend', 'suggest', 'find', 'show me', 'give me',
                'what should', 'can you', 'help me find'
            ],
            'comparison': [
                'compare', 'difference', 'better', 'worse', 'versus',
                'vs', 'which is', 'what\'s the difference'
            ],
            'information': [
                'what is', 'who is', 'tell me about', 'explain',
                'describe', 'information about'
            ],
            'exploration': [
                'explore', 'discover', 'browse', 'show me more',
                'what else', 'similar', 'related'
            ]
        }
        
        type_scores = {}
        for query_type, patterns in type_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                type_scores[query_type] = score
        
        # Determine primary query type
        if type_scores:
            primary_type = max(type_scores.keys(), key=lambda x: type_scores[x])
            confidence = min(0.9, type_scores[primary_type] * 0.4)
        else:
            primary_type = 'recommendation'  # Default
            confidence = 0.4
        
        # Check for question patterns
        is_question = query.strip().endswith('?') or any(
            query_lower.startswith(word) for word in [
                'what', 'who', 'where', 'when', 'why', 'how',
                'can', 'could', 'would', 'should', 'do', 'does'
            ]
        )
        
        # Check for imperative patterns
        is_imperative = any(
            query_lower.startswith(word) for word in [
                'find', 'show', 'give', 'recommend', 'suggest',
                'play', 'tell', 'help', 'get'
            ]
        )
        
        self.logger.debug(
            "Query type classified",
            primary_type=primary_type,
            confidence=confidence,
            is_question=is_question,
            is_imperative=is_imperative
        )
        
        return {
            'primary_type': primary_type,
            'confidence': confidence,
            'type_scores': type_scores,
            'is_question': is_question,
            'is_imperative': is_imperative,
            'query_structure': 'question' if is_question else ('imperative' if is_imperative else 'statement')
        }
    
    def _count_entities_in_query(self, query: str) -> int:
        """Count approximate number of entities in query."""
        # Simple heuristic: count proper nouns and quoted strings
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', query))
        quoted_strings = len(re.findall(r'["\'][^"\']+["\']', query))
        
        return proper_nouns + quoted_strings
    
    def extract_genre_hints(self, query: str) -> List[str]:
        """
        Extract genre hints from query text with enhanced R&B detection.
        
        Args:
            query: User query text
            
        Returns:
            List of detected genre hints
        """
        query_lower = query.lower()
        genre_hints = []
        
        # 🔧 ENHANCED: R&B specific detection patterns first
        rb_patterns = [
            r'\br&b\b', r'\brnb\b', r'\brhythm\s+and\s+blues\b',
            r'\br\s*&\s*b\b', r'\br\s*n\s*b\b', r'\br\s+and\s+b\b'
        ]
        
        for pattern in rb_patterns:
            if re.search(pattern, query_lower):
                genre_hints.append('r&b')
                self.logger.info(f"🎯 R&B DETECTED: Pattern '{pattern}' found in query")
                break
        
        # Enhanced genre list with R&B variants
        genres = [
            'rock', 'pop', 'electronic', 'jazz', 'classical', 'hip hop', 'hip-hop',
            'country', 'folk', 'metal', 'indie', 'alternative', 'blues',
            'reggae', 'punk', 'funk', 'soul', 'r&b', 'rnb', 'rhythm and blues',
            'techno', 'house', 'ambient', 'experimental', 'world', 'latin', 
            'gospel', 'motown', 'neo-soul', 'contemporary r&b'
        ]
        
        for genre in genres:
            if genre in query_lower and genre not in genre_hints:
                genre_hints.append(genre)
        
        # Genre-related patterns
        genre_patterns = [
            r'\b(\w+)\s+music\b',
            r'\b(\w+)\s+genre\b',
            r'\b(\w+)\s+style\b',
            r'\b(\w+)\s+sound\b',
            r'\b(\w+)\s+tracks?\b',
            r'\b(\w+)\s+songs?\b'
        ]
        
        for pattern in genre_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if len(match) > 2 and match not in genre_hints:
                    # Check if it's a known genre or genre-like term
                    if (match in genres or 
                        any(genre_word in match for genre_word in ['jazz', 'rock', 'pop', 'electronic', 'soul', 'funk']) or
                        match.endswith('y') and len(match) > 4):  # jazzy, rocky, etc.
                        genre_hints.append(match)
        
        # Remove duplicates while preserving order
        genre_hints = list(dict.fromkeys(genre_hints))
        
        self.logger.debug(
            "Genre hints extracted",
            genre_count=len(genre_hints),
            genres=genre_hints
        )
        
        return genre_hints
    
    def create_comprehensive_analysis(self, query: str) -> Dict[str, Any]:
        """
        Create comprehensive analysis combining all query analysis methods.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with comprehensive query analysis
        """
        # Ensure query is a string
        if isinstance(query, dict):
            query = query.get('query', str(query))
        elif not isinstance(query, str):
            query = str(query)
            
        analysis = {
            'original_query': query,
            'query_length': len(query),
            'word_count': len(query.split())
        }
        
        # Run all analysis methods
        analysis['intent_analysis'] = self.analyze_query_intent(query)
        analysis['complexity_analysis'] = self.analyze_query_complexity(query)
        analysis['mood_indicators'] = self.extract_mood_indicators(query)
        analysis['context_factors'] = self.extract_context_factors(query)
        analysis['urgency_analysis'] = self.detect_urgency_level(query)
        analysis['quality_preferences'] = self.detect_quality_preferences(query)
        analysis['query_classification'] = self.classify_query_type(query)
        analysis['genre_hints'] = self.extract_genre_hints(query)
        
        # Create summary
        analysis['summary'] = {
            'primary_intent': analysis['intent_analysis']['primary_intent'],
            'complexity_level': analysis['complexity_analysis']['complexity_level'],
            'urgency_level': analysis['urgency_analysis']['urgency_level'],
            'query_type': analysis['query_classification']['primary_type'],
            'has_mood_context': len(analysis['mood_indicators']) > 0,
            'has_activity_context': len(analysis['context_factors']) > 0,
            'has_genre_preferences': len(analysis['genre_hints']) > 0
        }
        
        self.logger.info(
            "Comprehensive query analysis completed",
            primary_intent=analysis['summary']['primary_intent'],
            complexity=analysis['summary']['complexity_level'],
            query_type=analysis['summary']['query_type'],
            total_indicators=len(analysis['mood_indicators']) + len(analysis['context_factors'])
        )
        
        return analysis 
    
    def detect_hybrid_subtype(self, query: str, entities: Dict[str, Any] = None) -> str:
        """
        Detect the primary intent within hybrid queries.
        
        Args:
            query: User query text
            entities: Extracted entities (optional)
            
        Returns:
            Hybrid sub-type: discovery_primary, similarity_primary, or genre_primary
        """
        import re
        
        query_lower = query.lower()
        
        # Discovery indicators - words that suggest underground/novelty focus
        discovery_terms = [
            'underground', 'new', 'hidden', 'unknown', 'discover', 'find',
            'gems', 'obscure', 'rare', 'experimental', 'unexplored',
            'fresh', 'latest', 'emerging', 'undiscovered'
        ]
        
        # Artist similarity indicators - phrases that suggest artist-based similarity
        similarity_phrases = [
            'like', 'similar', 'sounds like', 'reminds me of', 'in the style of',
            'comparable to', 'along the lines of', 'inspired by'
        ]
        
        # Count discovery indicators
        discovery_score = sum(1 for term in discovery_terms if term in query_lower)
        
        # 🔧 ENHANCED ARTIST SIMILARITY DETECTION
        has_artist_similarity = False
        artist_names = []
        
        # Method 1: Check entities first - handle both formats
        artists_data = None
        if entities:
            # Try nested format first (musical_entities wrapper)
            if entities.get('musical_entities', {}).get('artists', {}).get('primary'):
                artists_data = entities['musical_entities']['artists']['primary']
            # Try direct format (direct entities)
            elif entities.get('artists', {}).get('primary'):
                artists_data = entities['artists']['primary']
                
        if artists_data:
            artist_names = [
                artist.get('name', str(artist)) if isinstance(artist, dict) else str(artist)
                for artist in artists_data
            ]
            
            # Traditional similarity phrases
            has_similarity_phrase = any(phrase in query_lower for phrase in similarity_phrases)
            if has_similarity_phrase:
                has_artist_similarity = True
                self.logger.info("🔧 ARTIST SIMILARITY DETECTED: Found artists %s with similarity phrase in query", artist_names)
            
            # 🔧 NEW: Artist-focused patterns (even without similarity phrases)
            else:
                # Pattern 1: "Artist tracks that are Genre" -> artist-focused
                artist_track_patterns = [
                    r'\b\w+\s+tracks?\s+that\s+are\b',  # "X tracks that are"
                    r'\b\w+\s+songs?\s+that\s+are\b',   # "X songs that are" 
                    r'\b\w+\s+music\s+that\s+is\b',     # "X music that is"
                    r'\bmusic\s+by\s+\w+\s+that\b',     # "music by X that"
                    r'\b\w+\'s\s+\w+\s+tracks?\b',      # "X's jazz tracks"
                    r'\b\w+\'s\s+\w+\s+songs?\b'        # "X's rock songs"
                ]
                
                has_artist_pattern = any(re.search(pattern, query_lower) for pattern in artist_track_patterns)
                
                # Pattern 2: Artist mentioned with genre/style modifiers
                has_genre_mention = False
                if entities:
                    # Try both formats for genres
                    if entities.get('musical_entities', {}).get('genres', {}).get('primary'):
                        has_genre_mention = True
                    elif entities.get('genres', {}).get('primary'):
                        has_genre_mention = True
                
                if has_artist_pattern or has_genre_mention:
                    has_artist_similarity = True
                    self.logger.info("🔧 ARTIST-FOCUSED PATTERN DETECTED: Artists %s with track/genre pattern", artist_names)
        
        # Method 2: Fallback - look for direct artist names with similarity phrases
        if not has_artist_similarity:
            for phrase in similarity_phrases:
                if phrase in query_lower:
                    # Look for likely artist names (capitalized words) in the query
                    # Find all capitalized word sequences (potential artist names)
                    potential_artists = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
                    if potential_artists:
                        has_artist_similarity = True
                        artist_names = potential_artists
                        self.logger.info("🔧 FALLBACK ARTIST SIMILARITY: Found '%s' with '%s' in query", potential_artists, phrase)
                        break
        
        # Check for genre/mood emphasis beyond just artist similarity
        has_genre_mood_focus = self._has_genre_mood_emphasis(query_lower)
        
        # 🔧 PRIORITY LOGIC: Determine primary intent based on strength of indicators
        self.logger.info(
            "🔧 HYBRID DETECTION: discovery_score=%s, has_artist_similarity=%s, artist_names=%s, has_genre_mood_focus=%s",
            discovery_score, has_artist_similarity, artist_names, has_genre_mood_focus
        )
        
        # Discovery-primary: Strong discovery indicators override everything
        if discovery_score >= 2:
            self.logger.info("🔧 DISCOVERY-PRIMARY: %s discovery terms found", discovery_score)
            return 'discovery_primary'
        
        # 🔧 NEW: Similarity-primary: Artist-focused queries (with or without similarity phrases)
        if has_artist_similarity and artist_names:
            # Artist + genre = artist-focused with genre filtering
            if has_genre_mood_focus:
                self.logger.info("🔧 SIMILARITY-PRIMARY: Artist '%s' with genre filtering", artist_names)
                return 'similarity_primary'
            
            # Artist + style modifiers = artist-focused with style variation
            style_modifiers = ['but', 'with', 'and', 'plus', 'mixed with', 'combined with', 'featuring']
            has_style_modifier = any(modifier in query_lower for modifier in style_modifiers)
            
            if has_style_modifier:
                self.logger.info("🔧 SIMILARITY-PRIMARY: Artist '%s' with style modifier", artist_names)
                return 'similarity_primary'
            
            # Pure artist queries also go to similarity
            self.logger.info("🔧 SIMILARITY-PRIMARY: Artist-focused query for '%s'", artist_names)
            return 'similarity_primary'
        
        # Genre-primary: Strong genre/mood focus without clear artist similarity
        if has_genre_mood_focus and not has_artist_similarity:
            self.logger.info("🔧 GENRE-PRIMARY: Strong genre/mood focus without artist similarity")
            return 'genre_primary'
        
        # Discovery-primary: Even single discovery terms can indicate this intent
        if discovery_score >= 1:
            self.logger.info("🔧 DISCOVERY-PRIMARY: %s discovery term found", discovery_score)
            return 'discovery_primary'
        
        # Default fallback based on strongest signal
        if has_artist_similarity:
            self.logger.info("🔧 SIMILARITY-PRIMARY: Default for artist similarity queries")
            return 'similarity_primary'
        elif has_genre_mood_focus:
            self.logger.info("🔧 GENRE-PRIMARY: Default for genre/mood queries")
            return 'genre_primary'
        else:
            self.logger.info("🔧 GENRE-PRIMARY: Default fallback")
            return 'genre_primary'
    
    def _count_artist_mentions(self, query_lower: str) -> int:
        """Count explicit artist mentions in query text."""
        # Simple heuristic - count capitalized words that might be artist names
        words = query_lower.split()
        artist_indicators = ['by', 'from', 'artist']
        count = 0
        
        for i, word in enumerate(words):
            if word in artist_indicators and i + 1 < len(words):
                count += 1
        
        return count
    
    def _has_genre_mood_emphasis(self, query_lower: str) -> bool:
        """Check if query has strong genre or mood emphasis."""
        # Genre indicators
        genre_terms = [
            'jazz', 'jazzy', 'rock', 'electronic', 'indie', 'pop', 'hip-hop', 'rap',
            'ambient', 'classical', 'folk', 'country', 'metal', 'punk', 'reggae',
            'blues', 'soul', 'funk', 'disco', 'house', 'techno', 'dubstep'
        ]
        
        # Mood indicators  
        mood_terms = [
            'chill', 'relaxing', 'upbeat', 'energetic', 'sad', 'happy', 'dark',
            'bright', 'mellow', 'aggressive', 'calm', 'intense', 'smooth', 'rough'
        ]
        
        # Style modifiers that indicate genre/mood focus
        style_terms = [
            'vibes', 'style', 'sound', 'feeling', 'mood', 'atmosphere', 'energy'
        ]
        
        all_terms = genre_terms + mood_terms + style_terms
        found_terms = [term for term in all_terms if term in query_lower]
        
        # Strong emphasis if multiple terms or specific style modifiers
        return len(found_terms) >= 2 or any(term in style_terms for term in found_terms) 