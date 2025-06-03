"""
Query Understanding Engine for Planner Agent

Moved from root agents directory and simplified to use shared components.
Handles query analysis, entity extraction, and intent classification.
"""

from typing import Dict, Any, Optional
import structlog

from ...models.agent_models import QueryIntent, SimilarityType, QueryUnderstanding

logger = structlog.get_logger(__name__)


class QueryUnderstandingEngine:
    """
    Handles query understanding with both pattern-based and LLM-based analysis.
    
    Uses shared components for entity extraction, query analysis, and LLM interactions.
    """
    
    def __init__(self, llm_client, rate_limiter=None):
        """Initialize query understanding engine with shared components."""
        self.logger = logger
        self.llm_client = llm_client
        self.rate_limiter = rate_limiter
        
        # Build comprehensive system prompt for query understanding
        self.system_prompt = self._build_system_prompt()
        
        # Initialize shared LLM utilities with rate limiter
        try:
            from ..components.llm_utils import LLMUtils
            self.llm_utils = LLMUtils(llm_client, rate_limiter=rate_limiter)
        except ImportError:
            from components.llm_utils import LLMUtils
            self.llm_utils = LLMUtils(llm_client, rate_limiter=rate_limiter)
        
        # Initialize entity extraction utils
        try:
            from ..components.entity_extraction_utils import EntityExtractionUtils
            self.entity_utils = EntityExtractionUtils()
        except ImportError:
            from components.entity_extraction_utils import EntityExtractionUtils
            self.entity_utils = EntityExtractionUtils()
        
        # Initialize query analysis utilities
        try:
            from ..components.query_analysis_utils import QueryAnalysisUtils
            self.query_utils = QueryAnalysisUtils()
        except ImportError:
            from components.query_analysis_utils import QueryAnalysisUtils
            self.query_utils = QueryAnalysisUtils()
        
        self.logger.info("Query Understanding Engine initialized with shared components")
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt for query understanding."""
        return """You are a music query understanding assistant. Analyze user queries about music recommendations and extract structured information.

CRITICAL: Classify queries into these specific intent types based on the design document:

1. BY_ARTIST ("Music by [Artist]", "Give me tracks by [Artist]", "[Artist] songs")
   - Focus on finding tracks BY the specified artist
   - User wants the artist's own discography/tracks

2. ARTIST_SIMILARITY ("Music like [Artist]", "Similar to [Artist]", "Artists that sound like [Artist]")
   - Focus on finding artists/tracks that sound similar to the target artist
   - User wants OTHER artists that are similar, NOT the target artist's own tracks
   - Extract artist names EXACTLY as written (e.g., "Mk.gee", "BROCKHAMPTON", "!!!")

3. DISCOVERY ("Find me underground indie rock", "Something new and different")
   - Focus on discovering truly new/unknown music
   - Emphasis on novelty and underground tracks

4. GENRE_MOOD ("Upbeat electronic music", "Sad indie songs")
   - Focus on specific vibes, genres, or moods
   - No specific artist reference, just style/feel

5. CONTEXTUAL ("Music for studying", "Workout playlist", "Road trip songs")
   - Focus on functional music for specific activities
   - Context-driven recommendations

6. HYBRID ("Chill songs like Bon Iver", "Upbeat music similar to Daft Punk")
   - Combines artist similarity with mood/genre requirements
   - Both artist reference AND style/context requirements

Return a JSON object with this exact structure:
{
    "intent": "by_artist|artist_similarity|discovery|genre_mood|contextual|hybrid",
    "musical_entities": {
        "artists": ["artist1", "artist2"],
        "genres": ["genre1", "genre2"], 
        "tracks": ["track1", "track2"],
        "moods": ["mood1", "mood2"]
    },
    "context_factors": ["context1", "context2"],
    "complexity_level": "simple|medium|complex",
    "similarity_type": "light|moderate|strong",
    "confidence": 0.8
}

EXAMPLES:
- "Music by Mk.gee" → intent: "by_artist", artists: ["Mk.gee"]
- "Give me tracks by Radiohead" → intent: "by_artist", artists: ["Radiohead"]
- "Mk.gee songs" → intent: "by_artist", artists: ["Mk.gee"]
- "Music like Mk.gee" → intent: "artist_similarity", artists: ["Mk.gee"]
- "Artists similar to Radiohead" → intent: "artist_similarity", artists: ["Radiohead"]
- "Find underground electronic music" → intent: "discovery", genres: ["electronic"]
- "Happy music for working out" → intent: "contextual", moods: ["happy"], context_factors: ["workout"]
- "Chill songs like Bon Iver" → intent: "hybrid", artists: ["Bon Iver"], moods: ["chill"]
- "Upbeat electronic music" → intent: "genre_mood", genres: ["electronic"], moods: ["upbeat"]

Be specific about genres and extract moods from emotional language."""
    
    async def understand_query(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> QueryUnderstanding:
        """
        Understand user query using hybrid approach with shared components.
        
        Args:
            query: User's music query
            conversation_context: Optional conversation context
            
        Returns:
            QueryUnderstanding object with extracted information
        """
        self.logger.info("Starting query understanding", query_length=len(query))
        
        try:
            # Phase 1: Pattern-based analysis using shared utilities for fallback
            pattern_analysis = self._pattern_based_analysis(query)
            
            # Phase 2: LLM-based understanding for ALL queries (not just complex)
            # LLM is much better at entity extraction, especially for artist names
            try:
                llm_analysis = await self._llm_based_understanding(query)
                # Merge pattern and LLM analysis, prioritizing LLM for entities
                final_analysis = self._merge_analyses(pattern_analysis, llm_analysis, prioritize_llm_entities=True)
            except Exception as e:
                self.logger.warning("LLM understanding failed, using pattern analysis only", error=str(e))
                final_analysis = pattern_analysis
            
            # Phase 3: Validate and enhance with shared utilities
            final_analysis = self.entity_utils.validate_and_enhance_entities(
                final_analysis, query
            )
            
            # Convert to QueryUnderstanding object
            understanding = self._convert_to_understanding(final_analysis, query)
            
            self.logger.info(
                "Query understanding completed",
                intent=understanding.intent.value,
                confidence=understanding.confidence,
                entity_count=len(understanding.artists)
            )
            
            return understanding
            
        except Exception as e:
            self.logger.error("Query understanding failed", error=str(e))
            # Return fallback understanding
            return self._create_fallback_understanding(query)
    
    def _pattern_based_analysis(self, query: str) -> Dict[str, Any]:
        """Use shared utilities for pattern-based analysis."""
        # Use shared query analysis utilities
        comprehensive_analysis = self.query_utils.create_comprehensive_analysis(query)
        
        # Extract entities using shared utilities - fix the entity extraction
        try:
            entities = self.entity_utils.validate_and_enhance_entities({}, query)
        except Exception as e:
            self.logger.warning("Entity extraction failed, using fallback", error=str(e))
            entities = {"musical_entities": {"artists": {"primary": []}, "genres": {"primary": []}, "tracks": {"primary": []}, "moods": {"primary": []}}}
        
        # Extract similarity indicators
        try:
            similarity_info = self.entity_utils.extract_similarity_indicators(query)
        except Exception as e:
            self.logger.warning("Similarity extraction failed, using fallback", error=str(e))
            similarity_info = {"similarity_type": None}
        
        # Combine all analyses
        pattern_analysis = {
            'intent': comprehensive_analysis['intent_analysis']['primary_intent'],
            'similarity_type': similarity_info.get('similarity_type'),
            'musical_entities': entities.get('musical_entities', {}),
            'context_factors': comprehensive_analysis['context_factors'],
            'complexity_level': comprehensive_analysis['complexity_analysis']['complexity_level'],
            'confidence': 0.7,  # Base confidence for pattern matching
            'mood_indicators': comprehensive_analysis['mood_indicators'],
            'genre_hints': comprehensive_analysis['genre_hints']
        }
        
        # 🔧 SET FLAG: Track if pattern analysis detected hybrid intent
        self._pattern_detected_hybrid = (comprehensive_analysis['intent_analysis']['primary_intent'] == 'hybrid')
        if self._pattern_detected_hybrid:
            self.logger.info(f"🔧 FLAG SET: Pattern analysis detected hybrid intent for query: '{query}'")
        
        return pattern_analysis
    
    async def _llm_based_understanding(self, query: str) -> Dict[str, Any]:
        """Use shared LLM utilities for comprehensive understanding."""
        user_prompt = f"""Analyze this music query and return the structured JSON response:

Query: "{query}"

Remember to return ONLY the JSON object with no additional text."""
        
        try:
            # Use shared LLM utilities with JSON parsing
            llm_data = await self.llm_utils.call_llm_with_json_response(
                user_prompt=user_prompt,
                system_prompt=self.system_prompt,
                max_retries=2
            )
            
            # 🔧 DEBUG: Log what LLM actually returned
            self.logger.info(f"🔧 LLM RAW RESPONSE: {llm_data}")
            
            # Validate JSON structure using shared utilities
            required_keys = ['intent', 'musical_entities', 'context_factors', 'complexity_level']
            optional_keys = ['similarity_type', 'confidence']
            
            validated_data = self.llm_utils.validate_json_structure(
                llm_data, required_keys, optional_keys
            )
            
            # 🔧 DEBUG: Log validated data  
            self.logger.info(f"🔧 LLM VALIDATED RESPONSE: {validated_data}")
            
            return validated_data
            
        except Exception as e:
            self.logger.warning("LLM understanding failed", error=str(e))
            raise e
    
    def _merge_analyses(
        self, pattern_analysis: Dict[str, Any], llm_analysis: Dict[str, Any], prioritize_llm_entities: bool = False
    ) -> Dict[str, Any]:
        """Merge pattern-based and LLM-based analyses."""
        merged = pattern_analysis.copy()
        
        # 🔧 DEBUG: Log what we're merging
        self.logger.info(f"🔧 MERGE DEBUG: prioritize_llm_entities={prioritize_llm_entities}")
        self.logger.info(f"🔧 PATTERN ENTITIES: {pattern_analysis.get('musical_entities', {})}")
        self.logger.info(f"🔧 LLM ENTITIES: {llm_analysis.get('musical_entities', {})}")
        
        # Use LLM intent if it has higher confidence
        llm_confidence = llm_analysis.get('confidence', 0.5)
        if llm_confidence > merged.get('confidence', 0.0):
            merged['intent'] = llm_analysis.get('intent', merged['intent'])
            merged['confidence'] = llm_confidence
        
        # Merge musical entities
        llm_entities = llm_analysis.get('musical_entities', {})
        pattern_entities = merged.get('musical_entities', {})
        
        if prioritize_llm_entities:
            # 🔧 FIX: When prioritizing LLM entities, convert them to the expected structure
            converted_entities = {}
            for entity_type in ['artists', 'genres', 'tracks', 'moods']:
                if entity_type in llm_entities:
                    llm_data = llm_entities[entity_type]
                    if isinstance(llm_data, list):
                        # Convert simple list to structured format
                        converted_entities[entity_type] = {
                            'primary': [{'name': item, 'confidence': 0.9} if isinstance(item, str) else item 
                                        for item in llm_data],
                            'secondary': [],
                            'similar_to': []
                        }
                    elif isinstance(llm_data, dict):
                        # Already in structured format
                        converted_entities[entity_type] = llm_data
                    else:
                        # Fallback for other types
                        converted_entities[entity_type] = {
                            'primary': [{'name': str(llm_data), 'confidence': 0.9}],
                            'secondary': [],
                            'similar_to': []
                        }
                else:
                    # Keep existing pattern entities for this type
                    if entity_type in pattern_entities:
                        converted_entities[entity_type] = pattern_entities[entity_type]
            
            merged['musical_entities'] = converted_entities
            self.logger.info(f"🔧 CONVERTED LLM entities to structured format: {converted_entities}")
        else:
            # Only combine when NOT prioritizing LLM entities
            for entity_type in ['artists', 'genres', 'tracks', 'moods']:
                if entity_type in llm_entities:
                    if entity_type not in pattern_entities:
                        pattern_entities[entity_type] = {'primary': [], 'secondary': [], 'similar_to': []}
                    
                    llm_data = llm_entities[entity_type]
                    if isinstance(llm_data, list):
                        # Convert simple list items to structured format
                        for item in llm_data:
                            structured_item = {'name': item, 'confidence': 0.9} if isinstance(item, str) else item
                            pattern_entities[entity_type]['primary'].append(structured_item)
                    elif isinstance(llm_data, dict):
                        # Merge structured data
                        for category in ['primary', 'secondary', 'similar_to']:
                            if category in llm_data:
                                existing = pattern_entities[entity_type].get(category, [])
                                new_items = llm_data[category]
                                combined = existing + new_items
                                pattern_entities[entity_type][category] = combined
            merged['musical_entities'] = pattern_entities
        
        # Merge context factors
        llm_context = llm_analysis.get('context_factors', [])
        pattern_context = merged.get('context_factors', [])
        merged['context_factors'] = list(dict.fromkeys(pattern_context + llm_context))
        
        # Use LLM similarity type if available
        if llm_analysis.get('similarity_type'):
            merged['similarity_type'] = llm_analysis['similarity_type']
        
        return merged
    
    def _convert_to_understanding(
        self, analysis: Dict[str, Any], original_query: str
    ) -> QueryUnderstanding:
        """Convert analysis to QueryUnderstanding object."""
        try:
            # Ensure original_query is a string
            if isinstance(original_query, dict):
                original_query = original_query.get('query', str(original_query))
            elif not isinstance(original_query, str):
                original_query = str(original_query)
            
            # Extract and validate intent
            intent_str = analysis.get('intent', 'discovery')
            try:
                # Map common intent values to valid enum values
                intent_mapping = {
                    'by_artist': 'by_artist',
                    'discovery': 'discovery',
                    'similarity': 'artist_similarity',
                    'artist_similarity': 'artist_similarity',
                    'mood_based': 'genre_mood',
                    'activity_based': 'contextual',
                    'genre_specific': 'genre_mood',
                    'contextual': 'contextual',  # 🔧 FIX: Add missing contextual mapping
                    'hybrid': 'hybrid'  # ✅ FIXED: Use lowercase to match enum value
                }
                
                # 🔧 FIX: Override LLM intent if pattern analysis detected hybrid
                if hasattr(self, '_pattern_detected_hybrid') and self._pattern_detected_hybrid:
                    self.logger.info(f"🔧 OVERRIDE: Pattern analysis detected hybrid, overriding LLM intent '{intent_str}' -> 'hybrid'")
                    intent_str = 'hybrid'
                
                mapped_intent = intent_mapping.get(intent_str.lower(), 'discovery')  # 🔧 FIX: Fallback to 'discovery' not 'DISCOVERY'
                self.logger.debug(f"🔧 INTENT MAPPING: '{intent_str}' -> '{mapped_intent}'")
                intent = QueryIntent(mapped_intent)
                self.logger.debug(f"🔧 INTENT CREATED: {intent} (value: {intent.value})")
            except ValueError as e:
                self.logger.warning(f"Invalid intent: {intent_str}, error: {e}")
                intent = QueryIntent.DISCOVERY
            
            # Extract similarity type if present
            similarity_type = None
            if analysis.get('similarity_type'):
                try:
                    # Map similarity types to valid enum values
                    similarity_mapping = {
                        'exact': 'STYLISTIC',
                        'moderate': 'STYLISTIC',  # ✅ FIXED! Artist similarity should be stylistic
                        'loose': 'MOOD'
                    }
                    similarity_str = analysis.get('similarity_type')
                    mapped_similarity = similarity_mapping.get(similarity_str.lower(), None)
                    if mapped_similarity:
                        similarity_type = SimilarityType(mapped_similarity)
                except ValueError:
                    self.logger.warning("Invalid similarity_type", similarity_type=analysis['similarity_type'])
            
            # Extract musical entities and convert to separate lists
            musical_entities = analysis.get('musical_entities', {})
            
            # Helper function to extract names from entity lists
            def extract_names(entity_list):
                """Extract names from entity list that may contain dicts or strings."""
                names = []
                if isinstance(entity_list, list):
                    for item in entity_list:
                        if isinstance(item, dict):
                            # Handle confidence score format: {'name': 'Artist', 'confidence': 0.8}
                            names.append(item.get('name', str(item)))
                        elif isinstance(item, str):
                            names.append(item)
                        else:
                            names.append(str(item))
                return names
            
            # Extract artists from musical entities
            artists = []
            if 'artists' in musical_entities:
                artists_data = musical_entities['artists']
                if isinstance(artists_data, dict):
                    artists.extend(extract_names(artists_data.get('primary', [])))
                    artists.extend(extract_names(artists_data.get('similar_to', [])))
                elif isinstance(artists_data, list):
                    artists.extend(extract_names(artists_data))
            
            # ✅ FORCE ARTIST_SIMILARITY intent when artists found with similarity indicators
            # BUT NOT for hybrid queries that have additional genre/mood constraints
            genres_found = []
            if 'genres' in musical_entities:
                genres_data = musical_entities['genres']
                if isinstance(genres_data, dict):
                    genres_found.extend(extract_names(genres_data.get('primary', [])))
                    genres_found.extend(extract_names(genres_data.get('secondary', [])))
                elif isinstance(genres_data, list):
                    genres_found.extend(extract_names(genres_data))
            
            # Check for mood constraints too
            moods_found = []
            if 'moods' in musical_entities:
                moods_data = musical_entities['moods']
                if isinstance(moods_data, dict):
                    moods_found.extend(extract_names(moods_data.get('primary', [])))
                    moods_found.extend(extract_names(moods_data.get('secondary', [])))
                    moods_found.extend(extract_names(moods_data.get('energy', [])))
                    moods_found.extend(extract_names(moods_data.get('emotion', [])))
                elif isinstance(moods_data, list):
                    moods_found.extend(extract_names(moods_data))
            
            # FIXED: Only consider it a constraint if there are ACTUAL genres or moods
            has_genre_mood_constraints = bool(genres_found) or bool(moods_found)
            
            if (artists and 
                any(phrase in original_query.lower() for phrase in ['like', 'similar to', 'sounds like', 'reminds me of']) and
                not has_genre_mood_constraints and  # 🎯 NEW: Don't override hybrid queries with constraints
                intent != QueryIntent.HYBRID):  # 🎯 NEW: Don't override correctly detected hybrid intent
                
                intent = QueryIntent.ARTIST_SIMILARITY
                # Set default similarity type for artist similarity if not already set
                if similarity_type is None:
                    similarity_type = SimilarityType.STYLISTIC
                self.logger.info("Detected pure artist similarity query, forcing ARTIST_SIMILARITY intent", artists=artists)
            elif (artists and has_genre_mood_constraints and
                  any(phrase in original_query.lower() for phrase in ['like', 'similar to', 'sounds like', 'reminds me of'])):
                # Keep as hybrid for queries like "Music like X but Y"
                self.logger.info(f"🎯 HYBRID query detected: artist similarity + constraints (genres: {genres_found}, moods: {musical_entities.get('moods', [])})")
                if intent != QueryIntent.HYBRID:
                    intent = QueryIntent.HYBRID
                    self.logger.info("🔧 Converted to HYBRID intent due to genre/mood constraints")
            
            # 🔧 NEW: Detect hybrid sub-types for better scoring
            hybrid_subtype = None
            if intent == QueryIntent.HYBRID:
                hybrid_subtype = self.query_utils.detect_hybrid_subtype(
                    original_query, 
                    analysis.get('musical_entities', {})
                )
                self.logger.info(f"🔧 HYBRID SUB-TYPE DETECTED: {hybrid_subtype} for query: '{original_query}'")
            
            # Extract genres from musical entities
            genres = []
            if 'genres' in musical_entities:
                genres_data = musical_entities['genres']
                if isinstance(genres_data, dict):
                    genres.extend(extract_names(genres_data.get('primary', [])))
                    genres.extend(extract_names(genres_data.get('secondary', [])))
                elif isinstance(genres_data, list):
                    genres.extend(extract_names(genres_data))
            
            # Extract moods from musical entities
            moods = []
            if 'moods' in musical_entities:
                moods_data = musical_entities['moods']
                if isinstance(moods_data, dict):
                    moods.extend(extract_names(moods_data.get('primary', [])))
                    # Also check other mood categories
                    moods.extend(extract_names(moods_data.get('energy', [])))
                    moods.extend(extract_names(moods_data.get('emotion', [])))
                elif isinstance(moods_data, list):
                    moods.extend(extract_names(moods_data))
            
            # Extract activities (if any)
            activities = []
            if 'activities' in musical_entities:
                activities_data = musical_entities['activities']
                if isinstance(activities_data, dict):
                    activities.extend(extract_names(activities_data.get('primary', [])))
                    activities.extend(extract_names(activities_data.get('physical', [])))
                    activities.extend(extract_names(activities_data.get('mental', [])))
                elif isinstance(activities_data, list):
                    activities.extend(extract_names(activities_data))
            
            # Also check contextual entities for activities
            contextual_entities = analysis.get('contextual_entities', {})
            if 'activities' in contextual_entities:
                activities_data = contextual_entities['activities']
                if isinstance(activities_data, dict):
                    activities.extend(extract_names(activities_data.get('physical', [])))
                    activities.extend(extract_names(activities_data.get('mental', [])))
                elif isinstance(activities_data, list):
                    activities.extend(extract_names(activities_data))
            
            # Extract confidence
            confidence = analysis.get('confidence', 0.5)
            if isinstance(confidence, dict):
                confidence = confidence.get('overall', 0.5)
            
            # Create QueryUnderstanding object with correct parameters
            understanding = QueryUnderstanding(
                intent=intent,
                confidence=confidence,
                artists=artists,
                genres=genres,
                moods=moods,
                activities=activities,
                similarity_type=similarity_type,
                original_query=original_query,
                normalized_query=original_query.lower().strip(),
                reasoning=(f"Analysis completed with {confidence:.1%} confidence" + 
                           (f" | Hybrid sub-type: {genres_found}" if genres_found else "") +
                           (f" | Mood constraints: {has_genre_mood_constraints}" if has_genre_mood_constraints else ""))
            )
            
            return understanding
            
        except Exception as e:
            self.logger.error("Failed to convert analysis to understanding", error=str(e))
            return self._create_fallback_understanding(original_query)
    
    def _create_fallback_understanding(self, query: str) -> QueryUnderstanding:
        """Create fallback understanding when analysis fails."""
        # Ensure query is a string
        if isinstance(query, dict):
            query = query.get('query', str(query))
        elif not isinstance(query, str):
            query = str(query)
        
        return QueryUnderstanding(
            intent=QueryIntent.DISCOVERY,
            confidence=0.3,
            artists=[],
            genres=[],
            moods=[],
            activities=[],
            similarity_type=None,
            original_query=query,
            normalized_query=query.lower(),
            reasoning="Fallback understanding due to processing error"
        )
    
    def get_understanding_summary(self, understanding: QueryUnderstanding) -> Dict[str, Any]:
        """Get summary of understanding for logging/debugging."""
        return {
            "intent": understanding.intent.value,
            "similarity_type": understanding.similarity_type.value if understanding.similarity_type else None,
            "confidence": understanding.confidence,
            "complexity": getattr(understanding, 'complexity_level', 'unknown'),
            "has_artists": bool(understanding.artists),
            "has_genres": bool(understanding.genres),
            "has_moods": bool(understanding.moods),
            "has_activities": bool(understanding.activities)
        } 