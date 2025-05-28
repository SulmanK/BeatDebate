"""
Query Understanding System for BeatDebate

Production-level query parsing and intent extraction for music recommendation queries.
Handles ambiguous natural language and extracts structured intent for agent coordination.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class QueryIntent(Enum):
    """Primary intent types for music queries."""
    ARTIST_SIMILARITY = "artist_similarity"      # "Music like X"
    GENRE_EXPLORATION = "genre_exploration"      # "I want jazz music"
    MOOD_MATCHING = "mood_matching"              # "Happy music for workout"
    ACTIVITY_CONTEXT = "activity_context"        # "Music for studying"
    DISCOVERY = "discovery"                      # "Something new and different"
    PLAYLIST_BUILDING = "playlist_building"      # "Songs for my road trip"
    SPECIFIC_REQUEST = "specific_request"        # "Play Bohemian Rhapsody"


class SimilarityType(Enum):
    """Types of similarity for artist-based queries."""
    STYLISTIC = "stylistic"        # Similar sound/production
    GENRE = "genre"               # Same genre family
    ERA = "era"                   # Same time period
    MOOD = "mood"                 # Similar emotional feel
    ENERGY = "energy"             # Similar energy level


@dataclass
class QueryUnderstanding:
    """Structured representation of understood query."""
    intent: QueryIntent
    confidence: float
    
    # Core entities
    artists: List[str]
    genres: List[str]
    moods: List[str]
    activities: List[str]
    
    # Intent-specific details
    similarity_type: Optional[SimilarityType] = None
    exploration_level: str = "moderate"  # strict, moderate, broad
    temporal_context: Optional[str] = None
    energy_level: Optional[str] = None
    
    # Agent coordination hints
    primary_agent: str = "genre_mood"  # Which agent should lead
    agent_weights: Dict[str, float] = None
    search_strategy: Dict[str, Any] = None
    
    # Original query context
    original_query: str = ""
    normalized_query: str = ""
    reasoning: str = ""


class QueryUnderstandingEngine:
    """
    Production-level query understanding system using Pure LLM Approach.
    
    Uses LLM for all query understanding with:
    1. Comprehensive prompt engineering for music domain
    2. Artist/genre knowledge base enhancement
    3. Structured JSON output with confidence scoring
    4. Fallback strategies for LLM failures
    """
    
    def __init__(self, llm_client=None):
        """Initialize query understanding engine."""
        if not llm_client:
            raise ValueError("LLM client is required for Pure LLM approach")
            
        self.llm_client = llm_client
        self.logger = logger.bind(component="QueryUnderstanding")
        
        # Initialize knowledge base for enhancement
        self._init_artist_knowledge_base()
        self._init_genre_mappings()
        
        # Initialize LLM system prompt
        self._init_llm_system_prompt()
        
        self.logger.info("Pure LLM Query Understanding Engine initialized")
    
    def _init_llm_system_prompt(self):
        """Initialize comprehensive system prompt for music query understanding."""
        self.system_prompt = """You are an expert music query understanding system for a music recommendation platform. Your job is to analyze user queries and extract structured information to coordinate music recommendation agents.

**Your Task:**
Analyze the user's music request and return a JSON object with detailed understanding.

**Required JSON Structure:**
{
    "intent": "artist_similarity|genre_exploration|mood_matching|activity_context|discovery|playlist_building|specific_request",
    "confidence": 0.0-1.0,
    "artists": ["artist1", "artist2"],
    "genres": ["genre1", "genre2"], 
    "moods": ["mood1", "mood2"],
    "activities": ["activity1"],
    "similarity_type": "stylistic|genre|era|mood|energy",
    "exploration_level": "strict|moderate|broad",
    "temporal_context": "90s|2000s|contemporary|classic|modern",
    "energy_level": "low|medium|high|very_high",
    "reasoning": "detailed explanation of your understanding"
}

**Intent Definitions:**
- **artist_similarity**: User wants music similar to specific artist(s) - "Music like Radiohead", "Similar to Mk.gee"
- **genre_exploration**: User wants to explore a specific genre - "Jazz music", "Some good electronic tracks"
- **mood_matching**: User wants music matching a specific mood - "Happy music", "Sad songs", "Energetic tracks"
- **activity_context**: User wants music for specific activity - "Music for studying", "Workout playlist", "Driving songs"
- **discovery**: User wants to discover new/different music - "Something new", "Surprise me", "Underground gems"
- **playlist_building**: User wants to build a themed playlist - "Road trip playlist", "Party mix"
- **specific_request**: User wants a specific song/album - "Play Bohemian Rhapsody"

**Similarity Types (for artist_similarity intent):**
- **stylistic**: Similar sound, production style, musical approach
- **genre**: Same or related genre family
- **era**: Same time period or musical era
- **mood**: Similar emotional feel or atmosphere
- **energy**: Similar energy level or intensity

**Exploration Levels:**
- **strict**: Stay very close to the request, minimal variation
- **moderate**: Some exploration while staying relevant
- **broad**: Wide exploration, embrace discovery and variety

**Confidence Scoring Guidelines:**
- **0.9-1.0**: Very clear, unambiguous request with specific entities
- **0.7-0.9**: Clear intent with some specific information
- **0.5-0.7**: Somewhat clear but missing some context
- **0.3-0.5**: Ambiguous request requiring interpretation
- **0.1-0.3**: Very unclear or incomplete request

**Examples:**

Query: "Music like Mk.gee"
Response: {
    "intent": "artist_similarity",
    "confidence": 0.85,
    "artists": ["Mk.gee"],
    "genres": [],
    "moods": [],
    "activities": [],
    "similarity_type": "stylistic",
    "exploration_level": "moderate",
    "temporal_context": "contemporary",
    "energy_level": "medium",
    "reasoning": "Clear artist similarity request for Mk.gee, an experimental pop artist. User wants stylistically similar music with moderate exploration."
}

Query: "Happy workout music"
Response: {
    "intent": "mood_matching",
    "confidence": 0.9,
    "artists": [],
    "genres": [],
    "moods": ["happy", "energetic"],
    "activities": ["workout"],
    "similarity_type": null,
    "exploration_level": "moderate",
    "temporal_context": null,
    "energy_level": "high",
    "reasoning": "Clear mood and activity request combining happiness with workout context. High energy music needed for exercise motivation."
}

Query: "Some good jazz for studying"
Response: {
    "intent": "genre_exploration",
    "confidence": 0.85,
    "artists": [],
    "genres": ["jazz"],
    "moods": ["calm", "focused"],
    "activities": ["studying"],
    "similarity_type": null,
    "exploration_level": "moderate",
    "temporal_context": null,
    "energy_level": "low",
    "reasoning": "Genre exploration request for jazz with studying context. Implies need for calm, non-distracting jazz suitable for concentration."
}

Query: "Surprise me with something new"
Response: {
    "intent": "discovery",
    "confidence": 0.8,
    "artists": [],
    "genres": [],
    "moods": [],
    "activities": [],
    "similarity_type": null,
    "exploration_level": "broad",
    "temporal_context": null,
    "energy_level": null,
    "reasoning": "Clear discovery request with emphasis on novelty and surprise. User wants broad exploration of unfamiliar music."
}

Query: "Play that song by Queen"
Response: {
    "intent": "specific_request",
    "confidence": 0.6,
    "artists": ["Queen"],
    "genres": [],
    "moods": [],
    "activities": [],
    "similarity_type": null,
    "exploration_level": "strict",
    "temporal_context": "classic",
    "energy_level": null,
    "reasoning": "Specific song request but lacks song title. User wants a particular Queen song but didn't specify which one."
}

**Important Guidelines:**
1. Always return valid JSON - no extra text before or after
2. Be generous with confidence for clear requests
3. Extract all relevant entities (artists, genres, moods, activities)
4. Consider context clues for energy level and temporal context
5. Provide detailed reasoning for your interpretation
6. Handle typos and variations in artist/genre names
7. Recognize implicit moods from activities (e.g., "workout" implies "energetic")
8. Consider cultural and linguistic variations in music terminology"""

    async def understand_query(self, query: str) -> QueryUnderstanding:
        """
        Main entry point for LLM-based query understanding.
        
        Args:
            query: Raw user query
            
        Returns:
            Structured query understanding
        """
        self.logger.info("Understanding query with LLM", query=query)
        
        try:
            # Step 1: Get LLM understanding
            llm_result = await self._llm_based_understanding(query)
            
            # Step 2: Enhance with knowledge base context
            enhanced_result = self._enhance_with_knowledge_base(llm_result)
            
            # Step 3: Generate agent coordination strategy
            final_result = self._generate_agent_strategy(enhanced_result)
            
            self.logger.info(
                "LLM query understanding completed",
                intent=final_result.intent.value,
                confidence=final_result.confidence,
                artists=final_result.artists,
                primary_agent=final_result.primary_agent
            )
            
            return final_result
            
        except Exception as e:
            self.logger.error("LLM understanding failed", error=str(e), query=query)
            # Fallback to basic understanding
            return self._create_fallback_understanding(query)
    
    async def _llm_based_understanding(self, query: str) -> QueryUnderstanding:
        """Use LLM for comprehensive query understanding."""
        
        user_prompt = f"""Analyze this music query and return the structured JSON response:

Query: "{query}"

Remember to return ONLY the JSON object with no additional text."""

        try:
            # Make LLM request
            response = await self._make_llm_request(user_prompt)
            
            # Parse JSON response
            llm_data = self._parse_llm_response(response)
            
            # Convert to QueryUnderstanding object
            return self._convert_llm_data_to_understanding(llm_data, query)
            
        except Exception as e:
            self.logger.warning("LLM request failed", error=str(e))
            raise e
    
    async def _make_llm_request(self, user_prompt: str) -> str:
        """Make request to LLM with proper error handling."""
        try:
            # Use the system prompt + user prompt
            full_prompt = f"{self.system_prompt}\n\n{user_prompt}"
            response = self.llm_client.generate_content(full_prompt)
            return response.text
        except Exception as e:
            self.logger.error("LLM API call failed", error=str(e))
            raise e
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response with robust JSON extraction."""
        try:
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Remove any markdown code blocks
            if cleaned_text.startswith('```'):
                lines = cleaned_text.split('\n')
                # Remove first and last lines if they're markdown
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].startswith('```'):
                    lines = lines[:-1]
                cleaned_text = '\n'.join(lines)
            
            # Remove any text before the first {
            start_idx = cleaned_text.find('{')
            if start_idx == -1:
                raise ValueError("No JSON object found in response")
            
            # Find matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(cleaned_text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            # Extract JSON string
            json_str = cleaned_text[start_idx:end_idx]
            
            # Additional JSON cleaning for common LLM issues
            json_str = self._clean_json_string(json_str)
            
            # Parse JSON
            llm_data = json.loads(json_str)
            
            self.logger.debug("Successfully parsed LLM response", keys=list(llm_data.keys()))
            return llm_data
            
        except json.JSONDecodeError as e:
            self.logger.error("JSON parsing failed", error=str(e), response_preview=response_text[:200])
            
            # Try alternative parsing approaches
            try:
                # Attempt 1: Try fixing common JSON issues
                fixed_json = self._fix_common_json_issues(response_text)
                return json.loads(fixed_json)
            except:
                pass
            
            try:
                # Attempt 2: Use regex to extract JSON-like structure
                extracted_json = self._extract_json_with_regex(response_text)
                if extracted_json:
                    return json.loads(extracted_json)
            except:
                pass
            
            # If all parsing attempts fail, raise original error
            raise ValueError(f"Invalid JSON in LLM response: {e}")
        except Exception as e:
            self.logger.error("Response parsing failed", error=str(e))
            raise e
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string to fix common LLM formatting issues."""
        # Remove any trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Remove any comments (// or /* */)
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Replace single quotes with double quotes (but be careful with content)
        # This is a simple approach - for more complex cases, we'd need a proper parser
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Keys
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # String values
        
        # Fix common typos in boolean/null values
        json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bnull\b', 'null', json_str, flags=re.IGNORECASE)
        
        return json_str
    
    def _fix_common_json_issues(self, response_text: str) -> str:
        """Attempt to fix common JSON formatting issues."""
        # Find JSON boundaries more aggressively
        start_idx = response_text.find('{')
        if start_idx == -1:
            return response_text
        
        # Extract everything from first { to last }
        end_idx = response_text.rfind('}')
        if end_idx == -1:
            return response_text
        
        json_candidate = response_text[start_idx:end_idx + 1]
        
        # Apply cleaning
        json_candidate = self._clean_json_string(json_candidate)
        
        return json_candidate
    
    def _extract_json_with_regex(self, response_text: str) -> Optional[str]:
        """Extract JSON using regex patterns as a last resort."""
        # Look for JSON-like structure with balanced braces
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if matches:
            # Return the longest match (most likely to be complete)
            longest_match = max(matches, key=len)
            return self._clean_json_string(longest_match)
        
        return None
    
    def _convert_llm_data_to_understanding(self, llm_data: Dict[str, Any], original_query: str) -> QueryUnderstanding:
        """Convert LLM response data to QueryUnderstanding object."""
        try:
            # Extract and validate intent
            intent_str = llm_data.get('intent', 'discovery')
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                self.logger.warning("Invalid intent from LLM", intent=intent_str)
                intent = QueryIntent.DISCOVERY
            
            # Extract similarity type if present
            similarity_type = None
            if llm_data.get('similarity_type'):
                try:
                    similarity_type = SimilarityType(llm_data['similarity_type'])
                except ValueError:
                    self.logger.warning("Invalid similarity_type from LLM", similarity_type=llm_data['similarity_type'])
            
            # Create QueryUnderstanding object
            understanding = QueryUnderstanding(
                intent=intent,
                confidence=float(llm_data.get('confidence', 0.5)),
                artists=llm_data.get('artists', []),
                genres=llm_data.get('genres', []),
                moods=llm_data.get('moods', []),
                activities=llm_data.get('activities', []),
                similarity_type=similarity_type,
                exploration_level=llm_data.get('exploration_level', 'moderate'),
                temporal_context=llm_data.get('temporal_context'),
                energy_level=llm_data.get('energy_level'),
                original_query=original_query,
                normalized_query=original_query.lower().strip(),
                reasoning=llm_data.get('reasoning', 'LLM-based understanding')
            )
            
            return understanding
            
        except Exception as e:
            self.logger.error("Failed to convert LLM data", error=str(e), llm_data=llm_data)
            raise e
    
    def _init_artist_knowledge_base(self):
        """Initialize artist knowledge base for context."""
        self.artist_knowledge = {
            # Experimental/Art Pop
            'mk.gee': {
                'canonical_name': 'Mk.gee',
                'genres': ['experimental pop', 'art pop', 'indie R&B', 'bedroom pop'],
                'characteristics': ['dreamy', 'atmospheric', 'lo-fi', 'ethereal', 'minimal'],
                'similar_artists': ['FKA twigs', 'James Blake', 'Solange', 'Dijon', 'Frank Ocean'],
                'era': 'contemporary',
                'popularity': 'emerging',
                'discovery_strategy': 'underground_experimental'
            },
            'fka twigs': {
                'canonical_name': 'FKA twigs',
                'genres': ['experimental R&B', 'art pop', 'electronic'],
                'characteristics': ['ethereal', 'avant-garde', 'sensual', 'innovative'],
                'similar_artists': ['Mk.gee', 'James Blake', 'Björk', 'Solange'],
                'era': 'contemporary',
                'popularity': 'established',
                'discovery_strategy': 'experimental_mainstream'
            },
            'radiohead': {
                'canonical_name': 'Radiohead',
                'genres': ['alternative rock', 'experimental rock', 'electronic'],
                'characteristics': ['innovative', 'melancholic', 'complex', 'atmospheric'],
                'similar_artists': ['Thom Yorke', 'Atoms for Peace', 'Portishead'],
                'era': '90s-contemporary',
                'popularity': 'mainstream',
                'discovery_strategy': 'alternative_deep_cuts'
            },
            # Add more artists as needed...
        }
    
    def _init_genre_mappings(self):
        """Initialize genre understanding mappings."""
        self.genre_mappings = {
            # Normalize genre variations
            'electronic': ['electronic', 'electronica', 'edm', 'techno', 'house', 'ambient'],
            'indie': ['indie', 'independent', 'indie rock', 'indie pop', 'indie folk'],
            'experimental': ['experimental', 'avant-garde', 'art rock', 'art pop'],
            'hip_hop': ['hip hop', 'rap', 'hip-hop', 'urban', 'trap'],
            'jazz': ['jazz', 'smooth jazz', 'contemporary jazz', 'fusion'],
            'rock': ['rock', 'alternative rock', 'indie rock', 'classic rock'],
            'pop': ['pop', 'pop rock', 'electropop', 'synth-pop'],
            'folk': ['folk', 'acoustic', 'singer-songwriter', 'americana']
        }
    
    def _enhance_with_knowledge_base(self, understanding: QueryUnderstanding) -> QueryUnderstanding:
        """Enhance understanding with artist/genre knowledge base."""
        
        # Enhance artist information
        if understanding.artists:
            for i, artist in enumerate(understanding.artists):
                normalized_artist = artist.lower().replace('.', '').replace(' ', '')
                if normalized_artist in self.artist_knowledge:
                    artist_info = self.artist_knowledge[normalized_artist]
                    
                    # Update canonical name
                    understanding.artists[i] = artist_info['canonical_name']
                    
                    # Add genres if not already present
                    if not understanding.genres:
                        understanding.genres = artist_info['genres'][:2]  # Top 2 genres
                    
                    # Boost confidence for known artists
                    understanding.confidence = min(1.0, understanding.confidence + 0.2)
                    
                    # Update reasoning
                    understanding.reasoning += f" | Enhanced with knowledge of {artist_info['canonical_name']}"
        
        return understanding
    
    def _generate_agent_strategy(self, understanding: QueryUnderstanding) -> QueryUnderstanding:
        """Generate agent coordination strategy based on understanding."""
        
        # Set agent weights based on intent
        if understanding.intent == QueryIntent.ARTIST_SIMILARITY:
            understanding.agent_weights = {
                'discovery': 0.6,    # Primary for similarity exploration
                'genre_mood': 0.3,   # Secondary for genre context
                'judge': 0.1         # Final selection
            }
            understanding.search_strategy = {
                'similarity_depth': 3,
                'underground_bias': 0.4,
                'genre_exploration': True
            }
            
        elif understanding.intent == QueryIntent.GENRE_EXPLORATION:
            understanding.agent_weights = {
                'genre_mood': 0.6,   # Primary for genre matching
                'discovery': 0.3,    # Secondary for variety
                'judge': 0.1         # Final selection
            }
            understanding.search_strategy = {
                'genre_focus': understanding.genres,
                'mood_matching': True,
                'underground_bias': 0.2
            }
            
        elif understanding.intent == QueryIntent.DISCOVERY:
            understanding.agent_weights = {
                'discovery': 0.7,    # Primary for exploration
                'genre_mood': 0.2,   # Minimal genre context
                'judge': 0.1         # Final selection
            }
            understanding.search_strategy = {
                'exploration_breadth': 'wide',
                'underground_bias': 0.8,
                'novelty_priority': True
            }
        
        else:  # Mood/Activity
            understanding.agent_weights = {
                'genre_mood': 0.7,   # Primary for mood matching
                'discovery': 0.2,    # Some variety
                'judge': 0.1         # Final selection
            }
            understanding.search_strategy = {
                'mood_focus': understanding.moods,
                'activity_context': understanding.activities,
                'energy_matching': True
            }
        
        return understanding
    
    def _create_fallback_understanding(self, query: str) -> QueryUnderstanding:
        """Create fallback understanding for unclear queries."""
        
        return QueryUnderstanding(
            intent=QueryIntent.DISCOVERY,
            confidence=0.3,  # Low confidence
            artists=[],
            genres=[],
            moods=[],
            activities=[],
            exploration_level="moderate",
            primary_agent="genre_mood",  # Default to GenreMoodAgent
            original_query=query,
            normalized_query=query,
            reasoning="Unclear query, defaulting to general discovery"
        )


# Integration with PlannerAgent
class EnhancedPlannerAgent:
    """Enhanced PlannerAgent with integrated query understanding."""
    
    def __init__(self, config, gemini_client):
        self.config = config
        self.gemini_client = gemini_client
        self.query_engine = QueryUnderstandingEngine(gemini_client)
        self.logger = logger.bind(component="EnhancedPlannerAgent")
    
    async def process(self, state) -> Dict[str, Any]:
        """Process query with enhanced understanding."""
        
        # Step 1: Understand the query
        query_understanding = await self.query_engine.understand_query(state.user_query)
        
        # Step 2: Generate strategy based on understanding
        strategy = self._generate_strategy_from_understanding(query_understanding)
        
        # Step 3: Update state with enhanced information
        state.planning_strategy = strategy
        state.query_understanding = query_understanding
        state.confidence = query_understanding.confidence
        
        self.logger.info(
            "Enhanced planning completed",
            intent=query_understanding.intent.value,
            confidence=query_understanding.confidence,
            primary_agent=query_understanding.primary_agent,
            strategy_components=len(strategy)
        )
        
        return state
    
    def _generate_strategy_from_understanding(self, understanding: QueryUnderstanding) -> Dict[str, Any]:
        """Generate detailed strategy from query understanding."""
        
        strategy = {
            "task_analysis": {
                "primary_goal": understanding.intent.value,
                "confidence_level": understanding.confidence,
                "complexity_level": "high" if understanding.confidence > 0.8 else "medium",
                "context_factors": understanding.artists + understanding.genres + understanding.moods
            },
            "coordination_strategy": {
                "primary_agent": understanding.primary_agent,
                "agent_weights": understanding.agent_weights or {},
                "execution_order": self._determine_execution_order(understanding)
            },
            "search_strategy": understanding.search_strategy or {},
            "evaluation_framework": {
                "primary_weights": self._generate_evaluation_weights(understanding),
                "quality_thresholds": self._generate_quality_thresholds(understanding),
                "diversity_requirements": self._generate_diversity_requirements(understanding)
            }
        }
        
        return strategy
    
    def _determine_execution_order(self, understanding: QueryUnderstanding) -> List[str]:
        """Determine optimal agent execution order."""
        if understanding.intent == QueryIntent.ARTIST_SIMILARITY:
            return ["discovery", "genre_mood", "judge"]
        elif understanding.intent == QueryIntent.GENRE_EXPLORATION:
            return ["genre_mood", "discovery", "judge"]
        else:
            return ["genre_mood", "discovery", "judge"]  # Default order
    
    def _generate_evaluation_weights(self, understanding: QueryUnderstanding) -> Dict[str, float]:
        """Generate evaluation weights based on understanding."""
        if understanding.intent == QueryIntent.ARTIST_SIMILARITY:
            return {
                "similarity_score": 0.4,
                "quality_score": 0.3,
                "novelty_score": 0.2,
                "diversity_score": 0.1
            }
        elif understanding.intent == QueryIntent.DISCOVERY:
            return {
                "novelty_score": 0.4,
                "quality_score": 0.3,
                "diversity_score": 0.2,
                "similarity_score": 0.1
            }
        else:  # Genre/Mood
            return {
                "genre_mood_fit": 0.4,
                "quality_score": 0.3,
                "diversity_score": 0.2,
                "novelty_score": 0.1
            }
    
    def _generate_quality_thresholds(self, understanding: QueryUnderstanding) -> Dict[str, float]:
        """Generate quality thresholds based on understanding."""
        if understanding.intent == QueryIntent.DISCOVERY:
            return {"min_quality": 0.3, "min_novelty": 0.6}
        else:
            return {"min_quality": 0.5, "min_novelty": 0.3}
    
    def _generate_diversity_requirements(self, understanding: QueryUnderstanding) -> Dict[str, Any]:
        """Generate diversity requirements based on understanding."""
        return {
            "max_same_artist": 2,
            "min_genres": 3 if understanding.intent == QueryIntent.DISCOVERY else 2,
            "temporal_spread": understanding.exploration_level == "broad"
        } 