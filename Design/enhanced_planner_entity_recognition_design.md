# Enhanced PlannerAgent Entity Recognition System Design

## Problem Statement

The current BeatDebate system has fragmented query understanding with basic entity recognition scattered across agents. We need to centralize and enhance entity recognition in the PlannerAgent to create a single source of truth for query understanding that supports:

1. **Comprehensive Entity Types**: Artists, albums, decades, activities, and more
2. **LLM-Based Recognition**: Sophisticated semantic understanding beyond regex patterns
3. **Complex Query Handling**: Multi-faceted requests with contextual relationships
4. **Conversation Context**: Session-aware understanding of previous recommendations and user preferences

## Design Goals

### Primary Objectives
- **Centralized Authority**: PlannerAgent as the sole query understanding component
- **Rich Entity Extraction**: Support for diverse entity types and relationships
- **Context Awareness**: Maintain conversation history and user preference evolution
- **Intelligent Coordination**: Use entity understanding to create smarter agent strategies
- **Extensibility**: Easy addition of new entity types and recognition patterns

### Success Criteria
- All agents receive consistent, pre-processed entity information
- Complex queries are decomposed into actionable agent strategies
- Conversation context influences recommendation strategies
- Entity recognition accuracy improves through LLM integration
- System maintains backward compatibility during migration

## Enhanced Entity Recognition Framework

### Core Entity Types

#### 1. **Musical Entities**
```python
{
    "artists": {
        "primary": ["The Beatles", "Radiohead"],
        "similar_to": ["Pink Floyd"],  # "like Pink Floyd"
        "avoid": ["Taylor Swift"],     # "but not Taylor Swift"
        "era_context": "1960s"         # "60s Beatles"
    },
    "tracks": {
        "specific": ["Bohemian Rhapsody", "Stairway to Heaven"],
        "referenced": ["the last song", "that track you played"],
        "style_reference": ["something like Yesterday"]
    },
    "albums": {
        "specific": ["Abbey Road", "OK Computer"],
        "era": ["their early albums", "latest release"],
        "type": ["concept albums", "live recordings"]
    },
    "genres": {
        "primary": ["rock", "jazz", "electronic"],
        "sub_genres": ["progressive rock", "bebop jazz"],
        "fusion": ["jazz-rock", "electro-swing"],
        "avoid": ["country"]
    }
}
```

#### 2. **Contextual Entities**
```python
{
    "moods": {
        "energy": ["high", "medium", "low", "chill", "energetic"],
        "emotion": ["happy", "melancholic", "nostalgic", "angry"],
        "atmosphere": ["dark", "bright", "mysterious", "uplifting"]
    },
    "activities": {
        "physical": ["workout", "running", "yoga", "dancing"],
        "mental": ["studying", "reading", "coding", "meditating"],
        "social": ["party", "dinner", "road trip", "date night"],
        "temporal": ["morning", "evening", "late night", "commute"]
    },
    "temporal": {
        "decades": ["60s", "70s", "80s", "90s", "2000s"],
        "eras": ["classic rock era", "golden age of hip hop"],
        "periods": ["early career", "latest work", "peak period"]
    }
}
```

#### 3. **Preference Entities**
```python
{
    "similarity_requests": {
        "type": "artist_similarity",
        "target": "The Beatles",
        "relationship": "similar_to",
        "intensity": "somewhat",  # "somewhat like", "exactly like"
        "aspects": ["style", "energy", "instrumentation"]
    },
    "discovery_preferences": {
        "novelty": ["underground", "mainstream", "hidden gems"],
        "familiarity": ["new discoveries", "familiar artists"],
        "popularity": ["popular", "obscure", "trending"]
    },
    "quality_preferences": {
        "audio_quality": ["high quality", "lo-fi", "studio"],
        "production": ["well-produced", "raw", "polished"]
    }
}
```

#### 4. **Conversation Context Entities**
```python
{
    "session_references": {
        "previous_tracks": ["the last song", "that track", "the first one"],
        "previous_artists": ["that artist", "the band you mentioned"],
        "previous_recommendations": ["like before", "similar to earlier"]
    },
    "preference_evolution": {
        "liked": ["I loved that", "more like this"],
        "disliked": ["not that style", "too heavy"],
        "adjustments": ["but jazzier", "more upbeat", "less electronic"]
    },
    "conversation_flow": {
        "continuation": ["also", "and", "plus"],
        "contrast": ["but", "however", "instead"],
        "refinement": ["more specifically", "actually", "I mean"]
    }
}
```

## LLM-Based Entity Recognition System

### Architecture Overview

```python
class EnhancedEntityRecognizer:
    """
    LLM-powered entity recognition with fallback mechanisms.
    """
    
    def __init__(self, gemini_client, fallback_patterns):
        self.llm_client = gemini_client
        self.fallback_patterns = fallback_patterns
        self.entity_cache = {}
        
    async def extract_entities(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract comprehensive entities using LLM with context awareness.
        """
```

### LLM Prompt Engineering

#### System Prompt Template
```python
ENTITY_EXTRACTION_SYSTEM_PROMPT = """
You are an expert music entity recognition system. Extract and categorize all musical and contextual entities from user queries.

ENTITY CATEGORIES:
1. Musical: artists, tracks, albums, genres, sub-genres
2. Contextual: moods, activities, temporal references
3. Preferences: similarity requests, discovery preferences, quality preferences
4. Conversational: session references, preference evolution, conversation flow

CONTEXT AWARENESS:
- Consider previous recommendations and user feedback
- Understand relative references ("like the last song")
- Detect preference evolution ("but jazzier", "more upbeat")

OUTPUT FORMAT: Structured JSON with confidence scores and relationships.
"""
```

#### Query Analysis Prompt
```python
async def _create_entity_extraction_prompt(
    self, 
    query: str, 
    context: Optional[Dict] = None
) -> str:
    """Create context-aware entity extraction prompt."""
    
    base_prompt = f"""
    QUERY: "{query}"
    
    CONVERSATION CONTEXT:
    {self._format_conversation_context(context)}
    
    Extract entities in this JSON format:
    {{
        "musical_entities": {{
            "artists": {{"primary": [], "similar_to": [], "avoid": []}},
            "tracks": {{"specific": [], "referenced": [], "style_reference": []}},
            "albums": {{"specific": [], "era": [], "type": []}},
            "genres": {{"primary": [], "sub_genres": [], "fusion": [], "avoid": []}}
        }},
        "contextual_entities": {{
            "moods": {{"energy": [], "emotion": [], "atmosphere": []}},
            "activities": {{"physical": [], "mental": [], "social": [], "temporal": []}},
            "temporal": {{"decades": [], "eras": [], "periods": []}}
        }},
        "preference_entities": {{
            "similarity_requests": [],
            "discovery_preferences": [],
            "quality_preferences": []
        }},
        "conversation_entities": {{
            "session_references": [],
            "preference_evolution": [],
            "conversation_flow": []
        }},
        "confidence_scores": {{
            "overall": 0.0-1.0,
            "entity_specific": {{"entity_name": 0.0-1.0}}
        }},
        "relationships": [
            {{"source": "entity1", "target": "entity2", "relationship": "similar_to"}}
        ]
    }}
    """
    return base_prompt
```

### Fallback Mechanisms

```python
class FallbackEntityExtractor:
    """
    Regex-based fallback for when LLM extraction fails.
    """
    
    def __init__(self):
        self.patterns = {
            "artist_similarity": [
                r"(?i)(?:like|similar to)\s+([A-Z][a-zA-Z0-9\s&]+)",
                r"(?i)sounds?\s+like\s+([A-Z][a-zA-Z0-9\s&]+)",
                r"(?i)reminds?\s+me\s+of\s+([A-Z][a-zA-Z0-9\s&]+)"
            ],
            "genres": [
                r"(?i)(rock|jazz|pop|hip.hop|electronic|classical|country|blues|folk|metal)",
                r"(?i)(indie|alternative|progressive|experimental|ambient)"
            ],
            "decades": [
                r"(?i)(60s|70s|80s|90s|2000s|2010s)",
                r"(?i)(sixties|seventies|eighties|nineties)"
            ],
            "activities": [
                r"(?i)(workout|exercise|running|studying|party|driving|cooking)",
                r"(?i)(gym|work|sleep|relax|focus)"
            ]
        }
```

## Complex Query Handling

### Query Decomposition Strategy

#### Example: "Beatles-style but for working out"
```python
{
    "query_complexity": "multi_faceted",
    "primary_intent": "artist_similarity_with_activity_context",
    "decomposition": {
        "base_similarity": {
            "target_artist": "The Beatles",
            "similarity_aspects": ["style", "instrumentation", "songwriting"]
        },
        "contextual_modification": {
            "activity": "workout",
            "required_adjustments": ["higher_energy", "stronger_beat", "motivational"]
        },
        "coordination_strategy": {
            "discovery_agent": {
                "focus": "Beatles-influenced artists",
                "filter": "high_energy_tracks",
                "underground_bias": 0.3
            },
            "genre_mood_agent": {
                "base_genres": ["rock", "pop_rock"],
                "energy_level": "high",
                "activity_context": "workout"
            }
        }
    }
}
```

#### Example: "More like the last song but jazzier"
```python
{
    "query_complexity": "conversational_refinement",
    "primary_intent": "session_reference_with_style_modification",
    "decomposition": {
        "session_reference": {
            "target": "last_recommended_track",
            "track_id": "previous_session_track_1",
            "base_attributes": ["tempo", "mood", "instrumentation"]
        },
        "style_modification": {
            "target_genre": "jazz",
            "modification_type": "genre_fusion",
            "intensity": "moderate"
        },
        "coordination_strategy": {
            "discovery_agent": {
                "seed_track": "previous_session_track_1",
                "style_direction": "jazz_influenced",
                "similarity_weight": 0.7
            },
            "genre_mood_agent": {
                "base_track_analysis": "previous_session_track_1",
                "genre_shift": "towards_jazz",
                "preserve_attributes": ["tempo", "energy"]
            }
        }
    }
}
```

## Conversation Context Management

### Session State Architecture

```python
class ConversationContextManager:
    """
    Manages conversation history and user preference evolution.
    """
    
    def __init__(self):
        self.session_store = {}
        
    async def update_session_context(
        self, 
        session_id: str, 
        query: str,
        entities: Dict[str, Any],
        recommendations: List[Dict],
        user_feedback: Optional[Dict] = None
    ):
        """Update session with new interaction data."""
        
        if session_id not in self.session_store:
            self.session_store[session_id] = {
                "interaction_history": [],
                "preference_profile": {},
                "recommendation_history": [],
                "entity_evolution": {}
            }
            
        session = self.session_store[session_id]
        
        # Add interaction
        interaction = {
            "timestamp": datetime.now(),
            "query": query,
            "extracted_entities": entities,
            "recommendations": recommendations,
            "user_feedback": user_feedback
        }
        session["interaction_history"].append(interaction)
        
        # Update preference profile
        await self._update_preference_profile(session, entities, user_feedback)
        
        # Track entity evolution
        await self._track_entity_evolution(session, entities)
```

### Preference Evolution Tracking

```python
async def _analyze_preference_evolution(
    self, 
    session_history: List[Dict]
) -> Dict[str, Any]:
    """
    Analyze how user preferences have evolved during the session.
    """
    
    evolution_patterns = {
        "genre_drift": [],      # How genre preferences changed
        "energy_adjustment": [], # Energy level modifications
        "discovery_tolerance": [], # Openness to new music
        "artist_affinity": [],   # Artist preference patterns
        "activity_correlation": [] # Activity-music correlations
    }
    
    for interaction in session_history:
        # Analyze feedback patterns
        if interaction.get("user_feedback"):
            await self._extract_preference_signals(
                interaction, evolution_patterns
            )
            
        # Analyze query modifications
        await self._analyze_query_refinements(
            interaction, evolution_patterns
        )
    
    return {
        "preference_trends": evolution_patterns,
        "confidence_scores": self._calculate_evolution_confidence(evolution_patterns),
        "recommendations": self._generate_preference_recommendations(evolution_patterns)
    }
```

## Enhanced Agent Coordination

### Strategy Generation with Entity Context

```python
async def _create_entity_aware_coordination_strategy(
    self, 
    entities: Dict[str, Any],
    conversation_context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create agent coordination strategy based on extracted entities.
    """
    
    strategy = {
        "discovery_agent": await self._create_discovery_strategy(entities, conversation_context),
        "genre_mood_agent": await self._create_genre_mood_strategy(entities, conversation_context),
        "judge_agent": await self._create_judge_strategy(entities, conversation_context)
    }
    
    return strategy

async def _create_discovery_strategy(
    self, 
    entities: Dict[str, Any],
    context: Optional[Dict] = None
) -> Dict[str, Any]:
    """Create DiscoveryAgent strategy based on entities."""
    
    strategy = {
        "seed_artists": entities.get("musical_entities", {}).get("artists", {}).get("primary", []),
        "similarity_targets": entities.get("musical_entities", {}).get("artists", {}).get("similar_to", []),
        "avoid_artists": entities.get("musical_entities", {}).get("artists", {}).get("avoid", []),
        "novelty_preference": self._extract_novelty_preference(entities),
        "underground_bias": self._calculate_underground_bias(entities),
        "activity_context": entities.get("contextual_entities", {}).get("activities", {}),
        "session_continuity": self._extract_session_continuity(entities, context)
    }
    
    return strategy
```

## Implementation Plan

### Phase 1: Foundation (Week 1-2)
1. **Create Enhanced Entity Models**
   - Define comprehensive entity data structures
   - Update MusicRecommenderState to include entity store
   - Create ConversationContextManager

2. **Basic LLM Integration**
   - Implement EnhancedEntityRecognizer class
   - Create entity extraction prompts
   - Add fallback mechanisms

3. **Migrate Existing Logic**
   - Move artist extraction from DiscoveryAgent to PlannerAgent
   - Update agent coordination to use centralized entities
   - Ensure backward compatibility

### Phase 2: Advanced Features (Week 3-4)
1. **Complex Query Handling**
   - Implement query decomposition logic
   - Add multi-faceted intent recognition
   - Create contextual modification handling

2. **Conversation Context**
   - Implement session state management
   - Add preference evolution tracking
   - Create context-aware entity resolution

### Phase 3: Optimization (Week 5-6)
1. **Performance Tuning**
   - Add entity caching mechanisms
   - Optimize LLM prompt efficiency
   - Implement batch processing for multiple queries

2. **Advanced Coordination**
   - Enhance agent strategy generation
   - Add cross-agent entity validation
   - Implement confidence-based fallbacks

### Phase 4: Testing & Refinement (Week 7-8)
1. **Comprehensive Testing**
   - Unit tests for entity extraction
   - Integration tests for agent coordination
   - End-to-end conversation flow tests

2. **Performance Monitoring**
   - Add entity recognition metrics
   - Monitor conversation context accuracy
   - Track user satisfaction improvements

## Success Metrics

### Technical Metrics
- **Entity Recognition Accuracy**: >90% for common entity types
- **Query Understanding Completeness**: >85% of query intents captured
- **Conversation Context Retention**: >95% of session references resolved
- **Agent Coordination Consistency**: 100% of agents receive same entity understanding

### User Experience Metrics
- **Query Satisfaction**: User feedback on recommendation relevance
- **Conversation Flow**: Natural progression through multi-turn interactions
- **Discovery Quality**: Balance between familiar and novel recommendations
- **Context Awareness**: Successful handling of relative references

## Risk Mitigation

### Technical Risks
1. **LLM Latency**: Implement caching and fallback mechanisms
2. **Entity Ambiguity**: Use confidence scores and validation
3. **Context Complexity**: Gradual rollout of advanced features
4. **Memory Usage**: Efficient session state management

### User Experience Risks
1. **Over-Engineering**: Maintain simple query handling for basic requests
2. **Context Confusion**: Clear session boundaries and reset mechanisms
3. **Preference Drift**: Allow explicit preference reset options

## Conclusion

This Enhanced PlannerAgent Entity Recognition System will transform BeatDebate into a sophisticated, context-aware music recommendation system. By centralizing entity recognition and adding conversation context, we create a single source of truth that enables more intelligent agent coordination and better user experiences.

The phased implementation approach ensures we can deliver value incrementally while building toward the full vision of contextual, conversational music discovery. 