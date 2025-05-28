# Agents Directory Refactoring Design Document

## Problem Statement

**Current State:**
- Large, monolithic agent files (1000+ lines) that are difficult to maintain and test
- Mixed responsibilities within single agent classes (query understanding, execution, evaluation)
- Tightly coupled components making individual testing challenging
- **SIGNIFICANT CODE DUPLICATION IDENTIFIED:**
  - Two separate candidate generators (`EnhancedCandidateGenerator` vs `EnhancedDiscoveryGenerator`)
  - Duplicate entity extraction methods across agents (`_extract_artists_from_*`, `_extract_*_genres`)
  - Duplicate JSON parsing methods (`_parse_json_response`)
  - Duplicate LLM calling methods (`_make_llm_call`)
  - Duplicate query analysis patterns across `PlannerAgent`, `EntityRecognizer`, and `JudgeAgent`

**Desired State:**
- Modular agent architecture with focused, smaller files (<500 lines each)
- **UNIFIED SHARED COMPONENTS** eliminating duplication
- Clear separation of concerns between planning, execution, and evaluation
- Consolidated candidate generation with configurable strategies
- Shared utility components for common operations

**Value Proposition:**
- **Eliminate ~30% code duplication** through consolidation
- Single source of truth for entity extraction, JSON parsing, and LLM calls
- Faster development cycles through improved code organization
- Easier testing of shared utilities
- Reduced maintenance burden from duplicate logic

## Architecture Design

### Overall Philosophy
- `PlannerAgent` remains the central query understander and strategy coordinator
- Advocate agents (`GenreMoodAgent`, `DiscoveryAgent`) execute strategies without duplicate query parsing
- `JudgeAgent` evaluates based on Planner's output, not its own analysis
- **SHARED COMPONENTS** handle all common operations
- **UNIFIED CANDIDATE GENERATOR** with strategy-based configuration

### Target Directory Structure

```
src/agents/
├── __init__.py                     # Exports main agent classes
├── base_agent.py                   # BaseAgent class (unchanged)
├── components/                     # Shared components
│   ├── __init__.py
│   ├── unified_candidate_generator.py  # UNIFIED: Replaces both generators
│   ├── quality_scorer.py               # MOVED: From root agents/
│   ├── entity_extraction_utils.py      # NEW: Consolidated extraction methods
│   ├── llm_utils.py                    # NEW: Shared LLM calling & JSON parsing
│   └── query_analysis_utils.py         # NEW: Shared query analysis patterns
├── planner/
│   ├── __init__.py                 # Exports PlannerAgent
│   ├── agent.py                    # PlannerAgent class (simplified)
│   ├── query_understanding_engine.py   # QueryUnderstandingEngine
│   └── entity_recognizer.py            # Enhanced entity recognizer (if kept)
├── genre_mood/
│   ├── __init__.py                 # Exports GenreMoodAgent
│   ├── agent.py                    # GenreMoodAgent class (simplified)
│   ├── mood_logic.py               # Mood mapping helpers (optional)
│   └── tag_generator.py            # Search tag generation (optional)
├── discovery/
│   ├── __init__.py                 # Exports DiscoveryAgent
│   ├── agent.py                    # DiscoveryAgent class (simplified)
│   ├── similarity_explorer.py      # MultiHopSimilarityExplorer
│   └── underground_detector.py     # UndergroundDetector
└── judge/
    ├── __init__.py                 # Exports JudgeAgent
    ├── agent.py                    # JudgeAgent class (simplified)
    ├── ranking_logic.py            # Scoring components
    └── explainer.py                # ConversationalExplainer
```

### **NEW: Unified Shared Components**

#### 1. Unified Candidate Generator (`src/agents/components/unified_candidate_generator.py`)
**Consolidates**: `EnhancedCandidateGenerator` + `EnhancedDiscoveryGenerator`

```python
class UnifiedCandidateGenerator:
    """Single generator with strategy-based configuration"""
    
    def __init__(self, lastfm_client, strategy_config: str = "genre_mood"):
        self.strategy_configs = {
            'genre_mood': {
                'primary_search': 40,
                'similar_artists': 30, 
                'genre_exploration': 20,
                'underground_gems': 10
            },
            'discovery': {
                'multi_hop_similarity': 50,
                'underground_detection': 30,
                'serendipitous_discovery': 20
            }
        }
        self.strategy = strategy_config
        
    async def generate_candidate_pool(self, entities, intent_analysis, agent_type):
        # Use appropriate strategy based on agent_type
        config = self.strategy_configs[agent_type]
        # Unified generation logic with configurable sources
```

#### 2. Entity Extraction Utils (`src/agents/components/entity_extraction_utils.py`)
**Consolidates**: Duplicate extraction methods from all agents

```python
class EntityExtractionUtils:
    @staticmethod
    def extract_artists_from_entities(entities: Dict[str, Any]) -> List[str]:
        """Single method for extracting artists from entities"""
        
    @staticmethod
    def extract_artists_from_query(query: str) -> List[str]:
        """Single method for extracting artists from query text"""
        
    @staticmethod
    def extract_genres_from_entities(entities: Dict[str, Any]) -> List[str]:
        """Single method for extracting genres from entities"""
        
    @staticmethod
    def extract_target_genres(entities: Dict[str, Any]) -> List[str]:
        """Single method for target genre extraction"""
```

#### 3. LLM Utils (`src/agents/components/llm_utils.py`)
**Consolidates**: All LLM calling and JSON parsing logic

```python
class LLMUtils:
    @staticmethod
    async def make_llm_call(client, prompt: str, system_prompt: str = None) -> str:
        """Single LLM calling method used by all agents"""
        
    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        """Single JSON parsing method with robust error handling"""
        
    @staticmethod
    def clean_json_string(json_str: str) -> str:
        """Shared JSON cleaning utilities"""
```

#### 4. Query Analysis Utils (`src/agents/components/query_analysis_utils.py`)
**Consolidates**: Common query analysis patterns

```python
class QueryAnalysisUtils:
    @staticmethod
    def initialize_query_patterns() -> Dict[str, Dict[str, List[str]]]:
        """Single source for all query patterns"""
        
    @staticmethod
    def extract_primary_intent(prompt: str) -> str:
        """Unified intent extraction logic"""
        
    @staticmethod
    def identify_activity(prompt: str) -> Optional[str]:
        """Unified activity identification"""
```

### **Component Responsibilities (Updated)**

#### 1. Planner Agent (`src/agents/planner/`)
- **Primary Role**: Query understanding and strategy generation
- **Simplified**: Remove duplicate JSON parsing, use shared LLM utils
- **Key Components**:
  - `agent.py`: Main PlannerAgent class with strategy coordination (< 400 lines)
  - `query_understanding_engine.py`: QueryUnderstandingEngine using shared utils
  - `entity_recognizer.py`: Enhanced entity recognizer (if kept as fallback)

#### 2. Genre/Mood Agent (`src/agents/genre_mood/`)
- **Primary Role**: Genre and mood-based recommendations
- **Simplified**: Use unified candidate generator, shared extraction utils
- **Key Components**:
  - `agent.py`: Main GenreMoodAgent class (< 300 lines)
  - `mood_logic.py`: Mood mapping helpers (optional split)
  - `tag_generator.py`: Search tag generation (optional split)

#### 3. Discovery Agent (`src/agents/discovery/`)
- **Primary Role**: Similarity-based discovery and underground recommendations
- **Simplified**: Use unified candidate generator, remove duplicate extraction
- **Key Components**:
  - `agent.py`: Main DiscoveryAgent class (< 400 lines)
  - `similarity_explorer.py`: MultiHopSimilarityExplorer
  - `underground_detector.py`: UndergroundDetector

#### 4. Judge Agent (`src/agents/judge/`)
- **Primary Role**: Evaluation and final selection
- **Simplified**: Remove internal prompt analysis, use shared query utils
- **Key Components**:
  - `agent.py`: Main JudgeAgent class (< 300 lines)
  - `ranking_logic.py`: Scoring components
  - `explainer.py`: ConversationalExplainer

#### 5. Shared Components (`src/agents/components/`)
- **Primary Role**: All shared functionality
- **Key Components**:
  - `unified_candidate_generator.py`: Single configurable generator
  - `quality_scorer.py`: Comprehensive quality scoring (moved from root)
  - `entity_extraction_utils.py`: All entity extraction methods
  - `llm_utils.py`: LLM calling and JSON parsing
  - `query_analysis_utils.py`: Query analysis patterns

### **Duplication Elimination Strategy**

#### Phase 1: Create Shared Components
1. **Create unified candidate generator** combining both existing generators
2. **Extract common LLM utilities** from all agents
3. **Consolidate entity extraction methods** into shared utils
4. **Consolidate query analysis patterns** into shared utils

#### Phase 2: Agent Simplification
1. **Remove duplicate methods** from individual agents
2. **Replace with shared component calls**
3. **Simplify agent logic** to focus on core responsibilities
4. **Update imports** to use shared components

#### Phase 3: Validation & Testing
1. **Verify functionality preservation** through comprehensive testing
2. **Performance validation** to ensure no degradation
3. **Integration testing** of shared components

### **File Size Reduction Targets**

| Component | Current Size | Target Size | Reduction |
|-----------|-------------|-------------|-----------|
| PlannerAgent | 1276 lines | <400 lines | 68% |
| JudgeAgent | 1361 lines | <300 lines | 78% |
| DiscoveryAgent | 1316 lines | <400 lines | 70% |
| GenreMoodAgent | 985 lines | <300 lines | 70% |

**Total Reduction**: ~70% in main agent files + elimination of duplicate utility code

### Technical Considerations

#### Backward Compatibility
- Maintain existing API interfaces during transition
- Gradual migration to shared components
- Test compatibility with existing service layer

#### Import Management
- Clear import structure for shared components
- Avoid circular dependencies through careful layering
- Use relative imports within agent subdirectories

#### Performance Considerations
- Shared components should not introduce performance overhead
- Lazy loading for optional components
- Efficient caching strategies for repeated operations

## Implementation Plan

### Step 1: Create Shared Components Foundation
```bash
mkdir -p src/agents/components
```
- Create `unified_candidate_generator.py`
- Create `entity_extraction_utils.py`  
- Create `llm_utils.py`
- Create `query_analysis_utils.py`

### Step 2: Extract and Consolidate Utilities
1. **LLM Utils**: Extract all `_make_llm_call` and `_parse_json_response` methods
2. **Entity Utils**: Consolidate all `_extract_*` methods
3. **Query Utils**: Consolidate query analysis patterns
4. **Unified Generator**: Merge both candidate generators

### Step 3: Simplify Agent Classes
1. Replace duplicate methods with shared component calls
2. Remove internal utility methods
3. Focus agent logic on core responsibilities

### Step 4: Create Agent Subdirectories
1. Move simplified agents to subdirectories
2. Extract remaining helper components
3. Update all imports

### Step 5: Integration & Testing
1. Update `recommendation_engine.py` imports
2. Run comprehensive test suite
3. Performance validation

## Success Criteria

1. **Duplication Elimination**: No duplicate utility methods across agents
2. **File Size Reduction**: All main agent files <500 lines
3. **Shared Component Adoption**: All agents use unified components
4. **Functionality Preservation**: All existing capabilities maintained
6. **Performance Maintenance**: No significant performance degradation

## Risk Mitigation

1. **Shared Component Bugs**: Comprehensive testing of shared utilities
2. **Import Conflicts**: Careful dependency management and testing
3. **Functionality Loss**: Step-by-step validation during migration
4. **Performance Issues**: Benchmarking before/after changes

## Expected Impact

- **Code Reduction**: ~3000 lines of duplicate code eliminated
- **Maintenance**: Single source of truth for common operations
- **Testing**: Shared components can be tested in isolation
- **Development**: Easier to add new agents using shared components
- **Architecture**: Clean separation of concerns with shared utilities

This refactoring will transform the agents from monolithic, duplicated code into a clean, modular architecture with significant reduction in complexity and maintenance burden. 