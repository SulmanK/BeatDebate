# Comprehensive Codebase Refactoring Design Document

## Problem Statement

**Current State:**
- Large, monolithic agent files (1000+ lines) that are difficult to maintain and test
- Mixed responsibilities within single agent classes (query understanding, execution, evaluation)
- Tightly coupled components making individual testing challenging
- **SIGNIFICANT CODE DUPLICATION ACROSS ENTIRE CODEBASE:**
  - **Agents**: Two separate candidate generators, duplicate entity extraction, JSON parsing, LLM calls
  - **API Clients**: Duplicate `_make_request` implementations in LastFM and Spotify clients
  - **Rate Limiting**: Duplicate rate limiting logic across multiple clients
  - **Error Handling**: Similar error handling patterns across API clients
  - **Client Instantiation**: Repeated client creation patterns across agents
  - **Metadata Models**: Similar track/artist metadata structures across clients

**Desired State:**
- Modular architecture with focused, smaller files (<500 lines each)
- **UNIFIED SHARED COMPONENTS** eliminating duplication across entire codebase
- **CONSOLIDATED API LAYER** with shared HTTP client and rate limiting
- **UNIFIED METADATA MODELS** across all services
- **SHARED UTILITIES** for common operations (logging, caching, error handling)
- Clear separation of concerns between all layers

**Value Proposition:**
- **Eliminate ~40% code duplication** through comprehensive consolidation
- Single source of truth for HTTP requests, rate limiting, and error handling
- Unified metadata models across all services
- Faster development cycles through improved code organization
- Easier testing of shared utilities and API components
- Reduced maintenance burden from duplicate logic across entire codebase

## Comprehensive Architecture Design

### Overall Philosophy
- **Layered Architecture**: Clear separation between agents, services, API clients, and utilities
- **Shared Components**: All common functionality consolidated into reusable components
- **Unified Interfaces**: Consistent interfaces across all API clients and services
- **Single Responsibility**: Each component has a focused, well-defined purpose

### **EXPANDED: Target Directory Structure**

```
src/
├── api/                           # API Layer - Unified HTTP & External Services
│   ├── __init__.py
│   ├── base_client.py             # NEW: Base HTTP client with shared request logic
│   ├── rate_limiter.py            # NEW: Unified rate limiting for all APIs
│   ├── lastfm_client.py           # REFACTORED: Uses base_client
│   ├── spotify_client.py          # REFACTORED: Uses base_client
│   └── client_factory.py          # NEW: Factory for creating configured clients
├── models/                        # Data Models & Schemas
│   ├── __init__.py
│   ├── metadata_models.py         # NEW: Unified track/artist metadata models
│   ├── api_models.py             # NEW: API request/response models
│   ├── agent_models.py           # EXISTING: Agent workflow models
│   └── recommendation_models.py   # EXISTING: Recommendation result models
├── services/                      # Business Logic Services
│   ├── __init__.py
│   ├── recommendation_engine.py   # REFACTORED: Uses unified components
│   ├── smart_context_manager.py   # EXISTING: Context management service
│   ├── cache_manager.py          # EXISTING: Caching service
│   └── metadata_service.py       # NEW: Unified metadata operations
├── agents/                        # Agent Layer - Simplified with Shared Components
│   ├── __init__.py
│   ├── base_agent.py             # EXISTING: Base agent class
│   ├── components/               # Shared Agent Components
│   │   ├── __init__.py
│   │   ├── unified_candidate_generator.py  # UNIFIED: Replaces both generators
│   │   ├── quality_scorer.py               # MOVED: From root agents/
│   │   ├── entity_extraction_utils.py      # NEW: Consolidated extraction methods
│   │   ├── llm_utils.py                    # NEW: Shared LLM calling & JSON parsing
│   │   └── query_analysis_utils.py         # NEW: Shared query analysis patterns
│   ├── planner/
│   │   ├── __init__.py
│   │   ├── agent.py                        # SIMPLIFIED: PlannerAgent class
│   │   ├── query_understanding_engine.py   # MOVED: QueryUnderstandingEngine
│   │   └── entity_recognizer.py            # CONDITIONAL: Enhanced entity recognizer
│   ├── genre_mood/
│   │   ├── __init__.py
│   │   ├── agent.py                        # SIMPLIFIED: GenreMoodAgent class
│   │   ├── mood_logic.py                   # OPTIONAL: Mood mapping helpers
│   │   └── tag_generator.py                # OPTIONAL: Search tag generation
│   ├── discovery/
│   │   ├── __init__.py
│   │   ├── agent.py                        # SIMPLIFIED: DiscoveryAgent class
│   │   ├── similarity_explorer.py          # MOVED: MultiHopSimilarityExplorer
│   │   └── underground_detector.py         # MOVED: UndergroundDetector
│   └── judge/
│       ├── __init__.py
│       ├── agent.py                        # SIMPLIFIED: JudgeAgent class
│       ├── ranking_logic.py                # MOVED: Scoring components
│       └── explainer.py                    # MOVED: ConversationalExplainer
└── utils/                         # Shared Utilities
    ├── __init__.py
    ├── logging_config.py          # EXISTING: Logging configuration
    ├── http_utils.py              # NEW: HTTP utilities and error handling
    ├── retry_utils.py             # NEW: Retry logic and exponential backoff
    └── validation_utils.py        # NEW: Data validation and parsing utilities
```

## **NEW: API Layer Consolidation**

### **1. Base HTTP Client (`src/api/base_client.py`)**
**Consolidates**: Duplicate `_make_request` implementations

```python
class BaseAPIClient:
    """Base HTTP client with unified request handling, rate limiting, and error handling."""
    
    def __init__(
        self, 
        base_url: str, 
        rate_limiter: "RateLimiter",
        timeout: int = 10
    ):
        self.base_url = base_url
        self.rate_limiter = rate_limiter
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
        retries: int = 3
    ) -> Dict[str, Any]:
        """Unified request method with rate limiting, retries, and error handling."""
        
    async def _handle_api_error(self, response: aiohttp.ClientResponse, endpoint: str):
        """Unified API error handling across all clients."""
        
    async def _exponential_backoff(self, attempt: int, base_delay: float = 1.0):
        """Unified exponential backoff strategy."""
```

### **2. Unified Rate Limiter (`src/api/rate_limiter.py`)**
**Consolidates**: All rate limiting logic across clients

```python
class UnifiedRateLimiter:
    """Unified rate limiter supporting multiple rate limit strategies."""
    
    def __init__(self, calls_per_second: float = None, calls_per_hour: int = None):
        self.calls_per_second = calls_per_second
        self.calls_per_hour = calls_per_hour
        # Unified rate limiting implementation
        
    async def wait_if_needed(self):
        """Unified rate limiting logic for all API clients."""
```

### **3. Refactored API Clients**
Both `LastFmClient` and `SpotifyClient` will:
- Inherit from `BaseAPIClient`
- Remove duplicate `_make_request` implementations
- Use unified rate limiting
- Use shared error handling
- Focus only on API-specific logic

### **4. Client Factory (`src/api/client_factory.py`)**
**Consolidates**: Repeated client instantiation patterns

```python
class APIClientFactory:
    """Factory for creating configured API clients."""
    
    @staticmethod
    async def create_lastfm_client(api_key: str, rate_limit: float = 3.0) -> LastFmClient:
        """Create configured Last.fm client."""
        
    @staticmethod  
    async def create_spotify_client(
        client_id: str, 
        client_secret: str, 
        rate_limit: int = 50
    ) -> SpotifyClient:
        """Create configured Spotify client."""
```

## **NEW: Models Layer Consolidation**

### **1. Unified Metadata Models (`src/models/metadata_models.py`)**
**Consolidates**: Similar track/artist metadata across clients

```python
@dataclass
class UnifiedTrackMetadata:
    """Unified track metadata across all services."""
    name: str
    artist: str
    album: Optional[str] = None
    # Common fields across LastFM and Spotify
    # Service-specific data in 'source_data' field
    
@dataclass  
class UnifiedArtistMetadata:
    """Unified artist metadata across all services."""
    name: str
    # Common fields with service-specific extensions
```

### **2. API Models (`src/models/api_models.py`)**
**NEW**: Request/response models for API operations

```python
class SearchRequest(BaseModel):
    """Unified search request model."""
    
class MetadataRequest(BaseModel):
    """Unified metadata request model."""
```

## **NEW: Services Layer Enhancement**

### **1. Metadata Service (`src/services/metadata_service.py`)**
**NEW**: Unified metadata operations across all clients

```python
class MetadataService:
    """Unified service for metadata operations across all APIs."""
    
    def __init__(self, lastfm_client: LastFmClient, spotify_client: SpotifyClient):
        self.lastfm_client = lastfm_client
        self.spotify_client = spotify_client
        
    async def get_unified_track_metadata(
        self, 
        artist: str, 
        track: str
    ) -> UnifiedTrackMetadata:
        """Get unified track metadata from multiple sources."""
        
    async def search_tracks_unified(self, query: str) -> List[UnifiedTrackMetadata]:
        """Search tracks across multiple services with unified results."""
```

## **NEW: Utils Layer Expansion**

### **1. HTTP Utils (`src/utils/http_utils.py`)**
**NEW**: Shared HTTP utilities

```python
class HTTPUtils:
    @staticmethod
    def build_query_params(params: Dict[str, Any]) -> str:
        """Build query parameters with proper encoding."""
        
    @staticmethod
    def parse_api_error(response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Parse API error responses with consistent format."""
```

### **2. Retry Utils (`src/utils/retry_utils.py`)**
**NEW**: Shared retry logic

```python
class RetryUtils:
    @staticmethod
    async def exponential_backoff(
        attempt: int, 
        base_delay: float = 1.0, 
        max_delay: float = 60.0
    ):
        """Exponential backoff with jitter."""
        
    @staticmethod
    def should_retry(exception: Exception, attempt: int, max_attempts: int) -> bool:
        """Determine if operation should be retried."""
```

## **EXPANDED: Agent Layer Consolidation**

### **Updated: Agent Simplification Strategy**

#### **Remove Client Instantiation from Agents**
All agents currently create their own `LastFmClient` instances:
```python
# REMOVE THIS PATTERN from all agents:
from ..api.lastfm_client import LastFmClient
async with LastFmClient(api_key=self.lastfm.api_key) as client:
```

**Replace with**:
```python
# Agents receive configured clients via dependency injection
# Use MetadataService for unified operations
```

#### **Eliminate Rate Limit Duplication**
All agents currently store: `self.lastfm_rate_limit = lastfm_client.rate_limiter.calls_per_second`

**Replace with**: Shared configuration through `MetadataService`

## **EXPANDED: Duplication Elimination Strategy**

### **Phase 1: Core Infrastructure**
1. **Create base HTTP client** with unified request handling
2. **Create unified rate limiter** for all APIs  
3. **Create unified metadata models** replacing service-specific ones
4. **Create client factory** for standardized client creation

### **Phase 2: API Layer Refactoring**
1. **Refactor LastFM client** to use base client
2. **Refactor Spotify client** to use base client
3. **Remove duplicate HTTP and error handling code**
4. **Implement unified rate limiting**

### **Phase 3: Services Layer Enhancement**
1. **Create metadata service** for unified operations
2. **Update recommendation engine** to use unified services
3. **Integrate with existing cache manager** and context manager

### **Phase 4: Agent Layer Simplification**
1. **Remove client instantiation** from agents
2. **Replace with dependency injection** of configured services
3. **Implement unified candidate generator**
4. **Consolidate utility methods** into shared components

### **Phase 5: Utils Layer Completion**
1. **Extract HTTP utilities** from API clients
2. **Extract retry logic** into shared utilities
3. **Create validation utilities** for consistent data handling

## **EXPANDED: File Size Reduction Targets**

| Component | Current Size | Target Size | Reduction |
|-----------|-------------|-------------|-----------|
| **Agents Layer** |
| PlannerAgent | 1276 lines | <400 lines | 68% |
| JudgeAgent | 1361 lines | <300 lines | 78% |
| DiscoveryAgent | 1316 lines | <400 lines | 70% |
| GenreMoodAgent | 985 lines | <300 lines | 70% |
| **API Layer** |
| LastFmClient | 630 lines | <400 lines | 37% |
| SpotifyClient | 511 lines | <350 lines | 32% |
| **Services Layer** |
| RecommendationEngine | 809 lines | <500 lines | 38% |

**Total Reduction**: ~60% in main files + elimination of duplicate utility code across entire codebase

## **EXPANDED: Implementation Plan**

### **Step 1: Core Infrastructure (API Layer Foundation)**
```bash
mkdir -p src/api src/models src/utils
```
- Create `src/api/base_client.py` with unified HTTP handling
- Create `src/api/rate_limiter.py` with unified rate limiting
- Create `src/models/metadata_models.py` with unified data models
- Create `src/utils/http_utils.py` and `src/utils/retry_utils.py`

### **Step 2: API Layer Consolidation**
1. **Extract common patterns** from existing API clients
2. **Refactor LastFM and Spotify clients** to use base infrastructure
3. **Create client factory** for standardized instantiation
4. **Test API compatibility** with existing functionality

### **Step 3: Services Layer Enhancement**
1. **Create metadata service** with unified operations
2. **Update recommendation engine** to use new services
3. **Integrate with cache manager** for performance
4. **Test service layer compatibility**

### **Step 4: Agent Layer Refactoring (As Previously Planned)**
1. **Create shared agent components**
2. **Simplify agent classes** removing duplicate utilities
3. **Move agents to subdirectories** with focused responsibilities
4. **Update all imports** and dependencies

### **Step 5: Integration & Validation**
1. **Update all import statements** across codebase
2. **Run comprehensive test suite** for all layers
3. **Performance validation** ensuring no degradation
4. **Integration testing** of complete workflow

## **EXPANDED: Success Criteria**

1. **API Layer**: No duplicate HTTP handling or rate limiting logic
2. **Models Layer**: Unified metadata models across all services  
3. **Services Layer**: Consolidated business logic with clear interfaces
4. **Agents Layer**: No duplicate utility methods, <500 lines per file
5. **Utils Layer**: Shared utilities for all common operations
6. **Functionality Preservation**: All existing capabilities maintained
7. **Performance Maintenance**: No significant performance degradation
8. **Test Compatibility**: All existing tests pass with minimal changes

## **EXPANDED: Expected Impact**

- **Code Reduction**: ~5000 lines of duplicate code eliminated across entire codebase
- **Maintenance**: Single source of truth for HTTP, rate limiting, metadata, and common operations
- **Testing**: All shared components can be tested in isolation
- **Development**: Consistent patterns across all layers, easier to add new features
- **Architecture**: Clean separation of concerns with comprehensive shared utilities
- **Performance**: Optimized HTTP handling and caching through unified components

This comprehensive refactoring will transform the entire codebase from having significant duplication across all layers into a clean, modular architecture with substantial reduction in complexity and maintenance burden throughout the system. 