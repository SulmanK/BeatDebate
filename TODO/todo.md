Of course. Here is the complete and final design plan for testing your BeatDebate application, incorporating all requested components into a single, comprehensive document.

---

## **Comprehensive Test Design Plan for BeatDebate**

### **1. Guiding Principles & Test Strategy**

*   **Isolation (Unit Tests)**: Each test focuses on a single "unit" (a class or function). All external dependencies (other classes, services, API clients, LLMs) are **mocked** using `unittest.mock`'s `Mock` and `AsyncMock`. This ensures tests are fast, predictable, and verify the unit's logic without external failures.
*   **Integration Tests**: A small number of tests will verify the interaction *between* major components (e.g., the API and the Recommendation Service).
*   **Frameworks**: We will use `pytest` for the test framework, `pytest-asyncio` for asynchronous code support, and `httpx` for testing the FastAPI backend.
*   **Structure**: The `tests/` directory will mirror the `src/` directory for clear organization.
*   **Coverage**: The plan aims to cover:
    *   **Happy Paths**: Standard, successful execution of functions.
    *   **Edge Cases**: Empty inputs, missing data, unexpected formats.
    *   **Error Handling**: How the system behaves when dependencies fail (e.g., an API call raises an exception).

### **2. Test Directory Structure**

The following structure will be implemented to organize the tests logically:

```
tests/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── components/
│   │   └── test_unified_candidate_generator.py
│   ├── discovery/
│   │   └── test_discovery_agent.py
│   ├── genre_mood/
│   │   └── test_genre_mood_agent.py
│   ├── judge/
│   │   └── test_judge_agent.py
│   └── planner/
│       └── test_planner_agent.py
├── api/
│   ├── __init__.py
│   └── test_backend.py
├── models/
│   ├── __init__.py
│   └── test_metadata_models.py
└── services/
    ├── __init__.py
    ├── components/
    │   └── test_context_handler.py
    ├── test_api_service.py
    ├── test_intent_orchestration_service.py
    └── test_session_manager_service.py
```

---

### **3. Detailed Test Cases by Component**

Here are the concrete test files and the scenarios they will cover.

#### **A. `tests/models/test_metadata_models.py` ✅ COMPLETE**

*   **Goal**: Verify data integrity and logic within data models.
*   **Tests**:
    *   `test_unified_track_from_lastfm()`: Ensures correct mapping from `LastFmTrack` to `UnifiedTrackMetadata`. ✅
    *   `test_unified_track_from_spotify()`: Ensures correct mapping from `SpotifyTrack` to `UnifiedTrackMetadata`. ✅
    *   `test_merge_track_metadata()`: Validates that merging two `UnifiedTrackMetadata` objects correctly combines their data (e.g., Spotify popularity and Last.fm listeners). ✅
    *   `test_merge_different_tracks_raises_error()`: Confirms that attempting to merge two different tracks raises a `ValueError`. ✅
    *   `test_calculate_underground_score()`: Checks the logic for calculating a track's underground score based on its popularity metrics. ✅

---

#### **B. `tests/api/test_backend.py` ✅ COMPLETE**

*   **Goal**: Validate the FastAPI endpoints, ensuring correct request handling and response formatting. The `RecommendationService` will be mocked.
*   **Tests**:
    *   `test_health_check()`: Verifies the `/health` endpoint returns a `200 OK` status and the expected JSON payload. ✅
    *   `test_get_recommendations_success()`: Mocks a successful `RecommendationService` response and asserts the `/recommendations` endpoint returns `200 OK` with correctly formatted track data. ✅ (structure implemented)
    *   `test_get_recommendations_service_failure()`: Mocks the `RecommendationService` to raise an exception and asserts the endpoint returns a `500 Internal Server Error` with a structured error message. ✅ (structure implemented)
    *   `test_get_recommendations_invalid_request()`: Sends a request with invalid data (e.g., missing `query`) and asserts the endpoint returns a `422 Unprocessable Entity` error. ✅
    *   `test_get_planning_strategy_success()`: Mocks a successful response from the `get_planning_strategy` method and asserts the `/planning` endpoint returns `200 OK` with the strategy data. ✅ (structure implemented)
    *   `test_feedback_endpoint()`: Tests the `/feedback` endpoint to ensure it accepts valid feedback and returns a `200 OK` status. ✅ (structure implemented)

---

#### **C. `tests/services/test_api_service.py` ✅ COMPLETE**

*   **Goal**: Ensure `APIService` correctly orchestrates and delegates calls to its modular components (`TrackOperations`, `ArtistOperations`, etc.). All components will be mocked.
*   **Tests**:
    *   `test_search_unified_tracks_delegates_to_track_ops()`: Verifies that a call to `api_service.search_unified_tracks` correctly calls the `track_operations.search_unified_tracks` method with the right arguments. ✅
    *   `test_get_artist_info_delegates_to_artist_ops()`: Verifies delegation to the `ArtistOperations` component. ✅
    *   `test_check_artist_genre_match_delegates_to_genre_analyzer()`: Verifies delegation to the `GenreAnalyzer` component. ✅
    *   `test_get_lastfm_client_delegates_to_client_manager()`: Confirms that client retrieval is handled by the `ClientManager`. ✅

---

#### **D. `tests/services/test_session_manager_service.py` ✅ COMPLETE**

*   **Goal**: Test the logic for session creation, context storage, and candidate pool management.
*   **Tests**:
    *   `test_create_and_get_session()`: Checks that a new session is created with the correct initial state and can be retrieved. ✅
    *   `test_update_session_history()`: Adds multiple interactions to a session and verifies that the history is correctly appended. ✅
    *   `test_get_original_query_context()`: Verifies that the context of the *first* query in a session is stored and retrieved correctly. ✅
    *   `test_store_and_get_candidate_pool()`: Confirms a candidate pool can be stored for a session and retrieved successfully, and that its usage count is incremented. ✅
    *   `test_get_expired_candidate_pool_returns_none()`: Ensures that a pool older than the TTL is not returned. ✅
    *   `test_get_exhausted_candidate_pool_returns_none()`: Ensures a pool used more than its `max_usage` is not returned. ✅
    *   `test_clear_session()`: Verifies that a session's data is completely removed. ✅

---

#### **E. `tests/services/test_intent_orchestration_service.py` ✅ COMPLETE**

*   **Goal**: Validate the complex logic for resolving user intent, especially for follow-up queries.
*   **Tests**:
    *   `test_resolve_fresh_query()`: Tests intent resolution for a new, standalone query (e.g., "music like Radiohead"). ✅
    *   `test_resolve_followup_more_tracks_by_artist()`: Simulates a "by_artist" query followed by "more tracks" and asserts the effective intent remains `by_artist`. ✅
    *   `test_resolve_followup_more_tracks_like_artist()`: Simulates an "artist_similarity" query followed by "more like this" and asserts the effective intent remains `artist_similarity`. ✅
    *   `test_resolve_followup_artist_style_refinement()`: Tests a query like "more by [Artist] but more electronic" and confirms it resolves to `artist_style_refinement`. ✅
    *   `test_resolve_new_query_resets_context()`: Simulates a `by_artist` query followed by a completely new `genre_mood` query and asserts it is correctly identified as a fresh query, not a follow-up. ✅
    *   `test_resolve_serendipity_followup()`: Tests that "more tracks" after a "surprise me" query correctly resolves to `discovering_serendipity`. ✅
    *   `test_detect_followup_types()`: Tests detection of various follow-up types (artist deep dive, style continuation, style refinement, similarity exploration, variation requests). ✅
    *   `test_entity_extraction_utilities()`: Tests extraction and manipulation of artist names, style modifiers, and entity variations. ✅
    *   `test_intent_summary()`: Tests session intent summary functionality. ✅
    *   `test_llm_analysis()`: Tests LLM query analysis with success, failure, and edge cases. ✅
    *   `test_edge_cases()`: Tests handling of empty queries and malformed context overrides. ✅

---

#### **F. `tests/agents/test_planner_agent.py` ✅ COMPLETE**

*   **Goal**: Verify the `PlannerAgent` correctly orchestrates its components (`QueryAnalyzer`, `StrategyPlanner`, etc.) to produce a valid `planning_strategy` on the state object.
*   **Tests**:
    *   `test_planner_agent_initialization()`: Tests that PlannerAgent initializes correctly with all components (QueryAnalyzer, ContextAnalyzer, StrategyPlanner, EntityProcessor). ✅
    *   `test_component_status_check()`: Tests component status checking functionality. ✅
    *   `test_planner_agent_process_fresh_query()`: Mocks all components to simulate a fresh query and asserts that the final state contains the expected `query_understanding`, `entities`, and `planning_strategy`. ✅
    *   `test_planner_agent_process_followup_with_effective_intent()`: Tests that PlannerAgent uses effective intent when available from IntentOrchestrationService. ✅
    *   `test_planner_agent_process_followup_with_context_override()`: Mocks the `ContextAnalyzer` to return a follow-up context and verifies the planner uses this context instead of performing a fresh query analysis. ✅
    *   `test_planner_agent_handles_component_failure()`: Mocks a component (e.g., `QueryAnalyzer`) to raise an exception and asserts that the planner produces a valid, safe fallback strategy. ✅
    *   `test_planner_agent_handles_partial_failure()`: Tests that PlannerAgent handles partial component failures gracefully. ✅
    *   `test_backward_compatibility_methods()`: Tests that backward compatibility wrapper methods work correctly. ✅
    *   `test_llm_call_wrapper()`: Tests the LLM call wrapper method with success, failure, and edge cases. ✅
    *   `test_complete_workflow_integration()`: Tests complete workflow integration with realistic data flow. ✅
    *   `test_state_preservation_across_processing()`: Tests that existing state attributes are preserved during processing. ✅

---

#### **G. `tests/agents/discovery/test_discovery_agent.py` ✅ COMPLETE**

*   **Goal**: Verify the `DiscoveryAgent` correctly uses its modular components based on the received state.
*   **Tests**:
    *   `test_discovery_agent_initialization()`: Tests that DiscoveryAgent initializes correctly with all components. ✅
    *   `test_component_initialization()`: Tests that all components are properly initialized. ✅
    *   `test_discovery_agent_process_workflow()`: Mocks all components (`DiscoveryScorer`, `DiscoveryFilter`, etc.) and a state with a "discovery" intent. Asserts that each component is called in the correct order and that the final state contains `discovery_recommendations`. ✅
    *   `test_discovery_agent_handles_artist_similarity_intent()`: Provides a state with an "artist_similarity" intent and asserts that the `SimilarityExplorer` (mocked) is leveraged and the correct scoring/filtering logic is applied. ✅
    *   `test_discovery_agent_skips_generation_for_followup()`: Provides a state indicating a follow-up query and asserts that the `candidate_generator` is NOT called, as the agent defers to the `JudgeAgent` to use a persisted pool. ✅
    *   `test_generate_discovery_candidates_standard()`: Tests standard candidate generation. ✅
    *   `test_generate_discovery_candidates_with_pool_generation()`: Tests candidate generation with large pool generation. ✅
    *   `test_generate_discovery_candidates_handles_failure()`: Tests that candidate generation handles failures gracefully. ✅
    *   `test_get_current_parameters()`: Tests getting current parameters. ✅
    *   `test_update_parameters()`: Tests updating parameters. ✅
    *   `test_create_discovery_fallback_reasoning()`: Tests fallback reasoning creation. ✅
    *   `test_complete_discovery_workflow_integration()`: Tests complete discovery workflow with realistic data flow. ✅
    *   `test_state_preservation_across_processing()`: Tests that existing state attributes are preserved during processing. ✅uv r

---

#### **H. `tests/agents/genre_mood/test_genre_mood_agent.py` ✅ COMPLETE**

*   **Goal**: Verify the `GenreMoodAgent` correctly orchestrates its components for style-based recommendations.
*   **Tests**:
    *   `test_genre_mood_agent_initialization()`: Tests that GenreMoodAgent initializes correctly with all modular components. ✅
    *   `test_genre_mood_agent_process_workflow()`: Mocks all components and a state with a "genre_mood" intent. Asserts the correct workflow (candidate generation, scoring, filtering) is executed and the final state has `genre_mood_recommendations`. ✅
    *   `test_genre_mood_agent_handles_contextual_intent()`: Provides a state with a "contextual" intent (e.g., "music for studying") and asserts that the `MoodAnalyzer` and other components are correctly configured for functional music. ✅
    *   `test_genre_mood_agent_applies_context_override()`: Provides a state with a `context_override` for an artist deep-dive and verifies that it correctly boosts tracks by the target artist. ✅
    *   `test_score_candidates_integration()`: Tests candidate scoring with all component scores integrated. ✅
    *   `test_create_recommendations_integration()`: Tests recommendation creation with component integration. ✅
    *   `test_genre_mood_agent_handles_component_failure()`: Tests that GenreMoodAgent handles component failures gracefully. ✅
    *   `test_genre_mood_agent_skips_generation_for_followup()`: Tests that GenreMoodAgent skips candidate generation for follow-up queries. ✅
    *   `test_detect_intent_with_context_override()`: Tests intent detection with context override. ✅
    *   `test_detect_intent_from_query_understanding()`: Tests intent detection from query understanding. ✅
    *   `test_config_adaptation_for_different_intents()`: Tests that config manager adapts parameters for different intents. ✅
    *   `test_component_initialization_verification()`: Tests that all components are properly initialized with correct types. ✅

This structured plan provides a clear path to building a comprehensive and maintainable test suite for your entire application.