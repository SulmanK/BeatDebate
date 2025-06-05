Okay, this is a substantial codebase with a good foundation, but there's definitely room for refinement to make it more production-ready, focusing on DRY principles and overall cleanliness.

Here's a design plan for refactoring:

## Design Plan: Production Refinement for BeatDebate

**Guiding Principles:**

1.  **DRY (Don't Repeat Yourself):** Consolidate common logic into base classes, utility functions, or shared components.
2.  **Single Responsibility Principle (SRP):** Ensure classes and modules have one primary reason to change.
3.  **Dependency Inversion Principle (DIP):** Depend on abstractions, not concretions. The use of `APIService` and DI in agents is a good start.
4.  **Clear Abstractions:** Make interfaces between components well-defined and intuitive.
5.  **Configuration Management:** Centralize and simplify how configurations are handled and passed.
6.  **Readability & Maintainability:** Improve code clarity through better organization and naming.
7.  **Preserve Functionality:** All existing features should work as before.

---

**Key Refactoring Areas & Action Plan:**

### 1. Agents (`src/agents/`)

The agent structure has a good base with shared components. The main goal here is to ensure maximum leverage of these components and the `BaseAgent`.

*   **`BaseAgent` Enhancement:**
    *   **Action:** Ensure `BaseAgent` handles common initialization patterns (e.g., logging setup, component instantiation if always the same type), standard error handling for `process`, and perhaps a standard way to access shared services like `APIService`, `MetadataService`.
    *   **Rationale:** Reduces boilerplate in concrete agent classes.

*   **Advocate Agents (`GenreMoodAgent`, `DiscoveryAgent`):**
    *   **Action (Candidate Generation):** Verify that *all* candidate generation logic is exclusively within `UnifiedCandidateGenerator`. Remove any local API calls or direct list building for candidates within these agents. They should only configure and call the generator.
    *   **Action (Scoring):** Ensure that all scoring uses `ComprehensiveQualityScorer`. Agent-specific scoring logic should be encapsulated as strategies *within* the `QualityScorer` or passed as parameters, rather than being implemented directly in the agent.
    *   **Action (Intent Adaptation):** The `_adapt_to_intent` methods are good. Consider if the `intent_parameters` dictionaries could be loaded from a configuration file or a shared constants module for easier management if they grow very large.
    *   **Rationale:** Maximizes reuse of shared components, makes agents leaner and focused on their specific advocacy role (configuring shared tools).

*   **Centralized Intent Logic for Agents:**
    *   **Action:** Reduce complex conditional intent handling within individual agents (`DiscoveryAgent`, `GenreMoodAgent`, `JudgeAgent`, `PlannerAgent`). Agents should receive a clear, pre-processed "effective intent" from a centralized service (see `IntentOrchestrationService` below).
    *   **Rationale:** Simplifies agent logic, reduces bugs, and centralizes intent-related decision-making, making the system easier to maintain and extend.

*   **`PlannerAgent`:**
    *   **Action (Query Understanding):** `QueryUnderstandingEngine` seems to be the core. Ensure it fully utilizes `LLMUtils`, `EntityExtractionUtils`, and `QueryAnalysisUtils`. The existing structure is good.
    *   **Action (Context Handling Logic):** Methods like `_is_followup_with_preserved_context`, `_create_understanding_from_context`, `_create_entities_from_context`, `_extract_entity_names` are specific to interpreting and transforming conversation context. These could potentially be moved to `ConversationContextService` or a new `ContextTransformationUtils` if they are general enough, or kept if tightly coupled with planner's decision-making. For now, let's assume they are planner-specific but review if they are used elsewhere.
    *   **Rationale:** Keeps planner focused on strategic planning, delegating detailed context interpretation.

*   **`JudgeAgent`:**
    *   **Action (Ranking & Explanation):** Relies on `RankingLogic` and `ConversationalExplainer`. This is good. Ensure `ConversationalExplainer` uses `LLMUtils`.
    *   **Action (Hybrid Subtype Detection):** The `_detect_hybrid_subtype` logic might be better placed within `QueryAnalysisUtils` or as part of the output from `QueryUnderstandingEngine`, as it's a core part of understanding the query's nuances.
    *   **Action (Filtering Shown Tracks):** The `_filter_out_recently_shown` logic should ideally be a utility provided by `ConversationContextService` (e.g., `get_recommendations_excluding_seen(candidates, session_id)`).
    *   **Rationale:** Centralizes query analysis and session-specific filtering.

*   **`src/agents/components/`:**
    *   **Action (General):** These are crucial. Review all agents to ensure they are *maximally* leveraging these utilities. Any local regex, list manipulation for entities, or direct LLM call patterns outside `LLMUtils` should be questioned and likely refactored to use these components.
    *   **`QualityScorer`:** The structure with sub-scorers and a `ComprehensiveQualityScorer` is good. Ensure it's the single point of entry for quality scoring.
    *   **`UnifiedCandidateGenerator`:** Ensure it's flexible enough to handle all generation strategies required by advocate agents. The intent-aware generation logic is a good pattern.
    *   **Rationale:** These components are the backbone of DRY for agent logic.

### 2. Services (`src/services/`)

*   **`EnhancedRecommendationService`:**
    *   **Action:** This is the main orchestrator. Its primary role should be to manage the LangGraph workflow and interact with other services. Keep business logic minimal here; it should delegate.
    *   **Action (Context Handling):** The `ContextAwareIntentAnalyzer` inside this service. This is a good place for the high-level decision of *how* to use context (e.g., is this a follow-up that needs special handling?). It should use `ConversationContextService` to fetch and update history.
    *   **Action (Agent Initialization):** Ensure `initialize_agents` correctly passes shared instances of `APIService`, `MetadataService`, `CacheManager`, and `rate_limiter` to all agents.
    *   **Rationale:** Clear separation of orchestration from core business logic.
    *   **Action (Slimming Down):** Ensure complex context analysis (especially original intent tracking for follow-ups) is fully delegated to the enhanced `ConversationContextService` and the new `IntentOrchestrationService`.
    *   **Rationale:** Keeps the orchestrator focused and delegates specialized tasks.

*   **`ConversationContextService` & `SmartContextManager`:**
    *   There's an overlap. `ConversationContextService` in `conversation_context_service.py` seems to be the intended manager for session data. `SmartContextManager` in `smart_context_manager.py` contains logic for *deciding* context use.
    *   **Action:**
        1.  Rename `ConversationContextService` to something more encompassing like `SessionManagerService` or `UserContextService`.
        2.  Merge the "smart" decision-making logic from `SmartContextManager` into this service or make `SmartContextManager` a sub-component that `EnhancedRecommendationService` consults. The goal is one primary service for all things related to user session and conversation context.
        3.  Ensure methods like `_extract_recently_shown_tracks` (currently in `EnhancedRecommendationService`) are part of this unified context service.
    *   **Rationale:** Consolidates all context-related logic into a single, authoritative service.
    *   **Action (Enhanced Context for Follow-ups):** This service (potentially renamed `SessionManagerService`) must store not just previous queries/recommendations, but also the *parsed original intent and entities* of earlier queries in the session. This is crucial for correct follow-up interpretation.
    *   **Action (Candidate Pool Persistence for Follow-ups):** This service should manage (potentially with `CacheManager`) the persistence of a larger initial candidate pool generated by advocate agents. "More tracks" follow-ups would then instruct the `JudgeAgent` to draw from this existing pool.
    *   **Rationale:** Enables accurate intent resolution for follow-ups and efficient "load more" style interactions.

*   **NEW: `IntentOrchestrationService` (Conceptual - could be merged or remain distinct):**
    *   **Action:** Create a new service dedicated to advanced intent management, or significantly enhance `ContextAwareIntentAnalyzer` and `ConversationContextService` to cover these responsibilities.
    *   **Responsibilities:**
        *   Managing the full lifecycle of intent: initial detection via `QueryUnderstandingEngine`.
        *   Storing the original query's intent and entities (working with `SessionManagerService`).
        *   Accurately interpreting follow-up queries by considering the original intent (e.g., "more tracks" after "songs *by* X" vs. "songs *like* X").
        *   Providing a clear, unambiguous "effective intent" to the `PlannerAgent` and subsequently to other agents for the current turn.
        *   Centralizing the logic that handles variations in follow-up phrasing and ensures they map to the correct underlying original intent.
    *   **Rationale:** Addresses the "hacky code" concern by creating a single source of truth for intent resolution, especially for complex follow-up scenarios. Drastically simplifies intent handling within individual agents.

*   **`APIService` & `MetadataService` & `CacheManager`:**
    *   **Action:** These seem well-defined. Ensure `APIService` is the *only* way agents/services interact with external APIs (Last.fm, Spotify). `MetadataService` should be the sole provider of `UnifiedTrackMetadata`, potentially using `APIService` and `CacheManager`.
    *   **Rationale:** Enforces a clean abstraction layer for external data and caching.

*   **`LLMFallbackService`:**
    *   **Action:** This is a distinct concern and is fine as a separate service. Ensure it uses `LLMUtils` for its Gemini calls.
    *   **Rationale:** Keeps fallback logic isolated.

### 3. API Layer (`src/api/`)

*   **`backend.py`:**
    *   **Action:** The `transform_unified_to_ui_format` function is UI-specific. Move it to `src/ui/response_formatter.py` or a new `src/ui/ui_utils.py`. The backend should ideally return data in a consistent internal format, and the UI layer transforms it.
    *   **Action:** Ensure all endpoints only call the `EnhancedRecommendationService` and do not contain business logic themselves.
    *   **Rationale:** Keeps backend clean and focused on API contracts, decouples it from UI presentation details.

*   **Other API Files (`base_client.py`, `client_factory.py`, `lastfm_client.py`, `spotify_client.py`, `rate_limiter.py`, `logging_middleware.py`):**
    *   **Action:** This structure is generally good. Ensure `LastFmClient` and `SpotifyClient` strictly adhere to `BaseAPIClient` and are always created via `APIClientFactory`. `UnifiedRateLimiter` is a good abstraction.
    *   **Rationale:** Maintains consistency and reusability for external API interactions.

### 4. Models (`src/models/`)

*   **Action:** Review `agent_models.py`, `metadata_models.py`, `recommendation_models.py`.
    *   Ensure `UnifiedTrackMetadata` and `UnifiedArtistMetadata` are the standard for passing track/artist data internally after fetching from external APIs.
    *   `MusicRecommenderState` is the central state object for LangGraph, which is good.
    *   Clarify if `TrackRecommendation` in `recommendation_models.py` is still needed or if `UnifiedTrackMetadata` (with its recommendation-specific fields) can serve that purpose. If `TrackRecommendation` is a UI-focused or Judge-output-focused model, ensure its distinction from `UnifiedTrackMetadata` is clear. It seems `TrackRecommendation` is used by `JudgeAgent`. This might be okay, but if it largely duplicates `UnifiedTrackMetadata`, consider merging or having it inherit. Given the current use, keeping it separate for the Judge's output and final UI presentation seems reasonable but ensure it's constructed from `UnifiedTrackMetadata`.
*   **Rationale:** Consistent data representation reduces errors and improves understanding.

### 5. UI Layer (`src/ui/`)

*   **Action:**
    *   `BeatDebateChatInterface` should interact cleanly with the `/recommendations` and `/planning` endpoints.
    *   The fallback logic in `BeatDebateChatInterface` using `LLMFallbackService` is good. Ensure the conditions for fallback (`_should_use_fallback`) are robust.
    *   Move `transform_unified_to_ui_format` here from `backend.py` (as mentioned above).
*   **Rationale:** Separates UI concerns.

### 6. Tests (`tests/`)


*   **Action:** Consolidate `tests/agents/test_enhanced_judge.py` and `tests/agents/test_judge_agent.py` if they test the same current `JudgeAgent`. If `test_enhanced_judge.py` was for an older version, it could be archived or removed if its tests are covered.
*   **Rationale:** Better test organization.

### 6.5. Follow-up Query Enhancements & Candidate Caching

*   **Problem:** Follow-up queries like "more tracks" need to:
    1.  Correctly interpret intent based on the *original* query (e.g., "more tracks BY artist X" vs. "more tracks LIKE artist X").
    2.  Efficiently provide more results without expensive regeneration if possible.
*   **Action (Intent Tracking):** The `SessionManagerService` (enhanced `ConversationContextService`) will store the original parsed intent and entities of queries. The `IntentOrchestrationService` (or equivalent) will use this stored original intent to correctly interpret follow-up queries.
*   **Action (Candidate Pool Persistence & Reuse):**
    *   Advocate agents (`DiscoveryAgent`, `GenreMoodAgent`), via `UnifiedCandidateGenerator`, will be capable of generating a larger initial pool of candidates (e.g., 50-100) if the `PlannerAgent` deems it appropriate (e.g., for intents likely to have "more like this" follow-ups).
    *   This initial, unranked pool will be stored by the `SessionManagerService` (possibly using `CacheManager`) for the session's duration.
    *   When a "more tracks" or similar follow-up occurs, and the `IntentOrchestrationService` confirms it's a continuation of the previous request:
        *   The `JudgeAgent` will be instructed to pull additional candidates from this *persisted pool* rather than advocate agents re-generating.
        *   The `JudgeAgent` will then apply its scoring, ranking, and diversity logic to the newly requested subset from the larger pool.
*   **Rationale:** Significantly improves user experience for follow-ups by providing contextually correct and efficiently loaded subsequent results. Reduces redundant API calls and processing.

### 7. Root Directory & Miscellaneous

*   **`TODO/todo.md`:**
    *   **Action:** Review and integrate these TODOs into this refactoring plan or a subsequent one. The items "Enhanced Planner Agent", "Agent Improvements", "Enhanced Judge Agent" are covered by this plan.
*   **Logging:**
    *   **Action:** `logging_config.py` and `logging.conf` provide good structured logging. Ensure all modules use `get_logger(__name__)` from `logging_config.py` for consistency.
    *   **Rationale:** Centralized and consistent logging.

---

**Code that is getting too large / needs most refactoring attention:**

1.  **`src/agents/components/quality_scorer.py` (and its sub-components):** While modular with sub-scorers, the main file `comprehensive_quality_scorer.py` can be the orchestrator.
    *   **Action (Reiteration):** Ensure individual scorer classes (e.g., `AudioQualityScorer`, `PopularityBalancer`, `EngagementScorer`, `GenreMoodFitScorer`, `IntentAwareScorer`) are in their own files within the `src/agents/components/scoring/` subdirectory. `ComprehensiveQualityScorer` acts as the composer.
    *   **Rationale:** Improves readability and maintainability of individual scoring components.

2.  **Advocate Agents (`src/agents/discovery/agent.py`, `src/agents/genre_mood/agent.py`):** These still contain significant internal logic for intent adaptation, filtering, and some aspects of scoring.
    *   **Action:** With the new `IntentOrchestrationService`, the intent adaptation logic within these agents should be drastically simplified. They should receive a clear "effective intent".
    *   **Action:** Push more of their specialized filtering or candidate manipulation logic into configurable strategies or utilities that `UnifiedCandidateGenerator` or the agents themselves can use, rather than large conditional blocks within the agent's `process` method.
    *   **Rationale:** Makes agents thinner, more focused on configuring common tools based on a clear intent, and reduces duplicated or "hacky" conditional logic.

3.  **`src/services/enhanced_recommendation_service.py`:** While it orchestrates the LangGraph flow, its direct business logic and context-handling should be minimal.
    *   **Action:** Ensure it primarily delegates to `PlannerAgent` for planning, advocate agents for candidate generation, `JudgeAgent` for selection, and the new `IntentOrchestrationService` (or enhanced `ContextAwareIntentAnalyzer` + `SessionManagerService`) for all complex intent and context decisions.
    *   **Rationale:** Keeps orchestration logic clean and focused on managing the workflow.

4.  **`src/agents/planner/agent.py`:**
    *   **Action:** If the context handling methods mentioned earlier (`_is_followup_with_preserved_context`, etc.) are deemed general enough, moving them to the unified `SessionManagerService` or `IntentOrchestrationService` would reduce this agent's size. Its role should be to take the "effective intent" and plan the agent sequence and high-level parameters.
    *   **Rationale:** Centralizes context transformation and intent resolution logic.

5.  **`src/agents/judge/agent.py`:**
    *   **Action:** Its logic for handling "more tracks" follow-ups will need to be adapted to request candidates from the persisted pool managed by `SessionManagerService` when appropriate.
    *   **Rationale:** Enables efficient "load more" functionality.

6.  **`src/services/api_service.py`:**
    *   **Action:** While generally well-structured, ensure it remains focused solely on external API interactions and does not creep into business logic related to candidate generation or scoring nuances that belong in agents or components. Review for any overly complex methods that could be simplified or broken down if they handle too many concerns.
    *   **Rationale:** Maintains a clean separation of concerns for external API interactions.

---

**Proposed Final Directory Structure (Illustrative Changes):**

```
BeatDebate/
├── Design/
├── scripts/
│   └── validate_lastfm.py
├── src/
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── entity_extraction_utils.py
│   │   │   ├── llm_utils.py
│   │   │   ├── query_analysis_utils.py
│   │   │   ├── unified_candidate_generator.py
│   │   │   └── scoring/                # New subdirectory for scorers
│   │   │       ├── __init__.py
│   │   │       ├── audio_quality_scorer.py
│   │   │       ├── popularity_balancer.py
│   │   │       ├── engagement_scorer.py
│   │   │       ├── genre_mood_fit_scorer.py
│   │   │       ├── intent_aware_scorer.py
│   │   │       └── comprehensive_quality_scorer.py # Main entry point, imports others
│   │   ├── planner/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── entity_recognizer.py
│   │   │   └── query_understanding_engine.py
│   │   ├── genre_mood/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── mood_logic.py
│   │   │   └── tag_generator.py
│   │   ├── discovery/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── similarity_explorer.py
│   │   │   └── underground_detector.py
│   │   └── judge/
│   │       ├── __init__.py
│   │       ├── agent.py
│   │       ├── explainer.py
│   │       └── ranking_logic.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── backend.py
│   │   ├── base_client.py
│   │   ├── client_factory.py
│   │   ├── lastfm_client.py
│   │   ├── logging_middleware.py
│   │   ├── rate_limiter.py
│   │   └── spotify_client.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── agent_models.py
│   │   ├── metadata_models.py
│   │   └── recommendation_models.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── api_service.py
│   │   ├── cache_manager.py
│   │   ├── conversation_context_service.py # Potentially renamed & merged
│   │   ├── enhanced_recommendation_service.py
│   │   ├── intent_orchestration_service.py # New or merged functionality
│   │   ├── llm_fallback_service.py
│   │   ├── metadata_service.py
│   │   # Removed smart_context_manager.py (logic merged or moved)
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── chat_interface.py
│   │   ├── planning_display.py
│   │   ├── response_formatter.py
│   │   └── ui_utils.py             # New for UI-specific transformations
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logging_config.py
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── agents/
│   │   ├── test_advocate_agents.py # Could be split into test_genre_mood.py, test_discovery.py
│   │   ├── test_enhanced_planner.py
│   │   ├── test_judge_agent.py    # Consolidated judge tests
│   │   └── test_planner_agent.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── test_enhanced_recommendation_service.py
│   │   ├── test_llm_fallback_service.py
│   │   ├── test_recommendation_engine_integration.py
│   │   └── test_recommendation_engine.py
│   ├── ui/
│   │   └── test_chat_interface_fallback.py
│   ├── integration/                 # New for broader integration tests
│   │   ├── test_intent_aware_backend_integration.py
│   │   └── test_phase3_demo.py
│   └── scenarios/                   # New for specific scenario tests
│       ├── test_followup_behavior.py
│       ├── test_followup_detection_fix.py
│       ├── test_history_filtering.py
│       ├── test_hybrid_context.py
│       ├── test_kendrick_followup.py
│       ├── test_simple_extraction.py
│       └── test_state_fix.py
├── TODO/
│   └── todo.md # To be updated or removed after refactoring
├── .flake8
├── .gitignore
├── app.py
├── env.example
├── logging.conf
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

**Proposed Refactoring Progression**

Here's a suggested way to progress through this refactoring plan, focusing on building foundational pieces first:

**Phase 1: Strengthen the Core - Context and Intent Services**

*   **Goal:** Establish robust services for managing conversation context and resolving user intent, especially for follow-ups.
*   **Key Actions from `TODO.md`:**
    1.  **Refactor `ConversationContextService` into `SessionManagerService`**:
        *   Implement storage for the *original parsed intent and entities* of previous queries.
        *   Add capabilities to store and retrieve a larger, unranked candidate pool generated by advocate agents (for "more tracks" scenarios).
    2.  **Develop `IntentOrchestrationService`** (or significantly enhance `ContextAwareIntentAnalyzer` within `EnhancedRecommendationService` and integrate its logic with `SessionManagerService`):
        *   Focus on accurately identifying follow-up queries.
        *   Implement logic to retrieve the stored original intent/entities from `SessionManagerService`.
        *   Develop the core mechanism to determine the "effective intent" for the current turn, especially differentiating "more tracks *by* X" from "more tracks *like* X" based on the original query's intent.

**Phase 2: Adapt Agents to the New Intent Paradigm**

*   **Goal:** Simplify agents by having them consume the "effective intent" from the centralized service.
*   **Key Actions from `TODO.md`:**
    1.  **Modify `PlannerAgent`**:
        *   Update it to receive and use the "effective intent" from `IntentOrchestrationService`.
        *   Delegate more of its specific context-handling logic to `SessionManagerService` and `IntentOrchestrationService`.
    2.  **Modify Advocate Agents (`DiscoveryAgent`, `GenreMoodAgent`)**:
        *   Simplify their internal intent adaptation logic (e.g., `_adapt_to_intent` methods) to rely on the clear "effective intent".
        *   Modify `UnifiedCandidateGenerator` to support generating a larger initial candidate pool when directed by the `PlannerAgent` (useful for intents prone to "load more" type follow-ups).
    3.  **Modify `JudgeAgent`**:
        *   Adapt its logic to request additional candidates from the persisted pool (managed by `SessionManagerService`) for relevant follow-up queries, instead of always triggering full regeneration.

**Phase 3: Implement Efficient Follow-up Candidate Handling**

*   **Goal:** Enable the "load more" functionality by using persisted candidate pools.
*   **Key Actions from `TODO.md`:**
    1.  **Integrate Candidate Pool Persistence**:
        *   Ensure `UnifiedCandidateGenerator` (via advocate agents) can store its larger initial candidate pool in `SessionManagerService` (potentially using `CacheManager`).
    2.  **Refine Follow-up Workflow**:
        *   Solidify the interaction between `IntentOrchestrationService`, `PlannerAgent` (to decide if a large pool is needed), advocate agents (to generate it), `SessionManagerService` (to store/retrieve it), and `JudgeAgent` (to use it for "more tracks" style follow-ups).

**Phase 4: Deep Dive into Modularization and Code Health**

*   **Goal:** Address the large code files and ensure adherence to DRY and SRP.
*   **Key Actions from `TODO.md`:**
    1.  **Refactor Large Agent Files**: Systematically review and refactor `DiscoveryAgent`, `GenreMoodAgent`, `PlannerAgent`, and `JudgeAgent`. Extract complex conditional logic, filtering strategies, or specific sub-tasks into smaller, reusable utility functions, helper classes, or strategy components.
    2.  **Slim Down `EnhancedRecommendationService`**: Ensure it's a lean orchestrator, delegating heavily to the specialized services and agents.
    3.  **Review `APIService`**: Ensure it remains strictly an API interaction layer and hasn't incorporated business logic.
    4.  **Finalize Scoring Component Modularization**: Double-check that all individual scorer classes (`AudioQualityScorer`, `PopularityBalancer`, etc.) are in their own files within `src/agents/components/scoring/` and that `ComprehensiveQualityScorer` correctly composes them.

**Phase 5: Testing, UI Adaptation, and Iteration**

*   **Goal:** Ensure all changes are robust, user-facing aspects are consistent, and gather feedback for further refinement.
*   **Key Actions from `TODO.md`:**
    1.  **Comprehensive Testing**:
        *   Write new unit and integration tests for the new services (`SessionManagerService`, `IntentOrchestrationService`) and modified agent interactions.
        *   Create specific scenario tests (in `tests/scenarios/`) for various follow-up queries (e.g., "more by X", "more like X", "more like these but upbeat") to validate correct intent handling and candidate pool reuse.
    2.  **UI Review**: Check if `BeatDebateChatInterface` needs any adjustments to how it displays or handles sequences of recommendations, especially with the "load more" style follow-ups.
    3.  **Review and Iterate**: Based on testing and usage, identify any remaining areas for improvement in clarity, efficiency, or intent handling.

---

This plan provides a structured approach. It's iterative; some decisions (like exactly where to move `_is_followup_with_preserved_context`) might become clearer during implementation. The key is to consistently apply the guiding principles.