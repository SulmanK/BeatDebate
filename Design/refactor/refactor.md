You're absolutely right! As agent logic grows, a single file can become unwieldy. Breaking each agent and its closely related, dedicated helper components into its own subdirectory is an excellent step for modularity and maintainability, especially if you're aiming to keep individual files under a certain line count (e.g., 500 lines).

Let's integrate this into the refactoring plan.

**Revised Refactoring Plan (Incorporating Agent Subdirectories):**

**Overall Philosophy (Still Applies):**
*   `PlannerAgent` is the central query understander.
*   Advocates execute; Judge evaluates based on Planner's output.
*   `MusicRecommenderState` is key.

---

**New Directory Structure for `src/agents/`:**

We will create subdirectories for each main agent. Shared components used by multiple agents will go into a common `components` subdirectory within `agents`.

```
src/
  agents/
    __init__.py                     # Exports BaseAgent and main agent classes
    base_agent.py                   # BaseAgent class
    components/                     # For components shared by multiple agents
        __init__.py
        enhanced_candidate_generator.py # Used by GenreMood & Discovery
        quality_scorer.py             # Used by GenreMood & Discovery
        # Potentially other shared logic units
    planner/
        __init__.py                 # Exports PlannerAgent from agent.py
        agent.py                    # PlannerAgent class definition
        query_understanding_engine.py # QueryUnderstandingEngine class
        # entity_recognizer.py      # If EnhancedEntityRecognizer is kept distinct
        # strategy_helpers.py     # For _generate_strategy_from_understanding, etc.
    genre_mood/
        __init__.py                 # Exports GenreMoodAgent from agent.py
        agent.py                    # GenreMoodAgent class definition
        # mood_logic.py             # For mood mappings, mood analysis helpers
        # tag_generator.py          # For _generate_search_tags helpers
    discovery/
        __init__.py                 # Exports DiscoveryAgent from agent.py
        agent.py                    # DiscoveryAgent class definition
        candidate_generator.py      # EnhancedDiscoveryGenerator
        similarity_explorer.py      # MultiHopSimilarityExplorer
        underground_detector.py     # UndergroundDetector
    judge/
        __init__.py                 # Exports JudgeAgent from agent.py
        agent.py                    # JudgeAgent (EnhancedJudgeAgent) class
        prompt_analyzer.py          # PromptAnalysisEngine (if specific to Judge and not planner)
                                    # NOTE: This should ideally NOT exist if planner provides full understanding
        ranking_logic.py            # For contextual_scorer, discovery_scorer, intent_alignment methods
        explainer.py                # ConversationalExplainer
    # Files to be RE-EVALUATED / POTENTIALLY MOVED/REMOVED from src/agents/ directly:
    # - conversation_context.py -> Potentially to src/services/ if SmartContextManager is the main user,
    #                              or its logic merged into SmartContextManager
    # - entity_recognizer.py -> Likely into src/agents/planner/ if still needed
    # - query_understanding.py -> Likely into src/agents/planner/ as query_understanding_engine.py
    # - multi_hop_similarity.py -> Into src/agents/discovery/ as similarity_explorer.py
    # - underground_detector.py -> Into src/agents/discovery/ as underground_detector.py
```

**Updated Refactoring Suggestions by Module (with new structure):**

**1. `src/agents/planner/` (New Directory for PlannerAgent)**

*   **Files:**
    *   `agent.py`: Contains the `PlannerAgent` class.
    *   `query_understanding_engine.py`: Move the `QueryUnderstandingEngine` class here from `src/agents/query_understanding.py`.
    *   `entity_recognizer.py`: (Conditional) If `EnhancedEntityRecognizer` is still needed as a distinct fallback or component alongside `QueryUnderstandingEngine`, move it here from `src/agents/entity_recognizer.py`. If `QueryUnderstandingEngine` makes it redundant, remove it.
    *   `strategy_helpers.py` (Optional): If `agent.py` becomes too long, internal helper methods for generating different parts of the planning strategy (e.g., `_default_agent_weights`, `_generate_evaluation_weights`) can move here.
*   **Refactoring within `agent.py` (PlannerAgent class):**
    *   Primary reliance on `QueryUnderstandingEngine` (from `query_understanding_engine.py`).
    *   Remove legacy query analysis methods and the `_merge_understanding_with_legacy` method.
    *   Streamline fallback planning.

**2. `src/agents/discovery/` (New Directory for DiscoveryAgent)**

*   **Files:**
    *   `agent.py`: Contains the `DiscoveryAgent` class.
    *   `candidate_generator.py`: Move `EnhancedDiscoveryGenerator` class here from `src/agents/enhanced_discovery_generator.py`.
    *   `similarity_explorer.py`: Move `MultiHopSimilarityExplorer` class here from `src/agents/multi_hop_similarity.py`.
    *   `underground_detector.py`: Move `UndergroundDetector` class here from `src/agents/underground_detector.py`.
*   **Refactoring within `agent.py` (DiscoveryAgent class):**
    *   Remove `_extract_artists_from_query`.
    *   Simplify `_analyze_discovery_requirements` and `_find_seed_artists` to use planner's strategy.
    *   `process` method uses `candidate_generator.EnhancedDiscoveryGenerator` (from the same directory).

**3. `src/agents/genre_mood/` (New Directory for GenreMoodAgent)**

*   **Files:**
    *   `agent.py`: Contains the `GenreMoodAgent` class.
    *   `mood_logic.py` (Optional): For `_initialize_mood_mappings`, `_initialize_energy_mappings`, `_initialize_genre_mappings`, and `_analyze_mood_requirements` if `agent.py` gets too long.
    *   `tag_generator.py` (Optional): For `_generate_search_tags` if `agent.py` gets too long.
*   **Refactoring within `agent.py` (GenreMoodAgent class):**
    *   Simplify `_analyze_mood_requirements` and `_generate_search_tags` to use planner's strategy.
    *   `process` method uses `EnhancedCandidateGenerator` (from `src/agents/components/`).

**4. `src/agents/judge/` (New Directory for JudgeAgent)**

*   **Files:**
    *   `agent.py`: Contains the `EnhancedJudgeAgent` class (rename file if class name is changed to just `JudgeAgent`).
    *   `ranking_logic.py`: Move `ContextualRelevanceScorer`, `DiscoveryAppropriatenessScorer`, and intent alignment scoring methods (`_score_concentration_intent`, etc.) here.
    *   `explainer.py`: Move `ConversationalExplainer` class here.
    *   `prompt_analyzer.py`: **REMOVE THIS.** The `PromptAnalysisEngine` should not be part of the Judge.
*   **Refactoring within `agent.py` (JudgeAgent class):**
    *   Remove internal `PromptAnalysisEngine`.
    *   Adapt `evaluate_and_select` to use `state.query_understanding` (provided by Planner).
    *   Utilize helpers from `ranking_logic.py` and `explainer.py`.

**5. `src/agents/components/` (New Directory for Shared Agent Components)**

*   **Files:**
    *   `enhanced_candidate_generator.py`: Move from `src/agents/`. This is used by both `GenreMoodAgent` and `DiscoveryAgent`.
    *   `quality_scorer.py`: Move from `src/agents/`. This is used by both `GenreMoodAgent` and `DiscoveryAgent`.
*   **Considerations:** Ensure these components are generic enough for shared use. Their constructors should take any specific clients (like `lastfm_client`) they need.

**6. `src/agents/base_agent.py`**

*   No structural change needed for this file itself, but ensure it's imported correctly by agents in their new subdirectories.
*   Initialize `self._reasoning_steps: List[str] = []` in `__init__`.

**7. `src/agents/__init__.py`**

*   Update this file to correctly export the main agent classes from their new subdirectories.
    ```python
    # src/agents/__init__.py
    from .base_agent import BaseAgent
    from .planner.agent import PlannerAgent  # Adjusted import
    from .genre_mood.agent import GenreMoodAgent # Adjusted import
    from .discovery.agent import DiscoveryAgent  # Adjusted import
    from .judge.agent import JudgeAgent # Adjusted import (assuming EnhancedJudgeAgent becomes JudgeAgent)

    __all__ = [
        "BaseAgent",
        "PlannerAgent",
        "GenreMoodAgent",
        "DiscoveryAgent",
        "JudgeAgent",
    ]
    ```

**8. `src/services/recommendation_engine.py`**

*   Update imports for agent classes to reflect their new locations (e.g., `from ..agents.planner.agent import PlannerAgent`).

**9. `src/services/smart_context_manager.py` and `src/agents/conversation_context.py`**

*   The `ConversationContextManager` from `src/agents/conversation_context.py` is closely related to session state.
*   The `SmartContextManager` is in `src/services/`.
*   **Recommendation:** Merge the functionality of `ConversationContextManager` into `SmartContextManager` or make `ConversationContextManager` a utility class used *by* `SmartContextManager`. The `SmartContextManager` should be the primary service handling all aspects of conversation context. If so, `src/agents/conversation_context.py` could be removed, and its logic integrated into `src/services/smart_context_manager.py`. This seems more aligned as context management is a service-level concern.

**Refactoring Steps with New Structure:**

1.  **Create New Directory Structure:** Create `planner/`, `genre_mood/`, `discovery/`, `judge/`, and `components/` under `src/agents/`.
2.  **Move Agent Files:**
    *   Move `PlannerAgent` class to `src/agents/planner/agent.py`.
    *   Move `QueryUnderstandingEngine` to `src/agents/planner/query_understanding_engine.py`.
    *   Move `EnhancedEntityRecognizer` (if kept) to `src/agents/planner/entity_recognizer.py`.
    *   Move `GenreMoodAgent` class to `src/agents/genre_mood/agent.py`.
    *   Move `DiscoveryAgent` class to `src/agents/discovery/agent.py`.
    *   Move `EnhancedDiscoveryGenerator` to `src/agents/discovery/candidate_generator.py`.
    *   Move `MultiHopSimilarityExplorer` to `src/agents/discovery/similarity_explorer.py`.
    *   Move `UndergroundDetector` to `src/agents/discovery/underground_detector.py`.
    *   Move `EnhancedJudgeAgent` class to `src/agents/judge/agent.py`.
    *   Move its helper classes (`ContextualRelevanceScorer`, etc.) to `src/agents/judge/ranking_logic.py` and `explainer.py`.
    *   Move `EnhancedCandidateGenerator` and `ComprehensiveQualityScorer` to `src/agents/components/`.
3.  **Update `__init__.py` Files:** For each new agent subdirectory, create an `__init__.py` that exports the main agent class (e.g., `from .agent import PlannerAgent`). Also, update `src/agents/__init__.py`.
4.  **Perform Core Refactoring:**
    *   **Planner Agent:** Focus on `QueryUnderstandingEngine`, remove legacy analysis.
    *   **Judge Agent:** Remove internal prompt analysis, use Planner's output.
    *   **Advocate Agents:** Simplify to execute Planner's strategy, remove direct query parsing.
5.  **Update Imports:** Go through all modified files and other files like `src/services/recommendation_engine.py`, `tests/*` and update import statements.
6.  **Test:** Run all tests and fix any issues arising from structural changes or refactoring. Create new tests for moved components if their interfaces changed.

This approach will significantly improve the organization of your `src/agents/` directory, making each agent's core logic and its specific dependencies easier to manage and understand, especially as they grow.