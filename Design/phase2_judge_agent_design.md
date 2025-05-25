# JudgeAgent - Design Document

**Date**: October 2023
**Author**: BeatDebate Team
**Status**: Revised (to match implementation)
**Review Status**: Pending

---

## 0. References

*   Overall Phase 2 Design: [phase2_planner_agent_design.md](phase2_planner_agent_design.md)
*   Main Project Design: [beatdebate-design-doc.md](Plans/beatdebate-design-doc.md)
*   Implemented Agent: `src/agents/judge_agent.py`
*   State Model: `src/models/agent_models.py` (MusicRecommenderState)
*   Recommendation Model: `src/models/recommendation_models.py` (TrackRecommendation)

---

## 1. Problem Statement

The `JudgeAgent` is responsible for making the final selection of music recommendations. It receives candidate tracks from the advocate agents (`GenreMoodAgent` and `DiscoveryAgent`) and an evaluation strategy (including criteria and weights) from the `PlannerAgent`. The `JudgeAgent` must:

1.  Parse candidate track data into `TrackRecommendation` models.
2.  Apply the `PlannerAgent`'s evaluation framework to score these `TrackRecommendation` objects.
3.  Ensure a degree of diversity in the final selections based on targets defined in the strategy.
4.  Select a final list of recommended `TrackRecommendation` objects (e.g., 3 tracks).
5.  Generate clear, transparent explanations for why each track was selected, populating the `explanation` field of each `TrackRecommendation`.

The core value of the `JudgeAgent` is to provide a reasoned, multi-criteria decision-making process that aligns with the overall strategy defined by the `PlannerAgent`, ensuring the final recommendations are not just a simple aggregation but a well-justified selection.

---

## 2. Goals & Non-Goals

### ✅ In Scope
-   Implementing the `JudgeAgent` class within the `src/agents/` directory.
-   Defining and implementing the key methods: `evaluate_and_select`, `_apply_evaluation_framework`, `_ensure_diversity`, and `_generate_final_explanations`.
-   Integrating with the `MusicRecommenderState` for accessing candidate tracks (as `List[Dict]`) and the `PlannerAgent`'s `evaluation_framework`.
-   Parsing candidate `List[Dict]` into `List[TrackRecommendation]`.
-   Producing a list of final recommendations (typically 3 `TrackRecommendation` objects) with detailed, strategy-aligned explanations.
-   Ensuring that all decisions and explanations are logged for transparency.
-   Writing unit tests for the `JudgeAgent`'s logic.

### ❌ Out of Scope
-   Defining the evaluation criteria themselves (this is the `PlannerAgent`'s responsibility).
-   Fetching track data (this is the advocate agents' responsibility).
-   Direct interaction with external APIs.
-   Complex real-time learning or adaptation beyond the provided strategy.

---

## 3. Technical Architecture

### 3.1 Agent Overview

The `JudgeAgent` acts as the final decision-maker in the recommendation pipeline. It takes a list of candidate tracks (collated from `GenreMoodAgent` and `DiscoveryAgent` recommendations stored as `List[Dict]` in the `MusicRecommenderState`) and the `evaluation_framework` (from the `planning_strategy` in `MusicRecommenderState`, which was defined by the `PlannerAgent`).

### 3.2 Data Flow

1.  **Input**:
    *   `state.genre_mood_recommendations`: `List[Dict]` - List of track data dictionaries recommended by `GenreMoodAgent`.
    *   `state.discovery_recommendations`: `List[Dict]` - List of track data dictionaries recommended by `DiscoveryAgent`.
    *   `state.planning_strategy["evaluation_framework"]`: `Dict` - Contains `primary_weights` and `diversity_targets` from the `PlannerAgent`. Example `diversity_targets` structure:
        ```python
        "evaluation_framework": {
            "primary_weights": {
                "concentration_friendliness_score": 0.4, # or 'concentration_friendliness'
                "novelty_score": 0.3,                    # or 'novelty'
                "quality_score": 0.3                     # or 'quality'
            },
            "diversity_targets": {
                "attributes": ["genres", "era"], # List of attributes on TrackRecommendation to diversify by
                "genres": 2, # Target number of unique genres
                "era": 1     # Target number of unique eras
            }
        }
        ```

2.  **Output**:
    *   Modifies `state.final_recommendations`: `List[TrackRecommendation]` - The selected `TrackRecommendation` objects, each with `judge_score` and `explanation` fields populated.
    *   Appends to `state.reasoning_log`: `List[str]` - Logs of the judging process.
    *   Returns the modified `MusicRecommenderState`.

### 3.3 Core `JudgeAgent` Class Structure

```python
# Location: src/agents/judge_agent.py
import logging # Standard Python logger
from typing import List, Dict, Any, Optional
from pydantic import ValidationError

from ..models.agent_models import MusicRecommenderState
from ..models.recommendation_models import TrackRecommendation

class JudgeAgent:
    def __init__(self, llm_client: Any = None): # Optional LLM client
        self.logger = logging.getLogger(__name__)
        self.llm_client = llm_client

    async def evaluate_and_select(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Main method to orchestrate the evaluation and selection process.
        Updates state.final_recommendations and state.reasoning_log.
        Returns the updated state.
        """
        self.logger.info("JudgeAgent: Starting evaluation and selection process.")
        
        all_candidate_dicts = state.genre_mood_recommendations + state.discovery_recommendations
        if not all_candidate_dicts:
            self.logger.warning("JudgeAgent: No candidate tracks provided for evaluation.")
            state.final_recommendations = []
            state.reasoning_log.append("JudgeAgent: No candidate tracks provided for evaluation.")
            return state

        evaluation_framework = state.planning_strategy.get("evaluation_framework")
        if not evaluation_framework or not evaluation_framework.get("primary_weights"):
            self.logger.error("JudgeAgent: Evaluation framework or primary_weights missing.")
            state.reasoning_log.append("JudgeAgent: Critical error - Evaluation framework missing.")
            state.final_recommendations = []
            return state

        primary_weights = evaluation_framework.get("primary_weights", {})
        diversity_targets = evaluation_framework.get("diversity_targets", {})

        # 1. Parse and Apply evaluation framework (scoring)
        scored_candidates = self._apply_evaluation_framework(
            all_candidate_dicts, 
            primary_weights
        ) # Returns List[TrackRecommendation]
        
        if not scored_candidates:
            self.logger.warning("JudgeAgent: No candidates were successfully scored.")
            state.final_recommendations = []
            return state

        # 2. Ensure diversity
        # Note: candidates are sorted by judge_score (desc) inside _ensure_diversity or before calling it
        selected_diverse_tracks = self._ensure_diversity(
            scored_candidates, # Already sorted List[TrackRecommendation]
            diversity_targets,
            num_recommendations=3 
        ) # Returns List[TrackRecommendation]
        
        # 3. Generate final explanations
        final_selections_with_explanations = await self._generate_final_explanations(
            selected_diverse_tracks, 
            evaluation_framework
        ) # Returns List[TrackRecommendation]
        
        state.final_recommendations = final_selections_with_explanations
        log_message = f"JudgeAgent: Selected {len(final_selections_with_explanations)} tracks."
        self.logger.info(log_message)
        state.reasoning_log.append(log_message)
        
        return state # Returns the updated state

    def _apply_evaluation_framework(self, candidates: List[Dict], primary_weights: Dict[str, float]) -> List[TrackRecommendation]:
        """
        Parses candidate track dictionaries into TrackRecommendation models,
        applies the scoring criteria, and adds a 'judge_score'.
        Tracks failing parsing or scoring are omitted.
        Looks for scorable attributes first on the model, then in `additional_scores`.
        """
        # ... (Implementation as in src/agents/judge_agent.py)
        # Handles ValidationError during TrackRecommendation(**track_dict)
        # Iterates criteria, attempts to getattr(track_model, criterion) or track_model.additional_scores.get(criterion)
        # Calculates and sets track_model.judge_score
        # Returns List[TrackRecommendation]
        pass

    def _ensure_diversity(self, candidates: List[TrackRecommendation], diversity_targets: Dict[str, int], num_recommendations: int = 3) -> List[TrackRecommendation]:
        """
        Selects tracks to meet diversity targets.
        `candidates` are expected to be pre-sorted by `judge_score` (descending).
        `diversity_targets` should include an "attributes" key listing fields to diversify by, 
        e.g., {"attributes": ["genres", "era"], "genres": 2}.
        """
        # ... (Implementation as in src/agents/judge_agent.py)
        # Handles empty candidates or no diversity_targets.get("attributes")
        # Iterates through attributes and candidates for greedy selection.
        # Fills remaining slots by highest score.
        # Returns List[TrackRecommendation]
        pass

    async def _generate_final_explanations(self, selections: List[TrackRecommendation], evaluation_framework: Dict) -> List[TrackRecommendation]:
        """
        Generates a human-readable explanation for each selected track,
        populating the `explanation` field of each TrackRecommendation.
        Can use an LLM for more nuanced explanations or a template-based approach.
        """
        # ... (Implementation as in src/agents/judge_agent.py)
        # Iterates selections, builds explanation string based on scores and diversity.
        # Optionally uses self.llm_client.
        # Sets track_copy.explanation.
        # Returns List[TrackRecommendation]
        pass
```

### 3.4 Assumptions
*   Candidate tracks provided by advocate agents (as `List[Dict]`) will be parsable into `TrackRecommendation` Pydantic models and contain necessary attributes (e.g., `genres`, `era`, and potentially pre-computed scores like `novelty_score`, `concentration_friendliness_score` either as direct attributes or within an `additional_scores` dictionary) that align with the keys in `evaluation_framework.primary_weights`.
*   Numerical scores/attributes used for weighting are ideally normalized (e.g., between 0 and 1).
*   The `evaluation_framework` provided by the `PlannerAgent` is well-formed and contains the necessary `primary_weights` and `diversity_targets` (including the `attributes` list within `diversity_targets`).

---

## 4. Implementation Plan

1.  **Develop `JudgeAgent` class structure**: Set up `src/agents/judge_agent.py` with the main class and method skeletons (✅ Done).
2.  **Implement `_apply_evaluation_framework`**:
    *   Logic for parsing `List[Dict]` into `List[TrackRecommendation]`.
    *   Handling `ValidationError` during parsing.
    *   Logic for iterating through `TrackRecommendation` objects and applying weighted scores.
    *   Robust handling of missing attributes or criteria (checking direct attributes and `additional_scores`).
    *   Add `judge_score` to each `TrackRecommendation` object (✅ Done).
3.  **Implement `_ensure_diversity`**:
    *   Algorithm for selecting tracks based on `diversity_targets` (using `diversity_targets["attributes"]`) while respecting `num_recommendations`.
    *   Strategy for prioritizing higher-scored tracks that also fulfill diversity.
    *   Handle cases where diversity targets cannot be fully met (✅ Done).
4.  **Implement `_generate_final_explanations`**:
    *   Initial template-based explanation generation.
    *   (Optional) Integrate with an LLM client for more narrative explanations if time permits and client is available.
    *   Populate `explanation` field of `TrackRecommendation` objects (✅ Done).
5.  **Implement `evaluate_and_select`**:
    *   Orchestrate the calls to the helper methods.
    *   Handle overall logic, including empty candidate lists or missing evaluation frameworks.
    *   Update `MusicRecommenderState` correctly and return the state (✅ Done).
6.  **Logging**: Integrate structured logging throughout the agent for transparency and debugging (✅ Done).
7.  **Unit Testing**: Create comprehensive unit tests for each method, covering various scenarios and edge cases (✅ Done, ongoing refinement).

---

## 5. LLM Implementation Prompts (Optional)

If using an LLM for explanation generation in `_generate_final_explanations`:

**Prompt for generating a track recommendation explanation (context is a `TrackRecommendation` object):**

```
Context:
You are part of a music recommendation system. A track has been selected by a JudgeAgent.
PlannerAgent provided the following evaluation framework:
Primary Weights: {primary_weights_json_string} 
Diversity Targets: {diversity_targets_json_string} # e.g., {"attributes": ["genres"], "genres": 2}

Track Details (TrackRecommendation model):
Title: {track.title}
Artist: {track.artist}
Judge Score: {track.judge_score}
Genres: {track.genres}
Era: {track.era}
Mood Tags: {track.tags} 
Key Attributes contributing to score (attribute: value, weighted_contribution): {weighted_attributes_string} 
Diversity Contribution (criterion: value): {diversity_contribution_string}
Other relevant track attributes from model: {other_track_model_attributes_json_string}

Task:
Generate a concise, compelling, and natural-sounding explanation (2-3 sentences) for why this track was recommended to the user.
The explanation should highlight:
1. The track's key strengths based on the scoring criteria it performed well on (use `judge_score` and `primary_weights`).
2. How it aligns with the user's inferred needs (if `primary_weights` give clues).
3. Mention its genres or mood_tags if relevant.
4. If it uniquely contributed to diversity goals (e.g., "offering a unique genre from {track.genres} to your recommendations").

Example:
"'{track.title}' by {track.artist} is a great choice if you're looking for {primary_goal_from_strategy}! It scored {track.judge_score:.2f} based on its strength in {top_criterion_1} and {top_criterion_2}, bringing a distinct {track.genres[0] if track.genres else 'vibe'}. Plus, it adds some excellent {diversity_attribute} variety to your suggestions."

Generate the explanation:
```

---

## 6. Testing Strategy

*   **Unit Tests (`tests/agents/test_judge_agent.py`)**:
    *   Test `_apply_evaluation_framework` with various `List[Dict]` candidates and weighting schemes. Verify correct calculation of `judge_score` on the resulting `TrackRecommendation` objects. Test with malformed dicts (handled by `ValidationError`) and missing attributes.
    *   Test `_ensure_diversity` with different `diversity_targets` (including the `attributes` key), numbers of `TrackRecommendation` candidates, and score distributions. Verify that diversity targets are met as best as possible and that higher-scoring tracks are prioritized. Test edge cases.
    *   Test `_generate_final_explanations` with template-based approach. If LLM is used, mock LLM calls and verify prompt construction and fallback mechanisms. Ensure `explanation` field of `TrackRecommendation` is populated.
    *   Test `evaluate_and_select` for overall orchestration:
        *   Empty candidate list.
        *   Missing `evaluation_framework` in state.
        *   Successful end-to-end flow.
        *   Correct updates to `MusicRecommenderState` (`final_recommendations` as `List[TrackRecommendation]`) and ensure the state object is returned.

---

## 7. Success Criteria

*   The `JudgeAgent` correctly parses candidate track dictionaries into `TrackRecommendation` models.
*   The `JudgeAgent` correctly applies the `PlannerAgent`'s evaluation framework and weights to `TrackRecommendation` objects.
*   The `JudgeAgent` successfully selects a final list of `TrackRecommendation` objects (defaulting to 3) that attempts to meet specified diversity goals (using `diversity_targets["attributes"]`).
*   The `JudgeAgent` generates clear and relevant explanations, populating the `explanation` field of each selected `TrackRecommendation`.
*   All operations are logged, and the `MusicRecommenderState` is updated correctly with `final_recommendations` (as `List[TrackRecommendation]`) and `reasoning_log`. The updated state is returned by `evaluate_and_select`.
*   Unit tests achieve high coverage for the agent's logic.

---

## 8. Risk Mitigation

*   **Risk**: Candidate track dictionaries lack the attributes expected by `TrackRecommendation` model or the `evaluation_framework`.
    *   **Mitigation**: `_apply_evaluation_framework` includes `try-except ValidationError` for parsing. Robust attribute checking (direct model attributes, then `additional_scores`) is implemented. Graceful fallbacks (e.g., assign a score of 0 for a criterion, log a warning) are in place. Clear contract for track data structure from advocate agents (should align with fields in `TrackRecommendation` or be placed in `additional_scores`).
*   **Risk**: `evaluation_framework` is missing or malformed.
    *   **Mitigation**: Checks in `evaluate_and_select` handle this, log an error, and return the state with empty `final_recommendations` (✅ Implemented).
*   **Risk**: Diversity algorithm is too complex or has unintended biases.
    *   **Mitigation**: Current greedy approach for diversity is relatively simple. Thoroughly test its behavior with various inputs. Log diversity decisions clearly (✅ Implemented).
*   **Risk**: LLM-generated explanations are inconsistent, too verbose, or expensive.
    *   **Mitigation**: Implement a solid template-based explanation as a fallback. Carefully craft LLM prompts and set token limits. Monitor LLM usage and cost if applicable (✅ Template fallback is implicit, LLM part is optional).

---

## 9. Open Questions & Discussion Points 
(Original decisions retained, but reflect that implementation uses `TrackRecommendation` and specific diversity structure)

*   **DECISION on Advocate Scores**: How should scores from advocate agents (e.g., a `GenreMoodAgent`'s confidence score) be incorporated into the `JudgeAgent`'s `judge_score`?
    *   **Resolution**: (As per original, but now these scores would be fields on `TrackRecommendation` or in its `additional_scores` dict).
        1.  **Primary Approach**: Advocate agents will strive to output tracks with attributes/scores that directly map to the high-level evaluation criteria defined by the `PlannerAgent`. These become fields on the `TrackRecommendation` model (e.g., `novelty_score`, `quality_score`).
        2.  **Flexibility**: If an advocate has a unique, valuable score, it can provide that score in the `additional_scores` dictionary of the `TrackRecommendation` object. The `PlannerAgent` can then choose to include this specific score name in its `evaluation_framework.primary_weights`.

*   **DECISION on Track Schema**: What is the exact schema for track dictionaries provided by advocate agents?
    *   **Resolution**: Advocate agents provide `List[Dict]`. The `JudgeAgent` parses these into the `TrackRecommendation` Pydantic model defined in `src/models/recommendation_models.py`. This model specifies all essential fields. The `JudgeAgent` relies on this standardized schema for all its operations. (✅ Aligned with implementation)

*   **DECISION on Diversity Tie-Breaking**: The `_ensure_diversity` method needs a robust tie-breaking mechanism.
    *   **Resolution**: (As per original)
        1.  **Primary Factor**: Candidate `TrackRecommendation` objects are pre-sorted by their calculated `judge_score` in descending order.
        2.  **Implicit Secondary Factor**: Iteration order through the sorted list provides deterministic tie-breaking. (✅ Aligned with implementation)

--- 