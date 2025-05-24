# JudgeAgent - Design Document

**Date**: October 2023
**Author**: BeatDebate Team
**Status**: Draft
**Review Status**: Pending

---

## 0. References

*   Overall Phase 2 Design: [phase2_planner_agent_design.md](phase2_planner_agent_design.md)
*   Main Project Design: [beatdebate-design-doc.md](Plans/beatdebate-design-doc.md)

---

## 1. Problem Statement

The `JudgeAgent` is responsible for making the final selection of music recommendations. It receives candidate tracks from the advocate agents (`GenreMoodAgent` and `DiscoveryAgent`) and an evaluation strategy (including criteria and weights) from the `PlannerAgent`. The `JudgeAgent` must:

1.  Apply the `PlannerAgent`'s evaluation framework to score the candidate tracks based on multiple criteria.
2.  Ensure a degree of diversity in the final selections based on targets defined in the strategy.
3.  Select a final list of recommended tracks (e.g., 3 tracks as per the overall system architecture).
4.  Generate clear, transparent explanations for why each track was selected, referencing the applied strategy and evaluation.

The core value of the `JudgeAgent` is to provide a reasoned, multi-criteria decision-making process that aligns with the overall strategy defined by the `PlannerAgent`, ensuring the final recommendations are not just a simple aggregation but a well-justified selection.

---

## 2. Goals & Non-Goals

### ✅ In Scope
-   Implementing the `JudgeAgent` class within the `src/agents/` directory.
-   Defining and implementing the key methods: `evaluate_and_select`, `_apply_evaluation_framework`, `_ensure_diversity`, and `_generate_final_explanations`.
-   Integrating with the `MusicRecommenderState` for accessing candidate tracks and the `PlannerAgent`'s `evaluation_framework`.
-   Producing a list of final recommendations (typically 3 tracks) with detailed, strategy-aligned explanations.
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

The `JudgeAgent` acts as the final decision-maker in the recommendation pipeline. It takes a list of candidate tracks (collated from `GenreMoodAgent` and `DiscoveryAgent` recommendations stored in the `MusicRecommenderState`) and the `evaluation_framework` (from the `planning_strategy` in `MusicRecommenderState`, which was defined by the `PlannerAgent`).

### 3.2 Data Flow

1.  **Input**:
    *   `state.genre_mood_recommendations`: `List[Dict]` - List of tracks recommended by `GenreMoodAgent`.
    *   `state.discovery_recommendations`: `List[Dict]` - List of tracks recommended by `DiscoveryAgent`.
    *   `state.planning_strategy["evaluation_framework"]`: `Dict` - Contains `primary_weights` and `diversity_targets` from the `PlannerAgent`. Example:
        ```python
        "evaluation_framework": {
            "primary_weights": {
                "concentration_friendliness": 0.4,
                "novelty": 0.3,
                "quality": 0.3 # Assuming 'quality' is a general score provided by advocates or derived
            },
            "diversity_targets": {"genre": 2, "era": 2, "energy": 1} # Target number of distinct items
        }
        ```

2.  **Output**:
    *   Modifies `state.final_recommendations`: `List[Dict]` - The selected tracks, each with an added `explanation` field.
    *   Appends to `state.reasoning_log`: `List[str]` - Logs of the judging process.

### 3.3 Core `JudgeAgent` Class Structure

```python
# Location: src/agents/judge_agent.py
from typing import List, Dict, Any
# from src.services.state_manager import MusicRecommenderState # Or LangGraph state definition

class JudgeAgent:
    def __init__(self, llm_client: Any = None): # Optional LLM client for explanation generation
        # self.logger = logging.getLogger(__name__) # Standard Python logger
        self.llm_client = llm_client

    async def evaluate_and_select(self, state: MusicRecommenderState) -> List[Dict]:
        """
        Main method to orchestrate the evaluation and selection process.
        Updates state.final_recommendations and state.reasoning_log.
        """
        # self.logger.info("JudgeAgent: Starting evaluation and selection.")
        
        all_candidate_tracks = state.genre_mood_recommendations + state.discovery_recommendations
        if not all_candidate_tracks:
            # self.logger.warning("JudgeAgent: No candidate tracks to evaluate.")
            state.final_recommendations = []
            state.reasoning_log.append("JudgeAgent: No candidate tracks provided for evaluation.")
            return []

        evaluation_framework = state.planning_strategy.get("evaluation_framework", {})
        if not evaluation_framework.get("primary_weights"):
            # self.logger.error("JudgeAgent: Evaluation framework or primary_weights missing from planning_strategy.")
            # Potentially select top N or random if no framework, or return error/empty
            state.reasoning_log.append("JudgeAgent: Critical error - Evaluation framework missing.")
            return [] # Or handle differently

        # 1. Apply evaluation framework (scoring)
        scored_candidates = self._apply_evaluation_framework(
            all_candidate_tracks, 
            evaluation_framework.get("primary_weights", {})
        )
        
        # 2. Ensure diversity
        # Sort by score before ensuring diversity to prioritize higher-scoring diverse tracks
        sorted_candidates = sorted(scored_candidates, key=lambda x: x.get("judge_score", 0), reverse=True)
        
        selected_diverse_tracks = self._ensure_diversity(
            sorted_candidates, 
            evaluation_framework.get("diversity_targets", {}),
            evaluation_framework.get("primary_weights", {}) # Pass weights for tie-breaking or context
        )
        
        # 3. Generate final explanations
        final_selections_with_explanations = await self._generate_final_explanations(
            selected_diverse_tracks, 
            evaluation_framework
        )
        
        state.final_recommendations = final_selections_with_explanations
        state.reasoning_log.append(f"JudgeAgent: Selected {len(final_selections_with_explanations)} tracks.")
        # self.logger.info(f"JudgeAgent: Successfully selected {len(final_selections_with_explanations)} tracks.")
        
        return final_selections_with_explanations

    def _apply_evaluation_framework(self, candidates: List[Dict], primary_weights: Dict[str, float]) -> List[Dict]:
        """
        Applies the scoring criteria from the PlannerAgent's strategy to each candidate track.
        Adds a 'judge_score' to each candidate dictionary.
        Assumes candidates have attributes that can be mapped to `primary_weights` keys.
        """
        # self.logger.debug(f"JudgeAgent: Applying evaluation framework with weights: {primary_weights}")
        scored_candidates = []
        for track in candidates:
            score = 0.0
            # Example: track might have {'novelty_score': 0.8, 'concentration_friendliness_score': 0.9, 'predicted_quality': 0.7}
            # These scores would ideally be normalized (0-1) by advocate agents or pre-processing.
            for criterion, weight in primary_weights.items():
                # Map criterion to a potential track attribute.
                # This mapping needs to be robust or clearly defined.
                # For example, 'novelty' weight applies to 'track_novelty_score' attribute.
                # We might need a mapping config if names aren't direct.
                track_attribute_value = track.get(f"{criterion}_score", track.get(criterion, 0.0)) 
                score += track_attribute_value * weight
            
            track_copy = track.copy()
            track_copy["judge_score"] = score
            scored_candidates.append(track_copy)
            # self.logger.debug(f"JudgeAgent: Track '{track.get('title', 'Unknown')}' scored: {score}")
        return scored_candidates

    def _ensure_diversity(self, candidates: List[Dict], diversity_targets: Dict[str, int], primary_weights: Dict[str, float], num_recommendations: int = 3) -> List[Dict]:
        """
        Selects tracks to meet diversity targets (e.g., number of unique genres, eras).
        Prioritizes higher-scoring tracks while fulfilling diversity.
        `candidates` are expected to be pre-sorted by `judge_score` (descending).
        """
        # self.logger.debug(f"JudgeAgent: Ensuring diversity with targets: {diversity_targets}. Num recommendations: {num_recommendations}")
        if not diversity_targets:
            # self.logger.debug("JudgeAgent: No diversity targets specified. Returning top N candidates.")
            return candidates[:num_recommendations]

        selected_tracks = []
        # Keep track of met diversity criteria, e.g., {"genre": {"rock", "pop"}, "era": {"90s"}}
        met_diversity_counts = {key: set() for key in diversity_targets.keys()}

        # Pass 1: Greedily select tracks that fulfill a new diversity criterion
        for track in candidates:
            if len(selected_tracks) >= num_recommendations:
                break

            fulfills_new_diversity = False
            current_track_diversity_values = {} # e.g. {"genre": "rock", "era": "90s"}

            for key, target_count in diversity_targets.items():
                track_value_for_key = track.get(key) # Assuming track has 'genre', 'era' attributes
                if track_value_for_key is not None:
                    current_track_diversity_values[key] = track_value_for_key
                    if len(met_diversity_counts[key]) < target_count and track_value_for_key not in met_diversity_counts[key]:
                        fulfills_new_diversity = True
            
            if fulfills_new_diversity and track not in selected_tracks:
                selected_tracks.append(track)
                for key, value in current_track_diversity_values.items():
                    if key in met_diversity_counts:
                         met_diversity_counts[key].add(value)
        
        # Pass 2: If not enough tracks selected, fill remaining slots with highest-scored tracks
        # that haven't been selected, even if they don't add new diversity,
        # or optionally, tracks that add to existing diversity counts if targets not fully met.
        
        idx = 0
        while len(selected_tracks) < num_recommendations and idx < len(candidates):
            track = candidates[idx]
            if track not in selected_tracks:
                selected_tracks.append(track)
                # Optionally update met_diversity_counts here too if needed for more complex logic
            idx += 1
            
        # self.logger.info(f"JudgeAgent: Selected {len(selected_tracks)} diverse tracks. Met diversity: {met_diversity_counts}")
        return selected_tracks[:num_recommendations] # Ensure we don't exceed num_recommendations

    async def _generate_final_explanations(self, selections: List[Dict], evaluation_framework: Dict) -> List[Dict]:
        """
        Generates a human-readable explanation for each selected track,
        referencing the evaluation criteria and why it was chosen.
        Can use an LLM for more nuanced explanations or a template-based approach.
        """
        # self.logger.debug("JudgeAgent: Generating final explanations.")
        explained_selections = []
        primary_weights = evaluation_framework.get("primary_weights", {})
        diversity_targets = evaluation_framework.get("diversity_targets", {})

        for track in selections:
            explanation_parts = []
            # Explanation based on score contributions
            if primary_weights:
                explanation_parts.append(f"This track, '{track.get('title', 'N/A')}' by {track.get('artist', 'N/A')}, was selected based on:")
                for criterion, weight in primary_weights.items():
                    track_attribute_value = track.get(f"{criterion}_score", track.get(criterion, 0.0))
                    if track_attribute_value > 0: # Only mention criteria that contributed positively or are present
                        explanation_parts.append(f"  - Its strength in '{criterion}' (contributing {track_attribute_value*weight:.2f} to its score of {track.get('judge_score',0):.2f}, with a weight of {weight*100}%).")
            else:
                explanation_parts.append(f"'{track.get('title', 'N/A')}' was selected as a good candidate.")

            # Explanation based on diversity contribution (if applicable)
            # This part requires knowing *why* a track was kept by _ensure_diversity.
            # For simplicity here, we'll just state its attributes.
            for key, target in diversity_targets.items():
                value = track.get(key)
                if value:
                    explanation_parts.append(f"  - It contributes to '{key}' diversity with value '{value}'.")
            
            track_copy = track.copy()
            track_copy["explanation"] = "
".join(explanation_parts)
            
            # If LLM client is available, could try to generate a more narrative explanation.
            # if self.llm_client:
            #     prompt = f"Generate a concise, compelling explanation for why the track '{track.get('title')}' by '{track.get('artist')}' was recommended. It scored {track.get('judge_score')} based on criteria {primary_weights}. It has attributes like genre: {track.get('genre')}, mood: {track.get('mood')}. Emphasize its key strengths according to the scoring."
            #     try:
            #         llm_explanation = await self.llm_client.generate_text(prompt) # Fictional method
            #         track_copy["explanation"] = llm_explanation
            #         self.logger.debug(f"LLM generated explanation for '{track.get('title')}'.")
            #     except Exception as e:
            #         self.logger.error(f"Error generating explanation with LLM: {e}. Falling back to template.")
            
            explained_selections.append(track_copy)
        return explained_selections

```

### 3.4 Assumptions
*   Candidate tracks provided by advocate agents will contain necessary attributes (e.g., `genre`, `era`, `mood`, and potentially pre-computed scores like `novelty_score`, `concentration_friendliness_score`) that align with the keys in `evaluation_framework.primary_weights`. If not direct matches, the `_apply_evaluation_framework` will need a mapping or more robust attribute fetching.
*   Numerical scores/attributes used for weighting are ideally normalized (e.g., between 0 and 1).
*   The `evaluation_framework` provided by the `PlannerAgent` is well-formed and contains the necessary `primary_weights` and `diversity_targets`.

---

## 4. Implementation Plan

1.  **Develop `JudgeAgent` class structure**: Set up `src/agents/judge_agent.py` with the main class and method skeletons.
2.  **Implement `_apply_evaluation_framework`**:
    *   Logic for iterating through candidates and applying weighted scores.
    *   Ensure robust handling of missing attributes or criteria.
    *   Add `judge_score` to each candidate.
3.  **Implement `_ensure_diversity`**:
    *   Algorithm for selecting tracks based on `diversity_targets` while respecting `num_recommendations`.
    *   Strategy for prioritizing higher-scored tracks that also fulfill diversity.
    *   Handle cases where diversity targets cannot be fully met.
4.  **Implement `_generate_final_explanations`**:
    *   Initial template-based explanation generation.
    *   (Optional) Integrate with an LLM client for more narrative explanations if time permits and client is available.
5.  **Implement `evaluate_and_select`**:
    *   Orchestrate the calls to the helper methods.
    *   Handle overall logic, including empty candidate lists or missing evaluation frameworks.
    *   Update `MusicRecommenderState` correctly.
6.  **Logging**: Integrate structured logging throughout the agent for transparency and debugging.
7.  **Unit Testing**: Create comprehensive unit tests for each method, covering various scenarios and edge cases.

---

## 5. LLM Implementation Prompts (Optional)

If using an LLM for explanation generation in `_generate_final_explanations`:

**Prompt for generating a track recommendation explanation:**

```
Context:
You are part of a music recommendation system. A track has been selected by a JudgeAgent.
PlannerAgent provided the following evaluation framework:
Primary Weights: {primary_weights_json_string}
Diversity Targets: {diversity_targets_json_string}

Track Details:
Title: {track_title}
Artist: {track_artist}
Judge Score: {track_judge_score}
Key Attributes contributing to score (attribute: value, weighted_contribution): {weighted_attributes_string} 
Diversity Contribution (criterion: value): {diversity_contribution_string}
Other relevant track attributes (genre, mood, era, etc.): {other_attributes_json_string}

Task:
Generate a concise, compelling, and natural-sounding explanation (2-3 sentences) for why this track was recommended to the user.
The explanation should highlight:
1. The track's key strengths based on the scoring criteria it performed well on.
2. How it aligns with the user's inferred needs (if primary_weights give clues, e.g., "concentration_friendliness").
3. Mention its genre or mood if relevant.
4. If it uniquely contributed to diversity goals, subtly weave that in if possible (e.g., "offering a unique [genre/era] to your recommendations").

Example:
"'{track_title}' by {track_artist} is a great choice if you're looking for {primary_goal_from_strategy}! It scored highly for {top_criterion_1} and {top_criterion_2}, bringing a distinct {genre/mood} vibe. Plus, it adds some excellent {diversity_attribute} variety to your suggestions."

Generate the explanation:
```

---

## 6. Testing Strategy

*   **Unit Tests (`tests/agents/test_judge_agent.py`)**:
    *   Test `_apply_evaluation_framework` with various candidate lists and weighting schemes. Verify correct calculation of `judge_score`. Test with missing attributes.
    *   Test `_ensure_diversity` with different diversity targets, numbers of candidates, and score distributions. Verify that diversity targets are met as best as possible and that higher-scoring tracks are prioritized. Test edge cases (e.g., no diversity targets, too few candidates to meet diversity).
    *   Test `_generate_final_explanations` with template-based approach. If LLM is used, mock LLM calls and verify prompt construction and fallback mechanisms.
    *   Test `evaluate_and_select` for overall orchestration:
        *   Empty candidate list.
        *   Missing `evaluation_framework` in state.
        *   Successful end-to-end flow.
        *   Correct updates to `MusicRecommenderState`.

---

## 7. Success Criteria

*   The `JudgeAgent` correctly applies the `PlannerAgent`'s evaluation framework and weights to candidate tracks.
*   The `JudgeAgent` successfully selects a final list of tracks (defaulting to 3) that attempts to meet specified diversity goals.
*   The `JudgeAgent` generates clear and relevant explanations for each selected track, referencing the strategy.
*   All operations are logged, and the `MusicRecommenderState` is updated correctly with `final_recommendations` and `reasoning_log`.
*   Unit tests achieve high coverage for the agent's logic.

---

## 8. Risk Mitigation

*   **Risk**: Candidate tracks lack the attributes expected by the `evaluation_framework`.
    *   **Mitigation**: Implement robust attribute checking in `_apply_evaluation_framework` with graceful fallbacks (e.g., assign a score of 0 for that criterion, log a warning). Define a clear contract for track data structure from advocate agents.
*   **Risk**: `evaluation_framework` is missing or malformed.
    *   **Mitigation**: Add checks in `evaluate_and_select` to handle this, log an error, and potentially return no recommendations or a default set.
*   **Risk**: Diversity algorithm is too complex or has unintended biases.
    *   **Mitigation**: Start with a simple greedy approach for diversity. Thoroughly test its behavior with various inputs. Log diversity decisions clearly.
*   **Risk**: LLM-generated explanations are inconsistent, too verbose, or expensive.
    *   **Mitigation**: Implement a solid template-based explanation as a fallback. Carefully craft LLM prompts and set token limits. Monitor LLM usage and cost if applicable.

---

## 9. Open Questions & Discussion Points

*   **DECISION on Advocate Scores**: How should scores from advocate agents (e.g., a `GenreMoodAgent`'s confidence score) be incorporated into the `JudgeAgent`'s `judge_score`?
    *   **Resolution**: We will use a hybrid approach:
        1.  **Primary Approach**: Advocate agents will strive to output tracks with attributes/scores that directly map to the high-level evaluation criteria defined by the `PlannerAgent` in its `evaluation_framework.primary_weights` (e.g., `novelty_score`, `quality_score`). Advocates are responsible for translating their internal metrics into these common criteria where possible.
        2.  **Flexibility**: If an advocate has a unique, valuable score that doesn't easily map to a general criterion (e.g., a hypothetical `DiscoveryAgent_raw_serendipity_index`), it can provide that score as a distinct attribute on the track. The `PlannerAgent` can then *choose* to include this specific score name (e.g., `"DiscoveryAgent_raw_serendipity_index": 0.15`) in its `evaluation_framework.primary_weights` if it deems it strategically important for a particular user query. This allows the `PlannerAgent` to maintain central control over evaluation criteria while enabling the use of specialized advocate insights when necessary.

*   **DECISION on Track Schema**: What is the exact schema for track dictionaries provided by advocate agents? Specifically, what fields related to scores (novelty, quality, etc.) and descriptive attributes (genre, era, mood, energy) will be reliably present?
    *   **Resolution**: A standardized Pydantic model for track data (e.g., `TrackRecommendation`) will be defined in `src/models/recommendation_models.py` (or a similar shared models file). This model will specify all essential fields, including:
        *   **Core Metadata**: e.g., `title: str`, `artist: str`, `lastfm_id: Optional[str]`, `spotify_id: Optional[str]`, `preview_url: Optional[str]`.
        *   **Descriptive Attributes for Diversity/Filtering**: e.g., `genre: Optional[str]`, `era: Optional[str]`, `mood_tags: List[str] = []`, `energy_level: Optional[str]` (could be categorical like "low", "medium", "high", or numerical if normalized).
        *   **Scorable Attributes (normalized, e.g., 0-1 range where applicable)**: e.g., `novelty_score: Optional[float]`, `quality_score: Optional[float]`, `concentration_friendliness_score: Optional[float]`, and any other common criteria the `PlannerAgent` may define weights for. Any advocate-specific scores that are not mapped to common criteria must also have clearly defined names and types (e.g., `gm_agent_custom_metric: Optional[float]`).
    *   Advocate agents are responsible for populating these fields as accurately and completely as possible. The `JudgeAgent` will rely on this standardized schema for all its operations.

*   **DECISION on Diversity Tie-Breaking**: The `_ensure_diversity` method needs a robust tie-breaking mechanism if multiple tracks could fulfill diversity criteria equally well.
    *   **Resolution**: For the MVP, the tie-breaking mechanism will be as follows:
        1.  **Primary Factor**: Candidate tracks are pre-sorted by their calculated `judge_score` in descending order before the diversity selection process begins. This means that, by default, higher-scoring tracks are preferred.
        2.  **Implicit Secondary Factor**: If multiple tracks could equally satisfy a diversity requirement (and have identical `judge_score`s, or during a secondary pass to fill remaining slots), the track that appears earlier in the pre-sorted list will be selected. This is a deterministic outcome of iterating through the sorted list.
    *   This approach is simple, transparent, and generally prioritizes overall track quality (as defined by `judge_score`) when resolving diversity ties. More complex tie-breaking mechanisms (e.g., considering secondary scores, randomness) can be explored post-MVP if deemed necessary.

--- 