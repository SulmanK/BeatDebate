import logging
from typing import List, Dict, Any, Optional

from pydantic import ValidationError

# Import the correct MusicRecommenderState model
from ..models.agent_models import MusicRecommenderState
from ..models.recommendation_models import TrackRecommendation

class JudgeAgent:
    """
    The JudgeAgent is responsible for making the final selection of music recommendations.
    It applies an evaluation framework from the PlannerAgent to candidate tracks from
    advocate agents, ensures diversity, and generates explanations.
    """
    def __init__(self, llm_client: Any = None):
        """
        Initializes the JudgeAgent.

        Args:
            llm_client (Any, optional): An optional LLM client for generating
                                       more nuanced explanations. Defaults to None.
        """
        self.logger = logging.getLogger(__name__)
        self.llm_client = llm_client
        self.logger.info("JudgeAgent initialized.")

    async def evaluate_and_select(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Main method to orchestrate the evaluation and selection process.
        Updates state.final_recommendations and state.reasoning_log.

        Args:
            state (MusicRecommenderState): The shared state object containing candidate 
                                           tracks and the planning strategy.

        Returns:
            MusicRecommenderState: The updated state with final recommendations.
        """
        self.logger.info("JudgeAgent: Starting evaluation and selection process.")
        
        # Consolidate all candidate tracks from different advocate agents
        # Ensure that these lists contain dictionaries that can be parsed into TrackRecommendation model
        all_candidate_dicts: List[Dict] = [] # Should be List[TrackRecommendation] after parsing
        all_candidate_dicts.extend(state.genre_mood_recommendations)
        all_candidate_dicts.extend(state.discovery_recommendations)

        if not all_candidate_dicts:
            self.logger.warning("JudgeAgent: No candidate tracks provided for evaluation.")
            state.final_recommendations = []
            state.reasoning_log.append("JudgeAgent: No candidate tracks provided for evaluation.")
            return state

        evaluation_framework = state.planning_strategy.get("evaluation_framework")
        if not evaluation_framework or not evaluation_framework.get("primary_weights"):
            self.logger.error("JudgeAgent: Evaluation framework or primary_weights missing from planning_strategy.")
            state.reasoning_log.append("JudgeAgent: Critical error - Evaluation framework or primary_weights missing.")
            # Decide on fallback: return empty, or try to rank without weights? For now, returning empty.
            state.final_recommendations = []
            return state

        primary_weights = evaluation_framework.get("primary_weights", {})
        diversity_targets = evaluation_framework.get("diversity_targets", {})

        # 1. Parse and Apply evaluation framework (scoring)
        scored_candidates = self._apply_evaluation_framework(
            all_candidate_dicts, 
            primary_weights
        )
        
        if not scored_candidates:
            self.logger.warning("JudgeAgent: No candidates were successfully scored.")
            state.final_recommendations = []
            # Log already happened in _apply_evaluation_framework if all failed parse
            return state

        # 2. Ensure diversity
        sorted_candidates = sorted(scored_candidates, key=lambda x: x.judge_score if x.judge_score is not None else 0.0, reverse=True)
        
        selected_diverse_tracks = self._ensure_diversity(
            sorted_candidates, 
            diversity_targets,
            num_recommendations=3 # Defaulting to 3 recommendations as per general architecture
        )
        
        # 3. Generate final explanations
        final_selections_with_explanations = await self._generate_final_explanations(
            selected_diverse_tracks, 
            evaluation_framework # Pass the whole framework for context in explanations
        )
        
        # Convert TrackRecommendation objects to dictionaries for state compatibility
        state.final_recommendations = [track.model_dump() for track in final_selections_with_explanations]
        log_message = f"JudgeAgent: Successfully selected {len(final_selections_with_explanations)} tracks out of {len(all_candidate_dicts)} candidates."
        self.logger.info(log_message)
        state.reasoning_log.append(log_message)
        
        return state

    def _apply_evaluation_framework(self, candidates: List[Dict], primary_weights: Dict[str, float]) -> List[TrackRecommendation]:
        """
        Parses candidate track dictionaries into TrackRecommendation models,
        applies the scoring criteria, and adds a 'judge_score'.

        Args:
            candidates (List[Dict]): Raw candidate tracks.
            primary_weights (Dict[str, float]): Criteria and their weights.

        Returns:
            List[TrackRecommendation]: Scored tracks as TrackRecommendation objects.
                                       Tracks failing parsing are omitted.
        """
        self.logger.debug(f"Applying evaluation framework with weights: {primary_weights} to {len(candidates)} candidates.")
        scored_recommendations: List[TrackRecommendation] = []
        for track_dict in candidates:
            try:
                track_model = TrackRecommendation(**track_dict)
            except ValidationError as e:
                self.logger.warning(
                    f"Failed to parse track data into TrackRecommendation model. "
                    f"Track: {track_dict.get('title', 'N/A')}. Error: {e}"
                )
                continue  # Skip this track

            current_score = 0.0
            for criterion, weight in primary_weights.items():
                track_attribute_value: Optional[Any] = None
                
                # Check if criterion is a direct attribute of TrackRecommendation
                if hasattr(track_model, criterion):
                    track_attribute_value = getattr(track_model, criterion)
                
                # If not a direct attribute, check in additional_scores
                if track_attribute_value is None and track_model.additional_scores:
                    track_attribute_value = track_model.additional_scores.get(criterion)

                if track_attribute_value is not None:
                    try:
                        current_score += float(track_attribute_value) * weight
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"Could not convert value '{track_attribute_value}' "
                            f"for criterion '{criterion}' to float for track "
                            f"'{track_model.title}'. Error: {e}"
                        )
                else:
                    self.logger.debug(
                        f"Criterion '{criterion}' not found for track "
                        f"'{track_model.title}'."
                    )
            
            track_model.judge_score = current_score
            scored_recommendations.append(track_model)
            self.logger.debug(
                f"Track '{track_model.title}' scored: {current_score:.4f}"
            )
        
        if not scored_recommendations and candidates:
            self.logger.error(
                "All candidate tracks failed parsing or scoring. "
                "Please check track data structure and evaluation criteria."
            )
        return scored_recommendations

    def _ensure_diversity(self, candidates: List[TrackRecommendation], diversity_targets: Dict[str, int], num_recommendations: int = 3) -> List[TrackRecommendation]:
        """
        Selects tracks to meet diversity targets (e.g., number of unique genres, eras).
        Prioritizes higher-scoring tracks (candidates are expected to be pre-sorted by `judge_score`).

        Args:
            candidates (List[TrackRecommendation]): Pre-sorted list of candidate tracks with 'judge_score'.
            diversity_targets (Dict[str, int]): Dictionary of diversity criteria (e.g., {"genres": 2, "era": 1}).
                                                 The value indicates the target number of unique items for that key.
            num_recommendations (int): The desired number of final recommendations.

        Returns:
            List[TrackRecommendation]: A list of selected tracks meeting diversity goals as best as possible.
        """
        self.logger.debug(f"Ensuring diversity for {num_recommendations} recommendations with targets: {diversity_targets} from {len(candidates)} candidates.")
        
        # Return early for empty cases or no diversity targets
        if not candidates:
            return []
            
        # If no diversity targets or no attributes, just return top N candidates by score
        if not diversity_targets or not diversity_targets.get("attributes"):
            self.logger.debug("No diversity targets or attributes. Returning top N candidates based on score.")
            sorted_candidates = sorted(candidates, key=lambda x: x.judge_score if x.judge_score is not None else 0.0, reverse=True)
            return sorted_candidates[:num_recommendations]

        # Make sure candidates are sorted by score in descending order
        sorted_candidates = sorted(candidates, key=lambda x: x.judge_score if x.judge_score is not None else 0.0, reverse=True)
        selected_tracks: List[TrackRecommendation] = []
        
        # Tracks unique values seen for each diversity dimension, e.g., {"genres": {"Rock", "Pop"}}
        met_diversity_values: Dict[str, set] = {}
        
        # Pass 1: Greedily select tracks that fulfill a new diversity criterion from highest scores
        for attribute in diversity_targets.get("attributes", []):
            if len(selected_tracks) >= num_recommendations:
                break
                
            # Initialize set for this attribute if not already present
            if attribute not in met_diversity_values:
                met_diversity_values[attribute] = set()
                
            # Go through each candidate for this attribute
            for track in sorted_candidates:
                if len(selected_tracks) >= num_recommendations:
                    break
                    
                if track in selected_tracks:
                    continue
                    
                # Track value for this diversity attribute
                track_value_for_key: Any = None
                if hasattr(track, attribute):
                    track_value_for_key = getattr(track, attribute)
                
                if track_value_for_key is not None:
                    # Handle list-based attributes (like genres) and single attributes (like era)
                    values_to_check = track_value_for_key if isinstance(track_value_for_key, list) else [track_value_for_key]
                    
                    # Check if any value is new for this attribute
                    for value in values_to_check:
                        if value and value not in met_diversity_values[attribute]:
                            selected_tracks.append(track)
                            # Add all values from this track to the seen set
                            for v in values_to_check:
                                if v:  # Only add non-empty values
                                    met_diversity_values[attribute].add(v)
                            break
        
        self.logger.debug(f"After Pass 1 (greedy diversity pick): {len(selected_tracks)} selected. Met diversity: { {k: len(v) for k,v in met_diversity_values.items()} }")

        # Pass 2: If not enough tracks selected, fill remaining slots with highest-scored available tracks
        remaining_candidates = [c for c in sorted_candidates if c not in selected_tracks]
        while len(selected_tracks) < num_recommendations and remaining_candidates:
            track_to_add = remaining_candidates.pop(0)  # Take highest scored remaining track
            selected_tracks.append(track_to_add)
            self.logger.debug(f"Pass 2: Adding track '{track_to_add.title}' (score: {track_to_add.judge_score}) to fill slots.")
            
        # Final sort by score to ensure order is maintained
        final_selected = sorted(selected_tracks, key=lambda x: x.judge_score if x.judge_score is not None else 0.0, reverse=True)
        final_selected_count = len(final_selected)
        
        self.logger.info(f"JudgeAgent: Selected {final_selected_count} diverse tracks. Met diversity values: {met_diversity_values}")
        return final_selected[:num_recommendations]

    async def _generate_final_explanations(
        self,
        selections: List[TrackRecommendation],
        evaluation_framework: Dict
    ) -> List[TrackRecommendation]:
        """
        Generates a human-readable explanation for each selected track.

        Args:
            selections (List[TrackRecommendation]): Final selected tracks.
            evaluation_framework (Dict): PlannerAgent's evaluation framework.

        Returns:
            List[TrackRecommendation]: Selections with 'explanation' field populated.
        """
        self.logger.debug(
            f"Generating final explanations for {len(selections)} selections."
        )
        primary_weights = evaluation_framework.get("primary_weights", {})
        diversity_targets = evaluation_framework.get("diversity_targets", {}) # For context

        for track_model in selections: # Iterate through TrackRecommendation objects
            explanation_parts = []
            judge_score_val = track_model.judge_score if track_model.judge_score is not None else 0.0

            explanation_parts.append(
                f"The track '{track_model.title}' by '{track_model.artist}' "
                f"(overall score: {judge_score_val:.2f}) was selected because:"
            )

            if primary_weights:
                has_criteria_explanation = False
                for criterion, weight in primary_weights.items():
                    value: Optional[Any] = None
                    # Check direct model attribute first
                    if hasattr(track_model, criterion):
                        value = getattr(track_model, criterion)
                    # Then check in additional_scores dictionary
                    elif track_model.additional_scores:
                        value = track_model.additional_scores.get(criterion)
                    
                    if value is not None and weight > 0: # Process only if value exists and weight is positive
                        try:
                            numeric_value = float(value)
                            contribution = numeric_value * weight
                            if contribution != 0: # Only mention if it made a non-zero contribution
                                explanation_parts.append(
                                    f"  - It performed well on '{criterion}', contributing {contribution:.2f} to its score "
                                    f"(value: {numeric_value}, weight: {weight:.2f})."
                                )
                                has_criteria_explanation = True
                        except (ValueError, TypeError):
                            # If value can't be float, it might be a boolean or string used differently
                            # For now, we only score floatable values. Log if needed.
                            self.logger.debug(f"Criterion '{criterion}' for track '{track_model.title}' has non-numeric value '{value}' or zero weight; not used in score contribution display.")
                if not has_criteria_explanation and primary_weights:
                     explanation_parts.append(
                         "  - Its overall score met the selection threshold based on the defined criteria."
                     )
            else:
                explanation_parts.append(
                    "  - It was identified as a strong candidate overall."
                )

            # Add diversity information if diversity_targets were considered
            if diversity_targets:
                diversity_contribution_parts = []
                for key in diversity_targets.keys(): # Iterate through targeted diversity keys
                    attr_value: Any = None
                    if hasattr(track_model, key):
                        attr_value = getattr(track_model, key)
                    
                    if attr_value is not None:
                        if isinstance(attr_value, list):
                            if attr_value: # If list is not empty
                                diversity_contribution_parts.append(f"{key}: {', '.join(map(str, attr_value))}")
                        elif isinstance(attr_value, (str, int, float, bool)):
                             diversity_contribution_parts.append(f"{key}: {str(attr_value)}")
                        # else: non-simple type, skip for this basic explanation
                
                if diversity_contribution_parts:
                    explanation_parts.append(
                        f"  - It contributes to recommendation diversity with attributes like: {'; '.join(diversity_contribution_parts)}."
                    )

            track_model.explanation = "\n".join(explanation_parts)
            
            # Placeholder for potential LLM-based explanation enhancement (from design doc)
            if self.llm_client:
                # Construct prompt using details from track_model, primary_weights, etc.
                # llm_explanation_prompt = f"Generate a concise, compelling explanation..."
                # self.logger.debug(f"Attempting LLM explanation for '{track_model.title}'.")
                # try:
                #     generated_explanation = await self.llm_client.generate_text(llm_explanation_prompt)
                #     track_model.explanation = generated_explanation # Overwrite template explanation
                #     self.logger.info(f"LLM generated explanation for '{track_model.title}'.")
                # except Exception as e:
                #     self.logger.error(f"Error generating explanation with LLM for '{track_model.title}': {e}. Using template-based explanation.")
                pass # Keep template-based explanation for MVP
            
        return selections

# Example Usage (conceptual):
# async def main():
#     # Setup basic logging
#     logging.basicConfig(level=logging.DEBUG)
#
#     # Create a mock state
#     mock_state = MusicRecommenderState(user_query="Find me some focus music")
#     mock_state.planning_strategy = {
#         "evaluation_framework": {
#             "primary_weights": {
#                 "concentration_friendliness_score": 0.5,
#                 "novelty_score": 0.3,
#                 "quality_score": 0.2
#             },
#             "diversity_targets": {"genre": 2, "era": 1}
#         }
#     }
#     mock_state.genre_mood_recommendations = [
#         {"title": "Ambient Wonder", "artist": " спокойствие", "id": "gm1", "source": "synth", "genres": ["Ambient"], "era": "2020s", "concentration_friendliness_score": 0.9, "novelty_score": 0.5, "quality_score": 0.8},
#         {"title": "Lo-Fi Beats", "artist": " Chill Cat", "id": "gm2", "source": "lofi_src", "genres": ["Lo-Fi Hip Hop"], "era": "2020s", "concentration_friendliness_score": 0.8, "novelty_score": 0.6, "quality_score": 0.7, "additional_scores": {"instrumentalness": 0.9}},
#     ]
#     mock_state.discovery_recommendations = [
#         {"title": "Forgotten Gem", "artist": "Obscurity", "id": "d1", "source": "bandcamp", "genres": ["IDM"], "era": "1990s", "concentration_friendliness_score": 0.7, "novelty_score": 0.9, "quality_score": 0.85},
#         {"title": "Ancient Echoes", "artist": "Etherea", "id": "d2", "source": "archive", "genres": ["Ambient"], "era": "1970s", "concentration_friendliness_score": 0.6, "novelty_score": 0.8, "quality_score": 0.7},
#         {"title": "Post-Rock Dreams", "artist": "Skyward", "id": "d3", "source": "local", "genres": ["Post-Rock"], "era": "2010s", "concentration_friendliness_score": 0.85, "novelty_score": 0.4, "quality_score": 0.9},
#     ]
#
#     # Initialize JudgeAgent
#     judge = JudgeAgent()
#
#     # Evaluate
#     updated_state = await judge.evaluate_and_select(mock_state)
#
#     print("\nFinal Recommendations:")
#     for rec in updated_state.final_recommendations:
#         print(f"  Title: {rec['title']}, Artist: {rec['artist']}, Judge Score: {rec['judge_score']:.2f}")
#         print(f"  Explanation: {rec['explanation']}")
#         print("  ----")
#     
#     print("\nReasoning Log:")
#     for log_entry in updated_state.reasoning_log:
#         print(f"  - {log_entry}")
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main()) 