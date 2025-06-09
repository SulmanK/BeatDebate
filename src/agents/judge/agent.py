"""
Refactored Judge Agent

Streamlined judge agent using modular components for better maintainability,
testability, and adherence to the Single Responsibility Principle.

This refactored version reduces the main agent to orchestration logic while
delegating specialized responsibilities to focused components.
"""

from typing import Dict, List, Any, Tuple
import structlog

from src.models.agent_models import MusicRecommenderState, AgentConfig
from src.models.recommendation_models import TrackRecommendation
from src.services.api_service import APIService
from src.services.metadata_service import MetadataService
from src.agents.base_agent import BaseAgent
from src.agents.components import QualityScorer

# Import the new modular components
from .components import (
    RankingEngine,
    ExplanationGenerator,
    CandidateSelector,
    DiversityOptimizer
)

logger = structlog.get_logger(__name__)


class JudgeAgent(BaseAgent):
    """
    Refactored Judge Agent with modular component architecture.
    
    Responsibilities:
    - Orchestrate the judge workflow
    - Coordinate between specialized components
    - Manage state transitions
    - Handle error recovery
    
    Delegates specialized tasks to:
    - CandidateSelector: Candidate collection and filtering
    - RankingEngine: Scoring and ranking algorithms
    - DiversityOptimizer: Diversity optimization
    - ExplanationGenerator: Explanation generation
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client,
        api_service: APIService,
        metadata_service: MetadataService,
        rate_limiter=None,
        session_manager=None
    ):
        """
        Initialize refactored judge agent with modular components.
        
        Args:
            config: Agent configuration
            llm_client: LLM client for explanations
            api_service: Unified API service
            metadata_service: Unified metadata service
            rate_limiter: Rate limiter for LLM API calls
            session_manager: SessionManagerService for candidate pool retrieval
        """
        super().__init__(
            config=config, 
            llm_client=llm_client, 
            api_service=api_service,
            metadata_service=metadata_service,
            rate_limiter=rate_limiter
        )
        
        # Initialize modular components
        self.candidate_selector = CandidateSelector(session_manager=session_manager)
        self.ranking_engine = RankingEngine()
        self.diversity_optimizer = DiversityOptimizer()
        self.explanation_generator = ExplanationGenerator(self.llm_utils)
        
        # Shared components from parent
        self.quality_scorer = QualityScorer()
        
        # Configuration (will be updated based on intent)
        self.final_recommendations = 20  # Default fallback
        
        self.logger.info("JudgeAgentRefactored initialized with modular components")
    
    async def process(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Main processing workflow using modular components.
        
        Args:
            state: Current workflow state with candidate recommendations
            
        Returns:
            Updated state with final ranked recommendations
        """
        try:
            self.logger.info("Starting refactored judge agent processing")
            
            # Phase 1: Candidate Collection and Filtering
            candidates = await self._collect_and_filter_candidates(state)
            if not candidates:
                self.logger.warning("No candidates available for evaluation")
                state.final_recommendations = []
                return state
            
            # Phase 2: Scoring and Ranking
            scored_candidates = await self._score_and_rank_candidates(candidates, state)
            if not scored_candidates:
                self.logger.warning("No candidates survived scoring")
                state.final_recommendations = []
                return state
            
            # Phase 3: Diversity Optimization
            diverse_selections = self._apply_diversity_optimization(scored_candidates, state)
            if not diverse_selections:
                self.logger.warning("Diversity optimization failed")
                # Fallback to top scored candidates
                diverse_selections = [track for track, _ in scored_candidates[:self.final_recommendations]]
            
            # Phase 4: Explanation Generation
            explained_selections = await self._generate_explanations(diverse_selections, state)
            
            # Update state with final results
            state.final_recommendations = explained_selections
            state.judge_metadata = self._create_judge_metadata(
                candidates, scored_candidates, diverse_selections, explained_selections
            )
            
            self.logger.info(
                "Judge processing completed successfully",
                input_candidates=len(candidates),
                scored_candidates=len(scored_candidates),
                final_recommendations=len(explained_selections)
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Judge agent processing failed: {e}")
            # Provide fallback recommendations if possible
            state.final_recommendations = self._create_fallback_recommendations(state)
            return state
    
    async def _collect_and_filter_candidates(
        self, 
        state: MusicRecommenderState
    ) -> List[TrackRecommendation]:
        """Phase 1: Collect and filter candidates using CandidateSelector."""
        try:
            candidates = await self.candidate_selector.collect_and_filter_candidates(state)
            
            # Validate candidates
            valid_candidates, validation_errors = self.candidate_selector.validate_candidates(candidates)
            
            if validation_errors:
                self.logger.warning(f"Candidate validation found {len(validation_errors)} issues")
            
            # Log candidate statistics
            stats = self.candidate_selector.get_candidate_statistics(valid_candidates)
            self.logger.info("Candidate collection completed", **stats)
            
            return valid_candidates
            
        except Exception as e:
            self.logger.error(f"Candidate collection failed: {e}")
            return []
    
    async def _score_and_rank_candidates(
        self,
        candidates: List[TrackRecommendation],
        state: MusicRecommenderState
    ) -> List[Tuple[TrackRecommendation, Dict[str, float]]]:
        """Phase 2: Score and rank candidates using RankingEngine."""
        try:
            # First, calculate additional scores for all candidates
            scored_candidates = []
            
            for candidate in candidates:
                try:
                    # Calculate contextual relevance
                    contextual_score = self.ranking_engine.calculate_contextual_relevance(
                        candidate, 
                        getattr(state, 'entities', {}),
                        getattr(state, 'intent_analysis', {})
                    )
                    
                    # Start with existing agent scores or defaults
                    agent_scores = getattr(candidate, '_scores', {})
                    if not agent_scores:
                        agent_scores = {
                            'quality': 0.5,
                            'novelty': 0.5,
                            'source_priority': 0.5
                        }
                    
                    # Add contextual relevance
                    agent_scores['contextual_relevance'] = contextual_score
                    
                    scored_candidates.append((candidate, agent_scores))
                    
                except Exception as e:
                    self.logger.warning(f"Individual candidate scoring failed: {e}")
                    # Use fallback scores
                    fallback_scores = {
                        'quality': 0.5,
                        'contextual_relevance': 0.3,
                        'novelty': 0.5,
                        'source_priority': 0.5
                    }
                    scored_candidates.append((candidate, fallback_scores))
            
            # Now rank all candidates
            ranked_candidates = await self.ranking_engine.rank_candidates(
                scored_candidates, state
            )
            
            self.logger.info(f"Scoring and ranking completed: {len(ranked_candidates)} candidates")
            return ranked_candidates
            
        except Exception as e:
            self.logger.error(f"Scoring and ranking failed: {e}")
            return []
    
    def _apply_diversity_optimization(
        self,
        scored_candidates: List[Tuple[TrackRecommendation, Dict[str, float]]],
        state: MusicRecommenderState
    ) -> List[TrackRecommendation]:
        """Phase 3: Apply diversity optimization using DiversityOptimizer."""
        try:
            # Get intent-specific final recommendations count
            target_count = self._get_target_recommendations_count(state)
            
            diverse_selections = self.diversity_optimizer.select_with_diversity(
                scored_candidates, state, target_count
            )
            
            # Calculate diversity metrics
            diversity_metrics = self.diversity_optimizer.calculate_diversity_score(diverse_selections)
            self.logger.info("Diversity optimization completed", target_count=target_count, **diversity_metrics)
            
            return diverse_selections
            
        except Exception as e:
            self.logger.error(f"Diversity optimization failed: {e}")
            return []
    
    async def _generate_explanations(
        self,
        selections: List[TrackRecommendation],
        state: MusicRecommenderState
    ) -> List[TrackRecommendation]:
        """Phase 4: Generate explanations using ExplanationGenerator."""
        try:
            explained_selections = await self.explanation_generator.generate_explanations(
                selections, state
            )
            
            self.logger.info(f"Explanation generation completed for {len(explained_selections)} tracks")
            return explained_selections
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            # Return selections with basic explanations
            for i, selection in enumerate(selections, 1):
                if not hasattr(selection, 'explanation') or not selection.explanation:
                    selection.explanation = f"Great track #{i} that matches your musical preferences!"
            return selections
    
    def _create_judge_metadata(
        self,
        candidates: List[TrackRecommendation],
        scored_candidates: List[Tuple[TrackRecommendation, Dict[str, float]]],
        diverse_selections: List[TrackRecommendation],
        final_selections: List[TrackRecommendation]
    ) -> Dict[str, Any]:
        """Create metadata about the judge's decision process."""
        try:
            # Calculate selection statistics
            candidate_stats = self.candidate_selector.get_candidate_statistics(candidates)
            diversity_metrics = self.diversity_optimizer.calculate_diversity_score(diverse_selections)
            
            # Calculate score statistics
            if scored_candidates:
                scores = [scores.get('final_score', 0.5) for _, scores in scored_candidates]
                score_stats = {
                    'mean_score': sum(scores) / len(scores),
                    'max_score': max(scores),
                    'min_score': min(scores),
                    'score_range': max(scores) - min(scores)
                }
            else:
                score_stats = {}
            
            return {
                'processing_summary': {
                    'input_candidates': len(candidates),
                    'scored_candidates': len(scored_candidates),
                    'diverse_selections': len(diverse_selections),
                    'final_selections': len(final_selections)
                },
                'candidate_statistics': candidate_stats,
                'diversity_metrics': diversity_metrics,
                'score_statistics': score_stats,
                'processing_timestamp': self._get_timestamp()
            }
            
        except Exception as e:
            self.logger.warning(f"Judge metadata creation failed: {e}")
            return {
                'processing_summary': {
                    'final_selections': len(final_selections)
                },
                'error': str(e)
            }
    
    def _create_fallback_recommendations(
        self, 
        state: MusicRecommenderState
    ) -> List[TrackRecommendation]:
        """Create fallback recommendations when main processing fails."""
        try:
            # Try to get any available candidates from state
            fallback_candidates = []
            
            # Collect from genre_mood_agent
            if hasattr(state, 'genre_mood_recommendations') and state.genre_mood_recommendations:
                fallback_candidates.extend(state.genre_mood_recommendations[:10])
            
            # Collect from discovery_agent
            if hasattr(state, 'discovery_recommendations') and state.discovery_recommendations:
                fallback_candidates.extend(state.discovery_recommendations[:10])
            
            # Add basic reasoning
            for i, candidate in enumerate(fallback_candidates[:self.final_recommendations], 1):
                if not hasattr(candidate, 'explanation') or not candidate.explanation:
                    candidate.explanation = f"Recommendation #{i} from our music discovery system."
            
            self.logger.info(f"Created {len(fallback_candidates)} fallback recommendations")
            return fallback_candidates[:self.final_recommendations]
            
        except Exception as e:
            self.logger.error(f"Fallback recommendation creation failed: {e}")
            return []
    
    def _get_target_recommendations_count(self, state: MusicRecommenderState) -> int:
        """Get the target recommendation count based on intent and configuration."""
        try:
            # Try to get from state's intent analysis first
            intent_analysis = getattr(state, 'intent_analysis', {})
            intent = intent_analysis.get('intent', 'discovery')
            
            # Check if the state has discovery parameters
            if hasattr(state, 'discovery_params') and state.discovery_params:
                final_recs = state.discovery_params.get('final_recommendations')
                if final_recs and isinstance(final_recs, int) and final_recs > 0:
                    self.logger.debug(f"Using final_recommendations from state: {final_recs}")
                    return final_recs
            
            # Intent-specific defaults
            intent_defaults = {
                'artist_similarity': 20,
                'by_artist': 20,  # Fixed: Changed from 25 to 20 to match expected track count
                'discovery': 20,
                'genre_mood': 20,
                'contextual': 20,
                'hybrid': 20
            }
            
            target_count = intent_defaults.get(intent, self.final_recommendations)
            self.logger.debug(f"Using intent-specific target count for {intent}: {target_count}")
            return target_count
            
        except Exception as e:
            self.logger.warning(f"Failed to determine target recommendations count: {e}")
            return self.final_recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        try:
            from datetime import datetime
            return datetime.now().isoformat()
        except Exception:
            return "unknown"
    
    # Compatibility methods for existing codebase
    async def evaluate_and_select(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """Compatibility method that delegates to process()."""
        return await self.process(state)
    
    async def run(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """Compatibility method that delegates to process()."""
        return await self.process(state)
    
    def _detect_hybrid_subtype(self, state: MusicRecommenderState) -> str:
        """
        Compatibility method for hybrid subtype detection.
        
        Note: This logic could be moved to QueryAnalysisUtils or PlannerAgent
        in future refactoring iterations.
        """
        try:
            entities = getattr(state, 'entities', {})
            
            # Check for genre combinations
            genres = entities.get('genres', [])
            if len(genres) >= 2:
                return 'genre_hybrid'
            
            # Check for mood + genre combinations
            moods = entities.get('moods', [])
            if genres and moods:
                return 'mood_genre_hybrid'
            
            # Check for artist + style combinations
            artists = entities.get('artists', [])
            if artists and (genres or moods):
                return 'artist_style_hybrid'
            
            return 'standard'
            
        except Exception as e:
            self.logger.warning(f"Hybrid subtype detection failed: {e}")
            return 'standard' 