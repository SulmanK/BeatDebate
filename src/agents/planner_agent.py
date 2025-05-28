"""
PlannerAgent for BeatDebate Multi-Agent Music Recommendation System

Strategic coordinator and planning engine that demonstrates sophisticated
agentic planning behavior for the AgentX competition.
"""

import json
import re
from typing import Dict, List, Any, Optional
import structlog

from .base_agent import BaseAgent
from .entity_recognizer import EnhancedEntityRecognizer
from .conversation_context import ConversationContextManager
from .query_understanding import QueryUnderstandingEngine, QueryUnderstanding
from ..models.agent_models import MusicRecommenderState, AgentConfig

logger = structlog.get_logger(__name__)


class PlannerAgent(BaseAgent):
    """
    Master planning agent that coordinates the entire music recommendation workflow.
    
    Enhanced with Pure LLM Query Understanding for superior query interpretation.
    
    Demonstrates agentic planning behavior required for AgentX competition:
    - Strategic task decomposition with LLM-powered query understanding
    - Resource allocation and coordination based on intent analysis
    - Success criteria definition with confidence scoring
    - Adaptive execution monitoring
    """
    
    def __init__(self, config: AgentConfig, gemini_client=None):
        """
        Initialize PlannerAgent with Pure LLM Query Understanding.
        
        Args:
            config: Agent configuration
            gemini_client: Gemini LLM client for reasoning
        """
        super().__init__(config)
        self.llm_client = gemini_client
        
        # Pure LLM Query Understanding Engine
        self.query_understanding_engine = QueryUnderstandingEngine(gemini_client)
        
        # Enhanced entity recognition and context management (legacy support)
        self.entity_recognizer = EnhancedEntityRecognizer(gemini_client)
        self.context_manager = ConversationContextManager()
        
        # Planning templates and patterns
        self.query_patterns = self._initialize_query_patterns()
        self.strategy_templates = self._initialize_strategy_templates()
        
        self.logger.info(
            "PlannerAgent initialized with Pure LLM Query Understanding Engine"
        )
    
    async def process(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Create comprehensive music discovery strategy using Pure LLM Query Understanding.
        
        Args:
            state: Current workflow state with user query
            
        Returns:
            Updated state with planning strategy and query understanding
        """
        self.add_reasoning_step("Starting Pure LLM query understanding and strategic planning")
        
        try:
            # Step 1: Pure LLM Query Understanding
            query_understanding = await self.query_understanding_engine.understand_query(state.user_query)
            self.add_reasoning_step(
                f"Query understanding completed: {query_understanding.intent.value} "
                f"(confidence: {query_understanding.confidence:.2f})"
            )
            
            # Step 2: Generate strategy from understanding
            planning_strategy = await self._generate_strategy_from_understanding(
                query_understanding, state.user_query
            )
            self.add_reasoning_step("Strategy generated from query understanding")
            
            # Step 3: Legacy entity extraction for backward compatibility
            entities, legacy_intent = await self._analyze_user_query_enhanced(
                state.user_query, state.conversation_context, state.session_id
            )
            self.add_reasoning_step("Legacy entity extraction completed for compatibility")
            
            # Step 4: Resolve session references if any
            if state.session_id and entities:
                entities = await self.context_manager.resolve_session_references(
                    state.session_id, entities
                )
                self.add_reasoning_step("Session references resolved")
            
            # Step 5: Update conversation context
            if state.session_id:
                await self.context_manager.update_session_context(
                    state.session_id, state.user_query, entities
                )
                self.add_reasoning_step("Conversation context updated")
            
            # Step 6: Merge LLM understanding with legacy data
            enhanced_strategy = await self._merge_understanding_with_legacy(
                planning_strategy, entities, legacy_intent
            )
            self.add_reasoning_step("LLM understanding merged with legacy systems")
            
            # Update state with comprehensive data
            state.query_understanding = query_understanding
            state.entities = entities  # Legacy compatibility
            state.intent_analysis = self._convert_understanding_to_legacy_intent(query_understanding)
            state.planning_strategy = enhanced_strategy
            state.coordination_strategy = enhanced_strategy.get("coordination_strategy", {})
            state.confidence = query_understanding.confidence
            
            # Add query understanding reasoning
            understanding_reasoning = {
                "method": "pure_llm",
                "intent": query_understanding.intent.value,
                "confidence": query_understanding.confidence,
                "artists_found": len(query_understanding.artists),
                "genres_found": len(query_understanding.genres),
                "moods_found": len(query_understanding.moods),
                "primary_agent": query_understanding.primary_agent,
                "reasoning": query_understanding.reasoning
            }
            state.entity_reasoning.append(understanding_reasoning)
            
            log_message = (
                f"PlannerAgent: Pure LLM understanding - {query_understanding.intent.value} "
                f"(confidence: {query_understanding.confidence:.2f}, "
                f"primary_agent: {query_understanding.primary_agent})"
            )
            state.reasoning_log.append(log_message)
            
            self.logger.info(
                "Pure LLM strategic planning completed",
                intent=query_understanding.intent.value,
                confidence=query_understanding.confidence,
                primary_agent=query_understanding.primary_agent,
                artists=query_understanding.artists,
                genres=query_understanding.genres,
                strategy_components=len(enhanced_strategy)
            )
            
            return state
            
        except Exception as e:
            self.logger.error("Pure LLM strategic planning failed", error=str(e))
            state.reasoning_log.append(f"PlannerAgent ERROR: {str(e)}")
            # Fall back to legacy planning if Pure LLM fails
            return await self._fallback_legacy_planning(state)
    
    async def _generate_strategy_from_understanding(
        self, 
        understanding: QueryUnderstanding, 
        user_query: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive strategy from Pure LLM query understanding.
        
        Args:
            understanding: Query understanding from LLM
            user_query: Original user query
            
        Returns:
            Complete planning strategy
        """
        # Convert understanding to task analysis format
        task_analysis = {
            "primary_goal": understanding.intent.value,
            "confidence_level": understanding.confidence,
            "complexity_level": "high" if understanding.confidence > 0.8 else "medium",
            "context_factors": understanding.activities + understanding.moods,
            "mood_indicators": understanding.moods,
            "genre_hints": understanding.genres,
            "artists_mentioned": understanding.artists,
            "similarity_type": understanding.similarity_type.value if understanding.similarity_type else None,
            "exploration_level": understanding.exploration_level,
            "temporal_context": understanding.temporal_context,
            "energy_level": understanding.energy_level,
            "reasoning": understanding.reasoning
        }
        
        # Generate coordination strategy based on understanding
        coordination_strategy = {
            "primary_agent": understanding.primary_agent,
            "agent_weights": understanding.agent_weights or self._default_agent_weights(understanding),
            "execution_order": self._determine_execution_order(understanding),
            "search_strategy": understanding.search_strategy or {},
            "parallel_execution": understanding.confidence > 0.8,  # High confidence allows parallel
            "fallback_strategy": self._generate_fallback_strategy(understanding)
        }
        
        # Create evaluation framework
        evaluation_framework = {
            "primary_weights": self._generate_evaluation_weights(understanding),
            "quality_thresholds": self._generate_quality_thresholds(understanding),
            "diversity_requirements": self._generate_diversity_requirements(understanding),
            "success_criteria": self._generate_success_criteria(understanding),
            "confidence_threshold": max(0.7, understanding.confidence - 0.1)
        }
        
        # Setup execution monitoring
        execution_monitoring = {
            "key_metrics": self._generate_key_metrics(understanding),
            "checkpoints": self._generate_checkpoints(understanding),
            "adaptation_triggers": self._generate_adaptation_triggers(understanding),
            "quality_gates": self._generate_quality_gates(understanding)
        }
        
        return {
            "task_analysis": task_analysis,
            "coordination_strategy": coordination_strategy,
            "evaluation_framework": evaluation_framework,
            "execution_monitoring": execution_monitoring,
            "query_understanding": {
                "intent": understanding.intent.value,
                "confidence": understanding.confidence,
                "artists": understanding.artists,
                "genres": understanding.genres,
                "moods": understanding.moods,
                "activities": understanding.activities,
                "similarity_type": understanding.similarity_type.value if understanding.similarity_type else None,
                "exploration_level": understanding.exploration_level,
                "temporal_context": understanding.temporal_context,
                "energy_level": understanding.energy_level,
                "original_query": understanding.original_query,
                "reasoning": understanding.reasoning
            }  # Store as dictionary to prevent serialization issues
        }
    
    def _default_agent_weights(self, understanding: QueryUnderstanding) -> Dict[str, float]:
        """Generate default agent weights based on understanding."""
        if understanding.intent.value == "artist_similarity":
            return {"discovery": 0.6, "genre_mood": 0.3, "judge": 0.1}
        elif understanding.intent.value == "genre_exploration":
            return {"genre_mood": 0.6, "discovery": 0.3, "judge": 0.1}
        elif understanding.intent.value == "discovery":
            return {"discovery": 0.7, "genre_mood": 0.2, "judge": 0.1}
        else:  # mood_matching, activity_context
            return {"genre_mood": 0.7, "discovery": 0.2, "judge": 0.1}
    
    def _determine_execution_order(self, understanding: QueryUnderstanding) -> List[str]:
        """Determine optimal agent execution order."""
        if understanding.intent.value == "artist_similarity":
            return ["discovery", "genre_mood", "judge"]
        elif understanding.intent.value == "genre_exploration":
            return ["genre_mood", "discovery", "judge"]
        else:
            return ["genre_mood", "discovery", "judge"]
    
    def _generate_evaluation_weights(self, understanding: QueryUnderstanding) -> Dict[str, float]:
        """Generate evaluation weights based on understanding."""
        if understanding.intent.value == "artist_similarity":
            return {
                "similarity_score": 0.4,
                "quality_score": 0.3,
                "novelty_score": 0.2,
                "diversity_score": 0.1
            }
        elif understanding.intent.value == "discovery":
            return {
                "novelty_score": 0.4,
                "quality_score": 0.3,
                "diversity_score": 0.2,
                "similarity_score": 0.1
            }
        else:  # Genre/Mood
            return {
                "genre_mood_fit": 0.4,
                "quality_score": 0.3,
                "diversity_score": 0.2,
                "novelty_score": 0.1
            }
    
    def _generate_quality_thresholds(self, understanding: QueryUnderstanding) -> Dict[str, float]:
        """Generate quality thresholds based on understanding."""
        if understanding.intent.value == "discovery":
            return {"min_quality": 0.3, "min_novelty": 0.6}
        else:
            return {"min_quality": 0.5, "min_novelty": 0.3}
    
    def _generate_diversity_requirements(self, understanding: QueryUnderstanding) -> Dict[str, Any]:
        """Generate diversity requirements based on understanding."""
        return {
            "max_same_artist": 2,
            "min_genres": 3 if understanding.intent.value == "discovery" else 2,
            "temporal_spread": understanding.exploration_level == "broad",
            "energy_variation": understanding.energy_level is not None
        }
    
    def _generate_success_criteria(self, understanding: QueryUnderstanding) -> Dict[str, Any]:
        """Generate success criteria based on understanding."""
        return {
            "min_tracks": 10,
            "confidence_threshold": understanding.confidence,
            "intent_fulfillment": understanding.intent.value,
            "entity_coverage": {
                "artists": len(understanding.artists) > 0,
                "genres": len(understanding.genres) > 0,
                "moods": len(understanding.moods) > 0
            }
        }
    
    def _generate_fallback_strategy(self, understanding: QueryUnderstanding) -> Dict[str, Any]:
        """Generate fallback strategy for low confidence scenarios."""
        return {
            "trigger_threshold": 0.3,
            "fallback_intent": "discovery",
            "fallback_agent": "genre_mood",
            "fallback_weights": {"genre_mood": 0.5, "discovery": 0.4, "judge": 0.1}
        }
    
    def _generate_key_metrics(self, understanding: QueryUnderstanding) -> List[str]:
        """Generate key metrics to monitor."""
        base_metrics = ["track_count", "diversity_score", "quality_score"]
        
        if understanding.intent.value == "artist_similarity":
            base_metrics.append("similarity_score")
        elif understanding.intent.value == "discovery":
            base_metrics.append("novelty_score")
        else:
            base_metrics.append("mood_fit_score")
            
        return base_metrics
    
    def _generate_checkpoints(self, understanding: QueryUnderstanding) -> List[Dict[str, Any]]:
        """Generate execution checkpoints."""
        return [
            {"stage": "entity_extraction", "required_confidence": 0.5},
            {"stage": "candidate_generation", "min_candidates": 20},
            {"stage": "quality_filtering", "min_quality": 0.4},
            {"stage": "final_selection", "target_count": 10}
        ]
    
    def _generate_adaptation_triggers(self, understanding: QueryUnderstanding) -> List[Dict[str, Any]]:
        """Generate adaptation triggers for dynamic strategy adjustment."""
        return [
            {"condition": "low_candidate_count", "threshold": 10, "action": "broaden_search"},
            {"condition": "low_quality_scores", "threshold": 0.3, "action": "adjust_filters"},
            {"condition": "poor_diversity", "threshold": 0.4, "action": "increase_exploration"}
        ]
    
    def _generate_quality_gates(self, understanding: QueryUnderstanding) -> List[Dict[str, Any]]:
        """Generate quality gates for execution control."""
        return [
            {"gate": "minimum_candidates", "threshold": 5, "action": "continue"},
            {"gate": "quality_threshold", "threshold": 0.3, "action": "filter"},
            {"gate": "diversity_check", "threshold": 0.3, "action": "rebalance"}
        ]
    
    async def _merge_understanding_with_legacy(
        self, 
        llm_strategy: Dict[str, Any], 
        legacy_entities: Dict[str, Any], 
        legacy_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge LLM understanding with legacy entity extraction for compatibility.
        
        Args:
            llm_strategy: Strategy from LLM understanding
            legacy_entities: Entities from legacy extraction
            legacy_intent: Intent from legacy analysis
            
        Returns:
            Enhanced strategy combining both approaches
        """
        # Use LLM strategy as base
        enhanced_strategy = llm_strategy.copy()
        
        # Enhance with legacy entity data where beneficial
        if legacy_entities.get("music_entities", {}).get("artists"):
            legacy_artists = legacy_entities["music_entities"]["artists"]
            
            # Get LLM artists from dictionary representation
            query_understanding = enhanced_strategy.get("query_understanding", {})
            llm_artists = query_understanding.get("artists", [])
                        
            # Merge artist lists, prioritizing LLM results
            if llm_artists:
                all_artists = list(set(llm_artists + legacy_artists))
                enhanced_strategy["task_analysis"]["artists_mentioned"] = all_artists
            else:
                # Fallback: use legacy artists if LLM artists are not available
                enhanced_strategy["task_analysis"]["artists_mentioned"] = legacy_artists
        
        # Add legacy session context if available
        if legacy_entities.get("conversation_entities", {}).get("session_references"):
            enhanced_strategy["coordination_strategy"]["session_context"] = (
                legacy_entities["conversation_entities"]["session_references"]
            )
        
        # Merge confidence scores
        llm_confidence = enhanced_strategy["task_analysis"]["confidence_level"]
        legacy_confidence = legacy_entities.get("confidence_scores", {}).get("overall", 0.0)
        
        # Use weighted average, favoring LLM
        combined_confidence = (llm_confidence * 0.8) + (legacy_confidence * 0.2)
        enhanced_strategy["task_analysis"]["confidence_level"] = combined_confidence
        
        return enhanced_strategy
    
    def _convert_understanding_to_legacy_intent(self, understanding: QueryUnderstanding) -> Dict[str, Any]:
        """Convert query understanding to legacy intent format for compatibility."""
        return {
            "primary_goal": understanding.intent.value,
            "complexity_level": "high" if understanding.confidence > 0.8 else "medium",
            "context_factors": understanding.activities + understanding.moods,
            "mood_indicators": understanding.moods,
            "genre_hints": understanding.genres,
            "confidence": understanding.confidence,
            "reasoning": understanding.reasoning
        }
    
    async def _fallback_legacy_planning(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Fallback to legacy planning if Pure LLM approach fails.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with legacy planning
        """
        self.add_reasoning_step("Falling back to legacy planning approach")
        
        try:
            # Use legacy enhanced planning
            return await self._fallback_basic_planning(state)
            
        except Exception as e:
            self.logger.error("Legacy fallback planning also failed", error=str(e))
            # Ultimate fallback - minimal planning
            state.planning_strategy = {
                "task_analysis": {"primary_goal": "music_discovery", "complexity_level": "simple"},
                "coordination_strategy": {"primary_agent": "genre_mood"},
                "evaluation_framework": {"primary_weights": {"quality_score": 1.0}},
                "execution_monitoring": {"key_metrics": ["track_count"]}
            }
            state.reasoning_log.append("PlannerAgent: Using minimal fallback planning")
            return state
    
    async def _analyze_user_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze user query for complexity, intent, and context factors.
        
        Args:
            user_query: User's music request
            
        Returns:
            Task analysis dictionary
        """
        system_prompt = (
            "You are a strategic music recommendation planner. Analyze the "
            "user's query to understand their intent, mood, and context."
            "\n\n"
            "Extract:\n"
            "1. Primary goal (what they want to achieve with music)\n"
            "2. Complexity level (simple/medium/complex)\n"
            "3. Context factors (activity, mood, setting)\n"
            "4. Mood indicators (energy level, emotional state)\n"
            "5. Genre hints (explicit or implicit preferences)\n\n"
            "Respond in JSON format."
        )

        user_prompt = f"""Analyze this music request:
"{user_query}"

Provide analysis in this JSON format:
{{
    "primary_goal": "brief description of main intent",
    "complexity_level": "simple|medium|complex",
    "context_factors": ["factor1", "factor2"],
    "mood_indicators": ["mood1", "mood2"],
    "genre_hints": ["genre1", "genre2"]
}}"""

        try:
            response = await self.call_llm(user_prompt, system_prompt)
            analysis = self._parse_json_response(response)
            
            # Validate and enhance analysis
            analysis = self._enhance_task_analysis(analysis, user_query)
            
            return analysis
            
        except Exception as e:
            self.logger.warning("LLM analysis failed, using pattern matching", error=str(e))
            return self._fallback_query_analysis(user_query)
    
    async def _analyze_user_query_enhanced(
        self, 
        user_query: str, 
        conversation_context: Optional[Dict] = None,
        session_id: Optional[str] = None
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Enhanced user query analysis with entity recognition and intent analysis.
        
        Args:
            user_query: User's music request
            conversation_context: Previous conversation context
            session_id: Session identifier for context retrieval
            
        Returns:
            Tuple of (entities, intent_analysis)
        """
        self.add_reasoning_step("Starting enhanced query analysis with entity recognition")
        
        try:
            # Get session context if available
            session_context = None
            if session_id:
                session_context = await self.context_manager.get_session_context(session_id)
            
            # Phase 2: Complex Query Decomposition
            query_decomposition = await self._decompose_complex_query(
                user_query, session_context
            )
            self.add_reasoning_step(f"Query decomposition: {query_decomposition.get('query_complexity', 'simple')}")
            
            # Phase 3: Extract entities using optimized recognizer
            entities = await self.entity_recognizer.extract_entities_optimized(
                user_query, session_context
            )
            
            # Phase 2: Enhanced entities with decomposition context
            entities = await self._enhance_entities_with_decomposition(
                entities, query_decomposition
            )
            
            # Perform intent analysis based on entities
            intent_analysis = await self._analyze_intent_from_entities(user_query, entities)
            
            # Phase 2: Add decomposition to intent analysis
            intent_analysis["query_decomposition"] = query_decomposition
            
            # Enhance with traditional analysis for backward compatibility
            traditional_analysis = await self._analyze_user_query(user_query)
            
            # Merge traditional analysis into intent analysis
            intent_analysis.update({
                "complexity_level": traditional_analysis.get("complexity_level", "medium"),
                "context_factors": traditional_analysis.get("context_factors", []),
                "mood_indicators": traditional_analysis.get("mood_indicators", []),
                "genre_hints": traditional_analysis.get("genre_hints", [])
            })
            
            self.add_reasoning_step(
                f"Enhanced analysis completed: {entities.get('confidence_scores', {}).get('overall', 0.0):.2f} confidence"
            )
            
            return entities, intent_analysis
            
        except Exception as e:
            self.logger.warning("Enhanced analysis failed, falling back to basic", error=str(e))
            # Fall back to basic analysis
            traditional_analysis = await self._analyze_user_query(user_query)
            
            # Create minimal entities structure
            minimal_entities = {
                "musical_entities": {"artists": {"primary": [], "similar_to": [], "avoid": []}, "genres": {"primary": traditional_analysis.get("genre_hints", [])}},
                "contextual_entities": {"moods": {"energy": [], "emotion": traditional_analysis.get("mood_indicators", [])}, "activities": {"mental": [], "physical": []}},
                "preference_entities": {"similarity_requests": [], "discovery_preferences": []},
                "conversation_entities": {"session_references": []},
                "confidence_scores": {"overall": 0.5},
                "extraction_method": "fallback"
            }
            
            return minimal_entities, traditional_analysis
    
    async def _analyze_intent_from_entities(self, user_query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user intent based on extracted entities.
        
        Args:
            user_query: Original user query
            entities: Extracted entities
            
        Returns:
            Intent analysis dictionary
        """
        # Extract key information from entities
        activities = entities.get("contextual_entities", {}).get("activities", {})
        similarity_requests = entities.get("preference_entities", {}).get("similarity_requests", [])
        discovery_prefs = entities.get("preference_entities", {}).get("discovery_preferences", [])
        
        # Determine primary intent
        primary_intent = "music_discovery"  # default
        
        if any(activities.get("mental", [])):
            primary_intent = "focus_music"
        elif any(activities.get("physical", [])):
            primary_intent = "workout_music"
        elif similarity_requests:
            primary_intent = "similarity_exploration"
        elif any("underground" in str(pref) for pref in discovery_prefs):
            primary_intent = "underground_discovery"
        
        # Determine activity context
        activity_context = "general"
        if activities.get("mental"):
            activity_context = activities["mental"][0]
        elif activities.get("physical"):
            activity_context = activities["physical"][0]
        elif activities.get("social"):
            activity_context = activities["social"][0]
        
        # Determine exploration openness
        exploration_openness = 0.5  # default
        if any("underground" in str(pref) for pref in discovery_prefs):
            exploration_openness = 0.8
        elif any("mainstream" in str(pref) for pref in discovery_prefs):
            exploration_openness = 0.3
        
        # Determine specificity level
        total_entities = self._count_extracted_entities(entities)
        if total_entities > 5:
            specificity_level = 0.8
        elif total_entities > 2:
            specificity_level = 0.6
        else:
            specificity_level = 0.3
        
        return {
            "primary_intent": primary_intent,
            "activity_context": activity_context,
            "exploration_openness": exploration_openness,
            "specificity_level": specificity_level,
            "primary_goal": primary_intent.replace("_", " "),
            "entity_confidence": entities.get("confidence_scores", {}).get("overall", 0.5)
        }
    
    def _count_extracted_entities(self, entities: Dict[str, Any]) -> int:
        """Count total number of extracted entities."""
        count = 0
        
        for category in ["musical_entities", "contextual_entities", "preference_entities"]:
            if category in entities:
                for entity_type, entity_data in entities[category].items():
                    if isinstance(entity_data, dict):
                        for sub_type, items in entity_data.items():
                            if isinstance(items, list):
                                count += len(items)
                    elif isinstance(entity_data, list):
                        count += len(entity_data)
        
        return count
    
    async def _decompose_complex_query(
        self, 
        user_query: str, 
        session_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Decompose complex queries into manageable components.
        
        Args:
            user_query: User's music request
            session_context: Session context for reference resolution
            
        Returns:
            Query decomposition analysis
        """
        # Detect query complexity patterns
        complexity_indicators = {
            "multi_faceted": ["but", "and", "with", "for", "during"],
            "conversational_refinement": ["like the last", "more like", "similar to that", "but more", "but less"],
            "style_modification": ["but jazzier", "more upbeat", "less heavy", "with more", "without the"],
            "activity_context": ["for working out", "while studying", "for the gym", "during", "when"]
        }
        
        query_lower = user_query.lower()
        detected_patterns = []
        
        for pattern_type, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                detected_patterns.append(pattern_type)
        
        # Determine primary complexity type
        if "conversational_refinement" in detected_patterns:
            query_complexity = "conversational_refinement"
            primary_intent = await self._analyze_conversational_refinement(user_query, session_context)
        elif "multi_faceted" in detected_patterns:
            query_complexity = "multi_faceted"
            primary_intent = await self._analyze_multi_faceted_query(user_query)
        elif "style_modification" in detected_patterns:
            query_complexity = "style_modification"
            primary_intent = await self._analyze_style_modification(user_query)
        else:
            query_complexity = "simple"
            primary_intent = "basic_recommendation"
        
        return {
            "query_complexity": query_complexity,
            "primary_intent": primary_intent,
            "detected_patterns": detected_patterns,
            "decomposition_confidence": len(detected_patterns) / len(complexity_indicators)
        }

    async def _analyze_conversational_refinement(
        self,
        user_query: str,
        session_context: Optional[Dict] = None
    ) -> str:
        """Analyze conversational refinement queries like 'more like the last song but jazzier'."""
        query_lower = user_query.lower()
        
        if "last song" in query_lower or "last track" in query_lower:
            return "session_reference_with_style_modification"
        elif "that artist" in query_lower or "that band" in query_lower:
            return "artist_reference_with_modification"
        elif "like before" in query_lower or "similar to earlier" in query_lower:
            return "previous_recommendation_refinement"
        else:
            return "conversational_refinement"

    async def _analyze_multi_faceted_query(self, user_query: str) -> str:
        """Analyze multi-faceted queries like 'Beatles-style but for working out'."""
        query_lower = user_query.lower()
        
        if any(activity in query_lower for activity in ["workout", "working out", "gym", "exercise", "running"]):
            return "artist_similarity_with_activity_context"
        elif any(mood in query_lower for mood in ["study", "studying", "focus", "relax", "sleep"]):
            return "similarity_with_mood_context"
        elif any(time in query_lower for time in ["morning", "evening", "night", "commute"]):
            return "similarity_with_temporal_context"
        else:
            return "multi_faceted_similarity"

    async def _analyze_style_modification(self, user_query: str) -> str:
        """Analyze style modification queries like 'but jazzier' or 'more upbeat'."""
        query_lower = user_query.lower()
        
        if any(genre in query_lower for genre in ["jazzier", "rockier", "more electronic", "more acoustic"]):
            return "genre_style_modification"
        elif any(energy in query_lower for energy in ["more upbeat", "more energetic", "calmer", "mellower"]):
            return "energy_style_modification"
        elif any(mood in query_lower for mood in ["happier", "sadder", "darker", "brighter"]):
            return "mood_style_modification"
        else:
            return "general_style_modification"

    async def _enhance_entities_with_decomposition(
        self, 
        entities: Dict[str, Any], 
        decomposition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance extracted entities with decomposition context.
        
        Args:
            entities: Base extracted entities
            decomposition: Query decomposition analysis
            
        Returns:
            Enhanced entities with decomposition context
        """
        # Add decomposition metadata
        entities["decomposition_metadata"] = {
            "query_complexity": decomposition.get("query_complexity", "simple"),
            "primary_intent": decomposition.get("primary_intent", "basic_recommendation"),
            "detected_patterns": decomposition.get("detected_patterns", []),
            "requires_session_context": "conversational_refinement" in decomposition.get("detected_patterns", [])
        }
        
        # Enhance conversation entities based on decomposition
        if decomposition.get("query_complexity") == "conversational_refinement":
            if "conversation_entities" not in entities:
                entities["conversation_entities"] = {"session_references": [], "preference_evolution": [], "conversation_flow": []}
            
            # Add session reference indicators
            entities["conversation_entities"]["session_references"].extend([
                {"type": "track_reference", "indicator": "last song"},
                {"type": "artist_reference", "indicator": "that artist"},
                {"type": "recommendation_reference", "indicator": "like before"}
            ])
        
        # Enhance contextual entities for multi-faceted queries
        if decomposition.get("query_complexity") == "multi_faceted":
            if "contextual_entities" not in entities:
                entities["contextual_entities"] = {"moods": {}, "activities": {}, "temporal": {}}
            
            # Mark as requiring contextual coordination
            entities["contextual_entities"]["coordination_required"] = True
        
        return entities
    
    async def _fallback_basic_planning(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """Fallback to basic planning if enhanced planning fails."""
        self.add_reasoning_step("Falling back to basic planning")
        
        try:
            # Use original process logic
            task_analysis = await self._analyze_user_query(state.user_query)
            coordination_strategy = await self._plan_agent_coordination(state.user_query, task_analysis)
            evaluation_framework = await self._create_evaluation_framework(state.user_query, task_analysis)
            execution_monitoring = await self._setup_execution_monitoring(task_analysis)
            
            planning_strategy = {
                "task_analysis": task_analysis,
                "coordination_strategy": coordination_strategy,
                "evaluation_framework": evaluation_framework,
                "execution_monitoring": execution_monitoring
            }
            
            state.planning_strategy = planning_strategy
            state.reasoning_log.append("PlannerAgent: Used fallback basic planning")
            
            return state
            
        except Exception as e:
            self.logger.error("Fallback planning also failed", error=str(e))
            state.reasoning_log.append(f"PlannerAgent CRITICAL ERROR: {str(e)}")
            return state
    
    async def _plan_agent_coordination(self, user_query: str, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan coordination strategy for GenreMoodAgent and DiscoveryAgent.
        
        Args:
            user_query: Original user query
            task_analysis: Analysis of the query
            
        Returns:
            Coordination strategy for advocate agents
        """
        system_prompt = (
            "You are planning coordination between two music recommendation agents:\n"
            "1. GenreMoodAgent: Specializes in genre and mood-based search\n"
            "2. DiscoveryAgent: Specializes in similarity and underground discovery\n\n"
            "Create specific strategies for each agent based on the user's "
            "request and analysis."
        )

        user_prompt = f"""User Query: "{user_query}"
Task Analysis: {json.dumps(task_analysis, indent=2)}

Create coordination strategy in this JSON format:
{{
    "genre_mood_agent": {{
        "focus_areas": ["area1", "area2"],
        "energy_level": "low|medium|high",
        "search_tags": ["tag1", "tag2"],
        "mood_priority": "primary mood to target",
        "genre_constraints": ["constraint1", "constraint2"]
    }},
    "discovery_agent": {{
        "novelty_priority": "low|medium|high",
        "similarity_base": "what to base similarity on",
        "underground_bias": 0.0-1.0,
        "discovery_scope": "narrow|medium|broad",
        "exploration_strategy": "strategy description"
    }}
}}"""

        try:
            response = await self.call_llm(user_prompt, system_prompt)
            coordination = self._parse_json_response(response)
            
            # Validate and enhance coordination strategy
            coordination = self._enhance_coordination_strategy(coordination, task_analysis)
            
            return coordination
            
        except Exception as e:
            self.logger.warning("LLM coordination planning failed, using templates", error=str(e))
            return self._fallback_coordination_strategy(task_analysis)
    
    async def _create_evaluation_framework(self, user_query: str, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create evaluation framework for JudgeAgent decision making.
        
        Args:
            user_query: Original user query
            task_analysis: Analysis of the query
            
        Returns:
            Evaluation framework for judge
        """
        system_prompt = (
            "You are creating an evaluation framework for a judge agent to "
            "select the best music recommendations.\n\n"
            "The framework should define:\n"
            "1. Primary weights for different criteria\n"
            "2. Diversity targets\n"
            "3. Explanation style preferences\n\n"
            "Consider the user's specific request and context."
        )

        user_prompt = f"""User Query: "{user_query}"
Task Analysis: {json.dumps(task_analysis, indent=2)}

Create evaluation framework in this JSON format:
{{
    "primary_weights": {{
        "confidence": 0.0-1.0,
        "novelty_score": 0.0-1.0,
        "quality_score": 0.0-1.0,
        "concentration_friendliness_score": 0.0-1.0
    }},
    "diversity_targets": {{
        "attributes": ["genres", "artist"],
        "genres": 1-3,
        "era": 1-3,
        "energy": 1-2,
        "artist": 2-3
    }},
    "explanation_style": "detailed|concise|technical|casual",
    "selection_criteria": ["criterion1", "criterion2"]
}}"""

        try:
            response = await self.call_llm(user_prompt, system_prompt)
            framework = self._parse_json_response(response)
            
            # Validate and enhance framework
            framework = self._enhance_evaluation_framework(framework, task_analysis)
            
            return framework
            
        except Exception as e:
            self.logger.warning("LLM framework creation failed, using templates", error=str(e))
            return self._fallback_evaluation_framework(task_analysis)
    
    async def _setup_execution_monitoring(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set up execution monitoring and adaptation protocols.
        
        Args:
            task_analysis: Analysis of the user query
            
        Returns:
            Execution monitoring configuration
        """
        complexity = task_analysis.get('complexity_level', 'medium')
        
        # Define quality thresholds based on complexity
        quality_thresholds = {
            'simple': {'min_confidence': 0.7, 'min_relevance': 0.8},
            'medium': {'min_confidence': 0.6, 'min_relevance': 0.7},
            'complex': {'min_confidence': 0.5, 'min_relevance': 0.6}
        }
        
        # Define fallback strategies
        fallback_strategies = [
            "Broaden search criteria if no results found",
            "Reduce novelty requirements if underground tracks unavailable",
            "Adjust mood constraints if mood-specific search fails",
            "Use genre similarity if exact genre match fails"
        ]
        
        # Define coordination protocols
        coordination_protocols = {
            "parallel_execution": True,
            "result_sharing": False,  # Agents work independently
            "conflict_resolution": "judge_decides",
            "timeout_handling": "partial_results_acceptable"
        }
        
        return {
            "quality_thresholds": quality_thresholds.get(complexity, quality_thresholds['medium']),
            "fallback_strategies": fallback_strategies,
            "coordination_protocols": coordination_protocols,
            "success_metrics": {
                "min_recommendations": 2,
                "target_recommendations": 3,
                "max_processing_time": 300  # 5 minutes
            }
        }
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, handling common formatting issues."""
        try:
            # Clean up response - remove markdown formatting
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*$', '', cleaned)
            cleaned = cleaned.strip()
            
            return json.loads(cleaned)
            
        except json.JSONDecodeError as e:
            self.logger.warning("Failed to parse JSON response", error=str(e), response=response[:200])
            raise
    
    def _enhance_task_analysis(self, analysis: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Enhance and validate task analysis."""
        # Ensure required fields exist
        analysis.setdefault('primary_goal', 'music_discovery')
        analysis.setdefault('complexity_level', 'medium')
        analysis.setdefault('context_factors', [])
        analysis.setdefault('mood_indicators', [])
        analysis.setdefault('genre_hints', [])
        
        # Add pattern-based enhancements
        query_lower = user_query.lower()
        
        # Detect activity context
        activity_patterns = {
            'work': ['work', 'coding', 'study', 'focus', 'concentration'],
            'exercise': ['workout', 'gym', 'running', 'exercise'],
            'relax': ['chill', 'relax', 'calm', 'peaceful'],
            'party': ['party', 'dance', 'energetic', 'upbeat']
        }
        
        for activity, keywords in activity_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                if activity not in analysis['context_factors']:
                    analysis['context_factors'].append(activity)
        
        return analysis
    
    def _enhance_coordination_strategy(self, coordination: Dict[str, Any], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance and validate coordination strategy."""
        # Ensure required structure
        coordination.setdefault('genre_mood_agent', {})
        coordination.setdefault('discovery_agent', {})
        
        # Set defaults for GenreMoodAgent
        gma = coordination['genre_mood_agent']
        gma.setdefault('focus_areas', task_analysis.get('genre_hints', ['indie', 'alternative']))
        gma.setdefault('energy_level', 'medium')
        gma.setdefault('search_tags', task_analysis.get('mood_indicators', ['chill']))
        
        # Set defaults for DiscoveryAgent
        da = coordination['discovery_agent']
        da.setdefault('novelty_priority', 'medium')
        da.setdefault('underground_bias', 0.6)
        da.setdefault('discovery_scope', 'medium')
        
        return coordination
    
    def _enhance_evaluation_framework(self, framework: Dict[str, Any], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance and validate evaluation framework."""
        # Ensure required structure
        framework.setdefault('primary_weights', {})
        framework.setdefault('diversity_targets', {})
        
        # Set default weights using correct field names from TrackRecommendation model
        weights = framework['primary_weights']
        weights.setdefault('confidence', 0.3)  # Overall confidence/relevance
        weights.setdefault('novelty_score', 0.25)
        weights.setdefault('quality_score', 0.25)
        weights.setdefault('concentration_friendliness_score', 0.2)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] = weights[key] / total_weight
        
        # Set default diversity targets with correct attribute names
        diversity = framework['diversity_targets']
        diversity.setdefault('attributes', ['genres', 'artist'])  # Add attributes list
        diversity.setdefault('genres', 2)
        diversity.setdefault('era', 2)
        diversity.setdefault('energy', 1)
        diversity.setdefault('artist', 3)
        
        return framework
    
    def _fallback_query_analysis(self, user_query: str) -> Dict[str, Any]:
        """Fallback query analysis using pattern matching."""
        query_lower = user_query.lower()
        
        # Determine complexity
        complexity_indicators = {
            'simple': ['play', 'song', 'music'],
            'complex': ['discover', 'explore', 'recommend', 'find', 'suggest']
        }
        
        complexity = 'medium'  # default
        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                complexity = level
                break
        
        # Extract mood indicators
        mood_patterns = {
            'chill': ['chill', 'relax', 'calm', 'peaceful'],
            'energetic': ['energetic', 'upbeat', 'pump', 'hype'],
            'focus': ['focus', 'concentration', 'study', 'work'],
            'sad': ['sad', 'melancholy', 'depressing'],
            'happy': ['happy', 'joyful', 'cheerful']
        }
        
        mood_indicators = []
        for mood, keywords in mood_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                mood_indicators.append(mood)
        
        return {
            'primary_goal': 'music_discovery',
            'complexity_level': complexity,
            'context_factors': [],
            'mood_indicators': mood_indicators or ['general'],
            'genre_hints': []
        }
    
    def _fallback_coordination_strategy(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback coordination strategy using templates."""
        return {
            'genre_mood_agent': {
                'focus_areas': task_analysis.get('genre_hints', ['indie', 'alternative']),
                'energy_level': 'medium',
                'search_tags': task_analysis.get('mood_indicators', ['chill']),
                'mood_priority': task_analysis.get('mood_indicators', ['general'])[0],
                'genre_constraints': []
            },
            'discovery_agent': {
                'novelty_priority': 'medium',
                'similarity_base': 'genre_and_mood',
                'underground_bias': 0.6,
                'discovery_scope': 'medium',
                'exploration_strategy': 'balanced_discovery'
            }
        }
    
    def _fallback_evaluation_framework(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback evaluation framework using templates."""
        return {
            'primary_weights': {
                'confidence': 0.3,
                'novelty_score': 0.25,
                'quality_score': 0.25,
                'concentration_friendliness_score': 0.2
            },
            'diversity_targets': {
                'attributes': ['genres', 'artist'],
                'genres': 2,
                'era': 2,
                'energy': 1,
                'artist': 3
            },
            'explanation_style': 'detailed',
            'selection_criteria': ['confidence', 'novelty_score', 'quality_score']
        }
    
    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for query analysis."""
        return {
            'activity_context': {
                'work': ['work', 'coding', 'study', 'focus', 'concentration', 'productivity'],
                'exercise': ['workout', 'gym', 'running', 'exercise', 'fitness'],
                'relax': ['chill', 'relax', 'calm', 'peaceful', 'unwind'],
                'party': ['party', 'dance', 'energetic', 'upbeat', 'celebration'],
                'sleep': ['sleep', 'bedtime', 'lullaby', 'peaceful'],
                'drive': ['driving', 'road trip', 'car', 'travel']
            },
            'mood_indicators': {
                'happy': ['happy', 'joyful', 'cheerful', 'uplifting', 'positive'],
                'sad': ['sad', 'melancholy', 'depressing', 'emotional', 'heartbreak'],
                'energetic': ['energetic', 'pump up', 'hype', 'motivational'],
                'calm': ['calm', 'peaceful', 'serene', 'tranquil'],
                'nostalgic': ['nostalgic', 'throwback', 'memories', 'classic'],
                'romantic': ['romantic', 'love', 'intimate', 'passionate']
            },
            'genre_hints': {
                'rock': ['rock', 'guitar', 'band', 'alternative'],
                'electronic': ['electronic', 'edm', 'techno', 'house', 'ambient'],
                'hip_hop': ['hip hop', 'rap', 'beats', 'urban'],
                'indie': ['indie', 'independent', 'underground', 'alternative'],
                'classical': ['classical', 'orchestra', 'symphony', 'instrumental'],
                'jazz': ['jazz', 'blues', 'swing', 'improvisation'],
                'folk': ['folk', 'acoustic', 'singer-songwriter', 'country']
            }
        }
    
    def _initialize_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize strategy templates for different scenarios."""
        return {
            'work_focus': {
                'genre_mood_agent': {
                    'focus_areas': ['instrumental', 'ambient', 'post-rock'],
                    'energy_level': 'medium-low',
                    'search_tags': ['focus', 'study', 'instrumental', 'concentration']
                },
                'discovery_agent': {
                    'novelty_priority': 'medium',
                    'underground_bias': 0.7,
                    'discovery_scope': 'narrow'
                }
            },
            'workout_energy': {
                'genre_mood_agent': {
                    'focus_areas': ['electronic', 'rock', 'hip-hop'],
                    'energy_level': 'high',
                    'search_tags': ['energetic', 'pump', 'workout', 'motivational']
                },
                'discovery_agent': {
                    'novelty_priority': 'low',
                    'underground_bias': 0.3,
                    'discovery_scope': 'broad'
                }
            },
            'chill_discovery': {
                'genre_mood_agent': {
                    'focus_areas': ['indie', 'alternative', 'folk'],
                    'energy_level': 'low',
                    'search_tags': ['chill', 'relax', 'mellow', 'peaceful']
                },
                'discovery_agent': {
                    'novelty_priority': 'high',
                    'underground_bias': 0.8,
                    'discovery_scope': 'broad'
                }
            }
        }
    
    async def _make_llm_call(self, prompt: str, system_prompt: str = None) -> str:
        """
        Make LLM call using Gemini client.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            LLM response
        """
        if not self.llm_client:
            raise RuntimeError("Gemini client not initialized")
        
        try:
            # Combine system and user prompts
            full_prompt = (
                f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            )
            
            # Call Gemini - handle both sync and async mocks
            response = self.llm_client.generate_content(full_prompt)
            
            # If it's a coroutine (async mock), await it
            if hasattr(response, '__await__'):
                response = await response
            
            return response.text
            
        except Exception as e:
            self.logger.error("Gemini API call failed", error=str(e))
            raise
    
    def _extract_output_data(self, state: MusicRecommenderState) -> Dict[str, Any]:
        """Extract PlannerAgent output data."""
        return {
            "planning_strategy_created": state.planning_strategy is not None,
            "strategy_components": (
                len(state.planning_strategy) if state.planning_strategy else 0
            )
        }
    
    def _calculate_confidence(self, state: MusicRecommenderState) -> float:
        """Calculate confidence in planning strategy."""
        if not state.planning_strategy:
            return 0.0
        
        # Base confidence
        confidence = 0.7
        
        # Increase confidence based on strategy completeness
        required_components = [
            'task_analysis', 'coordination_strategy', 'evaluation_framework'
        ]
        present_components = sum(
            1 for comp in required_components if comp in state.planning_strategy
        )
        confidence += (present_components / len(required_components)) * 0.3
        
        return min(confidence, 1.0)