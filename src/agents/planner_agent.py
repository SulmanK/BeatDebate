"""
PlannerAgent for BeatDebate Multi-Agent Music Recommendation System

Strategic coordinator and planning engine that demonstrates sophisticated
agentic planning behavior for the AgentX competition.
"""

import json
import re
from typing import Dict, List, Any
import structlog

from .base_agent import BaseAgent
from ..models.agent_models import MusicRecommenderState, AgentConfig

logger = structlog.get_logger(__name__)


class PlannerAgent(BaseAgent):
    """
    Master planning agent that coordinates the entire music recommendation workflow.
    
    Demonstrates agentic planning behavior required for AgentX competition:
    - Strategic task decomposition
    - Resource allocation and coordination
    - Success criteria definition
    - Adaptive execution monitoring
    """
    
    def __init__(self, config: AgentConfig, gemini_client=None):
        """
        Initialize PlannerAgent with Gemini LLM client.
        
        Args:
            config: Agent configuration
            gemini_client: Gemini LLM client for reasoning
        """
        super().__init__(config)
        self.llm_client = gemini_client
        
        # Planning templates and patterns
        self.query_patterns = self._initialize_query_patterns()
        self.strategy_templates = self._initialize_strategy_templates()
        
        self.logger.info("PlannerAgent initialized with strategic planning capabilities")
    
    async def process(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Create comprehensive music discovery strategy.
        
        Args:
            state: Current workflow state with user query
            
        Returns:
            Updated state with planning strategy
        """
        self.add_reasoning_step("Starting strategic planning for music discovery")
        
        try:
            # Step 1: Analyze user query complexity and intent
            task_analysis = await self._analyze_user_query(state.user_query)
            self.add_reasoning_step(f"Query analysis completed: {task_analysis['primary_goal']}")
            
            # Step 2: Create coordination strategy for advocate agents
            coordination_strategy = await self._plan_agent_coordination(state.user_query, task_analysis)
            self.add_reasoning_step("Agent coordination strategy developed")
            
            # Step 3: Define evaluation framework for judge
            evaluation_framework = await self._create_evaluation_framework(state.user_query, task_analysis)
            self.add_reasoning_step("Evaluation framework established")
            
            # Step 4: Set up execution monitoring
            execution_monitoring = await self._setup_execution_monitoring(task_analysis)
            self.add_reasoning_step("Execution monitoring protocols defined")
            
            # Combine into comprehensive strategy
            planning_strategy = {
                "task_analysis": task_analysis,
                "coordination_strategy": coordination_strategy,
                "evaluation_framework": evaluation_framework,
                "execution_monitoring": execution_monitoring
            }
            
            # Update state with strategy
            state.planning_strategy = planning_strategy
            log_message = (
                f"PlannerAgent: Created comprehensive strategy for "
                f"'{task_analysis['primary_goal']}'"
            )
            state.reasoning_log.append(log_message)
            
            self.logger.info(
                "Strategic planning completed",
                primary_goal=task_analysis['primary_goal'],
                complexity=task_analysis['complexity_level'],
                strategy_components=len(planning_strategy)
            )
            
            return state
            
        except Exception as e:
            self.logger.error("Strategic planning failed", error=str(e))
            state.reasoning_log.append(f"PlannerAgent ERROR: {str(e)}")
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
        "relevance": 0.0-1.0,
        "novelty": 0.0-1.0,
        "quality": 0.0-1.0,
        "mood_match": 0.0-1.0,
        "context_fit": 0.0-1.0
    }},
    "diversity_targets": {{
        "genre": 1-3,
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
        
        # Set default weights
        weights = framework['primary_weights']
        weights.setdefault('relevance', 0.3)
        weights.setdefault('novelty', 0.25)
        weights.setdefault('quality', 0.25)
        weights.setdefault('mood_match', 0.2)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] = weights[key] / total_weight
        
        # Set default diversity targets
        diversity = framework['diversity_targets']
        diversity.setdefault('genre', 2)
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
                'relevance': 0.3,
                'novelty': 0.25,
                'quality': 0.25,
                'mood_match': 0.2
            },
            'diversity_targets': {
                'genre': 2,
                'era': 2,
                'energy': 1,
                'artist': 3
            },
            'explanation_style': 'detailed',
            'selection_criteria': ['relevance', 'novelty', 'quality']
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
                f"{system_prompt}\\n\\n{prompt}" if system_prompt else prompt
            )
            
            # Call Gemini (this will be implemented when we integrate Gemini)
            # For now, return a placeholder
            response = await self.llm_client.generate_content(full_prompt)
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