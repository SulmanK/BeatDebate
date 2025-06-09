"""
Strategy Planner Component for Planner Agent

Handles agent strategy and parameter planning logic for the planner agent.
Extracted from PlannerAgent for better modularization and single responsibility.
"""

from typing import Dict, Any
import structlog

from ...models.agent_models import QueryUnderstanding

logger = structlog.get_logger(__name__)


class StrategyPlanner:
    """
    Handles strategy and parameter planning for the planner agent.
    
    Responsibilities:
    - Creating planning strategies based on understanding and analysis
    - Determining agent sequences and coordination
    - Setting quality thresholds and diversity targets
    - Planning candidate pool strategies for efficient follow-ups
    """
    
    def __init__(self):
        """Initialize the StrategyPlanner."""
        self.logger = logger
        self.logger.info("StrategyPlanner initialized")
    
    async def create_planning_strategy(
        self, understanding: QueryUnderstanding, task_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create planning strategy based on understanding and analysis.
        
        Args:
            understanding: Query understanding results
            task_analysis: Task complexity analysis
            
        Returns:
            Planning strategy dictionary
        """
        try:
            strategy = {
                'intent': understanding.intent.value,
                'complexity_level': task_analysis.get(
                    'complexity_level', 'medium'
                ),
                'confidence': understanding.confidence,
                'agent_sequence': self.determine_agent_sequence(
                    understanding, task_analysis
                ),
                'quality_thresholds': self.determine_quality_thresholds(
                    task_analysis
                ),
                'diversity_targets': self.determine_diversity_targets(
                    understanding
                ),
                'explanation_style': self.determine_explanation_style(
                    task_analysis
                ),
                'fallback_strategies': self.create_fallback_strategies(
                    understanding
                ),
                # Phase 3: Candidate pool persistence strategy
                'generate_large_pool': self.should_generate_large_pool(
                    understanding, task_analysis
                ),
                'pool_size_multiplier': self.determine_pool_size_multiplier(
                    understanding, task_analysis
                )
            }
            
            self.logger.debug(
                "Planning strategy created",
                agent_sequence=strategy['agent_sequence'],
                quality_thresholds=strategy['quality_thresholds']
            )
            
            return strategy
            
        except Exception as e:
            self.logger.error(
                "Planning strategy creation failed", error=str(e)
            )
            return self.create_fallback_strategy()
    
    def determine_agent_sequence(
        self, understanding: QueryUnderstanding, task_analysis: Dict[str, Any]
    ) -> list:
        """
        Determine the sequence of agents to use based on intent and complexity.
        
        Args:
            understanding: Query understanding results
            task_analysis: Task analysis dictionary
            
        Returns:
            List of agent names in execution order
        """
        intent = understanding.intent.value
        complexity = task_analysis.get('complexity_level', 'medium')
        
        # 🔧 NEW: Check for follow-up queries that should use stored pools
        is_followup = (hasattr(understanding, 'reasoning') and understanding.reasoning and 
                       ('follow-up:' in understanding.reasoning or 'Context override:' in understanding.reasoning))
        
        if is_followup:
            # For follow-up queries, check if we should skip discovery and use stored pools
            # This is more efficient and provides consistent recommendations
            self.logger.info(
                "🔄 Follow-up query detected - considering pool retrieval",
                intent=intent,
                reasoning=understanding.reasoning if hasattr(understanding, 'reasoning') else None
            )
            # For artist follow-ups, skip discovery and go straight to judge for pool retrieval
            if intent in ['by_artist', 'artist_similarity']:
                self.logger.info("✨ Using stored pool for artist follow-up - skipping discovery agent")
                return ['judge_agent']
        
        # Base sequences by intent
        intent_sequences = {
            'discovery': ['discovery_agent', 'judge_agent'],
            'discovering_serendipity': ['discovery_agent', 'judge_agent'],  # Pure serendipitous discovery
            'similarity': ['discovery_agent', 'judge_agent'],
            'genre_mood': ['genre_mood_agent', 'judge_agent'],
            'activity_based': ['genre_mood_agent', 'judge_agent'],
            'contextual': ['genre_mood_agent', 'judge_agent'],  # Fix: Route contextual queries to genre_mood agent
            'hybrid': ['genre_mood_agent', 'discovery_agent', 'judge_agent'],
            'hybrid_similarity_genre': ['discovery_agent', 'judge_agent'],  # ✅ NEW: Artist similarity + genre filtering
            'by_artist': ['discovery_agent', 'judge_agent'],
            'artist_similarity': ['discovery_agent', 'judge_agent'],  # DiscoveryAgent handles Enhanced Similar Artist Strategy
            'artist_genre': ['discovery_agent', 'judge_agent']  # ✅ NEW: DiscoveryAgent generates artist tracks, then filters by genre
        }
        
        base_sequence = intent_sequences.get(intent, ['discovery_agent', 'judge_agent'])
        
        # Modify sequence based on complexity
        if complexity == 'complex':
            # For complex queries, use both advocate agents for broader coverage
            if intent in ['discovery', 'similarity', 'by_artist', 'artist_similarity']:
                base_sequence = ['genre_mood_agent', 'discovery_agent', 'judge_agent']
                # Ensure we have specific entities for genre/mood agent
        # Exception: Don't remove genre_mood_agent for contextual intents as they extract implicit moods
        if ('genre_mood_agent' in base_sequence and 
                not understanding.genres and not understanding.moods and
                intent != 'contextual'):  # Fix: Keep genre_mood_agent for contextual intents
            # Remove genre_mood_agent if no relevant entities
            base_sequence = [agent for agent in base_sequence if agent != 'genre_mood_agent']
        
        self.logger.debug(
            "Agent sequence determined",
            intent=intent,
            complexity=complexity,
            sequence=base_sequence,
            is_followup=is_followup
        )
        
        return base_sequence
    
    def determine_quality_thresholds(self, task_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Determine quality thresholds based on task complexity.
        
        Args:
            task_analysis: Task analysis dictionary
            
        Returns:
            Dictionary of quality thresholds
        """
        complexity = task_analysis.get('complexity_level', 'medium')
        specificity = task_analysis.get('specificity', 'moderate')
        
        # Base thresholds
        thresholds = {
            'minimum_quality': 0.3,
            'preferred_quality': 0.6,
            'diversity_threshold': 0.4,
            'popularity_threshold': 0.2
        }
        
        # Adjust based on complexity
        if complexity == 'simple':
            thresholds['minimum_quality'] = 0.2
            thresholds['preferred_quality'] = 0.5
        elif complexity == 'complex':
            thresholds['minimum_quality'] = 0.4
            thresholds['preferred_quality'] = 0.7
        
        # Adjust based on specificity
        if specificity == 'specific':
            thresholds['minimum_quality'] += 0.1
            thresholds['preferred_quality'] += 0.1
        elif specificity == 'vague':
            thresholds['minimum_quality'] -= 0.1
            thresholds['diversity_threshold'] += 0.2
        
        # Ensure thresholds stay within bounds
        for key in thresholds:
            thresholds[key] = max(0.0, min(1.0, thresholds[key]))
        
        return thresholds
    
    def determine_diversity_targets(self, understanding: QueryUnderstanding) -> Dict[str, Any]:
        """
        Determine diversity targets based on query understanding.
        
        Args:
            understanding: Query understanding results
            
        Returns:
            Dictionary of diversity targets
        """
        targets = {
            'genre_diversity': 0.6,
            'artist_diversity': 0.8,
            'era_diversity': 0.4,
            'energy_diversity': 0.5,
            'popularity_spread': 0.7
        }
        
        # Adjust based on intent
        if understanding.intent.value in ['discovery', 'similarity']:
            targets['genre_diversity'] = 0.8
            targets['artist_diversity'] = 0.9
        elif understanding.intent.value in ['genre_mood', 'activity_based']:
            targets['genre_diversity'] = 0.4
            targets['energy_diversity'] = 0.7
        elif understanding.intent.value in ['by_artist', 'artist_similarity', 'artist_genre']:
            targets['artist_diversity'] = 0.3  # Focus on one artist or similar artists
            targets['genre_diversity'] = 0.5
        
        # Adjust based on entities
        if understanding.artists:
            targets['artist_diversity'] = 0.4  # Lower when specific artists mentioned
        if understanding.genres:
            targets['genre_diversity'] = 0.3  # Lower when specific genres mentioned
        
        return targets
    
    def determine_explanation_style(self, task_analysis: Dict[str, Any]) -> str:
        """
        Determine explanation style based on task analysis.
        
        Args:
            task_analysis: Task analysis dictionary
            
        Returns:
            Explanation style string
        """
        complexity = task_analysis.get('complexity_level', 'medium')
        specificity = task_analysis.get('specificity', 'moderate')
        
        if complexity == 'simple' and specificity == 'vague':
            return 'casual'
        elif complexity == 'complex' or specificity == 'specific':
            return 'detailed'
        else:
            return 'balanced'
    
    def create_fallback_strategies(self, understanding: QueryUnderstanding) -> list:
        """
        Create fallback strategies for the query.
        
        Args:
            understanding: Query understanding results
            
        Returns:
            List of fallback strategy dictionaries
        """
        strategies = []
        
        # Basic fallback: Lower thresholds
        strategies.append({
            'type': 'lower_thresholds',
            'adjustments': {
                'minimum_quality': 0.1,
                'preferred_quality': 0.3
            }
        })
        
        # Intent-specific fallbacks
        if understanding.intent.value in ['similarity', 'artist_similarity']:
            strategies.append({
                'type': 'broader_similarity',
                'adjustments': {
                    'similarity_threshold': 0.3,
                    'genre_expansion': True
                }
            })
        
        if understanding.genres or understanding.moods:
            strategies.append({
                'type': 'genre_expansion',
                'adjustments': {
                    'include_subgenres': True,
                    'mood_flexibility': 0.3
                }
            })
        
        # Ultimate fallback: Discovery mode
        strategies.append({
            'type': 'discovery_fallback',
            'adjustments': {
                'intent_override': 'discovery',
                'popularity_boost': 0.3
            }
        })
        
        return strategies
    
    def should_generate_large_pool(
        self, understanding: QueryUnderstanding, task_analysis: Dict[str, Any]
    ) -> bool:
        """
        Determine if a large candidate pool should be generated for follow-up efficiency.
        
        Phase 3: This supports the candidate pool persistence strategy for "more tracks" follow-ups.
        
        Args:
            understanding: Query understanding results
            task_analysis: Task analysis dictionary
            
        Returns:
            Boolean indicating whether to generate a large pool
        """
        # Generate large pools for intents likely to have follow-ups
        followup_prone_intents = [
            'discovery', 'discovering_serendipity', 'similarity', 'genre_mood', 'by_artist', 'by_artist_underground', 'artist_similarity', 'artist_genre', 'contextual'
        ]
        
        if understanding.intent.value not in followup_prone_intents:
            return False
        
        # Generate large pools for non-complex queries (complex ones are already expensive)
        complexity = task_analysis.get('complexity_level', 'medium')
        if complexity == 'complex':
            return False
        
        # Generate large pools for queries with good entity specificity
        # Include contextual entities for contextual intent queries
        has_specific_entities = bool(
            understanding.artists or understanding.genres or understanding.moods
        )
        
        # For contextual intent, check if we have contextual activities
        if understanding.intent.value == 'contextual' and not has_specific_entities:
            # Check if we have activities extracted (like 'study', 'workout', etc.)
            has_contextual_activities = bool(understanding.activities)
            has_specific_entities = has_contextual_activities
        
        if not has_specific_entities:
            return False
        
        self.logger.info(
            "🎯 Large pool generation recommended",
            intent=understanding.intent.value,
            complexity=complexity,
            has_entities=has_specific_entities
        )
        
        return True
    
    def determine_pool_size_multiplier(
        self, understanding: QueryUnderstanding, task_analysis: Dict[str, Any]
    ) -> int:
        """
        Determine the multiplier for candidate pool size.
        
        Phase 3: This determines how much larger the initial pool should be
        to support efficient "more tracks" follow-ups.
        
        Args:
            understanding: Query understanding results
            task_analysis: Task analysis dictionary
            
        Returns:
            Integer multiplier for pool size (2-5x typical)
        """
        if not self.should_generate_large_pool(understanding, task_analysis):
            return 1
        
        # Base multiplier
        multiplier = 3
        
        # Adjust based on intent
        if understanding.intent.value in ['discovery', 'discovering_serendipity', 'similarity', 'contextual']:
            multiplier = 4  # These benefit most from large pools
        elif understanding.intent.value in ['by_artist', 'by_artist_underground', 'artist_similarity', 'artist_genre']:
            multiplier = 5  # Artist-focused queries often have many good candidates
        
        # Adjust based on entity count
        entity_count = (
            len(understanding.artists) + len(understanding.genres) + len(understanding.moods)
        )
        if entity_count >= 3:
            multiplier += 1  # More entities = more potential candidates
        
        # Keep within reasonable bounds
        multiplier = max(2, min(5, multiplier))
        
        self.logger.debug(
            "Pool size multiplier determined",
            intent=understanding.intent.value,
            entity_count=entity_count,
            multiplier=multiplier
        )
        
        return multiplier
    
    async def plan_agent_coordination(
        self, user_query: str, task_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Plan agent coordination and communication strategies.
        
        Args:
            user_query: User's music request
            task_analysis: Task analysis dictionary
            
        Returns:
            Agent coordination plan dictionary
        """
        try:
            coordination = {
                'communication_strategy': self._determine_communication_strategy(
                    task_analysis
                ),
                'data_sharing': self._plan_data_sharing(task_analysis),
                'error_handling': self._plan_error_handling(task_analysis),
                'performance_targets': self._set_performance_targets(task_analysis),
                'coordination_timestamp': self._get_timestamp()
            }
            
            self.logger.debug(
                "Agent coordination planned",
                communication_strategy=coordination['communication_strategy'],
                data_sharing_enabled=coordination['data_sharing']['enabled']
            )
            
            return coordination
            
        except Exception as e:
            self.logger.error("Agent coordination planning failed", error=str(e))
            return self.create_fallback_coordination()
    
    def create_fallback_strategy(self) -> Dict[str, Any]:
        """Create a fallback planning strategy."""
        return {
            'intent': 'discovery',
            'complexity_level': 'medium',
            'confidence': 0.3,
            'agent_sequence': ['discovery_agent', 'judge_agent'],
            'quality_thresholds': {
                'minimum_quality': 0.2,
                'preferred_quality': 0.5,
                'diversity_threshold': 0.4
            },
            'diversity_targets': {
                'genre_diversity': 0.6,
                'artist_diversity': 0.8
            },
            'explanation_style': 'casual',
            'fallback_strategies': [
                {'type': 'lower_thresholds', 'adjustments': {'minimum_quality': 0.1}}
            ],
            'generate_large_pool': False,
            'pool_size_multiplier': 1
        }
    
    def create_fallback_coordination(self) -> Dict[str, Any]:
        """Create a fallback coordination plan."""
        return {
            'communication_strategy': 'basic',
            'data_sharing': {
                'enabled': True,
                'scope': 'basic_metadata'
            },
            'error_handling': {
                'strategy': 'graceful_degradation',
                'retry_attempts': 2
            },
            'performance_targets': {
                'max_total_time': 30.0,
                'max_agent_time': 10.0
            },
            'coordination_timestamp': self._get_timestamp()
        }
    
    def _determine_communication_strategy(self, task_analysis: Dict[str, Any]) -> str:
        """Determine communication strategy between agents."""
        complexity = task_analysis.get('complexity_level', 'medium')
        
        if complexity == 'simple':
            return 'basic'
        elif complexity == 'complex':
            return 'enhanced'
        else:
            return 'standard'
    
    def _plan_data_sharing(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan data sharing between agents."""
        return {
            'enabled': True,
            'scope': 'full_metadata',
            'include_scores': True,
            'include_reasoning': task_analysis.get('complexity_level') == 'complex'
        }
    
    def _plan_error_handling(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan error handling strategy."""
        complexity = task_analysis.get('complexity_level', 'medium')
        
        retry_attempts = 2 if complexity == 'simple' else 3
        
        return {
            'strategy': 'graceful_degradation',
            'retry_attempts': retry_attempts,
            'fallback_enabled': True,
            'timeout_handling': 'continue_with_partial'
        }
    
    def _set_performance_targets(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Set performance targets for the coordination."""
        complexity = task_analysis.get('complexity_level', 'medium')
        
        # Adjust timeouts based on complexity
        if complexity == 'simple':
            max_time = 20.0
            agent_time = 8.0
        elif complexity == 'complex':
            max_time = 45.0
            agent_time = 15.0
        else:
            max_time = 30.0
            agent_time = 10.0
        
        return {
            'max_total_time': max_time,
            'max_agent_time': agent_time,
            'quality_target': 0.7,
            'diversity_target': 0.6
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for coordination tracking."""
        from datetime import datetime
        return datetime.now().isoformat() 