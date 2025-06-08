"""
Strategy Factory for Candidate Generation

Maps QueryIntent enums to concrete strategy implementations.
This factory enforces type safety and fixes the 'str' vs enum bug.
"""

from typing import Dict, List, Union
import structlog

from ....services.api_service import APIService
from ....models.agent_models import QueryIntent

from .base_strategy import BaseGenerationStrategy
from .artist_strategies import TargetArtistStrategy, SimilarArtistStrategy
from .genre_strategies import GenreExplorationStrategy, GenreFocusedStrategy, RandomGenreStrategy
from .discovery_strategies import UndergroundGemsStrategy, SerendipitousDiscoveryStrategy
from .mood_strategies import MoodBasedSerendipityStrategy, MoodFilteredTracksStrategy, GenreMoodCombinedStrategy
from .contextual_strategies import ContextualActivityStrategy, FunctionalMoodStrategy


class StrategyFactory:
    """
    Factory for creating and managing generation strategies.
    
    This factory provides type-safe mapping between QueryIntent enums
    and strategy implementations, eliminating the 'str' vs enum bug.
    """
    
    def __init__(self, api_service: APIService, llm_client=None):
        """
        Initialize the strategy factory.
        
        Args:
            api_service: Unified API service for strategy initialization
            llm_client: LLM client for strategies that need AI classification
        """
        self.api_service = api_service
        self.llm_client = llm_client
        self.logger = structlog.get_logger(__name__)
        
        # Initialize strategy instances (reuse for performance)
        self._strategies = {
            # Artist strategies
            'target_artist': TargetArtistStrategy(api_service, llm_client=llm_client),
            'similar_artist': SimilarArtistStrategy(api_service, llm_client=llm_client),
            
            # Genre strategies  
            'genre_exploration': GenreExplorationStrategy(api_service, llm_client=llm_client),
            'genre_focused': GenreFocusedStrategy(api_service, llm_client=llm_client),
            'random_genre': RandomGenreStrategy(api_service, llm_client=llm_client),
            
            # Discovery strategies
            'underground_gems': UndergroundGemsStrategy(api_service, llm_client=llm_client),
            'serendipitous_discovery': SerendipitousDiscoveryStrategy(api_service, llm_client=llm_client),
            
            # Mood strategies
            'mood_serendipity': MoodBasedSerendipityStrategy(api_service, llm_client=llm_client),
            'mood_filtered': MoodFilteredTracksStrategy(api_service, llm_client=llm_client),
            'genre_mood_combined': GenreMoodCombinedStrategy(api_service, llm_client=llm_client),
            
            # Contextual strategies
            'contextual_activity': ContextualActivityStrategy(api_service, llm_client=llm_client),
            'functional_mood': FunctionalMoodStrategy(api_service, llm_client=llm_client)
        }
        
        # Intent to strategy mapping - THIS IS THE KEY FIX
        # Maps QueryIntent enums to strategy combinations
        self._intent_strategy_map = {
            QueryIntent.BY_ARTIST: ['target_artist'],
            QueryIntent.BY_ARTIST_UNDERGROUND: ['target_artist', 'underground_gems'],
            QueryIntent.ARTIST_SIMILARITY: ['similar_artist'],
            QueryIntent.ARTIST_GENRE: ['target_artist'],  # ✅ NEW: Generate artist tracks for genre filtering
            QueryIntent.DISCOVERY: ['underground_gems', 'serendipitous_discovery'],
            QueryIntent.DISCOVERING_SERENDIPITY: ['serendipitous_discovery'],  # Pure serendipitous discovery
            QueryIntent.GENRE_MOOD: ['genre_mood_combined', 'genre_focused'],  # Primary: combined strategy, fallback: genre-focused
            QueryIntent.CONTEXTUAL: ['contextual_activity', 'functional_mood'],
            QueryIntent.HYBRID_SIMILARITY_GENRE: ['similar_artist'],
        }
        
        self.logger.info(
            "StrategyFactory initialized",
            strategies_count=len(self._strategies),
            intent_mappings=len(self._intent_strategy_map)
        )
    
    def get_strategies_for_intent(
        self, 
        intent: Union[QueryIntent, str]
    ) -> List[BaseGenerationStrategy]:
        """
        Get strategy instances for a given intent.
        
        Args:
            intent: QueryIntent enum or string representation
            
        Returns:
            List of strategy instances for the intent
            
        Raises:
            ValueError: If intent is not recognized
        """
        # CRITICAL FIX: Handle both enum and string inputs safely
        if isinstance(intent, str):
            # Try to convert string to QueryIntent enum
            try:
                intent_enum = QueryIntent(intent)
            except ValueError:
                # Handle legacy string intents by mapping them
                intent_enum = self._map_legacy_intent(intent)
        elif isinstance(intent, QueryIntent):
            intent_enum = intent
        else:
            raise ValueError(f"Intent must be QueryIntent enum or string, got {type(intent)}")
        
        # Get strategy names for this intent
        strategy_names = self._intent_strategy_map.get(intent_enum, [])
        
        if not strategy_names:
            self.logger.warning(f"No strategies found for intent: {intent_enum}")
            # Fallback to discovery strategies
            strategy_names = ['serendipitous_discovery', 'genre_exploration']
        
        # Return strategy instances
        strategies = []
        for strategy_name in strategy_names:
            if strategy_name in self._strategies:
                strategies.append(self._strategies[strategy_name])
            else:
                self.logger.warning(f"Strategy '{strategy_name}' not found in factory")
        
        self.logger.debug(
            f"Selected strategies for intent {intent_enum}",
            strategies=[s.__class__.__name__ for s in strategies]
        )
        
        return strategies
    
    def _map_legacy_intent(self, intent_str: str) -> QueryIntent:
        """
        Map legacy string intents to QueryIntent enums.
        
        This provides backward compatibility for existing code
        that passes string intents.
        
        Args:
            intent_str: Legacy intent string
            
        Returns:
            Corresponding QueryIntent enum
        """
        legacy_mapping = {
            'by_artist': QueryIntent.BY_ARTIST,
            'by_artist_underground': QueryIntent.BY_ARTIST_UNDERGROUND,
            'artist_similarity': QueryIntent.ARTIST_SIMILARITY,
            'artist_genre': QueryIntent.ARTIST_GENRE,  # ✅ NEW: Legacy mapping for artist_genre
            'discovery': QueryIntent.DISCOVERY,
            'discovering_serendipity': QueryIntent.DISCOVERING_SERENDIPITY,
            'genre_mood': QueryIntent.GENRE_MOOD,
            'contextual': QueryIntent.CONTEXTUAL,
            'hybrid_similarity_genre': QueryIntent.HYBRID_SIMILARITY_GENRE,
            
            # Additional legacy mappings
            'genre_exploration': QueryIntent.GENRE_MOOD,
            'mood': QueryIntent.GENRE_MOOD,
            'similar': QueryIntent.ARTIST_SIMILARITY,
            'underground': QueryIntent.DISCOVERY,
            'serendipity': QueryIntent.DISCOVERING_SERENDIPITY,
            'surprise': QueryIntent.DISCOVERING_SERENDIPITY
        }
        
        intent_enum = legacy_mapping.get(intent_str.lower())
        if intent_enum is None:
            self.logger.warning(f"Unknown legacy intent '{intent_str}', defaulting to DISCOVERY")
            intent_enum = QueryIntent.DISCOVERY
        
        return intent_enum
    
    def get_strategy_by_name(self, strategy_name: str) -> BaseGenerationStrategy:
        """
        Get a specific strategy instance by name.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy instance
            
        Raises:
            KeyError: If strategy name is not found
        """
        if strategy_name not in self._strategies:
            raise KeyError(f"Strategy '{strategy_name}' not found. Available: {list(self._strategies.keys())}")
        
        return self._strategies[strategy_name]
    
    def list_available_strategies(self) -> List[str]:
        """
        List all available strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())
    
    def list_supported_intents(self) -> List[QueryIntent]:
        """
        List all supported QueryIntent enums.
        
        Returns:
            List of supported QueryIntent enums
        """
        return list(self._intent_strategy_map.keys()) 