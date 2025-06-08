"""
Generation Strategies Package for Unified Candidate Generator

This package contains the modular strategy components that implement
different candidate generation approaches using the Strategy Pattern.
"""

from .base_strategy import BaseGenerationStrategy
from .factory import StrategyFactory

# Strategy implementations
from .artist_strategies import TargetArtistStrategy, SimilarArtistStrategy
from .genre_strategies import GenreExplorationStrategy, GenreFocusedStrategy, RandomGenreStrategy
from .discovery_strategies import UndergroundGemsStrategy, SerendipitousDiscoveryStrategy
from .mood_strategies import MoodBasedSerendipityStrategy, MoodFilteredTracksStrategy

__all__ = [
    # Base and factory
    'BaseGenerationStrategy',
    'StrategyFactory',
    
    # Artist strategies
    'TargetArtistStrategy',
    'SimilarArtistStrategy',
    
    # Genre strategies
    'GenreExplorationStrategy',
    'GenreFocusedStrategy',
    'RandomGenreStrategy',
    
    # Discovery strategies
    'UndergroundGemsStrategy',
    'SerendipitousDiscoveryStrategy',
    
    # Mood strategies
    'MoodBasedSerendipityStrategy',
    'MoodFilteredTracksStrategy'
] 