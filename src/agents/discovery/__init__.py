"""
Discovery Agent Module

Phase 4: Modularized discovery agent with specialized components:
- Multi-hop similarity exploration
- Underground and hidden gem detection
- Serendipitous discovery beyond mainstream music

Components:
- DiscoveryAgent: Main agent (refactored)
- DiscoveryConfig: Intent parameter management
- DiscoveryScorer: Discovery-specific scoring
- DiscoveryFilter: Discovery-specific filtering
- DiscoveryDiversity: Diversity management
- SimilarityExplorer: Similarity exploration utilities
- UndergroundDetector: Underground detection utilities
"""

from .agent import DiscoveryAgent
from .discovery_config import DiscoveryConfig
from .discovery_scorer import DiscoveryScorer
from .discovery_filter import DiscoveryFilter
from .discovery_diversity import DiscoveryDiversity
from .similarity_explorer import SimilarityExplorer
from .underground_detector import UndergroundDetector

__all__ = [
    "DiscoveryAgent",
    "DiscoveryConfig",
    "DiscoveryScorer", 
    "DiscoveryFilter",
    "DiscoveryDiversity",
    "SimilarityExplorer",
    "UndergroundDetector",
] 