"""
Discovery Agent Module

Simplified discovery agent with focused responsibilities:
- Multi-hop similarity exploration
- Underground and hidden gem detection
- Serendipitous discovery beyond mainstream music
"""

from .agent import DiscoveryAgent
from .similarity_explorer import SimilarityExplorer
from .underground_detector import UndergroundDetector

__all__ = [
    "DiscoveryAgent",
    "SimilarityExplorer",
    "UndergroundDetector",
] 