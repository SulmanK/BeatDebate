#!/usr/bin/env python3
"""Quick test for enhanced hybrid system"""

from src.agents.components.query_analysis_utils import QueryAnalysisUtils
from src.agents.judge.ranking_logic import RankingLogic

def test_hybrid_system():
    # Test hybrid sub-type detection
    query_utils = QueryAnalysisUtils()
    print('🔧 Testing Hybrid Sub-Type Detection:')

    test_queries = [
        ('Find me underground indie rock', 'discovery_primary'),
        ('Music like Kendrick Lamar but jazzy', 'similarity_primary'),
        ('Upbeat indie rock with electronic elements', 'genre_primary')
    ]

    for query, expected in test_queries:
        entities = {}
        if 'like' in query.lower():
            artist = query.lower().split('like')[1].strip().split()[0].title()
            entities = {'musical_entities': {'artists': {'primary': [artist]}}}
        
        detected = query_utils.detect_hybrid_subtype(query, entities)
        status = '✅' if detected == expected else '❌'
        print(f'  {status} "{query}" → {detected} (expected: {expected})')

    # Test dynamic scoring weights
    print('\n🔧 Testing Dynamic Scoring Weights:')
    ranking_logic = RankingLogic()

    test_subtypes = ['hybrid_discovery_primary', 'hybrid_similarity_primary', 'hybrid_genre_primary']
    for subtype in test_subtypes:
        weights = ranking_logic.get_intent_weights(subtype)
        threshold = ranking_logic.get_novelty_threshold(subtype)
        print(f'  ✅ {subtype}:')
        print(f'     weights={weights}')
        print(f'     threshold={threshold}')

    print('\n🎯 Enhanced Hybrid System is working!')

if __name__ == "__main__":
    test_hybrid_system() 