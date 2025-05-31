#!/usr/bin/env python3
"""
Test Enhanced Hybrid System Implementation

Tests the new hybrid sub-type detection and dynamic scoring system to ensure:
1. Correct sub-type detection for different query patterns
2. Appropriate scoring weights are applied
3. Discovery-primary queries prioritize novelty
4. Similarity-primary queries prioritize artist similarity
"""

import asyncio
import structlog
from typing import Dict, Any
from src.agents.components.query_analysis_utils import QueryAnalysisUtils
from src.agents.judge.ranking_logic import RankingLogic
from src.agents.planner.query_understanding_engine import QueryUnderstandingEngine
from src.services.api_service import APIService
from src.models.recommendation_models import TrackRecommendation


# Test Query Set
TEST_QUERIES = {
    'discovery_primary': [
        "Find me underground indie rock",
        "New experimental electronic music",
        "Hidden electronic gems",
        "Discover obscure jazz artists",
        "Find me rare underground hip hop"
    ],
    'similarity_primary': [
        "Music like Kendrick Lamar but jazzy",
        "Chill songs like Bon Iver",
        "Electronic music similar to Aphex Twin",
        "Songs that sound like Radiohead but experimental",
        "Artists like Mk.gee with similar style"
    ],
    'genre_primary': [
        "Upbeat indie rock with electronic elements", 
        "Dark ambient with jazz influences",
        "Aggressive punk with melodic vocals",
        "Dreamy shoegaze with electronic textures",
        "Groovy funk with hip hop influences"
    ]
}

# Expected scoring weights for verification
EXPECTED_WEIGHTS = {
    'hybrid_discovery_primary': {
        'novelty': 0.5,
        'genre_mood_match': 0.4,
        'quality': 0.1
    },
    'hybrid_similarity_primary': {
        'similarity': 0.5,
        'genre_mood_match': 0.3,
        'quality': 0.2
    },
    'hybrid_genre_primary': {
        'genre_mood_match': 0.6,
        'novelty': 0.25,
        'quality': 0.15
    }
}

EXPECTED_THRESHOLDS = {
    'hybrid_discovery_primary': 0.6,   # Strict - underground focus
    'hybrid_similarity_primary': 0.25, # Relaxed - similarity focus  
    'hybrid_genre_primary': 0.35       # Moderate - balanced approach
}


async def test_hybrid_subtype_detection():
    """Test hybrid sub-type detection accuracy."""
    print("🔧 Testing Hybrid Sub-Type Detection\n")
    
    # Initialize query analysis utils
    query_utils = QueryAnalysisUtils()
    
    for expected_subtype, queries in TEST_QUERIES.items():
        print(f"📋 Testing {expected_subtype.replace('_', '-').title()} Queries:")
        
        for query in queries:
            # Mock entities for similarity queries
            entities = {}
            if 'like' in query.lower():
                # Extract artist name for similarity queries
                parts = query.lower().split('like')[1].strip().split()[0]
                entities = {
                    'musical_entities': {
                        'artists': {
                            'primary': [parts.title()]
                        }
                    }
                }
            
            detected_subtype = query_utils.detect_hybrid_subtype(query, entities)
            expected = expected_subtype.replace('_primary', '_primary').replace('_', '_')
            
            status = "✅" if detected_subtype == expected else "❌"
            print(f"  {status} '{query}' → {detected_subtype}")
            
            if detected_subtype != expected:
                print(f"     Expected: {expected}")
        
        print()


async def test_dynamic_scoring_weights():
    """Test that correct scoring weights are applied for each sub-type."""
    print("🔧 Testing Dynamic Scoring Weights\n")
    
    ranking_logic = RankingLogic()
    
    for subtype, expected_weights in EXPECTED_WEIGHTS.items():
        actual_weights = ranking_logic.get_intent_weights(subtype)
        
        print(f"📊 {subtype.replace('_', '-').title()}:")
        print(f"   Expected: {expected_weights}")
        print(f"   Actual:   {actual_weights}")
        
        # Check if weights match
        weights_match = actual_weights == expected_weights
        status = "✅" if weights_match else "❌"
        print(f"   {status} Weights {'match' if weights_match else 'do not match'}")
        
        # Test novelty thresholds
        expected_threshold = EXPECTED_THRESHOLDS[subtype]
        actual_threshold = ranking_logic.get_novelty_threshold(subtype)
        
        threshold_match = actual_threshold == expected_threshold
        status = "✅" if threshold_match else "❌"
        print(f"   {status} Threshold: {actual_threshold} (expected: {expected_threshold})")
        print()


async def test_full_query_understanding_flow():
    """Test the complete query understanding flow with hybrid detection."""
    print("🔧 Testing Full Query Understanding Flow\n")
    
    try:
        # Initialize with minimal config
        api_service = APIService()
        understanding_engine = QueryUnderstandingEngine(api_service=api_service)
        
        # Test key queries
        test_cases = [
            ("Find me underground indie rock", "discovery_primary"),
            ("Music like Kendrick Lamar but jazzy", "similarity_primary"),  
            ("Upbeat indie rock with electronic elements", "genre_primary")
        ]
        
        for query, expected_subtype in test_cases:
            print(f"🔍 Query: '{query}'")
            
            try:
                # Process query through understanding engine
                understanding = await understanding_engine.analyze_query(query)
                
                print(f"   Intent: {understanding.intent}")
                print(f"   Reasoning: {understanding.reasoning}")
                
                # Check if hybrid sub-type is in reasoning
                if 'Hybrid sub-type:' in understanding.reasoning:
                    detected_subtype = understanding.reasoning.split('Hybrid sub-type:')[1].strip()
                    status = "✅" if detected_subtype == expected_subtype else "❌"
                    print(f"   {status} Sub-type: {detected_subtype} (expected: {expected_subtype})")
                else:
                    print(f"   ❌ No sub-type detected in reasoning")
                
            except Exception as e:
                print(f"   ❌ Error processing query: {e}")
            
            print()
            
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")


def create_sample_candidates() -> list:
    """Create sample candidates for scoring tests."""
    return [
        TrackRecommendation(
            title="Underground Track",
            artist="Unknown Artist", 
            genres=["underground", "indie"],
            source="discovery_agent",
            confidence=0.8
        ),
        TrackRecommendation(
            title="Popular Hit",
            artist="Famous Artist",
            genres=["pop", "mainstream"],
            source="genre_mood_agent", 
            confidence=0.9
        ),
        TrackRecommendation(
            title="Similar Style",
            artist="Kendrick Lamar",
            genres=["hip hop", "jazz"],
            source="discovery_agent",
            confidence=0.85
        )
    ]


async def test_scoring_priorities():
    """Test that different sub-types prioritize different scoring components."""
    print("🔧 Testing Scoring Priorities\n")
    
    ranking_logic = RankingLogic()
    candidates = create_sample_candidates()
    
    # Mock scoring for each candidate
    mock_scores = [
        {'novelty_score': 0.9, 'quality_score': 0.6, 'contextual_relevance': 0.7},  # Underground
        {'novelty_score': 0.2, 'quality_score': 0.9, 'contextual_relevance': 0.8},  # Popular
        {'novelty_score': 0.4, 'quality_score': 0.8, 'contextual_relevance': 0.9},  # Similar
    ]
    
    scored_candidates = list(zip(candidates, mock_scores))
    
    for subtype in ['hybrid_discovery_primary', 'hybrid_similarity_primary', 'hybrid_genre_primary']:
        print(f"📊 Testing {subtype.replace('_', '-').title()} Scoring:")
        
        # Mock entities for similarity scoring
        entities = {
            'musical_entities': {
                'artists': {
                    'primary': ['Kendrick Lamar']
                }
            }
        }
        
        # Calculate scores for each candidate
        weights = ranking_logic.get_intent_weights(subtype)
        
        for i, (candidate, scores) in enumerate(scored_candidates):
            final_score = ranking_logic._calculate_intent_aware_score(
                candidate, scores.copy(), subtype, weights, entities, {}
            )
            
            print(f"   {candidate.title}: {final_score:.3f}")
            
        print()


async def main():
    """Run all hybrid system tests."""
    structlog.configure(
        processors=[structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.testing.LogCapture,
        logger_factory=structlog.testing.TestingLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    print("🚀 Enhanced Hybrid System Tests\n")
    print("=" * 50)
    
    # Run all tests
    await test_hybrid_subtype_detection()
    await test_dynamic_scoring_weights()  
    await test_full_query_understanding_flow()
    await test_scoring_priorities()
    
    print("✅ All tests completed!")
    print("\n💡 Key Success Indicators:")
    print("   • Discovery-primary queries should detect correctly") 
    print("   • Similarity-primary queries should boost target artists")
    print("   • Genre-primary queries should balance style and discovery")
    print("   • Scoring weights should match design specifications")


if __name__ == "__main__":
    asyncio.run(main()) 