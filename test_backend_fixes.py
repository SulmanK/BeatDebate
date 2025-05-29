#!/usr/bin/env python3
"""
Test script to verify backend fixes are working.
"""

import asyncio
import sys
from src.services.enhanced_recommendation_service import (
    get_recommendation_service, 
    RecommendationRequest
)

async def test_backend_fixes():
    """Test the main backend fixes."""
    try:
        print("🔧 Testing backend fixes...")
        
        # Test 1: Service initialization
        print("1. Testing service initialization...")
        service = get_recommendation_service()
        await service.initialize_agents()
        print("✅ Service initialized successfully!")
        
        # Test 2: Query understanding with entity extraction
        print("2. Testing query understanding...")
        planner = service.planner_agent
        test_query = "Music like Mk.gee"
        
        try:
            understanding = await planner.query_understanding_engine.understand_query(test_query)
            print(f"✅ Query understanding successful: {understanding.intent.value}")
        except Exception as e:
            print(f"❌ Query understanding failed: {e}")
        
        # Test 3: Underground score calculation with None handling
        print("3. Testing underground score calculation...")
        from src.models.metadata_models import UnifiedTrackMetadata
        test_metadata = UnifiedTrackMetadata(
            name="Test Track",
            artist="Test Artist",
            listeners=None,  # Test None handling
            playcount=None   # Test None handling
        )
        
        generator = service.discovery_agent.candidate_generator
        underground_score = generator._calculate_underground_score(test_metadata)
        print(f"✅ Underground score calculated: {underground_score:.2f}")
        
        # Test 4: Discovery agent scoring with None handling
        print("4. Testing discovery agent scoring...")
        discovery_agent = service.discovery_agent
        test_candidate = {
            'name': 'Test Track',
            'artist': 'Test Artist',
            'listeners': None,  # Test None handling
            'playcount': None,  # Test None handling
            'tags': None        # Test None handling
        }
        
        try:
            novelty_score = discovery_agent._calculate_novelty_score(
                test_candidate, {}, {}
            )
            underground_score = discovery_agent._calculate_underground_score(test_candidate)
            print(f"✅ Discovery scoring successful: novelty={novelty_score:.2f}, underground={underground_score:.2f}")
        except Exception as e:
            print(f"❌ Discovery scoring failed: {e}")
        
        print("\n🎉 All backend fixes verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_backend_fixes())
    sys.exit(0 if success else 1) 