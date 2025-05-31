#!/usr/bin/env python3
"""Test script to verify hybrid intent detection fixes."""

import asyncio
import sys
sys.path.append('src')

from agents.planner.query_understanding_engine import QueryUnderstandingEngine
from models.agent_models import QueryIntent

async def test_hybrid_intent_detection():
    """Test that hybrid intent is correctly detected and validated."""
    
    # Create mock LLM client and engine
    class MockLLMClient:
        async def generate_content_async(self, *args, **kwargs):
            return type('MockResponse', (), {
                'text': '''{"intent": "discovery", "musical_entities": {"artists": [], "genres": ["indie", "rock"], "tracks": [], "moods": []}, "context_factors": ["underground"], "complexity_level": "medium", "similarity_type": "moderate", "confidence": 0.9}'''
            })()
    
    engine = QueryUnderstandingEngine(MockLLMClient())
    
    # Test hybrid intent detection
    query = "Find me underground indie rock"
    print(f"\n🎯 Testing query: '{query}'")
    
    try:
        understanding = await engine.understand_query(query)
        
        print(f"✅ Intent detected: {understanding.intent}")
        print(f"✅ Intent value: {understanding.intent.value}")
        print(f"✅ Confidence: {understanding.confidence}")
        print(f"✅ Genres: {understanding.genres}")
        print(f"✅ Artists: {understanding.artists}")
        
        # Check if it's hybrid intent
        if understanding.intent == QueryIntent.HYBRID:
            print("🎉 SUCCESS: Hybrid intent correctly detected!")
            return True
        else:
            print(f"❌ FAILED: Expected hybrid, got {understanding.intent}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_hybrid_intent_detection())
    if result:
        print("\n🎉 Hybrid intent detection is working!")
    else:
        print("\n❌ Hybrid intent detection still has issues.") 