#!/usr/bin/env python3
"""Test script to verify hybrid intent detection fixes."""

import asyncio
import sys
import traceback
sys.path.append('src')

from services.enhanced_recommendation_service import EnhancedRecommendationService
from services.cache_manager import CacheManager
from services.api_service import APIService
from services.conversation_context_service import ConversationContextService


async def test_hybrid_intent_fix():
    """Test the fixed hybrid intent detection system."""
    cache_manager = CacheManager()
    api_service = APIService(cache_manager=cache_manager)
    context_service = ConversationContextService()
    
    service = EnhancedRecommendationService(
        api_service=api_service,
        context_service=context_service,
        cache_manager=cache_manager
    )
    
    try:
        print("🔧 Testing hybrid intent detection for: 'Find me underground indie rock'")
        print("Expected:")
        print("  - Intent: HYBRID (discovery + genre)")
        print("  - Agent sequence: ['discovery', 'genre_mood', 'judge']")
        print("  - Results: Underground indie rock tracks")
        print()
        
        result = await service.get_recommendations(
            query='Find me underground indie rock',
            session_id='test-hybrid-fix'
        )
        
        print(f"✅ Got {len(result)} recommendations:")
        for i, rec in enumerate(result, 1):
            print(f"  {i}. {rec.artist} - {rec.name}")
            print(f"     Genres: {rec.genres}")
            print(f"     Source: {rec.source}")
            print()
        
        # Test another hybrid query
        print("🔧 Testing: 'Chill songs like Radiohead'")
        result2 = await service.get_recommendations(
            query='Chill songs like Radiohead',
            session_id='test-hybrid-fix-2'
        )
        
        print(f"✅ Got {len(result2)} recommendations:")
        for i, rec in enumerate(result2[:3], 1):  # Show first 3
            print(f"  {i}. {rec.artist} - {rec.name}")
            print(f"     Genres: {rec.genres}")
            print(f"     Source: {rec.source}")
        print()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
    finally:
        await api_service.close()


if __name__ == "__main__":
    asyncio.run(test_hybrid_intent_fix()) 