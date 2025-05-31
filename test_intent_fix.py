#!/usr/bin/env python3
"""Test script for intent-aware recommendation fixes."""

import asyncio
import sys
sys.path.append('src')

from services.enhanced_recommendation_service import EnhancedRecommendationService
from services.cache_manager import CacheManager
from services.api_service import APIService
from services.conversation_context_service import ConversationContextService

async def test_fixed_system():
    """Test the fixed intent-aware recommendation system."""
    cache_manager = CacheManager()
    api_service = APIService(cache_manager=cache_manager)
    context_service = ConversationContextService()
    
    service = EnhancedRecommendationService(
        api_service=api_service,
        context_service=context_service,
        cache_manager=cache_manager
    )
    
    try:
        print("🎵 Testing: 'Music like Mk.gee'")
        result = await service.get_recommendations(
            query='Music like Mk.gee',
            session_id='test-session-fixed'
        )
        
        print(f'✅ Got {len(result)} recommendations:')
        target_artist_count = 0
        similar_artist_count = 0
        
        for i, rec in enumerate(result, 1):
            is_target = rec.artist.lower() == 'mk.gee'
            if is_target:
                target_artist_count += 1
                print(f'  {i}. {rec.artist} - {rec.name} 🎯 TARGET')
            else:
                similar_artist_count += 1
                print(f'  {i}. {rec.artist} - {rec.name} 🔍 SIMILAR')
        
        print(f'\n📊 Summary:')
        print(f'   Target artist tracks: {target_artist_count}')
        print(f'   Similar artist tracks: {similar_artist_count}')
        print(f'   Total recommendations: {len(result)}')
        
        if len(result) >= 3 and similar_artist_count > 0:
            print('✅ SUCCESS: System is working correctly!')
        else:
            print('❌ ISSUE: Still not generating enough diverse recommendations')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
    
    finally:
        await api_service.close()

if __name__ == '__main__':
    asyncio.run(test_fixed_system()) 