#!/usr/bin/env python3
"""
Quick test to verify the state management fix works
"""
import asyncio
import sys
import os
sys.path.append('src')

from services.enhanced_recommendation_service import get_recommendation_service, RecommendationRequest

async def test_state_management_fix():
    print('🧪 Testing state management fix...')
    
    # Set up environment
    os.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', 'test-key')
    os.environ['LASTFM_API_KEY'] = os.environ.get('LASTFM_API_KEY', 'test-key')
    os.environ['SPOTIFY_CLIENT_ID'] = os.environ.get('SPOTIFY_CLIENT_ID', 'test-id')
    os.environ['SPOTIFY_CLIENT_SECRET'] = os.environ.get('SPOTIFY_CLIENT_SECRET', 'test-secret')
    
    service = get_recommendation_service()
    await service.initialize_agents()
    
    # Simulate follow-up query with conversation history
    request = RecommendationRequest(
        query='More tracks by Kendrick Lamar',
        session_id='test_session',
        max_recommendations=10,
        context={
            'previous_queries': ['Music by Kendrick Lamar'],
            'previous_recommendations': [[
                {'artist': 'Kendrick Lamar', 'title': 'HUMBLE.', 'album': 'DAMN.'},
                {'artist': 'Kendrick Lamar', 'title': 'Alright', 'album': 'To Pimp a Butterfly'},
                {'artist': 'Kendrick Lamar', 'title': 'm.A.A.d city', 'album': 'good kid, m.A.A.d city'},
                {'artist': 'Kendrick Lamar', 'title': 'DNA.', 'album': 'DAMN.'},
                {'artist': 'Kendrick Lamar', 'title': 'King Kunta', 'album': 'To Pimp a Butterfly'}
            ]]
        }
    )
    
    print('📋 Testing with follow-up query and 5 previous Kendrick Lamar tracks...')
    print('🔍 Looking for logs showing:')
    print('   - 🎯 STATE PREPARED: recently_shown=5 tracks')
    print('   - 🎯 DISCOVERY AGENT STATE: recently_shown=5 tracks')
    print('   - 🎯 CANDIDATE SCALING: [scaled number] (from 100)')
    print()
    
    try:
        response = await service.get_recommendations(request)
        print(f'✅ SUCCESS: Got {len(response.recommendations)} recommendations')
        
        # Check if it detected follow-up correctly
        context_decision = response.metadata.get('context_decision', {})
        if context_decision.get('is_followup'):
            print(f'🎯 FOLLOW-UP DETECTED: {context_decision.get("target_entity")}')
        else:
            print(f'❌ FOLLOW-UP NOT DETECTED')
            
        return len(response.recommendations)
        
    except Exception as e:
        print(f'❌ ERROR: {e}')
        import traceback
        traceback.print_exc()
        return 0
    finally:
        await service.close()

if __name__ == "__main__":
    result = asyncio.run(test_state_management_fix())
    print(f'🏁 Final result: {result} recommendations') 