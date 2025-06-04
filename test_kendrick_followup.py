#!/usr/bin/env python3
"""Test Kendrick Lamar follow-up scenario that was failing."""

from src.services.enhanced_recommendation_service import EnhancedRecommendationService, RecommendationRequest
import asyncio

async def test_kendrick_followup():
    print('Testing Kendrick Lamar follow-up after Music by Kendrick Lamar...')
    
    service = EnhancedRecommendationService()
    
    # First query - Music by Kendrick Lamar
    request1 = RecommendationRequest(
        query='Music by Kendrick Lamar',
        max_recommendations=10,
        session_id='test_kendrick_session'
    )
    result1 = await service.get_recommendations(request1)
    
    print(f'First query returned {len(result1.recommendations)} tracks')
    first_track_ids = [f"{r.artist.lower()}::{r.name.lower()}" for r in result1.recommendations]
    print(f'First query track IDs: {first_track_ids[:3]}...')
    
    # Follow-up query - More Kendrick Lamar tracks  
    request2 = RecommendationRequest(
        query='More Kendrick Lamar tracks',
        max_recommendations=10,
        session_id='test_kendrick_session'
    )
    result2 = await service.get_recommendations(request2)
    
    print(f'Follow-up query returned {len(result2.recommendations)} tracks')
    second_track_ids = [f"{r.artist.lower()}::{r.name.lower()}" for r in result2.recommendations]
    
    # Check for duplicates
    duplicates = set(first_track_ids) & set(second_track_ids)
    print(f'Duplicates: {len(duplicates)}')
    if duplicates:
        print(f'Duplicate tracks: {list(duplicates)[:3]}...')
    
    if len(result2.recommendations) >= 8:
        print('✅ PASS: Follow-up returned sufficient tracks!')
    else:
        print(f'❌ FAIL: Follow-up only returned {len(result2.recommendations)} tracks, expected 8+')

if __name__ == '__main__':
    asyncio.run(test_kendrick_followup()) 