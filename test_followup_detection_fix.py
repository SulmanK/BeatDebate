#!/usr/bin/env python3
"""
Test script to verify follow-up detection fix handles different artists correctly
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.enhanced_recommendation_service import EnhancedRecommendationService, RecommendationRequest

async def test_followup_detection_fix():
    """Test that different artists are not detected as follow-ups"""
    print("🧪 Testing Follow-up Detection Fix")
    print("=" * 50)
    
    # Initialize the service
    service = EnhancedRecommendationService()
    await service.initialize_agents()
    
    try:
        # Test Case 1: First query - Mk.gee
        print("\n🎵 TEST CASE 1: First Query - 'Discover underground tracks by Mk.gee'")
        request1 = RecommendationRequest(
            query="Discover underground tracks by Mk.gee",
            max_recommendations=5,
            session_id="test_session_fix",
            context={
                "previous_queries": [],
                "previous_recommendations": []
            }
        )
        
        response1 = await service.get_recommendations(request1)
        print(f"✅ First query completed: {len(response1.recommendations)} tracks found")
        
        # Test Case 2: Different artist - should NOT be follow-up
        print("\n🎵 TEST CASE 2: Different Artist - 'Discover underground tracks by Kendrick Lamar'")
        request2 = RecommendationRequest(
            query="Discover underground tracks by Kendrick Lamar",
            max_recommendations=5,
            session_id="test_session_fix",
            context={
                "previous_queries": ["Discover underground tracks by Mk.gee"],
                "previous_recommendations": [
                    [
                        {"artist": track.artist, "title": track.name}
                        for track in response1.recommendations[:5]
                    ]
                ]
            }
        )
        
        response2 = await service.get_recommendations(request2)
        print(f"✅ Second query completed: {len(response2.recommendations)} tracks found")
        
        # Analyze results
        print("\n📊 RESULTS ANALYSIS:")
        print("=" * 30)
        
        # Check if any tracks are from Mk.gee (should be 0 for Kendrick query)
        mkgee_tracks_in_kendrick = [
            track for track in response2.recommendations 
            if "mk.gee" in track.artist.lower()
        ]
        
        # Check if any tracks are actually from Kendrick Lamar
        kendrick_tracks = [
            track for track in response2.recommendations 
            if "kendrick" in track.artist.lower()
        ]
        
        print(f"🔍 Mk.gee tracks in Kendrick query: {len(mkgee_tracks_in_kendrick)}")
        print(f"🔍 Kendrick tracks in response: {len(kendrick_tracks)}")
        
        # Success criteria
        if len(mkgee_tracks_in_kendrick) == 0:
            print("✅ SUCCESS: No Mk.gee tracks in Kendrick query (follow-up NOT detected)")
        else:
            print("❌ FAILURE: Mk.gee tracks found in Kendrick query (follow-up incorrectly detected)")
            for track in mkgee_tracks_in_kendrick:
                print(f"   - {track.artist} - {track.name}")
        
        if len(kendrick_tracks) > 0:
            print("✅ SUCCESS: Found actual Kendrick Lamar tracks")
            for track in kendrick_tracks[:3]:
                print(f"   - {track.artist} - {track.name}")
        else:
            print("⚠️  WARNING: No Kendrick Lamar tracks found (intent detection may need work)")
        
        # Test Case 3: Same artist follow-up - should BE detected as follow-up
        print("\n🎵 TEST CASE 3: Same Artist Follow-up - 'More from Mk.gee'")
        request3 = RecommendationRequest(
            query="More from Mk.gee",
            max_recommendations=5,
            session_id="test_session_fix",
            context={
                "previous_queries": ["Discover underground tracks by Mk.gee"],
                "previous_recommendations": [
                    [
                        {"artist": track.artist, "title": track.name}
                        for track in response1.recommendations[:5]
                    ]
                ]
            }
        )
        
        response3 = await service.get_recommendations(request3)
        print(f"✅ Third query completed: {len(response3.recommendations)} tracks found")
        
        # Check for duplicates with first query
        first_track_ids = {f"{track.artist.lower()}::{track.name.lower()}" for track in response1.recommendations}
        third_track_ids = {f"{track.artist.lower()}::{track.name.lower()}" for track in response3.recommendations}
        duplicates = first_track_ids.intersection(third_track_ids)
        
        print(f"🔍 Duplicates between first and third query: {len(duplicates)}")
        
        if len(duplicates) == 0:
            print("✅ SUCCESS: No duplicates in same-artist follow-up (duplicate prevention working)")
        else:
            print("⚠️  NOTE: Some duplicates found (may be expected if database is limited)")
            for dup in list(duplicates)[:3]:
                print(f"   - {dup}")
                
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await service.close()

if __name__ == "__main__":
    asyncio.run(test_followup_detection_fix()) 