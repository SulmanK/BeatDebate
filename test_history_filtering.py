#!/usr/bin/env python3
"""
Test script to verify history tracking and duplicate filtering for follow-up queries
"""

import asyncio
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_history_filtering_end_to_end():
    """Test that follow-up queries don't return duplicate recommendations."""
    
    # Import here to avoid circular dependencies
    from src.services.enhanced_recommendation_service import EnhancedRecommendationService, RecommendationRequest
    from src.services.api_service import APIService
    
    # Initialize services
    api_service = APIService()
    service = EnhancedRecommendationService(api_service)
    
    print("🧪 Testing End-to-End History Filtering")
    print("=" * 60)
    
    session_id = "test_session_e2e"
    
    # Step 1: Initial query
    print("🎵 STEP 1: Initial Query - 'Music like Mk.gee'")
    
    request1 = RecommendationRequest(
        query="Music like Mk.gee",
        max_recommendations=5,
        session_id=session_id
    )
    
    response1 = await service.get_recommendations(request1)
    initial_tracks = response1.recommendations
    
    print(f"✅ Initial query returned {len(initial_tracks)} recommendations")
    
    initial_track_names = []
    initial_track_ids = set()
    
    print("Initial recommendations:")
    for i, rec in enumerate(initial_tracks, 1):
        track_name = f"{rec.name} by {rec.artist}"
        initial_track_names.append(track_name)
        track_id = f"{rec.artist.lower()}::{rec.name.lower()}"
        initial_track_ids.add(track_id)
        print(f"  {i}. {track_name}")
    print()
    
    # Step 2: Follow-up query with correct context format
    print("🎵 STEP 2: Follow-up Query - 'More like that'")
    
    # ✅ CORRECT FORMAT: Use the request.context format that the real chat interface uses
    request2 = RecommendationRequest(
        query="More like that",
        max_recommendations=10,
        session_id="test_session_e2e",
        context={
            "previous_queries": ["Music like Mk.gee"],
            "previous_recommendations": [
                [
                    {"artist": track.artist, "title": track.name}
                    for track in initial_tracks
                ]
            ]
        }
    )
    
    response2 = await service.get_recommendations(request2)
    followup_tracks = response2.recommendations
    
    print(f"✅ Follow-up query returned {len(followup_tracks)} recommendations")
    
    print("Follow-up recommendations:")
    duplicates_found = []
    for i, rec in enumerate(followup_tracks, 1):
        track_name = f"{rec.name} by {rec.artist}"
        track_id = f"{rec.artist.lower()}::{rec.name.lower()}"
        
        # Check if this is a duplicate
        is_duplicate = track_id in initial_track_ids
        duplicate_indicator = " ❌ DUPLICATE" if is_duplicate else ""
        
        print(f"  {i}. {track_name}{duplicate_indicator}")
        
        if is_duplicate:
            duplicates_found.append(track_name)
    print()
    
    # Step 3: Analysis
    print("🔍 STEP 3: Duplicate Analysis")
    print(f"Initial track IDs: {len(initial_track_ids)}")
    print(f"Follow-up duplicates found: {len(duplicates_found)}")
    
    # Check response metadata for context decision
    context_decision = response2.metadata.get("context_decision", {})
    print(f"Context Analysis Results:")
    print(f"  - Is follow-up: {context_decision.get('is_followup', 'Unknown')}")
    print(f"  - Intent override: {context_decision.get('intent_override', 'None')}")
    print(f"  - Target entity: {context_decision.get('target_entity', 'None')}")
    print(f"  - Confidence: {context_decision.get('confidence', 'Unknown')}")
    
    if duplicates_found:
        print("❌ DUPLICATES FOUND:")
        for dup in duplicates_found:
            print(f"    - {dup}")
        return False
    else:
        print("✅ SUCCESS: No duplicates found in follow-up query!")
        return True

async def test_artist_deep_dive_filtering():
    """Test artist deep dive follow-up behavior."""
    
    from src.services.enhanced_recommendation_service import EnhancedRecommendationService, RecommendationRequest
    from src.services.api_service import APIService
    
    print("\n🧪 Testing Artist Deep Dive History Filtering")
    print("=" * 60)
    
    # Initialize services
    api_service = APIService()
    service = EnhancedRecommendationService(api_service)
    
    session_id = "test_session_artist_dive"
    
    # Step 1: Get some Mk.gee tracks
    print("🎵 STEP 1: Initial Query - 'Music by Mk.gee'")
    
    request1 = RecommendationRequest(
        query="Music by Mk.gee",
        max_recommendations=10,
        session_id=session_id
    )
    
    response1 = await service.get_recommendations(request1)
    initial_tracks = response1.recommendations
    
    print(f"✅ Initial query returned {len(initial_tracks)} recommendations")
    
    mkgee_tracks = []
    mkgee_track_ids = set()
    mkgee_track_count = 0
    
    for track in initial_tracks:
        if 'mk.gee' in track.artist.lower():
            mkgee_track_count += 1
            mkgee_tracks.append(track)
            track_id = f"{track.artist.lower()}::{track.name.lower()}"
            mkgee_track_ids.add(track_id)
            print(f"  - {track.name} by {track.artist}")
    
    print(f"Found {mkgee_track_count} tracks by Mk.gee")
    
    # Step 2: Follow-up query with correct context format
    print("🎵 STEP 2: Follow-up Query - 'More from Mk.gee'")
    
    # ✅ CORRECT FORMAT: Use the request.context format that the real chat interface uses
    request2 = RecommendationRequest(
        query="More from Mk.gee",
        max_recommendations=10,
        session_id="test_session_artist_dive",
        context={
            "previous_queries": ["Music by Mk.gee"],
            "previous_recommendations": [
                [
                    {"artist": track.artist, "title": track.name}
                    for track in initial_tracks
                ]
            ]
        }
    )
    
    response2 = await service.get_recommendations(request2)
    followup_tracks = response2.recommendations
    
    print(f"✅ Follow-up query returned {len(followup_tracks)} recommendations")
    
    # Check for duplicates specifically from Mk.gee
    mkgee_duplicates = []
    new_mkgee_tracks = 0
    
    for track in followup_tracks:
        if 'mk.gee' in track.artist.lower():
            new_mkgee_tracks += 1
            track_id = f"{track.artist.lower()}::{track.name.lower()}"
            
            if track_id in mkgee_track_ids:
                mkgee_duplicates.append(f"{track.name} by {track.artist}")
                print(f"  ❌ DUPLICATE: {track.name} by {track.artist}")
            else:
                print(f"  + NEW: {track.name} by {track.artist}")
    
    print(f"\nFound {new_mkgee_tracks} Mk.gee tracks in follow-up")
    print(f"Mk.gee duplicates: {len(mkgee_duplicates)}")
    
    # Check response metadata
    context_decision = response2.metadata.get("context_decision", {})
    print(f"Context Analysis Results:")
    print(f"  - Is follow-up: {context_decision.get('is_followup', 'Unknown')}")
    print(f"  - Intent override: {context_decision.get('intent_override', 'None')}")
    print(f"  - Target entity: {context_decision.get('target_entity', 'None')}")
    print(f"  - Confidence: {context_decision.get('confidence', 'Unknown')}")
    
    if mkgee_duplicates:
        print("❌ MK.GEE DUPLICATES FOUND:")
        for dup in mkgee_duplicates:
            print(f"    - {dup}")
        return False
    else:
        print("✅ SUCCESS: No duplicate Mk.gee tracks in follow-up!")
        return True

async def main():
    """Run comprehensive follow-up tests."""
    
    print("🚀 Starting Comprehensive History Filtering Tests")
    print("=" * 60)
    
    try:
        # Test 1: Style continuation ("More like that")
        style_success = await test_history_filtering_end_to_end()
        
        # Test 2: Artist deep dive ("More from [Artist]")
        artist_success = await test_artist_deep_dive_filtering()
        
        print("\n" + "=" * 60)
        print("📊 FINAL TEST RESULTS:")
        print(f"  Style Continuation ('More like that'): {'✅ PASS' if style_success else '❌ FAIL'}")
        print(f"  Artist Deep Dive ('More from Artist'): {'✅ PASS' if artist_success else '❌ FAIL'}")
        
        if style_success and artist_success:
            print("\n🎉 ALL HISTORY FILTERING TESTS PASSED!")
            print("✅ History filtering is working correctly.")
            print("✅ No duplicate recommendations in follow-up queries.")
        else:
            print("\n❌ SOME HISTORY FILTERING TESTS FAILED.")
            print("🔧 Check the history filtering implementation.")
            print("💡 This indicates the system may be returning duplicate tracks in follow-up queries.")
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 