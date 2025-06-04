#!/usr/bin/env python3
"""
Test follow-up query behavior to ensure no duplicates
"""

import asyncio
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_artist_deep_dive():
    """Test artist deep dive follow-up behavior."""
    
    from src.services.enhanced_recommendation_service import EnhancedRecommendationService, RecommendationRequest
    from src.services.api_service import APIService
    
    print("🧪 Testing Artist Deep Dive")
    print("=" * 60)
    
    # Initialize services
    api_service = APIService()
    service = EnhancedRecommendationService(api_service)
    
    # Step 1: Initial query
    print("🎵 STEP 1: Initial Query - 'Music by Mk.gee'")
    
    request1 = RecommendationRequest(
        query="Music by Mk.gee",
        max_recommendations=5,
        session_id="test_session_artist_dive"
    )
    
    response1 = await service.get_recommendations(request1)
    initial_tracks = response1.recommendations
    print(f"✅ Initial query returned {len(initial_tracks)} recommendations")
    
    print("Initial recommendations:")
    for i, rec in enumerate(initial_tracks, 1):
        print(f"  {i}. {rec.name} by {rec.artist}")
    print()
    
    # Convert to conversation history format 
    conversation_history = [
        {
            "query": "Music by Mk.gee",
            "recommendations": [
                {"artist": track.artist, "title": track.name}
                for track in initial_tracks
            ]
        }
    ]
    
    # Step 2: Follow-up query
    print("🎵 STEP 2: Follow-up Query - 'More from Mk.gee'")
    
    request2 = RecommendationRequest(
        query="More from Mk.gee",
        max_recommendations=10,
        session_id="test_session_artist_dive",
        context={
            "previous_queries": ["Music by Mk.gee"],
            "previous_recommendations": [
                [{"artist": track.artist, "title": track.name} for track in initial_tracks]
            ]
        }
    )
    
    response2 = await service.get_recommendations(request2)
    followup_tracks = response2.recommendations
    print(f"✅ Follow-up query returned {len(followup_tracks)} recommendations")
    
    # Check for duplicates
    initial_track_ids = {f"{track.artist.lower()}::{track.name.lower()}" for track in initial_tracks}
    duplicates = []
    new_tracks = []
    
    for track in followup_tracks:
        track_id = f"{track.artist.lower()}::{track.name.lower()}"
        if track_id in initial_track_ids:
            duplicates.append(f"{track.name} by {track.artist}")
        else:
            new_tracks.append(f"{track.name} by {track.artist}")
    
    print("Follow-up recommendations:")
    for i, rec in enumerate(followup_tracks, 1):
        status = "❌ DUPLICATE" if f"{rec.artist.lower()}::{rec.name.lower()}" in initial_track_ids else "+ NEW"
        print(f"  {status}: {rec.name} by {rec.artist}")
    print()
    
    print(f"Found {len(followup_tracks)} tracks in follow-up")
    print(f"Duplicates: {len(duplicates)}")
    
    # Results
    if len(duplicates) == 0:
        print("✅ PASS: No duplicates found!")
    else:
        print(f"❌ FAIL: Found {len(duplicates)} duplicates")
        for dup in duplicates:
            print(f"    - {dup}")

async def main():
    await test_artist_deep_dive()

if __name__ == "__main__":
    asyncio.run(main()) 