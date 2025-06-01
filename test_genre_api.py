#!/usr/bin/env python3
"""
Quick test of the new genre checking API methods.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append('src')

from services.api_service import get_api_service

async def test_genre_checking():
    """Test the new genre checking functionality."""
    
    print("🧪 Testing new genre checking API methods...")
    
    # Get API service
    api_service = get_api_service()
    
    # Test cases
    test_cases = [
        ("Michael Jackson", "r&b"),
        ("3T", "r&b"),
        ("The Jackson 5", "r&b"),
        ("Radiohead", "rock"),
        ("Frank Ocean", "r&b"),
    ]
    
    print("\n1. Testing artist genre checking:")
    for artist, genre in test_cases:
        try:
            result = await api_service.check_artist_genre_match(
                artist=artist,
                target_genre=genre,
                include_related_genres=True
            )
            
            match_status = "✅" if result['matches'] else "❌"
            print(f"{match_status} {artist} + {genre}: {result['confidence']:.2f} confidence")
            print(f"   Match type: {result['match_type']}")
            print(f"   Matched tags: {result['matched_tags']}")
            print(f"   Artist tags: {result.get('artist_tags', [])[:5]}")  # Show first 5 tags
            print()
            
        except Exception as e:
            print(f"❌ Error testing {artist} + {genre}: {e}")
    
    print("\n2. Testing track genre checking:")
    track_test_cases = [
        ("Michael Jackson", "Billie Jean", "r&b"),
        ("3T", "With You", "r&b"),
        ("The Jackson 5", "I'll Be There", "r&b"),
    ]
    
    for artist, track, genre in track_test_cases:
        try:
            result = await api_service.check_track_genre_match(
                artist=artist,
                track=track,
                target_genre=genre,
                include_related_genres=True
            )
            
            match_status = "✅" if result['matches'] else "❌"
            print(f"{match_status} {artist} - {track} + {genre}: {result['confidence']:.2f} confidence")
            print(f"   Match type: {result['match_type']}")
            print(f"   Matched tags: {result['matched_tags']}")
            print(f"   Track tags: {result.get('track_tags', [])}")
            print()
            
        except Exception as e:
            print(f"❌ Error testing {artist} - {track} + {genre}: {e}")
    
    print("🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_genre_checking()) 