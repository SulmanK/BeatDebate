#!/usr/bin/env python3
"""
Simple test to verify track extraction logic
"""

def test_extraction_logic():
    """Test the _extract_recently_shown_tracks logic separately."""
    
    from src.services.enhanced_recommendation_service import EnhancedRecommendationService
    from src.services.api_service import APIService
    
    print("🧪 Testing Extraction Logic")
    print("=" * 60)
    
    # Create service instance
    api_service = APIService()
    service = EnhancedRecommendationService(api_service)
    
    # Test conversation history
    conversation_history = [
        {
            "query": "Music like Mk.gee",
            "recommendations": [
                {"artist": "ML BUCH", "title": "I'm A Girl You Can Hold IRL"},
                {"artist": "Chanel Beads", "title": "Police Scanner"},
                {"artist": "DIJON", "title": "do you light up?"}
            ]
        }
    ]
    
    # Test context override for "More like that"
    context_override = {
        'is_followup': True,
        'intent_override': 'style_continuation',
        'target_entity': '',
        'confidence': 0.9
    }
    
    # Extract recently shown tracks
    recently_shown = service._extract_recently_shown_tracks(
        conversation_history, 
        context_override
    )
    
    print(f"Extracted {len(recently_shown)} recently shown track IDs:")
    for track_id in recently_shown:
        print(f"  - {track_id}")
    
    expected_count = 3  # Should match the 3 tracks in conversation history
    if len(recently_shown) == expected_count:
        print("✅ SUCCESS: Extraction logic working correctly!")
        return True
    else:
        print(f"❌ FAILURE: Expected {expected_count} tracks, got {len(recently_shown)}")
        return False

if __name__ == "__main__":
    test_extraction_logic() 