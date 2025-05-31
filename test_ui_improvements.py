#!/usr/bin/env python3
"""
Test script for UI improvements
"""


def test_ui_components():
    """Test that UI components can be imported and created."""
    try:
        from src.ui.chat_interface import create_chat_interface, QUERY_EXAMPLES
        from src.ui.response_formatter import ResponseFormatter
        
        print("✅ UI components imported successfully")
        
        # Test query examples
        print(f"✅ Query examples loaded: {len(QUERY_EXAMPLES)} categories")
        for category, examples in QUERY_EXAMPLES.items():
            print(f"   - {category}: {len(examples)} examples")
        
        # Test response formatter
        ResponseFormatter()
        print("✅ Response formatter created successfully")
        
        # Test interface creation (without launching)
        create_chat_interface()
        print("✅ Chat interface created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing UI components: {e}")
        return False


def test_preview_html():
    """Test the preview HTML generation with sample data."""
    try:
        from src.ui.chat_interface import BeatDebateChatInterface
        
        chat_interface = BeatDebateChatInterface()
        
        # Sample recommendations for testing
        sample_recommendations = [
            {
                "title": "Test Song",
                "artist": "Test Artist", 
                "confidence": 0.85,
                "source": "test"
            },
            {
                "title": "Another Song",
                "artist": "Another Artist",
                "confidence": 0.72,
                "source": "test"
            }
        ]
        
        html = chat_interface._create_lastfm_player_html(
            sample_recommendations
        )
        
        # Check that HTML contains expected elements
        assert "Last.fm" in html
        assert "Spotify" in html
        assert "YouTube" in html
        assert "Test Song" in html
        assert "Test Artist" in html
        
        print("✅ Preview HTML generation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing preview HTML: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing UI Improvements")
    print("=" * 40)
    
    success = True
    success &= test_ui_components()
    success &= test_preview_html()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!") 