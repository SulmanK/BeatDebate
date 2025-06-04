"""
Tests for Chat Interface Fallback Functionality

Integration tests to verify the chat interface correctly handles
fallback scenarios when the main 4-agent system fails.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.ui.chat_interface import BeatDebateChatInterface
from src.services.llm_fallback_service import FallbackTrigger


class TestChatInterfaceFallback:
    """Test suite for chat interface fallback functionality"""
    
    @pytest.fixture
    def mock_response_formatter(self):
        """Create mock response formatter"""
        formatter = Mock()
        formatter.format_recommendations = Mock(return_value="Formatted response")
        return formatter
    
    @pytest.fixture
    def mock_planning_display(self):
        """Create mock planning display"""
        display = Mock()
        return display
    
    @pytest.fixture
    def chat_interface(self, mock_response_formatter, mock_planning_display):
        """Create chat interface instance for testing"""
        with patch('src.ui.chat_interface.ResponseFormatter', return_value=mock_response_formatter), \
             patch('src.ui.chat_interface.PlanningDisplay', return_value=mock_planning_display):
            
            interface = BeatDebateChatInterface("http://test-backend:8000")
            return interface
    
    def test_should_use_fallback_api_error(self, chat_interface):
        """Test fallback trigger detection for API errors"""
        # Test with None response (API error)
        should_fallback, trigger = chat_interface._should_use_fallback(None)
        assert should_fallback is True
        assert trigger == FallbackTrigger.API_ERROR
    
    def test_should_use_fallback_unknown_intent(self, chat_interface):
        """Test fallback trigger detection for unknown intent"""
        response = {"intent": "unknown", "recommendations": []}
        should_fallback, trigger = chat_interface._should_use_fallback(response)
        assert should_fallback is True
        assert trigger == FallbackTrigger.UNKNOWN_INTENT
        
        response = {"intent": "unsupported", "recommendations": []}
        should_fallback, trigger = chat_interface._should_use_fallback(response)
        assert should_fallback is True
        assert trigger == FallbackTrigger.UNKNOWN_INTENT
    
    def test_should_use_fallback_no_recommendations(self, chat_interface):
        """Test fallback trigger detection for empty recommendations"""
        response = {"intent": "valid", "recommendations": []}
        should_fallback, trigger = chat_interface._should_use_fallback(response)
        assert should_fallback is True
        assert trigger == FallbackTrigger.NO_RECOMMENDATIONS
        
        response = {"intent": "valid"}  # Missing recommendations key
        should_fallback, trigger = chat_interface._should_use_fallback(response)
        assert should_fallback is True
        assert trigger == FallbackTrigger.NO_RECOMMENDATIONS
    
    def test_should_use_fallback_error_indicators(self, chat_interface):
        """Test fallback trigger detection for error indicators"""
        response = {"error": "Some error occurred", "recommendations": []}
        should_fallback, trigger = chat_interface._should_use_fallback(response)
        assert should_fallback is True
        assert trigger == FallbackTrigger.API_ERROR
        
        response = {"detail": "Error details", "recommendations": []}
        should_fallback, trigger = chat_interface._should_use_fallback(response)
        assert should_fallback is True
        assert trigger == FallbackTrigger.API_ERROR
    
    def test_should_not_use_fallback_valid_response(self, chat_interface):
        """Test that valid responses don't trigger fallback"""
        response = {
            "intent": "artist_similarity",
            "recommendations": [
                {"title": "Track 1", "artist": "Artist 1"},
                {"title": "Track 2", "artist": "Artist 2"}
            ]
        }
        should_fallback, trigger = chat_interface._should_use_fallback(response)
        assert should_fallback is False
        assert trigger is None
    
    def test_get_chat_context(self, chat_interface):
        """Test chat context extraction for fallback requests"""
        # Test with empty conversation history
        context = chat_interface._get_chat_context()
        assert context is None
        
        # Add some conversation history
        chat_interface.conversation_history = [
            {
                "user_message": "Music like Radiohead",
                "recommendations": [{"title": "Creep", "artist": "Radiohead"}]
            },
            {
                "user_message": "More electronic music",
                "recommendations": [{"title": "Aphex Twin", "artist": "Windowlicker"}]
            }
        ]
        
        context = chat_interface._get_chat_context()
        assert context is not None
        assert "previous_queries" in context
        assert "previous_recommendations" in context
        assert len(context["previous_queries"]) == 2
        assert "Music like Radiohead" in context["previous_queries"]
        assert "More electronic music" in context["previous_queries"]
    
    def test_create_emergency_response(self, chat_interface):
        """Test emergency response creation"""
        query = "test query"
        response = chat_interface._create_emergency_response(query)
        
        assert "SYSTEM TEMPORARILY UNAVAILABLE" in response
        assert query in response
        assert "Please try:" in response
        assert "🎵" in response  # Check for emoji
    
    @pytest.mark.asyncio
    async def test_get_fallback_recommendations_no_service(self, chat_interface):
        """Test fallback when service is not available"""
        # Ensure fallback service is None
        chat_interface.fallback_service = None
        
        result = await chat_interface._get_fallback_recommendations(
            "test query", 
            FallbackTrigger.UNKNOWN_INTENT
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_fallback_recommendations_success(self, chat_interface):
        """Test successful fallback recommendations"""
        # Mock fallback service
        mock_fallback_service = Mock()
        mock_fallback_response = {
            "recommendations": [
                {"title": "Test Track", "artist": "Test Artist", "confidence": 0.8}
            ],
            "explanation": "Test explanation",
            "fallback_used": True,
            "fallback_reason": "unknown_intent"
        }
        mock_fallback_service.get_fallback_recommendations = AsyncMock(
            return_value=mock_fallback_response
        )
        
        chat_interface.fallback_service = mock_fallback_service
        
        result = await chat_interface._get_fallback_recommendations(
            "test query",
            FallbackTrigger.UNKNOWN_INTENT
        )
        
        assert result is not None
        assert result["fallback_used"] is True
        assert "DEFAULTING TO REGULAR LLM" in result["explanation"]
        assert "Test explanation" in result["explanation"]
    
    @pytest.mark.asyncio
    async def test_get_fallback_recommendations_service_failure(self, chat_interface):
        """Test fallback service failure handling"""
        # Mock fallback service that throws exception
        mock_fallback_service = Mock()
        mock_fallback_service.get_fallback_recommendations = AsyncMock(
            side_effect=Exception("Service failed")
        )
        
        chat_interface.fallback_service = mock_fallback_service
        
        result = await chat_interface._get_fallback_recommendations(
            "test query",
            FallbackTrigger.UNKNOWN_INTENT
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_process_message_with_fallback_trigger(self, chat_interface):
        """Test process_message when fallback is triggered"""
        # Mock the _get_recommendations to return None (triggering fallback)
        chat_interface._get_recommendations = AsyncMock(return_value=None)
        
        # Mock successful fallback
        mock_fallback_response = {
            "recommendations": [
                {"title": "Fallback Track", "artist": "Fallback Artist"}
            ],
            "explanation": "Fallback explanation",
            "fallback_used": True
        }
        chat_interface._get_fallback_recommendations = AsyncMock(
            return_value=mock_fallback_response
        )
        
        # Mock response formatter
        formatted_response = "Formatted fallback response"
        chat_interface.response_formatter.format_recommendations.return_value = formatted_response
        
        # Test process_message
        result = await chat_interface.process_message("test query", [])
        
        # Verify fallback was used
        assert len(result) == 3  # (response, history, player_html)
        assert len(result[1]) == 1  # One message added to history
        assert result[1][0][0] == "test query"  # User message
        assert result[1][0][1] == formatted_response  # Bot response
        
        # Verify conversation history was updated with fallback flag
        assert len(chat_interface.conversation_history) == 1
        assert chat_interface.conversation_history[0]["used_fallback"] is True
    
    @pytest.mark.asyncio
    async def test_process_message_emergency_fallback(self, chat_interface):
        """Test process_message when all systems fail"""
        # Mock both systems to fail
        chat_interface._get_recommendations = AsyncMock(return_value=None)
        chat_interface._get_fallback_recommendations = AsyncMock(return_value=None)
        
        # Test process_message
        result = await chat_interface.process_message("test query", [])
        
        # Should return emergency response
        assert len(result) == 3
        assert len(result[1]) == 1
        assert "SYSTEM TEMPORARILY UNAVAILABLE" in result[1][0][1]
    
    def test_initialize_fallback_service_no_api_key(self):
        """Test fallback service initialization without API key"""
        with patch.dict('os.environ', {}, clear=True):  # Clear environment
            with patch('src.ui.chat_interface.ResponseFormatter'), \
                 patch('src.ui.chat_interface.PlanningDisplay'):
                
                interface = BeatDebateChatInterface()
                assert interface.fallback_service is None
    
    def test_initialize_fallback_service_with_demo_key(self):
        """Test fallback service initialization with demo key"""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'demo_gemini_key'}):
            with patch('src.ui.chat_interface.ResponseFormatter'), \
                 patch('src.ui.chat_interface.PlanningDisplay'):
                
                interface = BeatDebateChatInterface()
                assert interface.fallback_service is None  # Demo key should not create service 