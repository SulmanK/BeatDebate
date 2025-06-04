"""
Tests for LLM Fallback Service

Tests to verify the fallback service functionality, prompt engineering,
response parsing, and error handling.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch

from src.services.llm_fallback_service import (
    LLMFallbackService,
    FallbackRequest,
    FallbackTrigger
)


class TestLLMFallbackService:
    """Test suite for LLMFallbackService"""
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Create mock Gemini client"""
        client = Mock()
        client.generate_content = AsyncMock()
        client.generate_content_async = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_rate_limiter(self):
        """Create mock rate limiter"""
        limiter = Mock()
        limiter.acquire = AsyncMock()
        limiter.get_current_usage = Mock(return_value={"remaining_calls": 10})
        return limiter
    
    @pytest.fixture
    def fallback_service(self, mock_gemini_client, mock_rate_limiter):
        """Create LLMFallbackService instance for testing"""
        return LLMFallbackService(
            gemini_client=mock_gemini_client,
            rate_limiter=mock_rate_limiter
        )
    
    @pytest.fixture
    def sample_fallback_request(self):
        """Create sample fallback request"""
        return FallbackRequest(
            query="Music for studying",
            session_id="test_session_123",
            chat_context={
                "previous_queries": ["Music like Radiohead", "Chill electronic music"]
            },
            trigger_reason=FallbackTrigger.UNKNOWN_INTENT,
            max_recommendations=5
        )
    
    def test_fallback_service_initialization(self, mock_gemini_client, mock_rate_limiter):
        """Test that LLMFallbackService initializes correctly"""
        service = LLMFallbackService(mock_gemini_client, mock_rate_limiter)
        
        assert service.gemini_client == mock_gemini_client
        assert service.rate_limiter == mock_rate_limiter
        assert service.is_available() is True
        assert len(service._emergency_tracks) > 0
    
    def test_fallback_service_without_rate_limiter(self, mock_gemini_client):
        """Test initialization without rate limiter"""
        service = LLMFallbackService(mock_gemini_client, None)
        
        assert service.rate_limiter is None
        assert service.is_available() is True
    
    def test_fallback_service_without_client(self):
        """Test initialization without Gemini client"""
        service = LLMFallbackService(None, None)
        
        assert service.is_available() is False
    
    def test_build_fallback_prompt_basic(self, fallback_service, sample_fallback_request):
        """Test basic prompt building"""
        prompt = fallback_service._build_fallback_prompt(sample_fallback_request)
        
        assert "Music for studying" in prompt
        assert "5" in prompt  # max_recommendations
        assert "JSON format" in prompt
        assert "Radiohead" in prompt  # from context
        assert "Chill electronic music" in prompt  # from context
    
    def test_build_fallback_prompt_no_context(self, fallback_service):
        """Test prompt building without context"""
        request = FallbackRequest(
            query="Jazz music",
            session_id="test",
            chat_context=None,
            max_recommendations=3
        )
        
        prompt = fallback_service._build_fallback_prompt(request)
        
        assert "Jazz music" in prompt
        assert "3" in prompt
        assert "Conversation context" not in prompt
    
    @pytest.mark.asyncio
    async def test_successful_fallback_recommendations(
        self, 
        fallback_service, 
        sample_fallback_request,
        mock_gemini_client
    ):
        """Test successful generation of fallback recommendations"""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "tracks": [
                {
                    "title": "Weightless",
                    "artist": "Marconi Union",
                    "confidence": 0.9,
                    "explanation": "Scientifically designed to reduce anxiety"
                },
                {
                    "title": "Clair de Lune",
                    "artist": "Claude Debussy",
                    "confidence": 0.85,
                    "explanation": "Classical piece perfect for concentration"
                }
            ],
            "explanation": "Selected calming tracks ideal for studying and focus."
        })
        
        mock_gemini_client.generate_content_async.return_value = mock_response
        
        # Test the service
        result = await fallback_service.get_fallback_recommendations(sample_fallback_request)
        
        # Verify result structure
        assert result is not None
        assert result["fallback_used"] is True
        assert result["fallback_reason"] == "unknown_intent"
        assert len(result["recommendations"]) == 2
        assert result["recommendations"][0]["title"] == "Weightless"
        assert result["recommendations"][0]["artist"] == "Marconi Union"
        assert result["recommendations"][0]["source"] == "gemini_fallback"
        
        # Verify rate limiter was called
        fallback_service.rate_limiter.acquire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gemini_api_failure(
        self,
        fallback_service,
        sample_fallback_request,
        mock_gemini_client
    ):
        """Test handling of Gemini API failure"""
        # Mock API failure
        mock_gemini_client.generate_content_async.side_effect = Exception("API Error")
        
        # Test the service
        result = await fallback_service.get_fallback_recommendations(sample_fallback_request)
        
        # Should return emergency response
        assert result is not None
        assert result["fallback_reason"] == "emergency_fallback"
        assert "EMERGENCY FALLBACK" in result["explanation"]
        assert len(result["recommendations"]) > 0
    
    def test_parse_valid_gemini_response(self, fallback_service):
        """Test parsing of valid Gemini response"""
        response_text = json.dumps({
            "tracks": [
                {
                    "title": "Test Track",
                    "artist": "Test Artist",
                    "confidence": 0.8,
                    "explanation": "Good test track"
                }
            ],
            "explanation": "Test recommendations"
        })
        
        result = fallback_service._parse_gemini_response(response_text)
        
        assert "tracks" in result
        assert "explanation" in result
        assert len(result["tracks"]) == 1
        assert result["tracks"][0]["source"] == "gemini_fallback"
    
    def test_parse_gemini_response_with_markdown(self, fallback_service):
        """Test parsing response with markdown formatting"""
        response_text = "```json\n" + json.dumps({
            "tracks": [{"title": "Test", "artist": "Artist", "confidence": 0.7}],
            "explanation": "Test"
        }) + "\n```"
        
        result = fallback_service._parse_gemini_response(response_text)
        
        assert len(result["tracks"]) == 1
        assert result["tracks"][0]["title"] == "Test"
    
    def test_parse_invalid_json_response(self, fallback_service):
        """Test handling of invalid JSON response"""
        response_text = "Invalid JSON response"
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            fallback_service._parse_gemini_response(response_text)
    
    def test_parse_response_missing_fields(self, fallback_service):
        """Test handling of response missing required fields"""
        response_text = json.dumps({"wrong_field": "value"})
        
        with pytest.raises(ValueError, match="Missing required fields"):
            fallback_service._parse_gemini_response(response_text)
    
    def test_validate_track_data(self, fallback_service):
        """Test validation and cleaning of track data"""
        response_text = json.dumps({
            "tracks": [
                {
                    "title": "Valid Track",
                    "artist": "Valid Artist",
                    "confidence": 0.8,
                    "explanation": "Good track"
                },
                {
                    "title": "",  # Invalid - empty title
                    "artist": "Artist",
                    "confidence": 0.7
                },
                {
                    "title": "Track",
                    "artist": "",  # Invalid - empty artist
                    "confidence": 0.9
                },
                {
                    "title": "Another Track",
                    "artist": "Another Artist",
                    "confidence": 1.5,  # Invalid confidence (>1), should be clamped
                    "explanation": "Another good track"
                }
            ],
            "explanation": "Mixed valid and invalid tracks"
        })
        
        result = fallback_service._parse_gemini_response(response_text)
        
        # Should only have 2 valid tracks
        assert len(result["tracks"]) == 2
        assert result["tracks"][0]["title"] == "Valid Track"
        assert result["tracks"][1]["title"] == "Another Track"
        assert result["tracks"][1]["confidence"] == 0.9  # Clamped from 1.5
    
    def test_create_emergency_response(self, fallback_service, sample_fallback_request):
        """Test creation of emergency response"""
        result = fallback_service._create_emergency_response(sample_fallback_request)
        
        assert result["fallback_used"] is True
        assert result["fallback_reason"] == "emergency_fallback"
        assert "EMERGENCY FALLBACK" in result["explanation"]
        assert len(result["recommendations"]) > 0
        assert all("title" in track and "artist" in track for track in result["recommendations"])
    
    def test_get_service_status(self, fallback_service):
        """Test service status reporting"""
        status = fallback_service.get_service_status()
        
        assert "service_available" in status
        assert "has_rate_limiter" in status
        assert "emergency_tracks_available" in status
        assert status["service_available"] is True
        assert status["has_rate_limiter"] is True
        assert status["emergency_tracks_available"] > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_integration(
        self,
        fallback_service,
        sample_fallback_request,
        mock_gemini_client
    ):
        """Test rate limiter integration"""
        # Mock successful response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "tracks": [{"title": "Test", "artist": "Artist", "confidence": 0.8}],
            "explanation": "Test"
        })
        mock_gemini_client.generate_content_async.return_value = mock_response
        
        # Call service
        await fallback_service.get_fallback_recommendations(sample_fallback_request)
        
        # Verify rate limiter was called
        fallback_service.rate_limiter.acquire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_gemini_call_fallback(self, fallback_service):
        """Test fallback to sync method when async not available"""
        # Mock client without async method
        mock_client = Mock()
        mock_client.generate_content = Mock(return_value=Mock(text="Test response"))
        # Remove async method
        if hasattr(mock_client, 'generate_content_async'):
            delattr(mock_client, 'generate_content_async')
        
        fallback_service.gemini_client = mock_client
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock(return_value=Mock(text="Test response"))
            mock_loop.return_value.run_in_executor = mock_executor
            
            result = await fallback_service._call_gemini_async("test prompt")
            
            assert result == "Test response"
            mock_executor.assert_called_once()


class TestFallbackTrigger:
    """Test suite for FallbackTrigger enum"""
    
    def test_fallback_trigger_values(self):
        """Test FallbackTrigger enum values"""
        assert FallbackTrigger.UNKNOWN_INTENT.value == "unknown_intent"
        assert FallbackTrigger.NO_RECOMMENDATIONS.value == "no_recommendations"
        assert FallbackTrigger.API_ERROR.value == "api_error"
        assert FallbackTrigger.TIMEOUT.value == "timeout"
        assert FallbackTrigger.SYSTEM_ERROR.value == "system_error"


class TestFallbackRequest:
    """Test suite for FallbackRequest dataclass"""
    
    def test_fallback_request_creation(self):
        """Test FallbackRequest creation with defaults"""
        request = FallbackRequest(
            query="test query",
            session_id="test_session"
        )
        
        assert request.query == "test query"
        assert request.session_id == "test_session"
        assert request.chat_context is None
        assert request.trigger_reason == FallbackTrigger.SYSTEM_ERROR
        assert request.max_recommendations == 10
    
    def test_fallback_request_with_all_fields(self):
        """Test FallbackRequest creation with all fields"""
        context = {"previous_queries": ["test"]}
        
        request = FallbackRequest(
            query="test query",
            session_id="test_session",
            chat_context=context,
            trigger_reason=FallbackTrigger.UNKNOWN_INTENT,
            max_recommendations=5
        )
        
        assert request.chat_context == context
        assert request.trigger_reason == FallbackTrigger.UNKNOWN_INTENT
        assert request.max_recommendations == 5 