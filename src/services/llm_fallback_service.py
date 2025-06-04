"""
LLM Fallback Service for BeatDebate Music Recommendations

This service provides fallback music recommendations using Gemini Flash 2.0
when the main 4-agent system encounters unknown intents or fails to return results.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


class FallbackTrigger(Enum):
    """Enumeration of reasons why fallback was triggered."""
    UNKNOWN_INTENT = "unknown_intent"          # Backend returns unknown intent
    NO_RECOMMENDATIONS = "no_recommendations"   # Backend returns empty results
    API_ERROR = "api_error"                    # Backend returns error status
    TIMEOUT = "timeout"                        # Backend request timeout
    SYSTEM_ERROR = "system_error"              # General system error


@dataclass
class FallbackRequest:
    """Request structure for LLM fallback recommendations."""
    query: str
    session_id: str
    chat_context: Optional[Dict] = None
    trigger_reason: FallbackTrigger = FallbackTrigger.SYSTEM_ERROR
    max_recommendations: int = 10


class LLMFallbackService:
    """
    Service for handling LLM-based music recommendations when 4-agent system fails.
    
    This service uses Gemini Flash 2.0 to provide music recommendations as a fallback
    when the main recommendation system encounters edge cases or failures.
    """
    
    def __init__(self, gemini_client, rate_limiter=None):
        """
        Initialize the LLM fallback service.
        
        Args:
            gemini_client: Configured Gemini client for LLM interactions
            rate_limiter: Optional rate limiter for Gemini API calls
        """
        self.gemini_client = gemini_client
        self.rate_limiter = rate_limiter
        self.logger = logger.bind(service="LLMFallbackService")
        
        # Emergency fallback recommendations for total failures
        self._emergency_tracks = [
            {"title": "Breathe Me", "artist": "Sia", "confidence": 0.7},
            {"title": "Mad World", "artist": "Gary Jules", "confidence": 0.7},
            {"title": "Midnight City", "artist": "M83", "confidence": 0.7},
            {"title": "Holocene", "artist": "Bon Iver", "confidence": 0.7},
            {"title": "Black", "artist": "Pearl Jam", "confidence": 0.7},
        ]
        
        self.logger.info("LLM Fallback Service initialized")
    
    async def get_fallback_recommendations(
        self, 
        request: FallbackRequest
    ) -> Dict[str, Any]:
        """
        Get music recommendations from Gemini Flash 2.0 as fallback.
        
        Args:
            request: Fallback request with query and context
            
        Returns:
            Formatted response matching regular recommendation format
        """
        self.logger.info(
            "Generating fallback recommendations",
            query=request.query,
            trigger_reason=request.trigger_reason.value,
            has_context=bool(request.chat_context)
        )
        
        try:
            # Apply rate limiting if available
            if self.rate_limiter:
                await self.rate_limiter.acquire()
            
            # Build optimized prompt for music recommendations
            prompt = self._build_fallback_prompt(request)
            
            # Call Gemini API
            response = await self._call_gemini_async(prompt)
            
            # Parse the response
            parsed_response = self._parse_gemini_response(response)
            
            # Format as standard recommendation response
            formatted_response = {
                "recommendations": parsed_response["tracks"],
                "explanation": parsed_response["explanation"],
                "fallback_used": True,
                "fallback_reason": request.trigger_reason.value,
                "intent": "fallback",
                "reasoning": [
                    f"Fallback triggered: {request.trigger_reason.value}",
                    "Using Gemini Flash 2.0 for music recommendations",
                    parsed_response["explanation"]
                ],
                "processing_time": 0.0  # Will be set by caller
            }
            
            self.logger.info(
                "Fallback recommendations generated successfully",
                num_tracks=len(parsed_response["tracks"]),
                trigger_reason=request.trigger_reason.value
            )
            
            return formatted_response
            
        except Exception as e:
            self.logger.error(
                "Fallback service failed",
                error=str(e),
                query=request.query,
                trigger_reason=request.trigger_reason.value
            )
            
            # Return emergency fallback
            return self._create_emergency_response(request)
    
    def _build_fallback_prompt(self, request: FallbackRequest) -> str:
        """
        Build optimized prompt for music recommendations.
        
        Args:
            request: Fallback request with context
            
        Returns:
            Formatted prompt string for Gemini
        """
        # Extract conversation context if available
        context_info = ""
        if request.chat_context:
            previous_queries = request.chat_context.get("previous_queries", [])
            if previous_queries:
                recent_queries = previous_queries[-2:]  # Last 2 queries for context
                context_info = f"\nConversation context: Previous queries were {', '.join(recent_queries)}"
        
        # Build comprehensive prompt
        prompt = f"""You are a music recommendation assistant. The user asked: "{request.query}"{context_info}

Provide exactly {request.max_recommendations} music track recommendations in this JSON format:
{{
    "tracks": [
        {{
            "title": "Track Name",
            "artist": "Artist Name", 
            "confidence": 0.85,
            "explanation": "Brief explanation why this fits the request",
            "source": "gemini_fallback"
        }}
    ],
    "explanation": "Overall explanation of the recommendation approach and strategy used"
}}

Guidelines:
- Focus on diverse, high-quality music recommendations
- Include mix of popular and lesser-known tracks when appropriate
- Ensure artist and title are accurate and searchable on streaming platforms
- Provide confidence scores between 0.6-0.9 (be realistic about matches)
- Keep individual explanations concise but meaningful (1-2 sentences max)
- Consider the conversation context if provided
- Vary genres and artists for diversity
- Include tracks that match the user's intent and mood

The overall explanation should describe your recommendation strategy and why these tracks work well together for the user's request.

Respond ONLY with valid JSON. Do not include any other text."""
        
        return prompt
    
    async def _call_gemini_async(self, prompt: str) -> str:
        """
        Call Gemini API asynchronously.
        
        Args:
            prompt: Formatted prompt for the LLM
            
        Returns:
            Raw response text from Gemini
        """
        try:
            # Check if we have an async method
            if hasattr(self.gemini_client, 'generate_content_async'):
                response = await self.gemini_client.generate_content_async(prompt)
            else:
                # Fallback to sync method (wrap in async)
                import asyncio
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.gemini_client.generate_content, prompt
                )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("Unable to extract text from Gemini response")
                
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {e}")
            raise
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Gemini response text into structured format.
        
        Args:
            response_text: Raw response text from Gemini
            
        Returns:
            Parsed response dictionary
        """
        try:
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Remove potential markdown formatting
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            
            cleaned_text = cleaned_text.strip()
            
            # Parse JSON
            parsed = json.loads(cleaned_text)
            
            # Validate structure
            if "tracks" not in parsed or "explanation" not in parsed:
                raise ValueError("Missing required fields in response")
            
            if not isinstance(parsed["tracks"], list):
                raise ValueError("Tracks must be a list")
            
            # Validate and clean track data
            validated_tracks = []
            for track in parsed["tracks"]:
                if not isinstance(track, dict):
                    continue
                
                # Ensure required fields
                title = track.get("title", "").strip()
                artist = track.get("artist", "").strip()
                
                if not title or not artist:
                    continue
                
                validated_track = {
                    "title": title,
                    "artist": artist,
                    "confidence": min(0.9, max(0.6, float(track.get("confidence", 0.7)))),
                    "explanation": track.get("explanation", "AI-generated recommendation").strip(),
                    "source": "gemini_fallback"
                }
                
                validated_tracks.append(validated_track)
            
            return {
                "tracks": validated_tracks,
                "explanation": parsed.get("explanation", "AI-generated music recommendations")
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response from Gemini: {e}")
        except Exception as e:
            self.logger.error(f"Failed to parse Gemini response: {e}")
            raise ValueError(f"Failed to parse response: {e}")
    
    def _create_emergency_response(self, request: FallbackRequest) -> Dict[str, Any]:
        """
        Create emergency response when all else fails.
        
        Args:
            request: Original fallback request
            
        Returns:
            Emergency fallback response with static recommendations
        """
        self.logger.warning(
            "Creating emergency fallback response",
            query=request.query,
            trigger_reason=request.trigger_reason.value
        )
        
        # Select subset of emergency tracks
        num_tracks = min(request.max_recommendations, len(self._emergency_tracks))
        selected_tracks = self._emergency_tracks[:num_tracks]
        
        return {
            "recommendations": selected_tracks,
            "explanation": (
                "**⚠️ EMERGENCY FALLBACK ACTIVE** - Our systems are experiencing "
                "issues. Here are some popular tracks while we work to restore "
                "full functionality."
            ),
            "fallback_used": True,
            "fallback_reason": "emergency_fallback",
            "intent": "emergency",
            "reasoning": [
                f"Emergency fallback triggered due to: {request.trigger_reason.value}",
                "Providing static recommendations as last resort",
                "Please try again in a few moments for full AI recommendations"
            ],
            "processing_time": 0.0
        }
    
    def is_available(self) -> bool:
        """
        Check if the fallback service is available.
        
        Returns:
            True if service is available, False otherwise
        """
        return self.gemini_client is not None
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get current service status and statistics.
        
        Returns:
            Service status information
        """
        status = {
            "service_available": self.is_available(),
            "has_rate_limiter": self.rate_limiter is not None,
            "emergency_tracks_available": len(self._emergency_tracks)
        }
        
        if self.rate_limiter:
            status["rate_limiter_stats"] = self.rate_limiter.get_current_usage()
        
        return status 