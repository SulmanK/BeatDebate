"""
Gradio ChatInterface for BeatDebate Music Recommendation System

This module provides a ChatGPT-style interface that showcases the 4-agent
planning system with real-time progress indicators and planning visualization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import gradio as gr
import requests

from .response_formatter import ResponseFormatter
from .planning_display import PlanningDisplay

# Import fallback service components
from ..services.llm_fallback_service import (
    LLMFallbackService, 
    FallbackRequest, 
    FallbackTrigger
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Query examples from intent-aware recommendation system design document
QUERY_EXAMPLES = {
    "By Artist": [
        "Songs by Mk.gee",
        "Give me tracks by Radiohead", 
        "Play some Beatles songs"
    ],
    "Artist Similarity": [
        "Songs like Mk.gee",
        "Similar artists to BROCKHAMPTON", 
        "Songs that sound like Radiohead"
    ],
    "Discovery": [
        "Find me underground electronic music",
        "Something completely new and different",
        "Discover underground tracks by Kendrick Lamar"
    ],
    "Genre/Mood": [
        "Upbeat electronic music",
        "Sad indie songs",
        "Chill lo-fi hip hop"
    ],
    "Contextual": [
        "Music for studying",
        "Workout playlist songs", 
        "Background music for coding"
    ],
    "Hybrid": [
        "Songs like Kendrick Lamar but jazzy",
        "Songs by Michael Jackson that are R&B",
        "Electronic music similar to Aphex Twin"
    ],
    "Follow-ups": [
        "More tracks",
        "More like that",
        "Similar to these",
        "More from this artist",
        "More underground like these",
        "More for studying like these"
    ]
}


class BeatDebateChatInterface:
    """
    ChatGPT-style interface for BeatDebate music recommendations.
    
    Features:
    - Real-time agent progress indicators
    - Planning strategy visualization
    - Audio preview integration
    - Conversation history management
    - Last.fm player embeds
    - LLM fallback for unknown intents
    """
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        """
        Initialize the chat interface.
        
        Args:
            backend_url: URL of the FastAPI backend
        """
        self.backend_url = backend_url
        self.response_formatter = ResponseFormatter()
        self.planning_display = PlanningDisplay()
        # ✅ REMOVED: Global session management - now handled per-user via gr.State
        # ✅ REMOVED: Global conversation history - now managed by backend session store
        
        # Initialize fallback service
        self.fallback_service = None
        self._initialize_fallback_service()
        
        logger.info(
            f"BeatDebate Chat Interface initialized (multi-user safe), "
            f"fallback_available: {self.fallback_service is not None}"
        )
    
    def _initialize_fallback_service(self) -> None:
        """Initialize the LLM fallback service."""
        try:
            # Import Gemini client creation function
            from ..services.enhanced_recommendation_service import create_gemini_client
            from ..api.rate_limiter import UnifiedRateLimiter
            import os
            
            # Get Gemini API key
            gemini_api_key = os.getenv('GEMINI_API_KEY', 'demo_gemini_key')
            
            if gemini_api_key and gemini_api_key != 'demo_gemini_key':
                # Create Gemini client
                gemini_client = create_gemini_client(gemini_api_key)
                
                if gemini_client:
                    # Create rate limiter for fallback service
                    rate_limiter = UnifiedRateLimiter.for_gemini(calls_per_minute=8)
                    
                    # Initialize fallback service
                    self.fallback_service = LLMFallbackService(
                        gemini_client=gemini_client,
                        rate_limiter=rate_limiter
                    )
                    
                    logger.info("LLM fallback service initialized successfully")
                else:
                    logger.warning("Failed to create Gemini client for fallback service")
            else:
                logger.warning("No valid Gemini API key found, fallback service disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize fallback service: {e}")
            self.fallback_service = None
    
    async def process_message(
        self, 
        message: str, 
        history: List[Tuple[str, str]],
        session_id: str
    ) -> Tuple[str, List[Tuple[str, str]], str, str]:
        """
        Process user message and return response with track info.
        Enhanced with fallback support and per-user session management.
        
        Args:
            message: User input message
            history: Chat history as list of (user, assistant) tuples
            session_id: User-specific session identifier
            
        Returns:
            Tuple of (response, updated_history, lastfm_player_html, updated_session_id)
        """
        if not message.strip():
            return "", history, "", session_id
        
        logger.info(f"Processing message: {message} for session: {session_id}")
        
        try:
            # Primary: Get recommendations from 4-agent system with session ID
            recommendations_response = await self._get_recommendations(message, session_id)
            
            # Check if fallback is needed
            should_fallback, trigger_reason = self._should_use_fallback(
                recommendations_response
            )
            
            if should_fallback:
                logger.info(f"Using LLM fallback for session {session_id} due to: {trigger_reason.value}")
                recommendations_response = await self._get_fallback_recommendations(
                    message, trigger_reason, session_id
                )
            
            if recommendations_response:
                # Get potentially updated session_id from backend
                updated_session_id = recommendations_response.get("session_id", session_id)
                
                # Format the response
                formatted_response = (
                    self.response_formatter.format_recommendations(
                        recommendations_response
                    )
                )
                
                # Add to history using tuple format
                history.append((message, formatted_response))
                
                # Create Last.fm player HTML for latest recommendations
                lastfm_player_html = self._create_lastfm_player_html(
                    recommendations_response.get("recommendations", [])
                )
                
                return "", history, lastfm_player_html, updated_session_id
            else:
                # Final emergency fallback
                error_response = self._create_emergency_response(message)
                history.append((message, error_response))
                
                return "", history, "", session_id
                
        except Exception as e:
            logger.error(f"Error processing message for session {session_id}: {e}")
            error_response = f"An error occurred: {str(e)}"
            history.append((message, error_response))
            
            return "", history, "", session_id
    
    def _should_use_fallback(
        self, 
        response: Optional[Dict]
    ) -> Tuple[bool, FallbackTrigger]:
        """
        Determine if fallback should be used based on backend response.
        
        Args:
            response: Response from backend recommendation system
            
        Returns:
            Tuple of (should_fallback, trigger_reason)
        """
        if response is None:
            return True, FallbackTrigger.API_ERROR
        
        # Check for explicit unknown intent
        intent = response.get("intent", "").lower()
        if intent in ["unknown", "unsupported", "fallback"]:
            return True, FallbackTrigger.UNKNOWN_INTENT
        
        # Check for empty recommendations
        recommendations = response.get("recommendations", [])
        if not recommendations or len(recommendations) == 0:
            return True, FallbackTrigger.NO_RECOMMENDATIONS
        
        # Check for error indicators
        if response.get("error") or response.get("detail"):
            return True, FallbackTrigger.API_ERROR
        
        return False, None
    
    async def _get_fallback_recommendations(
        self, 
        query: str, 
        trigger_reason: FallbackTrigger,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get fallback recommendations from LLM service.
        
        Args:
            query: User query
            trigger_reason: Reason fallback was triggered
            session_id: User-specific session identifier
            
        Returns:
            Fallback recommendations response or None if unavailable
        """
        if not self.fallback_service:
            logger.warning("Fallback service not available")
            return None
        
        try:
            # Prepare fallback request
            fallback_request = FallbackRequest(
                query=query,
                session_id=session_id,
                chat_context=self._get_chat_context(),
                trigger_reason=trigger_reason,
                max_recommendations=10
            )
            
            # Get fallback recommendations
            fallback_response = await self.fallback_service.get_fallback_recommendations(
                fallback_request
            )
            
            # Add fallback disclaimer to explanation
            if fallback_response and fallback_response.get("fallback_used"):
                original_explanation = fallback_response.get("explanation", "")
                fallback_explanation = (
                    f"**⚠️ DEFAULTING TO REGULAR LLM** - This query is outside our "
                    f"specialized 4-agent system's scope.\n\n{original_explanation}"
                )
                fallback_response["explanation"] = fallback_explanation
            
            return fallback_response
            
        except Exception as e:
            logger.error(f"Fallback service failed: {e}")
            return None
    
    def _get_chat_context(self) -> Optional[Dict]:
        """Get chat context for fallback requests."""
        # ✅ UPDATED: Chat context now managed entirely by backend session store
        # The backend will retrieve session history based on session_id
        # Frontend no longer maintains global conversation history
        return None
    
    def _create_emergency_response(self, query: str) -> str:
        """Create emergency response when all systems fail."""
        return (
            "**🚨 SYSTEM TEMPORARILY UNAVAILABLE**\n\n"
            f"I apologize, but I'm unable to process your request for '{query}' "
            "at the moment. Our recommendation systems are experiencing issues.\n\n"
            "**Please try:**\n"
            "- Waiting a few moments and trying again\n"
            "- Simplifying your query (e.g., 'music like [artist name]')\n"
            "- Checking your internet connection\n\n"
            "We're working to restore full functionality. Thank you for your patience! 🎵"
        )
    
    async def _get_planning_strategy(self, query: str) -> Optional[Dict]:
        """Get planning strategy from backend."""
        try:
            response = requests.post(
                f"{self.backend_url}/planning",
                json={
                    "query": query,
                    "session_id": "planning-session"  # Planning doesn't need user session
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"Planning request failed: {response.status_code}"
                )
                return None
                
        except Exception as e:
            logger.error(f"Error getting planning strategy: {e}")
            return None
    
    async def _get_recommendations(self, query: str, session_id: str) -> Optional[Dict]:
        """Get recommendations from backend with user-specific session context."""
        try:
            # Prepare request with user-specific session ID
            request_data = {
                "query": query,
                "session_id": session_id,
                "max_recommendations": 10,
                "include_previews": True
            }
            
            # ✅ UPDATED: Chat context now retrieved by backend from session store
            # No need to pass chat_context explicitly - backend handles it
            
            response = requests.post(
                f"{self.backend_url}/recommendations",
                json=request_data,
                timeout=120
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data
            else:
                logger.error(
                    f"Recommendations request failed: {response.status_code} for session {session_id}"
                )
                return None
                
        except Exception as e:
            logger.error(f"Error getting recommendations for session {session_id}: {e}")
            return None
    
    def _create_lastfm_player_html(self, recommendations: List[Dict]) -> str:
        """Create HTML for track preview links and info."""
        if not recommendations:
            return """
            <div style="
                padding: 20px; 
                text-align: center;
                color: #cbd5e1;
                background: rgba(30, 41, 59, 0.5);
                border-radius: 0 0 12px 12px;
            ">
                <p><em>No tracks yet!</em></p>
                <p>Ask for music recommendations to see track info here</p>
            </div>
            """
        
        player_html = []
        
        # Container with scrolling for all tracks
        player_html.append("""
            <div style="
            max-height: 400px;
            overflow-y: auto;
            border-radius: 0 0 12px 12px;
            background: rgba(30, 41, 59, 0.5);
        ">
        """)
        
        # Show all tracks with rank numbers
        for i, rec in enumerate(recommendations):
            rank = i + 1
            title = rec.get("title", "Unknown Title")
            artist = rec.get("artist", "Unknown Artist")
            confidence = rec.get("confidence", 0.0)
            confidence_pct = int(confidence * 100)
            
            # Create search queries
            search_query = f"{artist} {title}".replace(" ", "+")
            lastfm_url = f"https://www.last.fm/search?q={search_query}"
            spotify_url = f"https://open.spotify.com/search/{search_query}"
            youtube_url = f"https://www.youtube.com/results?search_query={search_query}"
            
            # Dark mode compatible colors based on confidence
            if confidence_pct >= 70:
                border_color = "#10b981"  # emerald
                bg_gradient = "linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.15) 100%)"
            elif confidence_pct >= 50:
                border_color = "#f59e0b"  # amber
                bg_gradient = "linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.15) 100%)"
            else:
                border_color = "#ef4444"  # red
                bg_gradient = "linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%)"
            
            player_html.append(f"""
                <div style="
                    margin: 12px 20px;
                    padding: 15px;
                    background: {bg_gradient};
                    border-left: 4px solid {border_color};
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                    transition: all 0.2s ease;
                    border: 1px solid rgba(75, 85, 99, 0.3);
                    position: relative;
                ">
                    <!-- Rank Number -->
                    <div style="
                        position: absolute;
                        top: -8px;
                        left: 15px;
                        background: {border_color};
                        color: white;
                        width: 24px;
                        height: 24px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 12px;
                        font-weight: bold;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    ">
                        {rank}
                    </div>
                    
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 8px;
                        margin-top: 8px;
                    ">
                        <div style="
                            font-weight: 700; 
                            color: #ffffff; 
                            font-size: 15px; 
                            text-shadow: 
                                0 1px 3px rgba(0,0,0,0.8),
                                0 2px 6px rgba(0,0,0,0.6),
                                1px 1px 2px rgba(0,0,0,0.9);
                        ">
                            {artist}
                        </div>
                        <div style="
                            background: {border_color};
                            color: white;
                            padding: 3px 10px;
                            border-radius: 12px;
                            font-size: 12px;
                            font-weight: bold;
                            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                        ">
                            {confidence_pct}%
                        </div>
                    </div>
                    <div style="
                        color: #ffffff;
                        margin-bottom: 10px;
                        font-size: 14px;
                        font-weight: 600;
                        text-shadow: 
                            0 1px 3px rgba(0,0,0,0.8),
                            0 2px 6px rgba(0,0,0,0.6),
                            1px 1px 2px rgba(0,0,0,0.9);
                    ">
                        {title}
                    </div>
                    <div style="
                        display: flex;
                        gap: 8px;
                        font-size: 12px;
                    ">
                        <a href="{lastfm_url}" target="_blank" style="
                            color: #fca5a5;
                            text-decoration: none;
                            padding: 4px 8px;
                            border-radius: 4px;
                            background: rgba(0, 0, 0, 0.3);
                            border: 1px solid rgba(239, 68, 68, 0.5);
                            transition: all 0.2s ease;
                            font-weight: 500;
                        ">🎵 Last.fm</a>
                        <a href="{spotify_url}" target="_blank" style="
                            color: #86efac;
                            text-decoration: none;
                            padding: 4px 8px;
                            border-radius: 4px;
                            background: rgba(0, 0, 0, 0.3);
                            border: 1px solid rgba(34, 197, 94, 0.5);
                            transition: all 0.2s ease;
                            font-weight: 500;
                        ">🎧 Spotify</a>
                        <a href="{youtube_url}" target="_blank" style="
                            color: #fda4af;
                            text-decoration: none;
                            padding: 4px 8px;
                            border-radius: 4px;
                            background: rgba(0, 0, 0, 0.3);
                            border: 1px solid rgba(244, 63, 94, 0.5);
                            transition: all 0.2s ease;
                            font-weight: 500;
                        ">📺 YouTube</a>
                    </div>
                </div>
            """)
        
        # Close the scrollable container
        player_html.append("</div>")
        
        return ''.join(player_html)

    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="🎵 BeatDebate",
            theme=gr.themes.Soft(
                primary_hue="violet",
                secondary_hue="blue",
                neutral_hue="slate"
            ),
            css="""
            .main-container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            .header-section {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }
            .examples-section {
                margin: 20px 0;
                padding: 20px;
                background: rgba(55, 65, 81, 0.8);
                border-radius: 12px;
                border: 1px solid rgba(75, 85, 99, 0.5);
            }
            .examples-section h2 {
                color: #f8fafc !important;
                margin-bottom: 15px;
            }
            .examples-section h3 {
                color: #e2e8f0 !important;
                margin-bottom: 10px;
            }
            .example-chip {
                display: inline-block;
                margin: 4px;
                padding: 10px 16px;
                background: rgba(30, 41, 59, 0.9) !important;
                border: 1px solid rgba(100, 116, 139, 0.5) !important;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
                color: #f1f5f9 !important;
                cursor: pointer;
                transition: all 0.2s ease;
                letter-spacing: 0.025em;
            }
            .example-chip:hover {
                background: rgba(51, 65, 85, 0.9) !important;
                border-color: rgba(139, 92, 246, 0.7) !important;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2);
                font-weight: 600;
            }
            .chat-container {
                background: rgba(55, 65, 81, 0.8) !important;
                border: 1px solid rgba(75, 85, 99, 0.5) !important;
                border-radius: 15px 15px 0 0 !important;
                border-bottom: none !important;
                min-height: 400px !important;
            }
            .chat-header {
                padding: 15px 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-align: center;
                font-weight: 600;
                font-size: 16px;
                border-radius: 15px 15px 0 0;
                margin: 0;
                pointer-events: none;
            }
            .input-section {
                margin-top: 0;
                border-radius: 0 0 15px 15px;
                background: rgba(55, 65, 81, 0.8);
                padding: 20px;
                border-top: 1px solid rgba(75, 85, 99, 0.5);
            }
            .info-container {
                background: rgba(30, 41, 59, 0.6);
                border-radius: 12px;
                border: 1px solid rgba(75, 85, 99, 0.3);
                margin-left: 20px;
            }
            .agent-info {
                background: rgba(55, 65, 81, 0.8);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 15px;
                border: 1px solid rgba(75, 85, 99, 0.5);
                color: #f1f5f9;
            }
            .agent-info h3 {
                color: #f8fafc !important;
            }
            .agent-info ul li {
                color: #e2e8f0 !important;
            }
            /* Global dark mode overrides */
            .gradio-container {
                background: #0f172a !important;
                color: #f1f5f9 !important;
            }
            /* Input styling */
            .gr-textbox input {
                background: rgba(30, 41, 59, 0.9) !important;
                border: 1px solid rgba(100, 116, 139, 0.5) !important;
                color: #f1f5f9 !important;
            }
            .gr-textbox input::placeholder {
                color: #94a3b8 !important;
            }
            /* Button styling */
            .gr-button {
                background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
                border: none !important;
                color: white !important;
            }
            .gr-button:hover {
                background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
                box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3) !important;
            }
            """
        ) as interface:
            
            # ✅ ADD: Per-user session state management
            # NOTE: gr.State() without value generates unique session per user
            import uuid
            session_id_state = gr.State()
            
            with gr.Column(elem_classes=["main-container"]):
                # Header with gradient background
                with gr.Column(elem_classes=["header-section"]):
                    gr.Markdown("""
                    # 🎵 BeatDebate
                    ### AI Music Discovery with Intent-Aware 4-Agent Recommendation System
                    
                    **🏆 AgentX Competition Entry** | **🚀 Advanced Agentic Planning System**
                    
                    Discover perfect tracks using our sophisticated 4-agent AI system that understands your musical intent and collaborates intelligently!
                    
                    **🔗 Competition Links:** [AgentX Submission](https://rdi.berkeley.edu/agentx/) • [GitHub Repository](https://github.com/SulmanK/BeatDebate)
                    

                    """)
                
                # Query examples prominently displayed
                with gr.Column(elem_classes=["examples-section"]):
                    gr.Markdown("## 💡 **Try These Examples** - Click any to get started!")
                    gr.Markdown("*Our system recognizes all these intent types and optimizes agent coordination accordingly*")
                    
                    example_buttons = []
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("**🎯 By Artist**")
                            for example in QUERY_EXAMPLES["By Artist"]:
                                btn = gr.Button(
                                    example, 
                                    elem_classes=["example-chip"],
                                    size="sm",
                                    variant="secondary"
                                )
                                example_buttons.append((btn, example))
                        
                        with gr.Column(scale=1):
                            gr.Markdown("**🎯 Artist Similarity**")
                            for example in QUERY_EXAMPLES["Artist Similarity"]:
                                btn = gr.Button(
                                    example, 
                                    elem_classes=["example-chip"],
                                    size="sm",
                                    variant="secondary"
                                )
                                example_buttons.append((btn, example))
                        
                        with gr.Column(scale=1):
                            gr.Markdown("**🎯 Discovery**")
                            for example in QUERY_EXAMPLES["Discovery"]:
                                btn = gr.Button(
                                    example,
                                    elem_classes=["example-chip"], 
                                    size="sm",
                                    variant="secondary"
                                )
                                example_buttons.append((btn, example))
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("**🎵 Genre/Mood**")
                            for example in QUERY_EXAMPLES["Genre/Mood"]:
                                btn = gr.Button(
                                    example,
                                    elem_classes=["example-chip"],
                                    size="sm", 
                                    variant="secondary"
                                )
                                example_buttons.append((btn, example))
                        
                        with gr.Column(scale=1):
                            gr.Markdown("**📍 Contextual**")
                            for example in QUERY_EXAMPLES["Contextual"]:
                                btn = gr.Button(
                                    example,
                                    elem_classes=["example-chip"],
                                    size="sm",
                                    variant="secondary"
                                )
                                example_buttons.append((btn, example))

                        with gr.Column(scale=1):
                            gr.Markdown("**🎭 Hybrid Queries**")
                            for example in QUERY_EXAMPLES["Hybrid"]:
                                btn = gr.Button(
                                    example,
                                    elem_classes=["example-chip"],
                                    size="sm",
                                    variant="secondary"
                                )
                                example_buttons.append((btn, example))
                
                # Main content area
                with gr.Row():
                    with gr.Column(scale=7):
                        # Fixed chat header
                        gr.HTML(
                            """<div class="chat-header">🎵 Music Recommendations</div>""",
                            elem_classes=[]
                        )
                        
                        # Chat interface
                        chatbot = gr.Chatbot(
                            label="",
                            height=500,
                            elem_classes=["chat-container"],
                            show_label=False,
                            container=False,
                            render_markdown=True
                        )
                        
                        # Input area connected to chat
                        with gr.Column(elem_classes=["input-section"]):
                            with gr.Row():
                                msg_input = gr.Textbox(
                                    placeholder="What music are you in the mood for?",
                                    label="",
                                    scale=4,
                                    lines=1,
                                    show_label=False
                                )
                                send_btn = gr.Button(
                                    "Send", 
                                    scale=1, 
                                    variant="primary",
                                    size="lg"
                                )
                    
                    with gr.Column(scale=3):
                        # Track info with improved styling
                        with gr.Column(elem_classes=["info-container"]):
                            gr.HTML("""
                                <div style="
                                    padding: 20px; 
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    color: white;
                                    border-radius: 12px 12px 0 0;
                                    text-align: center;
                                    font-weight: 600;
                                    font-size: 16px;
                                ">
                                    <h3 style="margin: 0; font-size: 18px; font-weight: 600;">🎧 Latest Tracks</h3>
                                    <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 14px;">Click links to listen!</p>
                                </div>
                            """)
                            
                            player_display = gr.HTML(
                                label="",
                                elem_classes=[]
                            )
                        
                        # Agent system info with improved styling
                        with gr.Column(elem_classes=["agent-info"]):
                            gr.Markdown("""
                            ### 🤖 **AI Agent System**
                            - **🧠 Planner**: Analyzes your query intent
                            - **🎵 GenreMood**: Finds style/vibe matches  
                            - **🔍 Discovery**: Uncovers hidden gems
                            - **⚖️ Judge**: Ranks & selects best tracks
                            """)
            
            # Event handlers with session state management
            async def handle_message(message, history, session_id):
                # Generate session ID if not already set (first interaction)
                if session_id is None:
                    session_id = str(uuid.uuid4())
                    logger.info(f"🆕 Generated new session ID: {session_id}")
                
                return await self.process_message(message, history, session_id)
            
            # Example button handlers
            for btn, example_text in example_buttons:
                btn.click(
                    fn=lambda x=example_text: x,
                    inputs=[],
                    outputs=[msg_input]
                )
            
            # Submit on button click or enter with session state
            send_btn.click(
                fn=handle_message,
                inputs=[msg_input, chatbot, session_id_state],
                outputs=[msg_input, chatbot, player_display, session_id_state]
            )
            
            msg_input.submit(
                fn=handle_message,
                inputs=[msg_input, chatbot, session_id_state],
                outputs=[msg_input, chatbot, player_display, session_id_state]
            )
        
        return interface


def create_chat_interface(backend_url: str = "http://localhost:8000") -> gr.Blocks:
    """
    Factory function to create the BeatDebate chat interface.
    
    Args:
        backend_url: URL of the FastAPI backend
        
    Returns:
        Gradio Blocks interface
    """
    chat_interface = BeatDebateChatInterface(backend_url)
    return chat_interface.create_interface()


if __name__ == "__main__":
    # For testing the interface standalone
    interface = create_chat_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    ) 