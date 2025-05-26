"""
Gradio ChatInterface for BeatDebate Music Recommendation System

This module provides a ChatGPT-style interface that showcases the 4-agent
planning system with real-time progress indicators and planning visualization.
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any

import gradio as gr
import requests

from .response_formatter import ResponseFormatter
from .planning_display import PlanningDisplay

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeatDebateChatInterface:
    """
    ChatGPT-style interface for BeatDebate music recommendations.
    
    Features:
    - Real-time agent progress indicators
    - Planning strategy visualization
    - Audio preview integration
    - Conversation history management
    - Feedback collection
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
        self.session_id = str(uuid.uuid4())
        self.conversation_history: List[Dict[str, Any]] = []
        self.playlist: List[Dict[str, Any]] = []  # User's playlist
        
        logger.info(
            f"BeatDebate Chat Interface initialized with session: "
            f"{self.session_id}"
        )
    
    async def process_message(
        self, 
        message: str, 
        history: List[Tuple[str, str]]
    ) -> Tuple[str, List[Tuple[str, str]], str, str]:
        """
        Process user message and return formatted response.
        
        Args:
            message: User's music query
            history: Conversation history in tuple format
            
        Returns:
            Tuple of (response, updated_history, playlist_html, progress_html)
        """
        if not message.strip():
            return "", history, self._create_playlist_html(), ""
        
        logger.info(f"Processing message: {message}")
        
        try:
            # Show initial progress
            progress_html = self._create_progress_html(
                "🧠 PlannerAgent is analyzing your request..."
            )
            
            # Get recommendations (now includes chat history)
            recommendations_response = await self._get_recommendations(message)
            
            if recommendations_response:
                # Format the response
                formatted_response = (
                    self.response_formatter.format_recommendations(
                        recommendations_response
                    )
                )
                
                # Add to history using tuple format
                history.append((message, formatted_response))
                
                # Store in conversation history
                self.conversation_history.append({
                    "user_message": message,
                    "bot_response": formatted_response,
                    "recommendations": recommendations_response.get(
                        "recommendations", []
                    ),
                    "timestamp": time.time()
                })
                
                progress_html = self._create_progress_html(
                    "✅ Recommendations complete!"
                )
                playlist_html = self._create_playlist_html()
                
                return "", history, playlist_html, progress_html
            else:
                error_response = (
                    "I'm sorry, I couldn't generate recommendations right now. "
                    "Please try again."
                )
                history.append((message, error_response))
                progress_html = self._create_progress_html("❌ Error occurred")
                
                return "", history, self._create_playlist_html(), progress_html
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = f"An error occurred: {str(e)}"
            history.append((message, error_response))
            
            progress_html = self._create_progress_html("❌ Error occurred")
            return "", history, self._create_playlist_html(), progress_html
    
    async def _get_planning_strategy(self, query: str) -> Optional[Dict]:
        """Get planning strategy from backend."""
        try:
            response = requests.post(
                f"{self.backend_url}/planning",
                json={
                    "query": query,
                    "session_id": self.session_id
                },
                timeout=30
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
    
    async def _get_recommendations(self, query: str) -> Optional[Dict]:
        """Get recommendations from backend with chat history context."""
        try:
            # Prepare request with chat history context
            request_data = {
                "query": query,
                "session_id": self.session_id,
                "max_recommendations": 3,
                "include_previews": True
            }
            
            # Add chat history context if available
            if self.conversation_history:
                # Get last 3 interactions for context
                recent_history = self.conversation_history[-3:]
                request_data["chat_context"] = {
                    "previous_queries": [
                        h["user_message"] for h in recent_history
                    ],
                    "previous_recommendations": [
                        h.get("recommendations", [])[:1]  # Just first track
                        for h in recent_history
                    ]
                }
            
            response = requests.post(
                f"{self.backend_url}/recommendations",
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"Recommendations request failed: {response.status_code}"
                )
                return None
                
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return None
    
    def _create_progress_html(self, message: str) -> str:
        """Create HTML for progress indicator."""
        return f"""
        <div style="
            padding: 10px; 
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white; 
            border-radius: 8px; 
            margin: 5px 0;
            font-weight: bold;
            text-align: center;
        ">
            {message}
        </div>
        """
    
    def submit_feedback(self, recommendation_id: str, feedback: str) -> str:
        """Submit user feedback for a recommendation."""
        try:
            response = requests.post(
                f"{self.backend_url}/feedback",
                params={
                    "session_id": self.session_id,
                    "recommendation_id": recommendation_id,
                    "feedback": feedback
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return "✅ Feedback submitted successfully!"
            else:
                return "❌ Failed to submit feedback"
                
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return "❌ Error submitting feedback"
    
    def add_to_playlist(self, track_data: Dict[str, Any]) -> str:
        """Add a track to the user's playlist."""
        try:
            # Avoid duplicates
            track_id = f"{track_data.get('artist', '')}_{track_data.get('title', '')}"
            existing_ids = [
                f"{t.get('artist', '')}_{t.get('title', '')}" 
                for t in self.playlist
            ]
            
            if track_id not in existing_ids:
                self.playlist.append(track_data)
                return f"✅ Added '{track_data.get('title', 'Unknown')}' to playlist!"
            else:
                return "ℹ️ Track already in playlist"
                
        except Exception as e:
            logger.error(f"Error adding to playlist: {e}")
            return "❌ Error adding track"
    
    def _create_playlist_html(self) -> str:
        """Create HTML for the playlist builder."""
        if not self.playlist:
            return """
            <div style="
                padding: 15px; 
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px; 
                margin: 5px 0;
                text-align: center;
            ">
                <h4>🎵 Your Playlist</h4>
                <p><em>No tracks yet!</em></p>
                <p>👍 Like recommendations to add them here</p>
            </div>
            """
        
        playlist_html = [
            """
            <div style="
                padding: 15px; 
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px; 
                margin: 5px 0;
            ">
                <h4>🎵 Your Playlist ({} tracks)</h4>
            """.format(len(self.playlist))
        ]
        
        for i, track in enumerate(self.playlist[-5:], 1):  # Show last 5
            title = track.get('title', 'Unknown')[:30]
            artist = track.get('artist', 'Unknown')[:20]
            playlist_html.append(
                f"<p><strong>{i}.</strong> {title}<br>"
                f"<small>by {artist}</small></p>"
            )
        
        if len(self.playlist) > 5:
            playlist_html.append(f"<p><em>...and {len(self.playlist) - 5} more</em></p>")
        
        playlist_html.append("</div>")
        return "".join(playlist_html)

    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="🎵 BeatDebate",
            theme=gr.themes.Soft(),
            css="""
            .chat-container {
                max-height: 600px;
                overflow-y: auto;
            }
            .playlist-container {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                max-height: 400px;
                overflow-y: auto;
            }
            .progress-container {
                margin: 10px 0;
            }
            """
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🎵 BeatDebate
            ### AI Music Discovery with Playlist Builder
            
            Tell me what music you're in the mood for, and build your perfect playlist!
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        label="Music Recommendations",
                        height=500,
                        elem_classes=["chat-container"],
                        show_label=True,
                        container=True,
                        scale=1,
                        render_markdown=True
                    )
                    
                    # Progress indicator
                    progress_display = gr.HTML(
                        label="Agent Progress",
                        elem_classes=["progress-container"]
                    )
                    
                    # Input area
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="What music are you in the mood for?",
                            label="Your Message",
                            scale=4
                        )
                        send_btn = gr.Button("Send", scale=1, variant="primary")
                
                with gr.Column(scale=1):
                    # Playlist builder
                    playlist_display = gr.HTML(
                        label="🎵 Your Playlist",
                        elem_classes=["playlist-container"]
                    )
                    
                    # Playlist actions
                    gr.Markdown("""
                    **🎯 How to build your playlist:**
                    - 👍 Like tracks to add them
                    - Ask for "more like track #2"
                    - Build themed playlists
                    
                    **💡 Try these:**
                    - "Add some variety"
                    - "More underground tracks"
                    - "Something for working out"
                    """)
                    
                    # Export options (future feature)
                    gr.Markdown("""
                    **📤 Export Options:**
                    - Spotify playlist *(coming soon)*
                    - Apple Music *(coming soon)*
                    - Download as text
                    """)
            
            # Example queries
            gr.Examples(
                examples=[
                    "I need focus music for coding",
                    "Something chill for a rainy afternoon",
                    "Upbeat indie rock for working out",
                    "Ambient electronic for meditation",
                    "Underground hip-hop with good beats"
                ],
                inputs=msg_input,
                label="Try these examples:"
            )
            
            # Event handlers
            async def handle_message(message, history):
                return await self.process_message(message, history)
            
            # Submit on button click or enter
            send_btn.click(
                fn=handle_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, playlist_display, progress_display]
            )
            
            msg_input.submit(
                fn=handle_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, playlist_display, progress_display]
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