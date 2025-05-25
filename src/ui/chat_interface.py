"""
Gradio ChatInterface for BeatDebate Music Recommendation System

This module provides a ChatGPT-style interface that showcases the 4-agent
planning system with real-time progress indicators and planning visualization.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any

import gradio as gr
import requests
from gradio.components import Chatbot, Textbox, Button, HTML, Audio

from ..models.recommendation_models import RecommendationResponse
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
        
        logger.info(f"BeatDebate Chat Interface initialized with session: {self.session_id}")
    
    async def process_message(
        self, 
        message: str, 
        history: List[List[str]]
    ) -> Tuple[str, List[List[str]], str, str]:
        """
        Process user message and return formatted response.
        
        Args:
            message: User's music query
            history: Conversation history
            
        Returns:
            Tuple of (response, updated_history, planning_html, progress_html)
        """
        if not message.strip():
            return "", history, "", ""
        
        logger.info(f"Processing message: {message}")
        
        try:
            # Add user message to history
            history.append([message, ""])
            
            # Show initial progress
            progress_html = self._create_progress_html("🧠 PlannerAgent is analyzing your request...")
            
            # Get planning strategy first
            planning_response = await self._get_planning_strategy(message)
            planning_html = ""
            
            if planning_response:
                planning_html = self.planning_display.format_planning_strategy(
                    planning_response["strategy"]
                )
                progress_html = self._create_progress_html(
                    "🎸 GenreMoodAgent and 🔍 DiscoveryAgent are searching..."
                )
            
            # Get full recommendations
            recommendations_response = await self._get_recommendations(message)
            
            if recommendations_response:
                # Format the response
                formatted_response = self.response_formatter.format_recommendations(
                    recommendations_response
                )
                
                # Update history with bot response
                history[-1][1] = formatted_response
                
                # Store in conversation history
                self.conversation_history.append({
                    "user_message": message,
                    "bot_response": formatted_response,
                    "recommendations": recommendations_response.get("recommendations", []),
                    "timestamp": time.time()
                })
                
                progress_html = self._create_progress_html("✅ Recommendations complete!")
                
                return "", history, planning_html, progress_html
            else:
                error_response = "I'm sorry, I couldn't generate recommendations right now. Please try again."
                history[-1][1] = error_response
                progress_html = self._create_progress_html("❌ Error occurred")
                
                return "", history, planning_html, progress_html
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = f"An error occurred: {str(e)}"
            if history and len(history) > 0:
                history[-1][1] = error_response
            else:
                history.append([message, error_response])
            
            progress_html = self._create_progress_html("❌ Error occurred")
            return "", history, "", progress_html
    
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
                logger.error(f"Planning request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting planning strategy: {e}")
            return None
    
    async def _get_recommendations(self, query: str) -> Optional[Dict]:
        """Get recommendations from backend."""
        try:
            response = requests.post(
                f"{self.backend_url}/recommendations",
                json={
                    "query": query,
                    "session_id": self.session_id,
                    "max_recommendations": 3,
                    "include_previews": True
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Recommendations request failed: {response.status_code}")
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
            .planning-container {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            }
            .progress-container {
                margin: 10px 0;
            }
            """
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🎵 BeatDebate
            ### AI Music Discovery with Strategic Agent Planning
            
            Tell me what music you're in the mood for, and watch my 4 AI agents 
            debate to find you the perfect tracks!
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        label="Music Recommendations",
                        height=500,
                        elem_classes=["chat-container"]
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
                    # Planning strategy display
                    planning_display = gr.HTML(
                        label="🧠 Agent Planning Strategy",
                        elem_classes=["planning-container"]
                    )
                    
                    # Session info
                    gr.Markdown(f"""
                    **Session ID:** `{self.session_id[:8]}...`
                    
                    **How it works:**
                    1. 🧠 **PlannerAgent** analyzes your request
                    2. 🎸 **GenreMoodAgent** finds genre/mood matches
                    3. 🔍 **DiscoveryAgent** discovers hidden gems
                    4. ⚖️ **JudgeAgent** selects the best recommendations
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
                outputs=[msg_input, chatbot, planning_display, progress_display]
            )
            
            msg_input.submit(
                fn=handle_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, planning_display, progress_display]
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