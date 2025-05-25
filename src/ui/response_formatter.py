"""
Response Formatter for BeatDebate Chat Interface

This module formats recommendation responses with rich explanations,
audio previews, and agent reasoning transparency.
"""

import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Formats recommendation responses for the chat interface.
    
    Features:
    - Rich HTML formatting for recommendations
    - Audio preview integration
    - Agent reasoning display
    - Confidence scores and sources
    - Feedback buttons
    """
    
    def __init__(self):
        """Initialize the response formatter."""
        self.logger = logger.bind(component="ResponseFormatter")
    
    def format_recommendations(self, response: Dict[str, Any]) -> str:
        """
        Format a complete recommendation response for display.
        
        Args:
            response: Recommendation response from backend
            
        Returns:
            Formatted HTML string for display
        """
        try:
            recommendations = response.get("recommendations", [])
            reasoning_log = response.get("reasoning_log", [])
            agent_coordination = response.get("agent_coordination_log", [])
            response_time = response.get("response_time", 0)
            
            if not recommendations:
                return self._format_no_recommendations()
            
            # Build the response
            html_parts = []
            
            # Header with timing
            html_parts.append(self._format_header(len(recommendations), response_time))
            
            # Individual recommendations
            for i, rec in enumerate(recommendations, 1):
                html_parts.append(self._format_single_recommendation(rec, i))
            
            # Agent coordination summary
            if agent_coordination:
                html_parts.append(self._format_agent_coordination(agent_coordination))
            
            # Reasoning log (collapsible)
            if reasoning_log:
                html_parts.append(self._format_reasoning_log(reasoning_log))
            
            return "\n".join(html_parts)
            
        except Exception as e:
            self.logger.error(f"Error formatting recommendations: {e}")
            return self._format_error_response(str(e))
    
    def _format_header(self, count: int, response_time: float) -> str:
        """Format the response header."""
        return f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            text-align: center;
        ">
            <h3 style="margin: 0; font-size: 1.2em;">
                🎵 Found {count} Perfect Track{'' if count == 1 else 's'} for You!
            </h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                ⚡ Generated in {response_time:.1f}s by our AI agents
            </p>
        </div>
        """
    
    def _format_single_recommendation(self, rec: Dict[str, Any], index: int) -> str:
        """Format a single recommendation."""
        title = rec.get("title", "Unknown Track")
        artist = rec.get("artist", "Unknown Artist")
        explanation = rec.get("explanation", "No explanation available")
        confidence = rec.get("confidence", 0.0)
        preview_url = rec.get("preview_url")
        source = rec.get("source", "unknown")
        
        # Confidence color
        confidence_color = self._get_confidence_color(confidence)
        
        # Audio preview section
        audio_section = ""
        if preview_url:
            audio_section = f"""
            <div style="margin: 10px 0;">
                <audio controls style="width: 100%;">
                    <source src="{preview_url}" type="audio/mpeg">
                    Your browser does not support the audio element.
                </audio>
            </div>
            """
        
        return f"""
        <div style="
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                <div>
                    <h4 style="margin: 0; color: #333; font-size: 1.1em;">
                        {index}. "{title}" by {artist}
                    </h4>
                    <div style="margin: 5px 0;">
                        <span style="
                            background: {confidence_color};
                            color: white;
                            padding: 2px 8px;
                            border-radius: 12px;
                            font-size: 0.8em;
                            font-weight: bold;
                        ">
                            {confidence:.0%} match
                        </span>
                        <span style="
                            background: #f8f9fa;
                            color: #6c757d;
                            padding: 2px 8px;
                            border-radius: 12px;
                            font-size: 0.8em;
                            margin-left: 5px;
                        ">
                            via {source}
                        </span>
                    </div>
                </div>
                <div style="display: flex; gap: 5px;">
                    <button style="
                        background: #28a745;
                        color: white;
                        border: none;
                        padding: 5px 10px;
                        border-radius: 5px;
                        cursor: pointer;
                        font-size: 0.8em;
                    " onclick="submitFeedback('{rec.get('id', index)}', 'thumbs_up')">
                        👍
                    </button>
                    <button style="
                        background: #dc3545;
                        color: white;
                        border: none;
                        padding: 5px 10px;
                        border-radius: 5px;
                        cursor: pointer;
                        font-size: 0.8em;
                    " onclick="submitFeedback('{rec.get('id', index)}', 'thumbs_down')">
                        👎
                    </button>
                </div>
            </div>
            
            <div style="
                background: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                border-left: 4px solid #667eea;
                margin: 10px 0;
            ">
                <strong>Why this track:</strong> {explanation}
            </div>
            
            {audio_section}
        </div>
        """
    
    def _format_agent_coordination(self, coordination_log: List[str]) -> str:
        """Format agent coordination summary."""
        coordination_items = "\n".join([
            f"<li style='margin: 5px 0;'>{item}</li>" 
            for item in coordination_log
        ])
        
        return f"""
        <div style="
            background: #e3f2fd;
            border: 1px solid #bbdefb;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        ">
            <h4 style="margin: 0 0 10px 0; color: #1976d2;">
                🤖 Agent Coordination Summary
            </h4>
            <ul style="margin: 0; padding-left: 20px;">
                {coordination_items}
            </ul>
        </div>
        """
    
    def _format_reasoning_log(self, reasoning_log: List[str]) -> str:
        """Format reasoning log as collapsible section."""
        reasoning_items = "\n".join([
            f"<li style='margin: 5px 0; font-family: monospace; font-size: 0.9em;'>{item}</li>"
            for item in reasoning_log
        ])
        
        return f"""
        <details style="
            background: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            margin: 15px 0;
        ">
            <summary style="
                cursor: pointer;
                font-weight: bold;
                color: #555;
                padding: 5px;
            ">
                🔍 View Detailed Agent Reasoning
            </summary>
            <ul style="margin: 10px 0 0 0; padding-left: 20px;">
                {reasoning_items}
            </ul>
        </details>
        """
    
    def _format_no_recommendations(self) -> str:
        """Format response when no recommendations are found."""
        return """
        <div style="
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin: 15px 0;
        ">
            <h4 style="margin: 0 0 10px 0; color: #856404;">
                🎵 No Recommendations Found
            </h4>
            <p style="margin: 0; color: #856404;">
                I couldn't find any tracks matching your request. 
                Try being more specific or asking for a different genre/mood.
            </p>
        </div>
        """
    
    def _format_error_response(self, error_message: str) -> str:
        """Format error response."""
        return f"""
        <div style="
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin: 15px 0;
        ">
            <h4 style="margin: 0 0 10px 0; color: #721c24;">
                ❌ Error Occurred
            </h4>
            <p style="margin: 0; color: #721c24;">
                {error_message}
            </p>
        </div>
        """
    
    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence score."""
        if confidence >= 0.8:
            return "#28a745"  # Green
        elif confidence >= 0.6:
            return "#ffc107"  # Yellow
        else:
            return "#dc3545"  # Red
    
    def format_planning_preview(self, strategy: Dict[str, Any]) -> str:
        """
        Format a preview of the planning strategy.
        
        Args:
            strategy: Planning strategy from PlannerAgent
            
        Returns:
            Formatted HTML preview
        """
        try:
            task_analysis = strategy.get("task_analysis", {})
            primary_goal = task_analysis.get("primary_goal", "Music discovery")
            
            return f"""
            <div style="
                background: #e8f5e8;
                border: 1px solid #c3e6c3;
                border-radius: 8px;
                padding: 10px;
                margin: 5px 0;
            ">
                <strong>🧠 Planning:</strong> {primary_goal}
            </div>
            """
            
        except Exception as e:
            self.logger.error(f"Error formatting planning preview: {e}")
            return "" 