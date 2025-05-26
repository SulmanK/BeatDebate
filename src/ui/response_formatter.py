"""
Response Formatter for BeatDebate Music Recommendations

Formats recommendation responses into beautiful Markdown for Gradio display.
"""

from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class ResponseFormatter:
    """
    Formats music recommendation responses into beautiful Markdown.
    
    Converts recommendation data into Gradio-compatible Markdown format
    with proper styling and interactive elements.
    """
    
    def __init__(self):
        """Initialize the response formatter."""
        self.logger = logger
    
    def format_recommendations(self, response_data: Dict[str, Any]) -> str:
        """
        Format recommendations response into Markdown.
        
        Args:
            response_data: Response from recommendation engine
            
        Returns:
            Formatted Markdown string
        """
        try:
            recommendations = response_data.get("recommendations", [])
            processing_time = response_data.get("processing_time", 0)
            
            if not recommendations:
                return (
                    "❌ **No recommendations found.** "
                    "Please try a different query."
                )
            
            # Header
            markdown_parts = [
                f"# 🎵 Found {len(recommendations)} Perfect Tracks for You!",
                f"⚡ *Generated in {processing_time:.1f}s by our AI agents*",
                "",
            ]
            
            # Format each recommendation
            for i, rec in enumerate(recommendations, 1):
                rec_markdown = self._format_single_recommendation(rec, i)
                markdown_parts.append(rec_markdown)
                markdown_parts.append("---")  # Separator
            
            # Agent summary
            agent_summary = self._format_agent_summary(response_data)
            markdown_parts.append(agent_summary)
            
            # Reasoning details
            reasoning_details = self._format_reasoning_details(response_data)
            markdown_parts.append(reasoning_details)
            
            return "\n".join(markdown_parts)
            
        except Exception as e:
            self.logger.error("Failed to format recommendations", error=str(e))
            return f"❌ **Error formatting recommendations:** {str(e)}"
    
    def _format_single_recommendation(
        self, rec: Dict[str, Any], rank: int
    ) -> str:
        """Format a single recommendation as Markdown."""
        title = rec.get("title", "Unknown Title")
        artist = rec.get("artist", "Unknown Artist")
        confidence = rec.get("confidence", 0.0)
        source = rec.get("source", "unknown")
        
        # Convert confidence to percentage
        confidence_pct = int(confidence * 100)
        
        # Confidence badge color
        if confidence_pct >= 90:
            confidence_badge = f"🟢 **{confidence_pct}% match**"
        elif confidence_pct >= 70:
            confidence_badge = f"🟡 **{confidence_pct}% match**"
        else:
            confidence_badge = f"🔴 **{confidence_pct}% match**"
        
        markdown = [
            f"## {rank}. \"{title}\" by {artist}",
            f"{confidence_badge} • *via {source}*",
            ""
        ]
        
        # Add audio preview if available
        preview_url = rec.get("preview_url")
        if preview_url:
            markdown.extend([
                f"🎧 **[▶️ Preview]({preview_url})**",
                ""
            ])
        
        # Add thumbs up button (using track ID for identification)
        track_id = f"{artist}_{title}".replace(" ", "_")
        markdown.extend([
            "👍 **Like this track?** *(Add to playlist)*",
            f"*Track ID: {track_id}*",
            ""
        ])
        
        # Add reasoning if available
        reasoning = self._extract_reasoning(rec)
        if reasoning:
            markdown.extend([
                "### 🤔 Why this track:",
                reasoning,
                ""
            ])
        
        # Add genres and moods
        genres = rec.get("genres", [])
        moods = rec.get("moods", [])
        
        if genres or moods:
            tags = []
            if genres:
                tags.extend([f"🎼 {g}" for g in genres[:3]])
            if moods:
                tags.extend([f"😌 {m}" for m in moods[:3]])
            
            markdown.extend([
                f"**Tags:** {' • '.join(tags)}",
                ""
            ])
        
        return "\n".join(markdown)
    
    def _extract_reasoning(self, rec: Dict[str, Any]) -> str:
        """Extract and format reasoning for a recommendation."""
        # Try to get reasoning from different possible fields
        reasoning_sources = [
            rec.get("reasoning"),
            rec.get("explanation"),
            rec.get("why_recommended")
        ]
        
        for reasoning in reasoning_sources:
            if reasoning:
                return reasoning
        
        # Generate basic reasoning from scores
        confidence = rec.get("confidence", 0.0)
        novelty_score = rec.get("novelty_score", 0.0)
        quality_score = rec.get("quality_score", 0.0)
        
        reasoning_parts = []
        
        if confidence > 0.8:
            reasoning_parts.append("✅ High relevance to your request")
        elif confidence > 0.6:
            reasoning_parts.append("✅ Good match for your preferences")
        
        if novelty_score > 0.7:
            reasoning_parts.append("🌟 Unique discovery")
        elif novelty_score > 0.4:
            reasoning_parts.append("🎯 Balanced familiarity")
        
        if quality_score > 0.7:
            reasoning_parts.append("🏆 High quality track")
        
        return (
            " • ".join(reasoning_parts) 
            if reasoning_parts 
            else "Recommended by our AI agents"
        )
    
    def _format_agent_summary(self, response_data: Dict[str, Any]) -> str:
        """Format agent coordination summary."""
        markdown = [
            "## 🤖 Agent Coordination Summary",
            "",
            "✅ **PlannerAgent:** Strategic planning completed",
            "✅ **GenreMoodAgent:** Genre/mood recommendations generated", 
            "✅ **DiscoveryAgent:** Discovery recommendations generated",
            "✅ **JudgeAgent:** Final selection and ranking completed",
            ""
        ]
        
        return "\n".join(markdown)
    
    def _format_reasoning_details(self, response_data: Dict[str, Any]) -> str:
        """Format detailed reasoning log."""
        reasoning_log = response_data.get("reasoning_log", [])
        
        if not reasoning_log:
            return ""
        
        markdown = [
            "<details>",
            (
                "<summary><strong>🔍 View Detailed Agent Reasoning"
                "</strong></summary>"
            ),
            "",
        ]
        
        for entry in reasoning_log:
            markdown.append(f"• `{entry}`")
        
        markdown.extend([
            "",
            "</details>"
        ])
        
        return "\n".join(markdown)
    
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