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
    with proper styling and interactive elements. Enhanced with fallback support.
    """
    
    def __init__(self):
        """Initialize the response formatter."""
        self.logger = logger
    
    def format_recommendations(self, response_data: Dict[str, Any]) -> str:
        """
        Format recommendations response into Markdown.
        Enhanced to handle fallback responses with appropriate disclaimers.
        
        Args:
            response_data: Response from recommendation engine or fallback service
            
        Returns:
            Formatted Markdown string
        """
        try:
            recommendations = response_data.get("recommendations", [])
            processing_time = response_data.get("processing_time", 0)
            is_fallback = response_data.get("fallback_used", False)
            fallback_reason = response_data.get("fallback_reason", "unknown")
            
            if not recommendations:
                return (
                    "❌ **No recommendations found.** "
                    "Please try a different query."
                )
            
            markdown_parts = []
            
            # Add fallback disclaimer if applicable
            if is_fallback:
                disclaimer = self._create_fallback_disclaimer(fallback_reason)
                markdown_parts.extend([disclaimer, ""])
            
            # Header (adjusted for fallback)
            if is_fallback:
                markdown_parts.extend([
                    f"# 🔄 Found {len(recommendations)} Tracks via LLM Fallback",
                    f"⚡ *Generated in {processing_time:.1f}s using general AI assistance*",
                    "",
                ])
            else:
                markdown_parts.extend([
                    f"# 🎵 Found {len(recommendations)} Perfect Tracks for You!",
                    f"⚡ *Generated in {processing_time:.1f}s by our AI agents*",
                    "",
                ])
            
            # Format each recommendation
            for i, rec in enumerate(recommendations, 1):
                rec_markdown = self._format_single_recommendation(rec, i, is_fallback)
                markdown_parts.append(rec_markdown)
                markdown_parts.append("---")  # Separator
            
            # Agent summary (different for fallback)
            if is_fallback:
                agent_summary = self._format_fallback_summary(response_data)
            else:
                agent_summary = self._format_agent_summary(response_data)
            markdown_parts.append(agent_summary)
            
            # Reasoning details
            reasoning_details = self._format_reasoning_details(response_data)
            markdown_parts.append(reasoning_details)
            
            return "\n".join(markdown_parts)
            
        except Exception as e:
            self.logger.error("Failed to format recommendations", error=str(e))
            return f"❌ **Error formatting recommendations:** {str(e)}"
    
    def _create_fallback_disclaimer(self, reason: str) -> str:
        """
        Create styled fallback disclaimer.
        
        Args:
            reason: Reason why fallback was triggered
            
        Returns:
            Formatted disclaimer text
        """
        reason_descriptions = {
            "unknown_intent": "query intent not recognized by our specialized system",
            "no_recommendations": "specialized agents couldn't generate recommendations",
            "api_error": "temporary system issue",
            "timeout": "system response timeout",
            "emergency_fallback": "multiple system failures"
        }
        
        description = reason_descriptions.get(reason, "system limitation")
        
        return (
            "🔄 **FALLBACK MODE ACTIVE**\n"
            f"*Using general AI assistance due to {description}. "
            "For best results, try queries like 'music like [artist]' or '[genre] music'.*\n"
            "---"
        )
    
    def _format_single_recommendation(
        self, rec: Dict[str, Any], rank: int, is_fallback: bool = False
    ) -> str:
        """Format a single recommendation as Markdown."""
        title = rec.get("title", "Unknown Title")
        artist = rec.get("artist", "Unknown Artist")
        confidence = rec.get("confidence", 0.0)
        source = rec.get("source", "unknown")
        
        # Convert confidence to percentage
        confidence_pct = int(confidence * 100)
        
        # Confidence badge color (adjusted for fallback)
        if is_fallback:
            # More conservative confidence indicators for fallback
            if confidence_pct >= 80:
                confidence_badge = f"🟡 **{confidence_pct}% match (AI)**"
            elif confidence_pct >= 60:
                confidence_badge = f"🟠 **{confidence_pct}% match (AI)**"
            else:
                confidence_badge = f"🔴 **{confidence_pct}% match (AI)**"
        else:
            # Original confidence indicators for main system
            if confidence_pct >= 90:
                confidence_badge = f"🟢 **{confidence_pct}% match**"
            elif confidence_pct >= 70:
                confidence_badge = f"🟡 **{confidence_pct}% match**"
            else:
                confidence_badge = f"🔴 **{confidence_pct}% match**"
        
        # Source indicator
        source_indicator = " • *via LLM fallback*" if is_fallback else f" • *via {source}*"
        
        markdown = [
            f"## {rank}. \"{title}\" by {artist}",
            f"{confidence_badge}{source_indicator}",
            ""
        ]
        
        # Add Last.fm link for better preview integration
        lastfm_url = f"https://www.last.fm/music/{artist.replace(' ', '+')}/_/{title.replace(' ', '+')}"
        markdown.extend([
            f"🎧 **[Listen on Last.fm]({lastfm_url})**",
            ""
        ])
        
        # Track ID for reference (useful for research/debugging)
        track_id = f"{artist}_{title}".replace(" ", "_").replace("(", "").replace(")", "")
        markdown.extend([
            f"🔗 **Track ID:** `{track_id}`",
            ""
        ])
        
        # Add reasoning if available
        reasoning = self._extract_reasoning(rec, is_fallback)
        if reasoning:
            markdown.extend([
                "### 🤔 Why this track:",
                reasoning,
                ""
            ])
        
        # Add genres and moods with better formatting
        genres = rec.get("genres", [])
        moods = rec.get("moods", [])
        tags = rec.get("tags", [])
        
        if genres or moods or tags:
            tag_elements = []
            if genres:
                tag_elements.extend([f"😌 {g}" for g in genres[:3]])
            if moods:
                tag_elements.extend([f"😌 {m}" for m in moods[:3]])
            if tags:
                tag_elements.extend([f"😌 {t}" for t in tags[:3]])
            
            markdown.extend([
                f"**Tags:** {' • '.join(tag_elements)}",
                ""
            ])
        
        return "\n".join(markdown)
    
    def _extract_reasoning(self, rec: Dict[str, Any], is_fallback: bool = False) -> str:
        """Extract and format reasoning for a recommendation."""
        # Try to get reasoning from different possible fields
        reasoning_sources = [
            rec.get("reasoning"),
            rec.get("explanation"),
            rec.get("why_recommended")
        ]
        
        for reasoning in reasoning_sources:
            if reasoning:
                # Add fallback context if applicable
                if is_fallback and "AI-generated" not in reasoning:
                    return f"🤖 AI Analysis: {reasoning}"
                return reasoning
        
        # Generate basic reasoning from scores
        confidence = rec.get("confidence", 0.0) or 0.0
        novelty_score = rec.get("novelty_score", 0.0) or 0.0
        quality_score = rec.get("quality_score", 0.0) or 0.0
        
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
        
        default_reasoning = (
            " • ".join(reasoning_parts) 
            if reasoning_parts 
            else "Recommended by our AI system"
        )
        
        # Add fallback context for default reasoning
        if is_fallback:
            return f"🤖 AI Analysis: {default_reasoning}"
        
        return default_reasoning
    
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
    
    def _format_fallback_summary(self, response_data: Dict[str, Any]) -> str:
        """Format fallback system summary."""
        fallback_reason = response_data.get("fallback_reason", "unknown")
        
        markdown = [
            "## 🔄 AI Fallback System Summary",
            "",
            f"🤖 **Gemini Flash 2.0:** Generated recommendations via LLM fallback",
            f"⚠️ **Trigger Reason:** {fallback_reason.replace('_', ' ').title()}",
            "💡 **Note:** For specialized recommendations, try more specific queries",
            ""
        ]
        
        return "\n".join(markdown)
    
    def _format_reasoning_details(self, response_data: Dict[str, Any]) -> str:
        """Format detailed reasoning log."""
        reasoning_log = response_data.get("reasoning", [])
        
        if not reasoning_log:
            return ""
        
        # Handle both list and single string reasoning
        if isinstance(reasoning_log, str):
            reasoning_log = [reasoning_log]
        
        markdown = [
            "<details>",
            (
                "<summary><strong>🔍 View Detailed Reasoning"
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