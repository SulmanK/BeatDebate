"""
Planning Display for BeatDebate Chat Interface

This module visualizes the PlannerAgent's strategic thinking process
in real-time for the UI, showcasing the planning behavior for AgentX.
"""

import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlanningDisplay:
    """
    Visualizes planning strategies from the PlannerAgent.
    
    Features:
    - Strategic planning breakdown
    - Agent coordination visualization
    - Evaluation criteria display
    - Execution monitoring indicators
    """
    
    def __init__(self):
        """Initialize the planning display."""
        self.logger = logger.bind(component="PlanningDisplay")
    
    def format_planning_strategy(self, strategy: Dict[str, Any]) -> str:
        """
        Format a complete planning strategy for display.
        
        Args:
            strategy: Planning strategy from PlannerAgent
            
        Returns:
            Formatted HTML string for display
        """
        try:
            if not strategy:
                return self._format_no_strategy()
            
            html_parts = []
            
            # Header
            html_parts.append(self._format_strategy_header())
            
            # Task Analysis
            task_analysis = strategy.get("task_analysis", {})
            if task_analysis:
                html_parts.append(self._format_task_analysis(task_analysis))
            
            # Coordination Strategy
            coordination = strategy.get("coordination_strategy", {})
            if coordination:
                html_parts.append(self._format_coordination_strategy(coordination))
            
            # Evaluation Framework
            evaluation = strategy.get("evaluation_framework", {})
            if evaluation:
                html_parts.append(self._format_evaluation_framework(evaluation))
            
            # Execution Monitoring
            execution = strategy.get("execution_monitoring", {})
            if execution:
                html_parts.append(self._format_execution_monitoring(execution))
            
            return "\n".join(html_parts)
            
        except Exception as e:
            self.logger.error(f"Error formatting planning strategy: {e}")
            return self._format_error_display(str(e))
    
    def _format_strategy_header(self) -> str:
        """Format the strategy header."""
        return """
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 10px;
            text-align: center;
        ">
            <h4 style="margin: 0; font-size: 1.1em;">
                🧠 Strategic Planning Analysis
            </h4>
        </div>
        """
    
    def _format_task_analysis(self, task_analysis: Dict[str, Any]) -> str:
        """Format task analysis section."""
        primary_goal = task_analysis.get("primary_goal", "Music discovery")
        complexity = task_analysis.get("complexity_level", "medium")
        context_factors = task_analysis.get("context_factors", [])
        
        # Complexity indicator
        complexity_color = {
            "simple": "#28a745",
            "medium": "#ffc107", 
            "complex": "#dc3545"
        }.get(complexity, "#6c757d")
        
        # Context factors list
        context_html = ""
        if context_factors:
            context_items = "\n".join([
                f"<li style='margin: 2px 0; font-size: 0.85em;'>{factor}</li>"
                for factor in context_factors[:3]  # Limit to 3 items
            ])
            context_html = f"""
            <div style="margin-top: 8px;">
                <strong style="font-size: 0.9em;">Context Factors:</strong>
                <ul style="margin: 5px 0; padding-left: 15px;">
                    {context_items}
                </ul>
            </div>
            """
        
        return f"""
        <div style="
            background: #f8f9fa;
            border-left: 4px solid #4facfe;
            padding: 10px;
            margin: 8px 0;
            border-radius: 0 5px 5px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong style="color: #333; font-size: 0.95em;">🎯 Primary Goal</strong>
                <span style="
                    background: {complexity_color};
                    color: white;
                    padding: 2px 6px;
                    border-radius: 10px;
                    font-size: 0.75em;
                    font-weight: bold;
                ">
                    {complexity.upper()}
                </span>
            </div>
            <p style="margin: 5px 0 0 0; color: #555; font-size: 0.9em;">
                {primary_goal}
            </p>
            {context_html}
        </div>
        """
    
    def _format_coordination_strategy(self, coordination: Dict[str, Any]) -> str:
        """Format agent coordination strategy."""
        agents_html = []
        
        # GenreMoodAgent strategy
        genre_mood = coordination.get("genre_mood_agent", {})
        if genre_mood:
            focus = genre_mood.get("focus", "Genre and mood analysis")
            agents_html.append(f"""
            <div style="
                background: #e8f5e8;
                border: 1px solid #c3e6c3;
                border-radius: 5px;
                padding: 8px;
                margin: 5px 0;
            ">
                <strong style="color: #155724; font-size: 0.85em;">🎸 GenreMoodAgent</strong>
                <p style="margin: 3px 0 0 0; color: #155724; font-size: 0.8em;">
                    {focus}
                </p>
            </div>
            """)
        
        # DiscoveryAgent strategy
        discovery = coordination.get("discovery_agent", {})
        if discovery:
            focus = discovery.get("focus", "Discovery and novelty search")
            agents_html.append(f"""
            <div style="
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 5px;
                padding: 8px;
                margin: 5px 0;
            ">
                <strong style="color: #856404; font-size: 0.85em;">🔍 DiscoveryAgent</strong>
                <p style="margin: 3px 0 0 0; color: #856404; font-size: 0.8em;">
                    {focus}
                </p>
            </div>
            """)
        
        agents_content = "\n".join(agents_html)
        
        return f"""
        <div style="
            background: #f8f9fa;
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 8px 0;
            border-radius: 0 5px 5px 0;
        ">
            <strong style="color: #333; font-size: 0.95em;">🤝 Agent Coordination</strong>
            <div style="margin-top: 8px;">
                {agents_content}
            </div>
        </div>
        """
    
    def _format_evaluation_framework(self, evaluation: Dict[str, Any]) -> str:
        """Format evaluation framework."""
        weights = evaluation.get("primary_weights", {})
        diversity_targets = evaluation.get("diversity_targets", {})
        
        # Primary weights
        weights_html = ""
        if weights:
            weight_items = []
            for criterion, weight in list(weights.items())[:3]:  # Limit to 3
                weight_percent = int(weight * 100) if isinstance(weight, float) else weight
                weight_items.append(f"""
                <div style="display: flex; justify-content: space-between; margin: 3px 0;">
                    <span style="font-size: 0.8em;">{criterion.replace('_', ' ').title()}</span>
                    <span style="font-weight: bold; font-size: 0.8em;">{weight_percent}%</span>
                </div>
                """)
            weights_html = "\n".join(weight_items)
        
        # Diversity targets
        diversity_html = ""
        if diversity_targets:
            diversity_count = len(diversity_targets)
            diversity_html = f"""
            <div style="margin-top: 8px; font-size: 0.8em; color: #6c757d;">
                <strong>Diversity Targets:</strong> {diversity_count} criteria
            </div>
            """
        
        return f"""
        <div style="
            background: #f8f9fa;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 8px 0;
            border-radius: 0 5px 5px 0;
        ">
            <strong style="color: #333; font-size: 0.95em;">⚖️ Evaluation Criteria</strong>
            <div style="margin-top: 8px;">
                {weights_html}
                {diversity_html}
            </div>
        </div>
        """
    
    def _format_execution_monitoring(self, execution: Dict[str, Any]) -> str:
        """Format execution monitoring setup."""
        quality_thresholds = execution.get("quality_thresholds", {})
        fallback_strategies = execution.get("fallback_strategies", [])
        
        # Quality thresholds
        threshold_html = ""
        if quality_thresholds:
            min_confidence = quality_thresholds.get("min_confidence", 0.6)
            threshold_html = f"""
            <div style="font-size: 0.8em; color: #6c757d;">
                <strong>Quality Gate:</strong> {min_confidence:.0%} minimum confidence
            </div>
            """
        
        # Fallback strategies
        fallback_html = ""
        if fallback_strategies:
            fallback_count = len(fallback_strategies)
            fallback_html = f"""
            <div style="font-size: 0.8em; color: #6c757d; margin-top: 5px;">
                <strong>Fallback Plans:</strong> {fallback_count} strategies ready
            </div>
            """
        
        return f"""
        <div style="
            background: #f8f9fa;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 8px 0;
            border-radius: 0 5px 5px 0;
        ">
            <strong style="color: #333; font-size: 0.95em;">📊 Execution Monitoring</strong>
            <div style="margin-top: 8px;">
                {threshold_html}
                {fallback_html}
            </div>
        </div>
        """
    
    def _format_no_strategy(self) -> str:
        """Format display when no strategy is available."""
        return """
        <div style="
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            color: #6c757d;
        ">
            <p style="margin: 0; font-size: 0.9em;">
                🧠 Planning strategy will appear here...
            </p>
        </div>
        """
    
    def _format_error_display(self, error_message: str) -> str:
        """Format error display."""
        return f"""
        <div style="
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        ">
            <p style="margin: 0; color: #721c24; font-size: 0.85em;">
                ❌ Error displaying strategy: {error_message}
            </p>
        </div>
        """
    
    def format_planning_progress(self, stage: str, details: str = "") -> str:
        """
        Format planning progress indicator.
        
        Args:
            stage: Current planning stage
            details: Additional details about the stage
            
        Returns:
            Formatted HTML for progress display
        """
        stage_icons = {
            "analyzing": "🔍",
            "coordinating": "🤝", 
            "evaluating": "⚖️",
            "monitoring": "📊",
            "complete": "✅"
        }
        
        icon = stage_icons.get(stage, "🧠")
        
        return f"""
        <div style="
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            margin: 5px 0;
            font-size: 0.9em;
            text-align: center;
        ">
            {icon} <strong>{stage.title()}</strong>
            {f": {details}" if details else ""}
        </div>
        """ 