"""
Judge Agent Module

Simplified judge agent with focused responsibilities:
- Candidate evaluation and ranking
- Final recommendation selection
- Diversity optimization
- Conversational explanations
"""

from .agent import JudgeAgent
from .ranking_logic import RankingLogic
from .explainer import ConversationalExplainer

__all__ = [
    "JudgeAgent",
    "RankingLogic",
    "ConversationalExplainer",
] 