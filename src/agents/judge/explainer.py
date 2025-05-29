"""
Explainer for Judge Agent

Provides conversational explanation generation and reasoning capabilities
for the judge agent's final recommendations.
"""

from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class ConversationalExplainer:
    """
    Conversational explainer for judge agent recommendations.
    
    Provides:
    - Conversational explanation generation
    - Reasoning articulation
    - Context-aware explanations
    - Multi-style explanation formats
    """
    
    def __init__(self, llm_utils):
        """
        Initialize explainer with shared LLM utilities.
        
        Args:
            llm_utils: Shared LLM utilities for text generation
        """
        self.llm_utils = llm_utils
        self.explanation_templates = self._initialize_explanation_templates()
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        
        logger.debug("ConversationalExplainer initialized")
    
    async def generate_recommendation_explanation(
        self,
        recommendation: Any,
        context: Dict[str, Any],
        rank: int,
        explanation_style: str = 'conversational'
    ) -> str:
        """
        Generate explanation for a recommendation.
        
        Args:
            recommendation: The recommendation object
            context: Context including entities, intent, etc.
            rank: Recommendation rank
            explanation_style: Style of explanation
            
        Returns:
            Generated explanation text
        """
        try:
            if explanation_style == 'detailed':
                return await self._generate_detailed_explanation(recommendation, context, rank)
            elif explanation_style == 'concise':
                return await self._generate_concise_explanation(recommendation, context, rank)
            elif explanation_style == 'technical':
                return await self._generate_technical_explanation(recommendation, context, rank)
            else:  # conversational
                return await self._generate_conversational_explanation(recommendation, context, rank)
                
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")
            return self._create_fallback_explanation(recommendation, context, rank)
    
    async def _generate_conversational_explanation(
        self,
        recommendation: Any,
        context: Dict[str, Any],
        rank: int
    ) -> str:
        """Generate conversational explanation using LLM."""
        # Extract context information
        entities = context.get('entities', {})
        intent_analysis = context.get('intent_analysis', {})
        
        # Build explanation prompt
        prompt = self._build_conversational_prompt(recommendation, entities, intent_analysis, rank)
        
        # Generate explanation using LLM
        explanation = await self.llm_utils.call_llm(prompt)
        
        # Post-process explanation
        return self._post_process_explanation(explanation, recommendation, rank)
    
    def _build_conversational_prompt(
        self,
        recommendation: Any,
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        rank: int
    ) -> str:
        """Build conversational explanation prompt."""
        # Extract recommendation details
        name = getattr(recommendation, 'name', 'Unknown')
        artist = getattr(recommendation, 'artist', 'Unknown')
        genres = getattr(recommendation, 'genres', [])
        tags = getattr(recommendation, 'tags', [])
        source = getattr(recommendation, 'source', 'unknown')
        confidence = getattr(recommendation, 'confidence', 0.5)
        
        # Extract context details
        primary_intent = intent_analysis.get('primary_intent', 'discovery')
        target_genres = self._extract_target_genres(entities)
        target_moods = self._extract_target_moods(entities, intent_analysis)
        
        # Build prompt
        prompt = f"""Create an engaging, conversational explanation for why "{name}" by {artist} is ranked #{rank}.

User's Request Context:
- Primary Intent: {primary_intent}
- Target Genres: {', '.join(target_genres) if target_genres else 'Open to any'}
- Target Moods: {', '.join(target_moods) if target_moods else 'Any mood'}

Track Details:
- Genres: {', '.join(genres) if genres else 'Various'}
- Tags: {', '.join(tags[:5]) if tags else 'None specified'}
- Source: {source}
- Confidence: {confidence:.2f}

Write a natural, enthusiastic explanation (2-3 sentences) that:
1. Explains why this track fits the user's request
2. Highlights what makes it special or interesting
3. Uses conversational, engaging language
4. Avoids technical jargon

Focus on the musical qualities and why the user would enjoy this recommendation."""
        
        return prompt
    
    async def _generate_detailed_explanation(
        self,
        recommendation: Any,
        context: Dict[str, Any],
        rank: int
    ) -> str:
        """Generate detailed explanation with comprehensive reasoning."""
        entities = context.get('entities', {})
        intent_analysis = context.get('intent_analysis', {})
        
        prompt = f"""Create a detailed explanation for why "{getattr(recommendation, 'name', 'Unknown')}" by {getattr(recommendation, 'artist', 'Unknown')} is ranked #{rank}.

Provide a comprehensive explanation (4-5 sentences) covering:
1. How it matches the user's musical preferences
2. The specific qualities that make it a good fit
3. Its discovery value or uniqueness
4. Why it's ranked at this position

User Context: {intent_analysis.get('primary_intent', 'discovery')}
Track Genres: {', '.join(getattr(recommendation, 'genres', []))}
Track Tags: {', '.join(getattr(recommendation, 'tags', [])[:5])}
Confidence: {getattr(recommendation, 'confidence', 0.5):.2f}

Use informative but accessible language."""
        
        explanation = await self.llm_utils.call_llm(prompt)
        return self._post_process_explanation(explanation, recommendation, rank)
    
    async def _generate_concise_explanation(
        self,
        recommendation: Any,
        context: Dict[str, Any],
        rank: int
    ) -> str:
        """Generate concise explanation."""
        entities = context.get('entities', {})
        intent_analysis = context.get('intent_analysis', {})
        
        prompt = f"""Create a concise explanation for "{getattr(recommendation, 'name', 'Unknown')}" by {getattr(recommendation, 'artist', 'Unknown')} (#{rank}).

Write 1-2 sentences explaining:
- Why it fits the user's request
- What makes it noteworthy

User Intent: {intent_analysis.get('primary_intent', 'discovery')}
Track Info: {', '.join(getattr(recommendation, 'genres', [])[:2])}

Keep it brief but engaging."""
        
        explanation = await self.llm_utils.call_llm(prompt)
        return self._post_process_explanation(explanation, recommendation, rank)
    
    async def _generate_technical_explanation(
        self,
        recommendation: Any,
        context: Dict[str, Any],
        rank: int
    ) -> str:
        """Generate technical explanation with scoring details."""
        scores = getattr(recommendation, '_scores', {})
        
        explanation_parts = [
            f"#{rank}: {getattr(recommendation, 'name', 'Unknown')} by {getattr(recommendation, 'artist', 'Unknown')}"
        ]
        
        # Add scoring details
        if scores:
            if 'quality_score' in scores:
                explanation_parts.append(f"Quality: {scores['quality_score']:.2f}")
            if 'contextual_relevance' in scores:
                explanation_parts.append(f"Relevance: {scores['contextual_relevance']:.2f}")
            if 'intent_alignment' in scores:
                explanation_parts.append(f"Intent Alignment: {scores['intent_alignment']:.2f}")
        
        # Add source and confidence
        source = getattr(recommendation, 'source', 'unknown')
        confidence = getattr(recommendation, 'confidence', 0.5)
        explanation_parts.append(f"Source: {source}")
        explanation_parts.append(f"Confidence: {confidence:.2f}")
        
        return " | ".join(explanation_parts)
    
    def _create_fallback_explanation(
        self,
        recommendation: Any,
        context: Dict[str, Any],
        rank: int
    ) -> str:
        """Create fallback explanation when LLM is unavailable."""
        name = getattr(recommendation, 'name', 'Unknown')
        artist = getattr(recommendation, 'artist', 'Unknown')
        genres = getattr(recommendation, 'genres', [])
        source = getattr(recommendation, 'source', 'unknown')
        confidence = getattr(recommendation, 'confidence', 0.5)
        
        explanation_parts = [f"#{rank}: {name} by {artist}"]
        
        # Add genre information
        if genres:
            explanation_parts.append(f"A {'/'.join(genres[:2])} track")
        
        # Add source-based reasoning
        if source == 'discovery_agent':
            explanation_parts.append("Great for discovery")
        elif source == 'genre_mood_agent':
            explanation_parts.append("Matches your genre and mood preferences")
        elif source == 'planner_agent':
            explanation_parts.append("Fits your request well")
        
        # Add confidence information
        if confidence > 0.8:
            explanation_parts.append("High confidence match")
        elif confidence > 0.6:
            explanation_parts.append("Good match")
        else:
            explanation_parts.append("Interesting option")
        
        return ". ".join(explanation_parts) + "."
    
    def _post_process_explanation(
        self,
        explanation: str,
        recommendation: Any,
        rank: int
    ) -> str:
        """Post-process generated explanation."""
        # Clean up the explanation
        explanation = explanation.strip()
        
        # Ensure it starts with rank if not present
        name = getattr(recommendation, 'name', 'Unknown')
        if not explanation.startswith(f"#{rank}") and not explanation.startswith(name):
            explanation = f"#{rank}: {explanation}"
        
        # Ensure proper ending
        if not explanation.endswith('.') and not explanation.endswith('!'):
            explanation += "."
        
        return explanation
    
    async def generate_collection_summary(
        self,
        recommendations: List[Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate summary explanation for the entire recommendation collection."""
        try:
            # Extract collection characteristics
            total_count = len(recommendations)
            sources = [getattr(rec, 'source', 'unknown') for rec in recommendations]
            genres = []
            for rec in recommendations:
                genres.extend(getattr(rec, 'genres', []))
            
            # Count distributions
            source_counts = {}
            for source in sources:
                source_counts[source] = source_counts.get(source, 0) + 1
            
            genre_counts = {}
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            # Get top genres and sources
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # Extract user context
            intent_analysis = context.get('intent_analysis', {})
            primary_intent = intent_analysis.get('primary_intent', 'discovery')
            
            # Build summary prompt
            prompt = f"""Create a brief summary of this music recommendation collection.

Collection Details:
- Total Recommendations: {total_count}
- Primary Intent: {primary_intent}
- Top Genres: {', '.join([genre for genre, count in top_genres])}
- Main Sources: {', '.join([source.replace('_agent', '') for source, count in top_sources])}

Write 2-3 sentences that:
1. Summarize what type of music was recommended
2. Explain how it fits the user's request
3. Highlight the variety or focus of the collection

Use engaging, conversational language."""
            
            summary = await self.llm_utils.call_llm(prompt)
            return summary.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate collection summary: {e}")
            return self._create_fallback_summary(recommendations, context)
    
    def _create_fallback_summary(
        self,
        recommendations: List[Any],
        context: Dict[str, Any]
    ) -> str:
        """Create fallback collection summary."""
        total_count = len(recommendations)
        intent_analysis = context.get('intent_analysis', {})
        primary_intent = intent_analysis.get('primary_intent', 'discovery')
        
        if primary_intent == 'discovery':
            return f"Here are {total_count} discovery recommendations featuring a mix of underground gems and interesting finds to expand your musical horizons."
        elif primary_intent == 'genre_mood':
            return f"Here are {total_count} recommendations carefully selected to match your genre and mood preferences."
        else:
            return f"Here are {total_count} personalized music recommendations based on your request."
    
    def _extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target genres from entities."""
        musical_entities = entities.get('musical_entities', {})
        genres = musical_entities.get('genres', {})
        
        target_genres = []
        target_genres.extend(genres.get('primary', []))
        target_genres.extend(genres.get('secondary', []))
        
        return list(set(target_genres))  # Remove duplicates
    
    def _extract_target_moods(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[str]:
        """Extract target moods from entities and intent analysis."""
        moods = []
        
        # From entities
        contextual_entities = entities.get('contextual_entities', {})
        mood_entities = contextual_entities.get('moods', {})
        moods.extend(mood_entities.get('energy', []))
        moods.extend(mood_entities.get('emotion', []))
        
        # From intent analysis
        moods.extend(intent_analysis.get('mood_indicators', []))
        
        return list(set(moods))  # Remove duplicates
    
    def _initialize_explanation_templates(self) -> Dict[str, str]:
        """Initialize explanation templates for different scenarios."""
        return {
            'discovery_high_rank': "#{rank}: {name} by {artist} is a fantastic discovery that {reason}. {quality_note}",
            'genre_match': "#{rank}: {name} by {artist} perfectly captures the {genre} sound you're looking for. {additional_note}",
            'mood_match': "#{rank}: {name} by {artist} delivers exactly the {mood} energy you want. {context_note}",
            'underground_gem': "#{rank}: {name} by {artist} is a hidden gem that {discovery_reason}. {uniqueness_note}",
            'similarity_match': "#{rank}: {name} by {artist} shares the musical DNA of {similar_artists}. {connection_note}"
        }
    
    def _initialize_reasoning_patterns(self) -> Dict[str, List[str]]:
        """Initialize reasoning patterns for different recommendation types."""
        return {
            'discovery': [
                "offers something completely new to explore",
                "represents an exciting musical journey",
                "showcases underground talent worth discovering",
                "provides a fresh perspective on familiar sounds"
            ],
            'genre_mood': [
                "perfectly captures the essence of {genre}",
                "delivers the exact {mood} vibe you're seeking",
                "exemplifies the best of {genre} music",
                "creates the perfect {mood} atmosphere"
            ],
            'similarity': [
                "shares musical similarities with your favorites",
                "explores similar sonic territory",
                "offers a natural progression from your preferences",
                "connects to artists you already enjoy"
            ],
            'quality': [
                "demonstrates exceptional musical craftsmanship",
                "showcases outstanding production quality",
                "features memorable melodies and arrangements",
                "represents artistic excellence in its genre"
            ]
        } 