"""
Explanation Generator for Judge Agent

Enhanced explanation generation and reasoning capabilities
for the judge agent's final recommendations.

Extracted from judge/explainer.py and judge/agent.py for better modularity.
"""

from typing import Dict, List, Any
import structlog

logger = structlog.get_logger(__name__)


class ExplanationGenerator:
    """
    Enhanced explanation generator for judge agent recommendations.
    
    Responsibilities:
    - Conversational explanation generation
    - Reasoning articulation
    - Context-aware explanations
    - Multi-style explanation formats
    - Fallback explanation creation
    """
    
    def __init__(self, llm_utils):
        """
        Initialize explanation generator with shared LLM utilities.
        
        Args:
            llm_utils: Shared LLM utilities for text generation
        """
        self.llm_utils = llm_utils
        self.explanation_templates = self._initialize_explanation_templates()
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        
        self.logger = structlog.get_logger(__name__)
        self.logger.debug("ExplanationGenerator initialized")
    
    async def generate_explanations(
        self,
        selections: List[Any],
        state,
        explanation_style: str = 'conversational'
    ) -> List[Any]:
        """
        Generate explanations for all selected recommendations.
        
        Args:
            selections: List of selected recommendations
            state: Current workflow state with context
            explanation_style: Style of explanation ('conversational', 'detailed', 'concise', 'technical')
            
        Returns:
            List of recommendations with explanations added
        """
        try:
            # Extract context from state
            entities = getattr(state, 'entities', {})
            intent_analysis = getattr(state, 'intent_analysis', {})
            
            context = {
                'entities': entities,
                'intent_analysis': intent_analysis
            }
            
            # Generate explanations for all recommendations in a single batch call
            try:
                explanations = await self._generate_batch_explanations(
                    selections, context, explanation_style
                )
                
                # Apply explanations to recommendations
                explained_selections = []
                for i, (recommendation, explanation) in enumerate(zip(selections, explanations)):
                    recommendation.explanation = explanation
                    explained_selections.append(recommendation)
                
                self.logger.info(f"Generated explanations for {len(explained_selections)} recommendations")
                return explained_selections
                
            except Exception as e:
                self.logger.error(f"Batch explanation generation failed: {e}")
                # Fallback to simple explanations
                explained_selections = []
                for rank, recommendation in enumerate(selections, 1):
                    fallback_explanation = self._create_fallback_explanation(
                        recommendation, context, rank
                    )
                    recommendation.explanation = fallback_explanation
                    explained_selections.append(recommendation)
                
                self.logger.info(f"Generated fallback explanations for {len(explained_selections)} recommendations")
                return explained_selections
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            # Return selections with fallback explanations
            for rank, rec in enumerate(selections, 1):
                if not hasattr(rec, 'explanation') or not rec.explanation:
                    rec.explanation = f"Great track #{rank} that matches your musical taste!"
            return selections

    async def _generate_batch_explanations(
        self,
        selections: List[Any],
        context: Dict[str, Any],
        explanation_style: str = 'conversational'
    ) -> List[str]:
        """
        Generate explanations for all recommendations in a single LLM call.
        
        Args:
            selections: List of recommendations
            context: Context including entities, intent, etc.
            explanation_style: Style of explanations
            
        Returns:
            List of explanations corresponding to each recommendation
        """
        if not selections:
            return []
        
        # Extract context information
        entities = context.get('entities', {})
        intent_analysis = context.get('intent_analysis', {})
        primary_intent = intent_analysis.get('primary_intent', 'discovery')
        target_genres = self._extract_target_genres(entities)
        target_moods = self._extract_target_moods(entities, intent_analysis)
        
        # Build the batch prompt
        recommendations_text = []
        for i, rec in enumerate(selections, 1):
            name = getattr(rec, 'title', getattr(rec, 'name', 'Unknown'))
            artist = getattr(rec, 'artist', 'Unknown')
            genres = getattr(rec, 'genres', [])
            tags = getattr(rec, 'tags', [])
            confidence = getattr(rec, 'confidence', 0.5)
            
            rec_text = f"""#{i}: "{name}" by {artist}
- Genres: {', '.join(genres[:3]) if genres else 'Various'}
- Tags: {', '.join(tags[:3]) if tags else 'None specified'}
- Confidence: {confidence:.2f}"""
            
            recommendations_text.append(rec_text)
        
        prompt = f"""Generate brief, engaging explanations for why each of these music recommendations fits the user's request.

User's Request Context:
- Primary Intent: {primary_intent}
- Target Genres: {', '.join(target_genres) if target_genres else 'Open to any'}
- Target Moods: {', '.join(target_moods) if target_moods else 'Any mood'}

Recommendations to explain:
{chr(10).join(recommendations_text)}

Instructions:
- Write 1-2 sentences for each recommendation explaining why it's a good fit
- Use engaging, conversational language
- Focus on musical qualities and discovery value
- Number each explanation (1., 2., 3., etc.)
- Keep explanations concise but enthusiastic

Format your response as:
1. [explanation for first track]
2. [explanation for second track]
3. [explanation for third track]
etc."""

        # Make single LLM call
        response = await self.llm_utils.call_llm(prompt)
        
        # Debug: Log the actual LLM response
        self.logger.debug(f"🔍 LLM EXPLANATION RESPONSE: {response[:500]}...")
        
        # Parse the response into individual explanations
        explanations = self._parse_batch_explanations(response, len(selections))
        
        # Debug: Log the parsed explanations
        self.logger.debug(f"🔍 PARSED EXPLANATIONS: {[exp[:50] + '...' for exp in explanations[:3]]}")
        
        return explanations
    
    def _parse_batch_explanations(self, response: str, expected_count: int) -> List[str]:
        """
        Parse the batch LLM response into individual explanations.
        
        Args:
            response: The LLM response text
            expected_count: Expected number of explanations
            
        Returns:
            List of individual explanations
        """
        explanations = []
        lines = response.strip().split('\n')
        
        # Try numbered format first (1., 2., 3., etc.)
        current_explanation = ""
        found_numbered = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a number (1., 2., 3., etc.)
            import re
            if re.match(r'^\d+\.', line):
                found_numbered = True
                # Save previous explanation if we have one
                if current_explanation:
                    explanations.append(current_explanation.strip())
                
                # Start new explanation (remove the number prefix)
                current_explanation = re.sub(r'^\d+\.\s*', '', line)
            else:
                # Continue current explanation
                if current_explanation:
                    current_explanation += " " + line
        
        # Add the last explanation
        if current_explanation:
            explanations.append(current_explanation.strip())
        
        # If numbered format worked, return those explanations
        if found_numbered and explanations:
            # Filter out introductory text that doesn't contain track names
            filtered_explanations = []
            for exp in explanations:
                # Skip explanations that are just introductory text
                # Look for track names in quotes or "by Artist" patterns, or any substantial explanation
                if ('"' in exp and ('by ' in exp or ':' in exp)) or ('track' in exp.lower() and len(exp) > 30) or len(exp) > 40:
                    filtered_explanations.append(exp)
            
            if filtered_explanations:
                return filtered_explanations[:expected_count]
        
        # Fallback: Try track-name format ("Track Name" by Artist: explanation)
        explanations = []
        current_explanation = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a track name pattern
            if re.match(r'^"[^"]+"\s+by\s+[^:]+:', line):
                # Save previous explanation if we have one
                if current_explanation:
                    explanations.append(current_explanation.strip())
                
                # Start new explanation (remove the track name prefix)
                current_explanation = re.sub(r'^"[^"]+"\s+by\s+[^:]+:\s*', '', line)
            else:
                # Continue current explanation
                if current_explanation:
                    current_explanation += " " + line
        
        # Add the last explanation
        if current_explanation:
            explanations.append(current_explanation.strip())
        
        # Return up to expected count
        return explanations[:expected_count]
    
    async def generate_recommendation_explanation(
        self,
        recommendation: Any,
        context: Dict[str, Any],
        rank: int,
        explanation_style: str = 'conversational'
    ) -> str:
        """
        Generate explanation for a single recommendation.
        
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
            self.logger.warning(f"Failed to generate explanation: {e}")
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
        
        # Add scoring details if available
        if scores:
            high_scores = [(k, v) for k, v in scores.items() if isinstance(v, (int, float)) and v > 0.7]
            if high_scores:
                score_details = ', '.join([f"{k}: {v:.2f}" for k, v in high_scores[:3]])
                explanation_parts.append(f"Strong scores: {score_details}")
        
        # Add source and confidence
        source = getattr(recommendation, 'source', 'unknown')
        confidence = getattr(recommendation, 'confidence', 0.5)
        explanation_parts.append(f"Source: {source}, Confidence: {confidence:.2f}")
        
        return '. '.join(explanation_parts)
    
    def _create_fallback_explanation(
        self,
        recommendation: Any,
        context: Dict[str, Any],
        rank: int
    ) -> str:
        """Create fallback explanation when LLM generation fails."""
        try:
            # Fix: Try both 'title' and 'name' fields for track name
            name = getattr(recommendation, 'title', getattr(recommendation, 'name', 'Unknown'))
            artist = getattr(recommendation, 'artist', 'Unknown')
            
            entities = context.get('entities', {})
            intent_analysis = context.get('intent_analysis', {})
            
            # Basic template-based explanation
            templates = [
                f"'{name}' by {artist} is an excellent choice for your musical exploration.",
                f"This track by {artist} offers a great listening experience that matches your taste.",
                f"'{name}' brings unique qualities that align well with your musical preferences.",
                f"{artist}'s '{name}' is a solid recommendation that delivers on your request."
            ]
            
            base_explanation = templates[rank % len(templates)]
            
            # Add context-specific details if available
            target_genres = self._extract_target_genres(entities)
            if target_genres:
                base_explanation += f" It fits well within the {', '.join(target_genres[:2])} genre space."
            
            primary_intent = intent_analysis.get('primary_intent', '')
            if primary_intent == 'discovery':
                base_explanation += " It offers good discovery value for expanding your musical horizons."
            elif primary_intent == 'similarity':
                base_explanation += " It shares musical DNA with your reference tracks."
            
            return base_explanation
            
        except Exception as e:
            self.logger.warning(f"Fallback explanation creation failed: {e}")
            return f"Great track #{rank} that matches your musical taste!"
    
    def _post_process_explanation(
        self,
        explanation: str,
        recommendation: Any,
        rank: int
    ) -> str:
        """Post-process explanation to ensure quality and consistency."""
        try:
            # Clean up explanation
            explanation = explanation.strip()
            
            # Remove any unwanted prefixes/suffixes
            prefixes_to_remove = ["Here's why:", "Explanation:", "Because:"]
            for prefix in prefixes_to_remove:
                if explanation.startswith(prefix):
                    explanation = explanation[len(prefix):].strip()
            
            # Ensure it doesn't start with a quote
            if explanation.startswith('"') and explanation.endswith('"'):
                explanation = explanation[1:-1]
            
            # Ensure reasonable length (not too short or too long)
            if len(explanation) < 20:
                return self._create_fallback_explanation(recommendation, {}, rank)
            
            if len(explanation) > 500:
                # Truncate at sentence boundary
                sentences = explanation.split('. ')
                explanation = '. '.join(sentences[:3])
                if not explanation.endswith('.'):
                    explanation += '.'
            
            return explanation
            
        except Exception as e:
            self.logger.warning(f"Explanation post-processing failed: {e}")
            return explanation
    
    async def generate_collection_summary(
        self,
        recommendations: List[Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate summary for the entire recommendation collection."""
        try:
            entities = context.get('entities', {})
            intent_analysis = context.get('intent_analysis', {})
            
            primary_intent = intent_analysis.get('primary_intent', 'discovery')
            target_genres = self._extract_target_genres(entities)
            target_moods = self._extract_target_moods(entities, intent_analysis)
            
            prompt = f"""Create a brief, engaging summary for this collection of {len(recommendations)} music recommendations.

User's Request Context:
- Primary Intent: {primary_intent}
- Target Genres: {', '.join(target_genres) if target_genres else 'Various'}
- Target Moods: {', '.join(target_moods) if target_moods else 'Any mood'}

Collection Overview:
- Total Tracks: {len(recommendations)}
- Artists: {', '.join(set([rec.artist for rec in recommendations[:5]]))}
- Top Genres: {', '.join(set([g for rec in recommendations[:5] for g in (rec.genres or [])[:2]]))}

Write 2-3 sentences that:
1. Introduce the collection in relation to their request
2. Highlight the diversity or theme of selections
3. Encourage exploration

Keep it conversational and enthusiastic."""
            
            summary = await self.llm_utils.call_llm(prompt)
            return self._post_process_explanation(summary, None, 0)
            
        except Exception as e:
            self.logger.warning(f"Collection summary generation failed: {e}")
            return self._create_fallback_summary(recommendations, context)
    
    def _create_fallback_summary(
        self,
        recommendations: List[Any],
        context: Dict[str, Any]
    ) -> str:
        """Create fallback summary when LLM generation fails."""
        try:
            count = len(recommendations)
            entities = context.get('entities', {})
            intent_analysis = context.get('intent_analysis', {})
            
            primary_intent = intent_analysis.get('primary_intent', 'discovery')
            
            if primary_intent == 'discovery':
                return f"Here are {count} diverse tracks to expand your musical horizons! Each recommendation offers something unique to explore."
            elif primary_intent == 'similarity':
                return f"I've found {count} tracks that share musical qualities with your preferences. They should feel familiar yet offer new listening experiences."
            else:
                return f"Here's a curated collection of {count} tracks that match your request. Each one brings something special to your playlist!"
                
        except Exception as e:
            self.logger.warning(f"Fallback summary creation failed: {e}")
            return f"Here are {len(recommendations)} great tracks for you to explore!"
    
    def _extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target genres from entities."""
        return entities.get('genres', [])
    
    def _extract_target_moods(self, entities: Dict[str, Any], intent_analysis: Dict[str, Any]) -> List[str]:
        """Extract target moods from entities and intent analysis."""
        moods = entities.get('moods', [])
        mood_descriptors = intent_analysis.get('mood_descriptors', [])
        return list(set(moods + mood_descriptors))
    
    def _initialize_explanation_templates(self) -> Dict[str, str]:
        """Initialize explanation templates for different scenarios."""
        return {
            'discovery': "This track offers {qualities} that align with your taste for musical exploration.",
            'similarity': "'{track}' by {artist} shares {similarities} with your reference tracks.",
            'genre': "This {genre} track captures the essence of what you're looking for.",
            'mood': "'{track}' delivers the {mood} vibe you're seeking with {qualities}.",
            'underground': "This hidden gem by {artist} brings {unique_qualities} to your collection."
        }
    
    def _initialize_reasoning_patterns(self) -> Dict[str, List[str]]:
        """Initialize reasoning patterns for different contexts."""
        return {
            'genre_match': [
                "perfectly captures the {genre} sound",
                "exemplifies the best of {genre} music",
                "brings authentic {genre} elements"
            ],
            'mood_match': [
                "delivers the exact {mood} energy you're seeking",
                "captures that {mood} feeling brilliantly",
                "embodies the {mood} atmosphere"
            ],
            'discovery_value': [
                "offers great discovery potential",
                "introduces you to something fresh",
                "expands your musical horizons"
            ],
            'quality_indicators': [
                "showcases exceptional musicianship",
                "demonstrates high production quality",
                "features compelling songwriting"
            ]
        } 