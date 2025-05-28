import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import ValidationError

# Import the correct MusicRecommenderState model
from ..models.agent_models import MusicRecommenderState
from ..models.recommendation_models import TrackRecommendation


class PromptAnalysisEngine:
    """Analyzes user prompts to extract intent, context, and preferences"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Intent patterns for recognition
        self.intent_patterns = {
            "concentration": ["focus", "coding", "work", "study", "concentration", "productive", "programming"],
            "energy": ["workout", "exercise", "pump up", "energetic", "motivation", "gym", "running"],
            "relaxation": ["chill", "relax", "calm", "unwind", "peaceful", "mellow", "ambient"],
            "discovery": ["surprise", "new", "discover", "explore", "recommend", "find", "unknown"],
            "mood_enhancement": ["happy", "sad", "melancholic", "upbeat", "emotional", "feel good"],
            "background": ["background", "ambient", "while", "during", "playing"]
        }
        
        # Activity patterns
        self.activity_patterns = {
            "coding": ["coding", "programming", "development", "software", "computer"],
            "workout": ["workout", "exercise", "gym", "fitness", "training", "running"],
            "study": ["study", "studying", "learning", "reading", "homework"],
            "work": ["work", "working", "office", "meeting", "productivity"],
            "relaxation": ["relaxing", "chilling", "unwinding", "resting"],
            "driving": ["driving", "car", "road trip", "commute"]
        }
        
        # Mood indicators
        self.mood_indicators = {
            "upbeat": ["upbeat", "happy", "energetic", "positive", "cheerful"],
            "mellow": ["mellow", "calm", "peaceful", "soft", "gentle"],
            "intense": ["intense", "powerful", "strong", "aggressive", "heavy"],
            "melancholic": ["sad", "melancholic", "emotional", "nostalgic", "moody"]
        }
    
    def analyze_prompt(self, prompt: str, conversation_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze user prompt to extract intent, context, and preferences
        
        Args:
            prompt: User's natural language prompt
            conversation_context: Previous conversation context
            
        Returns:
            Dictionary containing prompt analysis results
        """
        prompt_lower = prompt.lower()
        
        analysis = {
            "primary_intent": self._extract_primary_intent(prompt_lower),
            "activity_context": self._identify_activity(prompt_lower),
            "mood_request": self._extract_mood_indicators(prompt_lower),
            "genre_preferences": self._identify_genre_mentions(prompt_lower),
            "exploration_openness": self._assess_discovery_intent(prompt_lower),
            "temporal_context": self._extract_time_context(prompt_lower),
            "specificity_level": self._measure_request_specificity(prompt_lower),
            "energy_level": self._extract_energy_level(prompt_lower),
            "conversation_continuity": self._analyze_conversation_continuity(prompt_lower, conversation_context)
        }
        
        self.logger.debug(f"Prompt analysis completed: {analysis}")
        return analysis
    
    def _extract_primary_intent(self, prompt: str) -> str:
        """Extract the primary intent from the prompt"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in prompt)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        return "discovery"  # Default intent
    
    def _identify_activity(self, prompt: str) -> Optional[str]:
        """Identify the activity context from the prompt"""
        for activity, patterns in self.activity_patterns.items():
            if any(pattern in prompt for pattern in patterns):
                return activity
        return None
    
    def _extract_mood_indicators(self, prompt: str) -> List[str]:
        """Extract mood indicators from the prompt"""
        found_moods = []
        for mood, patterns in self.mood_indicators.items():
            if any(pattern in prompt for pattern in patterns):
                found_moods.append(mood)
        return found_moods
    
    def _identify_genre_mentions(self, prompt: str) -> List[str]:
        """Identify genre mentions in the prompt"""
        # Common genre patterns
        genre_patterns = [
            "rock", "pop", "jazz", "classical", "electronic", "hip hop", "rap",
            "indie", "alternative", "metal", "folk", "country", "blues", "reggae",
            "techno", "house", "ambient", "experimental", "punk", "grunge"
        ]
        
        found_genres = []
        for genre in genre_patterns:
            if genre in prompt:
                found_genres.append(genre)
        return found_genres
    
    def _assess_discovery_intent(self, prompt: str) -> float:
        """Assess how open the user is to discovery (0.0 to 1.0)"""
        discovery_keywords = ["surprise", "new", "discover", "explore", "different", "unknown", "random"]
        specific_keywords = ["exactly", "specifically", "only", "just", "particular"]
        
        discovery_score = sum(1 for keyword in discovery_keywords if keyword in prompt)
        specificity_score = sum(1 for keyword in specific_keywords if keyword in prompt)
        
        # Normalize to 0-1 scale
        if discovery_score > 0 and specificity_score == 0:
            return min(1.0, discovery_score * 0.3 + 0.5)
        elif specificity_score > 0:
            return max(0.1, 0.5 - specificity_score * 0.2)
        else:
            return 0.5  # Neutral
    
    def _extract_time_context(self, prompt: str) -> Optional[str]:
        """Extract temporal context from the prompt"""
        time_patterns = {
            "morning": ["morning", "am", "wake up", "start day"],
            "afternoon": ["afternoon", "lunch", "midday"],
            "evening": ["evening", "night", "pm", "end of day"],
            "late_night": ["late night", "midnight", "sleep", "bedtime"]
        }
        
        for time_period, patterns in time_patterns.items():
            if any(pattern in prompt for pattern in patterns):
                return time_period
        return None
    
    def _measure_request_specificity(self, prompt: str) -> float:
        """Measure how specific the request is (0.0 = very open, 1.0 = very specific)"""
        specific_indicators = ["exactly", "specifically", "only", "just", "particular", "precise"]
        open_indicators = ["anything", "whatever", "surprise", "random", "any"]
        
        specific_count = sum(1 for indicator in specific_indicators if indicator in prompt)
        open_count = sum(1 for indicator in open_indicators if indicator in prompt)
        
        # Count specific mentions (artists, songs, albums)
        specific_mentions = len(re.findall(r'"[^"]*"', prompt))  # Quoted items
        
        specificity = (specific_count + specific_mentions * 0.5) / max(1, specific_count + open_count + specific_mentions)
        return min(1.0, specificity)
    
    def _extract_energy_level(self, prompt: str) -> Optional[str]:
        """Extract energy level from the prompt"""
        high_energy = ["high energy", "energetic", "pump up", "intense", "powerful"]
        low_energy = ["low energy", "calm", "peaceful", "mellow", "soft"]
        
        if any(pattern in prompt for pattern in high_energy):
            return "high"
        elif any(pattern in prompt for pattern in low_energy):
            return "low"
        return "medium"
    
    def _analyze_conversation_continuity(self, prompt: str, conversation_context: Optional[Dict]) -> Dict[str, Any]:
        """Analyze how this prompt relates to previous conversation"""
        continuity = {
            "is_follow_up": False,
            "references_previous": False,
            "builds_on_context": False
        }
        
        if conversation_context:
            follow_up_indicators = ["also", "another", "more", "similar", "like that", "continue"]
            reference_indicators = ["that", "those", "previous", "last", "earlier"]
            
            continuity["is_follow_up"] = any(indicator in prompt for indicator in follow_up_indicators)
            continuity["references_previous"] = any(indicator in prompt for indicator in reference_indicators)
            continuity["builds_on_context"] = continuity["is_follow_up"] or continuity["references_previous"]
        
        return continuity


class ContextualRelevanceScorer:
    """Scores tracks based on contextual relevance to the prompt"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Activity-music matching profiles
        self.activity_profiles = {
            "coding": {
                "instrumental_preference": 0.8,
                "energy_range": (0.3, 0.7),
                "vocal_preference": "minimal",
                "tempo_range": (80, 140),
                "genres_preferred": ["ambient", "post-rock", "electronic", "instrumental"]
            },
            "workout": {
                "energy_range": (0.7, 1.0),
                "tempo_range": (120, 180),
                "motivational": True,
                "genres_preferred": ["electronic", "hip hop", "rock", "pop"]
            },
            "study": {
                "distraction_level": "low",
                "consistency": "high",
                "tempo_range": (60, 120),
                "genres_preferred": ["classical", "ambient", "instrumental", "lo-fi"]
            },
            "relaxation": {
                "energy_range": (0.1, 0.5),
                "tempo_range": (60, 100),
                "genres_preferred": ["ambient", "folk", "jazz", "classical"]
            }
        }
    
    def calculate_contextual_relevance(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any]) -> float:
        """
        Calculate contextual relevance score for a track
        
        Args:
            track: Track to score
            prompt_analysis: Analysis of user prompt
            
        Returns:
            Contextual relevance score (0.0 to 1.0)
        """
        activity_context = prompt_analysis.get("activity_context")
        mood_request = prompt_analysis.get("mood_request", [])
        energy_level = prompt_analysis.get("energy_level")
        
        scores = []
        
        # Activity alignment score
        if activity_context:
            activity_score = self._calculate_activity_alignment(track, activity_context)
            scores.append(("activity", activity_score, 0.35))
        
        # Mood compatibility score
        if mood_request:
            mood_score = self._calculate_mood_match(track, mood_request)
            scores.append(("mood", mood_score, 0.30))
        
        # Energy alignment score
        if energy_level:
            energy_score = self._calculate_energy_match(track, energy_level)
            scores.append(("energy", energy_score, 0.25))
        
        # Temporal appropriateness
        temporal_context = prompt_analysis.get("temporal_context")
        if temporal_context:
            temporal_score = self._calculate_temporal_appropriateness(track, temporal_context)
            scores.append(("temporal", temporal_score, 0.10))
        
        # Calculate weighted average
        if scores:
            total_weight = sum(weight for _, _, weight in scores)
            weighted_sum = sum(score * weight for _, score, weight in scores)
            return weighted_sum / total_weight
        
        return 0.5  # Neutral score if no context available
    
    def _calculate_activity_alignment(self, track: TrackRecommendation, activity: str) -> float:
        """Calculate how well a track aligns with an activity"""
        if activity not in self.activity_profiles:
            return 0.5
        
        profile = self.activity_profiles[activity]
        alignment_score = 0.0
        factors = 0
        
        # Check instrumental preference
        if "instrumental_preference" in profile and track.instrumental is not None:
            if track.instrumental and profile["instrumental_preference"] > 0.5:
                alignment_score += profile["instrumental_preference"]
            elif not track.instrumental and profile["instrumental_preference"] <= 0.5:
                alignment_score += (1.0 - profile["instrumental_preference"])
            factors += 1
        
        # Check genre preferences
        if "genres_preferred" in profile and track.genres:
            genre_match = any(genre.lower() in [g.lower() for g in profile["genres_preferred"]] 
                            for genre in track.genres)
            alignment_score += 1.0 if genre_match else 0.3
            factors += 1
        
        # If we have energy information in additional_scores
        if "energy_range" in profile and track.additional_scores.get("energy"):
            energy_value = track.additional_scores["energy"]
            min_energy, max_energy = profile["energy_range"]
            if min_energy <= energy_value <= max_energy:
                alignment_score += 1.0
            else:
                # Penalize based on distance from range
                distance = min(abs(energy_value - min_energy), abs(energy_value - max_energy))
                alignment_score += max(0.0, 1.0 - distance)
            factors += 1
        
        return alignment_score / max(1, factors)
    
    def _calculate_mood_match(self, track: TrackRecommendation, mood_requests: List[str]) -> float:
        """Calculate mood compatibility between track and request"""
        if not mood_requests or not track.moods:
            return 0.5
        
        # Check for direct mood matches
        track_moods_lower = [mood.lower() for mood in track.moods]
        mood_matches = sum(1 for mood in mood_requests 
                          if mood.lower() in track_moods_lower)
        
        # Normalize by number of requested moods
        return min(1.0, mood_matches / len(mood_requests))
    
    def _calculate_energy_match(self, track: TrackRecommendation, requested_energy: str) -> float:
        """Calculate energy level match"""
        # Map energy levels to ranges
        energy_ranges = {
            "low": (0.0, 0.4),
            "medium": (0.3, 0.7),
            "high": (0.6, 1.0)
        }
        
        if requested_energy not in energy_ranges:
            return 0.5
        
        min_energy, max_energy = energy_ranges[requested_energy]
        
        # Check if track has energy information
        track_energy = track.additional_scores.get("energy")
        if track_energy is not None:
            if min_energy <= track_energy <= max_energy:
                return 1.0
            else:
                # Calculate distance penalty
                distance = min(abs(track_energy - min_energy), abs(track_energy - max_energy))
                return max(0.0, 1.0 - distance)
        
        # Fallback: use genre-based energy estimation
        high_energy_genres = ["electronic", "rock", "metal", "hip hop", "dance"]
        low_energy_genres = ["ambient", "classical", "folk", "jazz"]
        
        if track.genres:
            track_genres_lower = [g.lower() for g in track.genres]
            
            if requested_energy == "high":
                return 0.8 if any(g in high_energy_genres for g in track_genres_lower) else 0.3
            elif requested_energy == "low":
                return 0.8 if any(g in low_energy_genres for g in track_genres_lower) else 0.3
        
        return 0.5
    
    def _calculate_temporal_appropriateness(self, track: TrackRecommendation, temporal_context: str) -> float:
        """Calculate temporal appropriateness of track"""
        # Simple temporal matching based on energy and mood
        temporal_preferences = {
            "morning": {"energy_min": 0.4, "moods": ["upbeat", "energetic"]},
            "afternoon": {"energy_min": 0.3, "moods": ["focused", "productive"]},
            "evening": {"energy_max": 0.7, "moods": ["relaxing", "mellow"]},
            "late_night": {"energy_max": 0.5, "moods": ["calm", "ambient"]}
        }
        
        if temporal_context not in temporal_preferences:
            return 0.5
        
        prefs = temporal_preferences[temporal_context]
        score = 0.5
        
        # Check energy appropriateness
        track_energy = track.additional_scores.get("energy", 0.5)
        if "energy_min" in prefs and track_energy >= prefs["energy_min"]:
            score += 0.3
        elif "energy_max" in prefs and track_energy <= prefs["energy_max"]:
            score += 0.3
        
        # Check mood appropriateness
        if track.moods and "moods" in prefs:
            mood_match = any(mood.lower() in [m.lower() for m in prefs["moods"]] 
                           for mood in track.moods)
            if mood_match:
                score += 0.2
        
        return min(1.0, score)


class DiscoveryAppropriatenessScorer:
    """Scores tracks based on discovery appropriateness"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_discovery_score(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any]) -> float:
        """
        Calculate discovery appropriateness score
        
        Args:
            track: Track to score
            prompt_analysis: Analysis of user prompt
            
        Returns:
            Discovery appropriateness score (0.0 to 1.0)
        """
        exploration_openness = prompt_analysis.get("exploration_openness", 0.5)
        specificity_level = prompt_analysis.get("specificity_level", 0.5)
        
        # Calculate discovery factor based on prompt analysis
        discovery_factor = self._calculate_discovery_factor(exploration_openness, specificity_level)
        
        # Estimate track familiarity
        track_familiarity = self._estimate_track_familiarity(track)
        
        # Calculate genre adventurousness
        genre_adventurousness = self._calculate_genre_expansion(track, prompt_analysis)
        
        # Balance familiarity based on prompt openness
        if discovery_factor > 0.7:  # High discovery intent
            return (1 - track_familiarity) * 0.6 + genre_adventurousness * 0.4
        elif discovery_factor < 0.3:  # Low discovery intent
            return track_familiarity * 0.7 + genre_adventurousness * 0.3
        else:  # Balanced approach
            return 0.5 + (genre_adventurousness - 0.5) * discovery_factor
    
    def _calculate_discovery_factor(self, exploration_openness: float, specificity_level: float) -> float:
        """Calculate overall discovery factor from prompt analysis"""
        # More specific prompts = less discovery, more open prompts = more discovery
        return exploration_openness * (1.0 - specificity_level * 0.5)
    
    def _estimate_track_familiarity(self, track: TrackRecommendation) -> float:
        """Estimate how familiar/mainstream a track is"""
        # Use novelty score if available
        if track.novelty_score is not None:
            return 1.0 - track.novelty_score
        
        # Fallback: estimate based on genre and additional scores
        mainstream_genres = ["pop", "rock", "hip hop", "country"]
        underground_genres = ["experimental", "ambient", "post-rock", "drone"]
        
        if track.genres:
            track_genres_lower = [g.lower() for g in track.genres]
            
            if any(g in mainstream_genres for g in track_genres_lower):
                return 0.7
            elif any(g in underground_genres for g in track_genres_lower):
                return 0.2
        
        return 0.5  # Neutral if unknown
    
    def _calculate_genre_expansion(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any]) -> float:
        """Calculate how much the track expands beyond stated genre preferences"""
        stated_genres = prompt_analysis.get("genre_preferences", [])
        
        if not stated_genres or not track.genres:
            return 0.5
        
        # Check for exact matches
        track_genres_lower = [g.lower() for g in track.genres]
        stated_genres_lower = [g.lower() for g in stated_genres]
        
        exact_matches = sum(1 for g in track_genres_lower if g in stated_genres_lower)
        
        if exact_matches > 0:
            return 0.3  # Low expansion - matches stated preferences
        else:
            return 0.8  # High expansion - explores new genres


class ConversationalExplainer:
    """Generates prompt-based explanations for recommendations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_prompt_based_explanation(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any], ranking_factors: Dict[str, float]) -> str:
        """
        Generate explanation that references the original prompt
        
        Args:
            track: Recommended track
            prompt_analysis: Analysis of user prompt
            ranking_factors: Factors that contributed to ranking
            
        Returns:
            Human-readable explanation
        """
        explanation_parts = []
        
        # Reference the original request with artist context
        llm_intent = prompt_analysis.get("llm_intent")
        llm_artists = prompt_analysis.get("llm_artists", [])
        
        if llm_intent == "artist_similarity" and llm_artists:
            target_artist = llm_artists[0]
            if track.artist.lower() == target_artist.lower():
                explanation_parts.append(f"Direct match - this is a track by {target_artist} themselves. ")
            else:
                explanation_parts.append(f"Similar to {target_artist} - ")
        else:
            primary_intent = prompt_analysis.get("primary_intent")
            explanation_parts.append(f"Perfect for your {primary_intent} request - ")
        
        # Explain top ranking factors with specific scores
        top_factors = sorted(ranking_factors.items(), key=lambda x: x[1], reverse=True)[:2]
        
        factor_explanations = []
        for factor, score in top_factors:
            explanation = self._explain_ranking_factor(factor, score, track)
            factor_explanations.append(explanation)
        
        explanation_parts.append(" and ".join(factor_explanations))
        
        # Add source context
        if track.advocate_source_agent == "GenreMoodAgent":
            explanation_parts.append(". Found through genre/mood matching")
        elif track.advocate_source_agent == "DiscoveryAgent":
            if (track.additional_scores and 
                track.additional_scores.get("source") == "multi_hop_similarity"):
                explanation_parts.append(". Discovered through artist similarity network")
            else:
                explanation_parts.append(". Found through discovery exploration")
        
        return "".join(explanation_parts) + "."
    
    def _explain_ranking_factor(self, factor: str, score: float, track: TrackRecommendation) -> str:
        """Explain a specific ranking factor"""
        if factor == "intent_alignment":
            if score > 0.8:
                return f"excellent artist similarity match ({score:.0%})"
            elif score > 0.6:
                return f"good stylistic alignment ({score:.0%})"
            else:
                return f"moderate relevance ({score:.0%})"
        
        elif factor == "contextual_relevance":
            if score > 0.7:
                return f"perfect contextual fit ({score:.0%})"
            elif score > 0.5:
                return f"good contextual match ({score:.0%})"
            else:
                return f"basic contextual relevance ({score:.0%})"
        
        elif factor == "discovery_appropriateness":
            if score > 0.7:
                return f"ideal discovery balance ({score:.0%})"
            elif score > 0.5:
                return f"good exploration level ({score:.0%})"
            else:
                return f"conservative discovery ({score:.0%})"
        
        elif factor == "quality_score":
            if score > 0.8:
                return f"exceptional quality ({score:.0%})"
            elif score > 0.6:
                return f"high quality ({score:.0%})"
            else:
                return f"decent quality ({score:.0%})"
        
        elif factor == "conversational_fit":
            return f"conversational relevance ({score:.0%})"
        
        else:
            return f"{factor.replace('_', ' ')} ({score:.0%})"


class EnhancedJudgeAgent:
    """
    Enhanced JudgeAgent with prompt-driven ranking capabilities.
    Incorporates advanced ranking methodologies inspired by industry leaders
    while focusing on conversational, prompt-driven music discovery.
    """
    
    def __init__(self, llm_client: Any = None):
        """
        Initialize the Enhanced JudgeAgent with prompt analysis capabilities.

        Args:
            llm_client (Any, optional): An optional LLM client for generating
                                       more nuanced explanations. Defaults to None.
        """
        self.logger = logging.getLogger(__name__)
        self.llm_client = llm_client
        
        # Initialize prompt-driven components
        self.prompt_analyzer = PromptAnalysisEngine()
        self.contextual_scorer = ContextualRelevanceScorer()
        self.discovery_scorer = DiscoveryAppropriatenessScorer()
        self.explainer = ConversationalExplainer()
        
        # Enhanced ranking weights
        self.ranking_weights = {
            "intent_alignment": 0.40,      # How well it matches user intent
            "contextual_relevance": 0.25,  # Activity, mood, temporal fit
            "discovery_appropriateness": 0.20,  # Exploration vs familiarity balance
            "quality_score": 0.10,         # Basic quality metrics
            "conversational_fit": 0.05     # Conversational appropriateness
        }
        
        self.logger.info("Enhanced JudgeAgent initialized with prompt-driven capabilities.")

    async def evaluate_and_select(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Enhanced evaluation and selection process with prompt-driven ranking.

        Args:
            state (MusicRecommenderState): The shared state object containing candidate 
                                           tracks and the planning strategy.

        Returns:
            MusicRecommenderState: The updated state with final recommendations.
        """
        self.logger.info("Enhanced JudgeAgent: Starting prompt-driven evaluation and selection.")
        
        # Consolidate all candidate tracks
        all_candidate_dicts: List[Dict] = []
        all_candidate_dicts.extend(state.genre_mood_recommendations)
        all_candidate_dicts.extend(state.discovery_recommendations)

        if not all_candidate_dicts:
            self.logger.warning("Enhanced JudgeAgent: No candidate tracks provided for evaluation.")
            state.final_recommendations = []
            state.reasoning_log.append("Enhanced JudgeAgent: No candidate tracks provided for evaluation.")
            return state

        # Use Pure LLM Query Understanding if available, otherwise fallback to prompt analysis
        if state.query_understanding:
            # Convert Pure LLM Query Understanding to prompt analysis format
            prompt_analysis = self._convert_query_understanding_to_prompt_analysis(state.query_understanding)
            self.logger.info(f"Using Pure LLM Query Understanding: {prompt_analysis}")
            
            # Fix dictionary access for artists
            artists = state.query_understanding.get("artists", []) if isinstance(state.query_understanding, dict) else (state.query_understanding.artists if hasattr(state.query_understanding, 'artists') else [])
            state.reasoning_log.append(f"Enhanced JudgeAgent: Using Pure LLM understanding - Intent: {prompt_analysis.get('primary_intent')}, Artists: {artists}")
        else:
            # Fallback to basic prompt analysis
            prompt_analysis = self.prompt_analyzer.analyze_prompt(
                state.user_query, 
                state.conversation_context
            )
            self.logger.info(f"Fallback prompt analysis completed: {prompt_analysis}")
            state.reasoning_log.append(f"Enhanced JudgeAgent: Fallback analysis - Intent: {prompt_analysis.get('primary_intent')}, Activity: {prompt_analysis.get('activity_context')}")

        # Parse candidates into TrackRecommendation objects
        parsed_candidates = self._parse_candidates(all_candidate_dicts)
        
        if not parsed_candidates:
            self.logger.warning("Enhanced JudgeAgent: No candidates were successfully parsed.")
            state.final_recommendations = []
            state.reasoning_log.append("Enhanced JudgeAgent: No candidates were successfully parsed.")
            return state

        # Apply enhanced prompt-driven ranking
        ranked_candidates = await self._apply_prompt_driven_ranking(parsed_candidates, prompt_analysis)
        
        # Select top candidates with diversity consideration
        evaluation_framework = state.planning_strategy.get("evaluation_framework", {})
        diversity_targets = evaluation_framework.get("diversity_targets", {})
        
        selected_tracks = self._select_with_diversity(
            ranked_candidates, 
            diversity_targets,
            num_recommendations=20  # Increased from 3 to 20 as per design
        )
        
        # Generate enhanced explanations
        final_selections = await self._generate_enhanced_explanations(
            selected_tracks, 
            prompt_analysis
        )
        
        # Convert to dictionaries for state compatibility
        state.final_recommendations = [track.model_dump() for track in final_selections]
        
        log_message = f"Enhanced JudgeAgent: Selected {len(final_selections)} tracks using prompt-driven ranking."
        self.logger.info(log_message)
        state.reasoning_log.append(log_message)
        
        return state

    def _convert_query_understanding_to_prompt_analysis(self, query_understanding) -> Dict[str, Any]:
        """
        Convert Pure LLM Query Understanding to prompt analysis format

        Args:
            query_understanding: QueryUnderstanding object from Pure LLM system

        Returns:
            Dictionary in prompt analysis format for compatibility
        """
        # Map LLM intents to JudgeAgent intents
        intent_mapping = {
            "artist_similarity": "discovery",  # Artist similarity is a form of discovery
            "genre_exploration": "discovery",
            "mood_matching": "mood_enhancement",
            "activity_context": "concentration",  # Default activity mapping
            "discovery": "discovery",
            "playlist_building": "discovery",
            "specific_request": "discovery"
        }
        
        # Map activities from moods/activities
        activity_mapping = {
            "workout": "workout",
            "exercise": "workout", 
            "study": "study",
            "studying": "study",
            "work": "work",
            "coding": "coding",
            "focus": "coding",
            "relax": "relaxation",
            "chill": "relaxation"
        }
        
        # Handle both dict and object access for query_understanding
        if isinstance(query_understanding, dict):
            intent_value = query_understanding.get("intent", {}).get("value", "discovery")
            artists = query_understanding.get("artists", [])
            genres = query_understanding.get("genres", [])
            moods = query_understanding.get("moods", [])
            activities = query_understanding.get("activities", [])
            confidence = query_understanding.get("confidence", 0.5)
            exploration_level = query_understanding.get("exploration_level", "moderate")
            energy_level = query_understanding.get("energy_level")
            temporal_context = query_understanding.get("temporal_context")
            similarity_type = query_understanding.get("similarity_type", {}).get("value") if query_understanding.get("similarity_type") else None
        else:
            # Object access for backward compatibility
            intent_value = query_understanding.intent.value if hasattr(query_understanding, 'intent') else "discovery"
            artists = query_understanding.artists if hasattr(query_understanding, 'artists') else []
            genres = query_understanding.genres if hasattr(query_understanding, 'genres') else []
            moods = query_understanding.moods if hasattr(query_understanding, 'moods') else []
            activities = query_understanding.activities if hasattr(query_understanding, 'activities') else []
            confidence = query_understanding.confidence if hasattr(query_understanding, 'confidence') else 0.5
            exploration_level = query_understanding.exploration_level if hasattr(query_understanding, 'exploration_level') else "moderate"
            energy_level = query_understanding.energy_level if hasattr(query_understanding, 'energy_level') else None
            temporal_context = query_understanding.temporal_context if hasattr(query_understanding, 'temporal_context') else None
            similarity_type = query_understanding.similarity_type.value if hasattr(query_understanding, 'similarity_type') and query_understanding.similarity_type else None
        
        # Extract activity from understanding
        activity_context = None
        for activity in activities:
            if activity.lower() in activity_mapping:
                activity_context = activity_mapping[activity.lower()]
                break
        
        # For artist similarity, we want high specificity and focused discovery
        if intent_value == "artist_similarity":
            exploration_openness = 0.3  # Lower for focused similarity
            specificity_level = 0.9     # High for specific artist request
            primary_intent = "discovery"  # But treat as targeted discovery
        else:
            exploration_openness = 0.5 if exploration_level == "moderate" else (0.8 if exploration_level == "broad" else 0.2)
            specificity_level = 0.9 if artists else 0.3
            primary_intent = intent_mapping.get(intent_value, "discovery")
        
        # Convert to prompt analysis format
        prompt_analysis = {
            "primary_intent": primary_intent,
            "activity_context": activity_context,
            "mood_request": moods,
            "genre_preferences": genres,
            "exploration_openness": exploration_openness,
            "temporal_context": temporal_context,
            "specificity_level": specificity_level,
            "energy_level": energy_level or "medium",
            "conversation_continuity": {
                "is_follow_up": False,
                "references_previous": False,
                "builds_on_context": False
            },
            # Add LLM-specific fields for enhanced scoring
            "llm_intent": intent_value,
            "llm_artists": artists,
            "llm_confidence": confidence,
            "similarity_type": similarity_type
        }
        
        return prompt_analysis

    def _parse_candidates(self, candidates: List[Dict]) -> List[TrackRecommendation]:
        """Parse candidate dictionaries into TrackRecommendation objects"""
        parsed_candidates = []
        
        for track_dict in candidates:
            try:
                track_model = TrackRecommendation(**track_dict)
                parsed_candidates.append(track_model)
            except ValidationError as e:
                self.logger.warning(
                    f"Failed to parse track data: {track_dict.get('title', 'N/A')}. Error: {e}"
                )
                continue
        
        return parsed_candidates

    async def _apply_prompt_driven_ranking(self, candidates: List[TrackRecommendation], prompt_analysis: Dict[str, Any]) -> List[Tuple[TrackRecommendation, float, Dict[str, float]]]:
        """
        Apply enhanced prompt-driven ranking to candidates
        
        Args:
            candidates: List of candidate tracks
            prompt_analysis: Analysis of user prompt
            
        Returns:
            List of tuples (track, final_score, factor_breakdown)
        """
        ranked_candidates = []
        
        for track in candidates:
            # Calculate individual ranking factors
            intent_score = self._calculate_intent_alignment(track, prompt_analysis)
            contextual_score = self.contextual_scorer.calculate_contextual_relevance(track, prompt_analysis)
            discovery_score = self.discovery_scorer.calculate_discovery_score(track, prompt_analysis)
            quality_score = self._calculate_quality_score(track)
            conversational_score = self._calculate_conversational_fit(track, prompt_analysis)
            
            # Calculate weighted final score
            factor_scores = {
                "intent_alignment": intent_score,
                "contextual_relevance": contextual_score,
                "discovery_appropriateness": discovery_score,
                "quality_score": quality_score,
                "conversational_fit": conversational_score
            }
            
            final_score = sum(
                score * self.ranking_weights[factor] 
                for factor, score in factor_scores.items()
            )
            
            # Store the score in the track object
            track.judge_score = final_score
            
            ranked_candidates.append((track, final_score, factor_scores))
            
            self.logger.debug(
                f"Track '{track.title}' scored {final_score:.3f} - "
                f"Intent: {intent_score:.2f}, Context: {contextual_score:.2f}, "
                f"Discovery: {discovery_score:.2f}, Quality: {quality_score:.2f}"
            )
        
        # Sort by final score
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        return ranked_candidates

    def _calculate_intent_alignment(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any]) -> float:
        """Calculate how well the track aligns with user intent"""
        primary_intent = prompt_analysis.get("primary_intent")
        llm_intent = prompt_analysis.get("llm_intent")
        
        if not primary_intent:
            return 0.5
        
        # Special handling for artist similarity from Pure LLM
        if llm_intent == "artist_similarity":
            return self._score_artist_similarity_intent(track, prompt_analysis)
        
        # Intent-specific scoring logic
        intent_scorers = {
            "concentration": self._score_concentration_intent,
            "energy": self._score_energy_intent,
            "relaxation": self._score_relaxation_intent,
            "discovery": self._score_discovery_intent,
            "mood_enhancement": self._score_mood_intent,
            "background": self._score_background_intent
        }
        
        scorer = intent_scorers.get(primary_intent, lambda t, p: 0.5)
        return scorer(track, prompt_analysis)
    
    def _score_artist_similarity_intent(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any]) -> float:
        """Score track for artist similarity intent using data-driven approach"""
        llm_artists = prompt_analysis.get("llm_artists", [])
        
        if not llm_artists:
            return 0.5  # Neutral score if no target artists
        
        # 1. SOURCE-BASED SCORING (40% weight) - Prioritize by how track was found
        source_score = self._calculate_source_priority_score(track)
        
        # 2. ARTIST RELATIONSHIP SCORING (30% weight) - Use actual similarity data
        relationship_score = self._calculate_artist_relationship_score(track, llm_artists)
        
        # 3. MUSICAL FEATURE SIMILARITY (20% weight) - Genre and style matching
        feature_score = self._calculate_musical_feature_similarity(track, llm_artists)
        
        # 4. DISCOVERY QUALITY (10% weight) - Balance familiarity vs exploration
        discovery_score = self._calculate_discovery_quality_score(track)
        
        # Weighted final score
        final_score = (
            source_score * 0.4 +
            relationship_score * 0.3 +
            feature_score * 0.2 +
            discovery_score * 0.1
        )
        
        # Apply penalties for obviously unrelated tracks
        final_score = self._apply_similarity_penalties(track, final_score)
        
        return min(1.0, max(0.1, final_score))
    
    def _calculate_source_priority_score(self, track: TrackRecommendation) -> float:
        """Score based on how the track was discovered"""
        # Use the correct field for source information
        source_agent = track.advocate_source_agent
        
        # For artist similarity queries, prioritize GenreMoodAgent tracks
        # as they're more likely to contain similar artists
        if source_agent == "GenreMoodAgent":
            return 0.9  # High priority for artist similarity
        elif source_agent == "DiscoveryAgent":
            # Check if it's from multi-hop similarity (good for artist similarity)
            if (track.additional_scores and 
                track.additional_scores.get("source") == "multi_hop_similarity"):
                return 0.8
            # Check if it's underground detection (less relevant)
            elif (track.additional_scores and 
                  track.additional_scores.get("source") == "underground_detection"):
                return 0.4
            else:
                return 0.6  # Generic discovery
        else:
            # Fallback: Use track metadata to infer source quality
            if track.novelty_score and track.novelty_score > 0.7:
                return 0.6  # Likely from discovery agent
            else:
                return 0.5  # Generic track
    
    def _calculate_artist_relationship_score(self, track: TrackRecommendation, target_artists: List[str]) -> float:
        """Score based on actual artist relationships"""
        # Exact artist match (highest score)
        if track.artist in target_artists:
            return 1.0
        
        # Check for artist name similarity (fuzzy matching)
        for target in target_artists:
            similarity = self._calculate_artist_name_similarity(track.artist, target)
            if similarity > 0.8:
                return 0.9  # Very similar name
            elif similarity > 0.6:
                return 0.7  # Somewhat similar name
        
        # For GenreMoodAgent tracks, give higher base score as they're more
        # likely to be contextually similar even without direct name matches
        if track.advocate_source_agent == "GenreMoodAgent":
            return 0.6  # Higher base score for genre/mood matches
        
        # For multi-hop similarity tracks, give moderate score
        if (track.additional_scores and 
            track.additional_scores.get("source") == "multi_hop_similarity"):
            return 0.5  # Moderate score for algorithmic similarity
        
        # Check if track has rich metadata suggesting it's a real musical track
        # (not ambient noise, white noise, etc.)
        if self._has_genuine_similarity_indicators(track):
            return 0.4  # Some potential for similarity
        
        return 0.2  # Low relationship score
    
    def _calculate_musical_feature_similarity(self, track: TrackRecommendation, target_artists: List[str]) -> float:
        """Score based on musical features and style similarity"""
        score = 0.5  # Base score
        
        # Genre-based similarity (data-driven, not hardcoded)
        if track.genres:
            # Look for contemporary/alternative genres that suggest similar aesthetic
            contemporary_indicators = ["indie", "experimental", "electronic", "alternative", "art", "bedroom"]
            genre_matches = sum(1 for genre in track.genres 
                              for indicator in contemporary_indicators 
                              if indicator in genre.lower())
            
            if genre_matches > 0:
                score += min(0.3, genre_matches * 0.1)
        
        # Use quality scores as proxy for production style
        if track.quality_score and track.quality_score > 0.7:
            score += 0.1  # High quality suggests professional production
        
        # Instrumental vs vocal considerations
        if track.instrumental is not None:
            # For artist similarity, slight preference for vocal tracks (most artists have vocals)
            if not track.instrumental:
                score += 0.05
        
        return min(1.0, score)
    
    def _calculate_discovery_quality_score(self, track: TrackRecommendation) -> float:
        """Score based on discovery appropriateness"""
        if track.novelty_score is not None:
            # For artist similarity, we want moderate novelty (not too obscure, not too mainstream)
            if 0.4 <= track.novelty_score <= 0.8:
                return 0.8  # Sweet spot for discovery
            elif track.novelty_score > 0.8:
                return 0.6  # Might be too obscure
            else:
                return 0.4  # Might be too mainstream
        
        return 0.5  # Neutral if no novelty information
    
    def _apply_similarity_penalties(self, track: TrackRecommendation, base_score: float) -> float:
        """Apply penalties for obviously unrelated tracks"""
        penalty = 0.0
        
        # Heavy penalty for ambient/sleep/meditation tracks that aren't real music
        non_music_phrases = [
            "white noise", "sleep aid", "lullaby", "meditation", "nature sounds",
            "ambient eclipse", "experimental dental", "lo-fi beats"
        ]
        if track.title and any(phrase in track.title.lower() 
                              for phrase in non_music_phrases):
            penalty += 0.6
        
        # Penalty for artists that are clearly not musical artists
        non_artist_phrases = [
            "ambient eclipse", "experimental dental school", 
            "experimental products", "lo-fi beats", "white noise", 
            "sleep", "meditation"
        ]
        if track.artist and any(phrase in track.artist.lower() 
                               for phrase in non_artist_phrases):
            penalty += 0.5
        
        # Penalize tracks from different eras/styles for indie/experimental queries
        if track.genres:
            # Penalize genres very different from contemporary indie/experimental
            incompatible_genres = [
                "country", "classical", "opera", "folk", "bluegrass", "gospel"
            ]
            if any(genre.lower() in incompatible_genres 
                   for genre in track.genres):
                penalty += 0.3
        
        # Penalize obvious keyword-only matches without substance
        misleading_keywords = ["underground", "experimental", "indie"]
        if (track.title and 
            any(keyword in track.title.lower() 
                for keyword in misleading_keywords) and
            not self._has_genuine_similarity_indicators(track)):
            penalty += 0.3
        
        # Boost GenreMoodAgent tracks (reduce penalty)
        if track.advocate_source_agent == "GenreMoodAgent":
            penalty = max(0.0, penalty - 0.2)
        
        return max(0.1, base_score - penalty)
    
    def _has_genuine_similarity_indicators(self, track: TrackRecommendation) -> bool:
        """Check if track has genuine similarity indicators beyond keywords"""
        # Check if track has rich metadata suggesting it's a real musical track
        indicators = 0
        
        if track.genres and len(track.genres) > 1:
            indicators += 1
        if track.moods and len(track.moods) > 0:
            indicators += 1
        if track.quality_score and track.quality_score > 0.5:
            indicators += 1
        if track.novelty_score and track.novelty_score > 0.3:
            indicators += 1
        
        return indicators >= 2
    
    def _calculate_artist_name_similarity(self, artist1: str, artist2: str) -> float:
        """Calculate similarity between artist names (simple fuzzy matching)"""
        if not artist1 or not artist2:
            return 0.0
        
        artist1_lower = artist1.lower().strip()
        artist2_lower = artist2.lower().strip()
        
        if artist1_lower == artist2_lower:
            return 1.0
        
        # Simple substring matching
        if artist1_lower in artist2_lower or artist2_lower in artist1_lower:
            return 0.8
        
        # Check for common words
        words1 = set(artist1_lower.split())
        words2 = set(artist2_lower.split())
        
        if words1 & words2:  # Intersection
            return 0.6
        
        return 0.0
    
    def _score_concentration_intent(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any]) -> float:
        """Score track for concentration/focus intent"""
        score = 0.5
        
        # Prefer instrumental tracks
        if track.instrumental:
            score += 0.3
        
        # Check concentration-friendly score if available
        if track.concentration_friendliness_score is not None:
            score = track.concentration_friendliness_score
        
        # Prefer certain genres
        focus_genres = ["ambient", "post-rock", "instrumental", "classical", "electronic"]
        if track.genres:
            genre_match = any(g.lower() in focus_genres for g in track.genres)
            if genre_match:
                score += 0.2
        
        return min(1.0, score)
    
    def _score_energy_intent(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any]) -> float:
        """Score track for energy/workout intent"""
        score = 0.5
        
        # Check energy level
        energy_level = track.additional_scores.get("energy", 0.5)
        if energy_level > 0.7:
            score += 0.4
        
        # Prefer high-energy genres
        energy_genres = ["electronic", "rock", "metal", "hip hop", "dance"]
        if track.genres:
            genre_match = any(g.lower() in energy_genres for g in track.genres)
            if genre_match:
                score += 0.3
        
        return min(1.0, score)
    
    def _score_relaxation_intent(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any]) -> float:
        """Score track for relaxation intent"""
        score = 0.5
        
        # Check energy level (prefer lower)
        energy_level = track.additional_scores.get("energy", 0.5)
        if energy_level < 0.4:
            score += 0.4
        
        # Prefer relaxing genres
        relax_genres = ["ambient", "folk", "jazz", "classical", "chillout"]
        if track.genres:
            genre_match = any(g.lower() in relax_genres for g in track.genres)
            if genre_match:
                score += 0.3
        
        return min(1.0, score)
    
    def _score_discovery_intent(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any]) -> float:
        """Score track for discovery intent"""
        # Use novelty score if available
        if track.novelty_score is not None:
            return track.novelty_score
        
        # Fallback scoring
        score = 0.5
        
        # Prefer underground/experimental genres
        discovery_genres = ["experimental", "indie", "underground", "alternative"]
        if track.genres:
            genre_match = any(g.lower() in discovery_genres for g in track.genres)
            if genre_match:
                score += 0.3
        
        return min(1.0, score)
    
    def _score_mood_intent(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any]) -> float:
        """Score track for mood enhancement intent"""
        mood_requests = prompt_analysis.get("mood_request", [])
        
        if not mood_requests:
            return 0.5
        
        # Check mood alignment
        if track.moods:
            mood_matches = sum(1 for mood in mood_requests 
                             if any(m.lower() == mood.lower() for m in track.moods))
            return min(1.0, mood_matches / len(mood_requests) + 0.3)
        
        return 0.4
    
    def _score_background_intent(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any]) -> float:
        """Score track for background music intent"""
        score = 0.5
        
        # Prefer non-distracting characteristics
        if track.instrumental:
            score += 0.2
        
        # Moderate energy levels work best for background
        energy_level = track.additional_scores.get("energy", 0.5)
        if 0.3 <= energy_level <= 0.7:
            score += 0.3
        
        return min(1.0, score)
    
    def _calculate_quality_score(self, track: TrackRecommendation) -> float:
        """Calculate basic quality score for the track"""
        if track.quality_score is not None:
            return track.quality_score
        
        # Fallback quality estimation
        quality_factors = []
        
        # Check if track has rich metadata
        if track.genres:
            quality_factors.append(0.2)
        if track.moods:
            quality_factors.append(0.2)
        if track.album_title:
            quality_factors.append(0.1)
        
        # Check additional scores
        if track.additional_scores:
            quality_factors.append(0.3)
        
        return sum(quality_factors) if quality_factors else 0.5
    
    def _calculate_conversational_fit(self, track: TrackRecommendation, prompt_analysis: Dict[str, Any]) -> float:
        """Calculate how well the track fits the conversational context"""
        score = 0.5
        
        # Check conversation continuity
        continuity = prompt_analysis.get("conversation_continuity", {})
        
        if continuity.get("builds_on_context"):
            # This is a follow-up request, prefer tracks that complement previous selections
            score += 0.3
        
        # Check specificity alignment
        specificity = prompt_analysis.get("specificity_level", 0.5)
        if specificity > 0.7:
            # Very specific request - prefer exact matches
            score += 0.2 if track.genres else 0.0
        elif specificity < 0.3:
            # Open request - prefer diverse options
            score += 0.2
        
        return min(1.0, score)
    
    def _select_with_diversity(self, ranked_candidates: List[Tuple[TrackRecommendation, float, Dict[str, float]]], diversity_targets: Dict[str, Any], num_recommendations: int = 20) -> List[TrackRecommendation]:
        """Select tracks with diversity consideration"""
        if not ranked_candidates:
            return []
        
        # Extract just the tracks for diversity selection
        tracks_only = [candidate[0] for candidate in ranked_candidates]
        
        # Use existing diversity logic but with enhanced candidates
        return self._ensure_diversity(tracks_only, diversity_targets, num_recommendations)
    
    async def _generate_enhanced_explanations(self, selections: List[TrackRecommendation], prompt_analysis: Dict[str, Any]) -> List[TrackRecommendation]:
        """Generate enhanced explanations that reference the prompt"""
        for track in selections:
            # Get ranking factors for this track
            ranking_factors = {
                "intent_alignment": self._calculate_intent_alignment(track, prompt_analysis),
                "contextual_relevance": self.contextual_scorer.calculate_contextual_relevance(track, prompt_analysis),
                "discovery_appropriateness": self.discovery_scorer.calculate_discovery_score(track, prompt_analysis),
                "quality_score": self._calculate_quality_score(track),
                "conversational_fit": self._calculate_conversational_fit(track, prompt_analysis)
            }
            
            # Generate prompt-based explanation
            explanation = self.explainer.generate_prompt_based_explanation(
                track, prompt_analysis, ranking_factors
            )
            
            track.explanation = explanation
        
        return selections

    # Keep the existing diversity method from the original implementation
    def _ensure_diversity(self, candidates: List[TrackRecommendation], diversity_targets: Dict[str, int], num_recommendations: int = 20) -> List[TrackRecommendation]:
        """
        Selects tracks to meet diversity targets (e.g., number of unique genres, eras).
        Prioritizes higher-scoring tracks (candidates are expected to be pre-sorted by `judge_score`).

        Args:
            candidates (List[TrackRecommendation]): Pre-sorted list of candidate tracks with 'judge_score'.
            diversity_targets (Dict[str, int]): Dictionary of diversity criteria (e.g., {"genres": 2, "era": 1}).
                                                 The value indicates the target number of unique items for that key.
            num_recommendations (int): The desired number of final recommendations.

        Returns:
            List[TrackRecommendation]: A list of selected tracks meeting diversity goals as best as possible.
        """
        self.logger.debug(f"Ensuring diversity for {num_recommendations} recommendations with targets: {diversity_targets} from {len(candidates)} candidates.")
        
        # Return early for empty cases or no diversity targets
        if not candidates:
            return []
            
        # If no diversity targets or no attributes, just return top N candidates by score
        if not diversity_targets or not diversity_targets.get("attributes"):
            self.logger.debug("No diversity targets or attributes. Returning top N candidates based on score.")
            sorted_candidates = sorted(candidates, key=lambda x: x.judge_score if x.judge_score is not None else 0.0, reverse=True)
            return sorted_candidates[:num_recommendations]

        # Make sure candidates are sorted by score in descending order
        sorted_candidates = sorted(candidates, key=lambda x: x.judge_score if x.judge_score is not None else 0.0, reverse=True)
        selected_tracks: List[TrackRecommendation] = []
        
        # Tracks unique values seen for each diversity dimension, e.g., {"genres": {"Rock", "Pop"}}
        met_diversity_values: Dict[str, set] = {}
        
        # Pass 1: Greedily select tracks that fulfill a new diversity criterion from highest scores
        for attribute in diversity_targets.get("attributes", []):
            if len(selected_tracks) >= num_recommendations:
                break
                
            # Initialize set for this attribute if not already present
            if attribute not in met_diversity_values:
                met_diversity_values[attribute] = set()
                
            # Go through each candidate for this attribute
            for track in sorted_candidates:
                if len(selected_tracks) >= num_recommendations:
                    break
                    
                if track in selected_tracks:
                    continue
                    
                # Track value for this diversity attribute
                track_value_for_key: Any = None
                if hasattr(track, attribute):
                    track_value_for_key = getattr(track, attribute)
                
                if track_value_for_key is not None:
                    # Handle list-based attributes (like genres) and single attributes (like era)
                    values_to_check = track_value_for_key if isinstance(track_value_for_key, list) else [track_value_for_key]
                    
                    # Check if any value is new for this attribute
                    for value in values_to_check:
                        if value and value not in met_diversity_values[attribute]:
                            selected_tracks.append(track)
                            # Add all values from this track to the seen set
                            for v in values_to_check:
                                if v:  # Only add non-empty values
                                    met_diversity_values[attribute].add(v)
                            break
        
        self.logger.debug(f"After Pass 1 (greedy diversity pick): {len(selected_tracks)} selected. Met diversity: { {k: len(v) for k,v in met_diversity_values.items()} }")

        # Pass 2: If not enough tracks selected, fill remaining slots with highest-scored available tracks
        remaining_candidates = [c for c in sorted_candidates if c not in selected_tracks]
        while len(selected_tracks) < num_recommendations and remaining_candidates:
            track_to_add = remaining_candidates.pop(0)  # Take highest scored remaining track
            selected_tracks.append(track_to_add)
            self.logger.debug(f"Pass 2: Adding track '{track_to_add.title}' (score: {track_to_add.judge_score}) to fill slots.")
            
        # Final sort by score to ensure order is maintained
        final_selected = sorted(selected_tracks, key=lambda x: x.judge_score if x.judge_score is not None else 0.0, reverse=True)
        final_selected_count = len(final_selected)
        
        self.logger.info(f"Enhanced JudgeAgent: Selected {final_selected_count} diverse tracks. Met diversity values: {met_diversity_values}")
        return final_selected[:num_recommendations]


# Maintain backward compatibility by aliasing the enhanced version
JudgeAgent = EnhancedJudgeAgent 