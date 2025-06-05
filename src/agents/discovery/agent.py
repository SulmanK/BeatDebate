"""
Simplified Discovery Agent

Refactored to use dependency injection and shared components, eliminating:
- Client instantiation duplication
- Candidate generation duplication
- LLM calling duplication
- Underground detection duplication
"""

from typing import Dict, List, Any
import structlog

from ...models.agent_models import MusicRecommenderState, AgentConfig
from ...models.recommendation_models import TrackRecommendation
from ...services.api_service import APIService
from ...services.metadata_service import MetadataService
from ..base_agent import BaseAgent
from ..components.unified_candidate_generator import UnifiedCandidateGenerator
from ..components import QualityScorer

logger = structlog.get_logger(__name__)


class DiscoveryAgent(BaseAgent):
    """
    Simplified Discovery Agent with dependency injection.
    
    Responsibilities:
    - Multi-hop similarity exploration
    - Underground and hidden gem detection
    - Serendipitous discovery beyond mainstream music
    - Novelty-optimized recommendations
    
    Uses shared components to eliminate duplication:
    - UnifiedCandidateGenerator for discovery-focused candidate generation
    - QualityScorer for quality assessment
    - LLMUtils for LLM interactions
    - APIService for unified API access
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client,
        api_service: APIService,
        metadata_service: MetadataService,
        rate_limiter=None,
        session_manager=None
    ):
        """
        Initialize simplified discovery agent with injected dependencies.
        
        Args:
            config: Agent configuration
            llm_client: LLM client for reasoning
            api_service: Unified API service
            metadata_service: Unified metadata service
            rate_limiter: Rate limiter for LLM API calls
            session_manager: SessionManagerService for candidate pool persistence (Phase 3)
        """
        super().__init__(
            config=config, 
            llm_client=llm_client, 
            api_service=api_service,
            metadata_service=metadata_service,
            rate_limiter=rate_limiter
        )
        
        # Shared components (LLMUtils now initialized in parent with rate limiter)
        # Phase 3: Pass session_manager to UnifiedCandidateGenerator for candidate pool persistence
        self.candidate_generator = UnifiedCandidateGenerator(api_service, session_manager)
        self.quality_scorer = QualityScorer()
        
        # Base configuration - will be adapted based on intent
        self.target_candidates = 100
        self.final_recommendations = 20
        self.quality_threshold = 0.3   # Base threshold
        self.novelty_threshold = 0.4   # Base threshold
        
        # Discovery parameters - will be adapted based on intent
        self.underground_bias = 0.7
        self.similarity_depth = 2  # Multi-hop depth
        
        # Intent-specific parameter configurations from design document
        self.intent_parameters = {
            'by_artist': {
                'novelty_threshold': 0.0,     # 🔧 NEW: No novelty filtering for by_artist
                'quality_threshold': 0.25,    # Moderate quality for artist discography
                'underground_bias': 0.0,      # No underground bias for by_artist
                'similarity_depth': 1,        # Simple matching for target artist
                'max_per_artist': 20,         # Allow many tracks from target artist
                'candidate_focus': 'target_artist_discography'
            },
            'by_artist_underground': {
                'novelty_threshold': 0.6,     # 🔧 NEW: High novelty filtering for underground tracks
                'quality_threshold': 0.15,    # Lower quality threshold for underground gems
                'underground_bias': 0.8,      # Strong underground bias
                'similarity_depth': 1,        # Simple matching for target artist
                'max_per_artist': 20,         # Allow many tracks from target artist
                'candidate_focus': 'target_artist_deep_cuts'
            },
            'artist_similarity': {
                'novelty_threshold': 0.15,    # Very relaxed for similar artists
                'quality_threshold': 0.2,     # Lower quality for broader similarity search
                'underground_bias': 0.1,      # Slight underground preference
                'similarity_depth': 2,        # Deep similarity search
                'max_per_artist': 3,          # Allow more tracks per similar artist
                'candidate_focus': 'similar_artists'
            },
            'discovery': {
                'novelty_threshold': 0.6,     # Strict novelty requirement
                'quality_threshold': 0.25,    # Lower quality to find gems
                'underground_bias': 0.8,      # High underground preference
                'similarity_depth': 1,        # Less similarity depth
                'max_per_artist': 1,          # Diversity over repetition
                'candidate_focus': 'underground_gems'
            },
            'genre_mood': {
                'novelty_threshold': 0.3,     # Moderate novelty
                'quality_threshold': 0.35,    # Balanced quality
                'underground_bias': 0.5,      # Balanced underground bias
                'similarity_depth': 2,        # Standard depth
                'max_per_artist': 2,          # Moderate diversity
                'candidate_focus': 'style_match'
            },
            'contextual': {
                'novelty_threshold': 0.2,     # Relaxed for functional music
                'quality_threshold': 0.45,    # Higher quality for context
                'underground_bias': 0.4,      # Less underground for reliability
                'similarity_depth': 1,        # Simple matching
                'max_per_artist': 2,          # Moderate diversity
                'candidate_focus': 'functional_fit'
            },
            'hybrid': {
                'novelty_threshold': 0.25,    # Moderate novelty
                'quality_threshold': 0.35,    # Balanced approach
                'underground_bias': 0.6,      # Moderate underground bias
                'similarity_depth': 2,        # Standard depth
                'max_per_artist': 2,          # Balanced diversity
                'candidate_focus': 'balanced'
            }
        }
        
        self.logger.info("Simplified DiscoveryAgent initialized with intent-aware parameters")
    
    def _adapt_to_intent(self, intent: str) -> None:
        """Adapt agent parameters based on detected intent."""
        if intent in self.intent_parameters:
            params = self.intent_parameters[intent]
            
            self.novelty_threshold = params['novelty_threshold']
            self.quality_threshold = params['quality_threshold']
            self.underground_bias = params['underground_bias']
            self.similarity_depth = params['similarity_depth']
            
            self.logger.info(
                f"Adapted parameters for intent: {intent}",
                novelty_threshold=self.novelty_threshold,
                quality_threshold=self.quality_threshold,
                underground_bias=self.underground_bias,
                similarity_depth=self.similarity_depth
            )
        else:
            self.logger.warning(f"Unknown intent: {intent}, using default parameters")
    
    def _adapt_to_effective_intent(self, effective_intent: Dict[str, Any]) -> None:
        """
        Adapt agent parameters based on effective intent from IntentOrchestrationService.
        
        Phase 2: Simplified parameter adaptation using effective intent.
        """
        intent_str = effective_intent.get('intent', 'discovery')
        
        # Use existing intent parameter mapping
        self._adapt_to_intent(intent_str)
        
        # Additional Phase 2 adaptations based on effective intent metadata
        if effective_intent.get('is_followup'):
            followup_type = effective_intent.get('followup_type')
            if followup_type == 'load_more':
                # For load more requests, be more permissive with quality
                self.quality_threshold = max(0.1, self.quality_threshold - 0.1)
                self.logger.info("Phase 2: Relaxed quality threshold for load_more followup")
            elif followup_type == 'artist_deep_dive':
                # For artist deep dives, focus on target artist
                self.underground_bias = 0.2  # Less underground bias for artist focus
                self.novelty_threshold = 0.1  # More permissive novelty
                self.logger.info("Phase 2: Adapted for artist_deep_dive followup")
        
        # Adjust candidate pool size based on effective intent confidence
        confidence = effective_intent.get('confidence', 0.8)
        if confidence < 0.6:
            # Lower confidence - generate more candidates for better coverage
            self.target_candidates = int(self.target_candidates * 1.3)
            self.logger.info(f"Phase 2: Increased candidate pool to {self.target_candidates} due to low confidence ({confidence})")
        
        self.logger.info(
            "Phase 2: Adapted to effective intent",
            intent=intent_str,
            is_followup=effective_intent.get('is_followup', False),
            followup_type=effective_intent.get('followup_type'),
            confidence=confidence
        )
    
    async def process(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Generate discovery recommendations using shared components.
        
        Phase 2: Simplified to use effective intent from IntentOrchestrationService.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with discovery recommendations
        """
        try:
            self.logger.info("Starting discovery agent processing (Phase 2)")
            
            # Phase 2: Use effective intent if available, otherwise fall back to traditional approach
            if hasattr(state, 'effective_intent') and state.effective_intent:
                self.logger.info("🎯 Phase 2: Using effective intent from IntentOrchestrationService")
                effective_intent = state.effective_intent
                
                # Extract simplified parameters from effective intent
                intent_str = effective_intent.get('intent', 'discovery')
                entities = effective_intent.get('entities', {})
                is_followup = effective_intent.get('is_followup', False)
                
                # Adapt parameters based on effective intent
                self._adapt_to_effective_intent(effective_intent)
                
                # Use effective intent entities for processing
                intent_analysis = {
                    'intent': intent_str,
                    'confidence': effective_intent.get('confidence', 0.8),
                    'is_followup': is_followup,
                    'followup_type': effective_intent.get('followup_type'),
                    'preserves_original_context': effective_intent.get('preserves_original_context', False)
                }
                
                # Extract target artist if this is a follow-up
                target_artist_from_override = None
                if is_followup and 'artists' in entities:
                    artists = entities.get('artists', [])
                    if artists:
                        target_artist_from_override = artists[0] if isinstance(artists, list) else str(artists)
                
                context_override_applied = is_followup
                context_override = effective_intent  # Use effective intent as context
                
            else:
                # Fallback: Use traditional approach for backward compatibility
                self.logger.info("🔧 Phase 2: No effective intent available, using traditional approach")
                entities = state.entities or {}
                intent_analysis = state.intent_analysis or {}
                
                # Traditional context override handling
                context_override_applied = False
                target_artist_from_override = None
                context_override = None
                
                if hasattr(state, 'context_override') and state.context_override:
                    context_override = state.context_override
                    
                    # Handle both dictionary and object formats
                    is_followup = False
                    target_entity = None
                    intent_override = None
                    
                    if isinstance(context_override, dict):
                        is_followup = context_override.get('is_followup', False)
                        target_entity = context_override.get('target_entity')
                        intent_override = context_override.get('intent_override')
                    elif hasattr(context_override, 'is_followup'):
                        is_followup = context_override.is_followup
                        target_entity = getattr(context_override, 'target_entity', None)
                        intent_override = getattr(context_override, 'intent_override', None)
                
                if is_followup and target_entity:
                    target_artist_from_override = target_entity
                    context_override_applied = True
                    
                    self.logger.info(
                        "🚀 Discovery Agent: Context override detected",
                        intent_override=intent_override,
                        target_entity=target_entity,
                        is_followup=is_followup
                    )
                    
                    # Override detected intent for followup queries
                    if intent_override in ['artist_deep_dive', 'artist_similarity', 'hybrid_artist_genre']:
                        if intent_override == 'hybrid_artist_genre':
                            intent_analysis['intent'] = 'by_artist'  # Use by_artist logic but preserve genre filtering
                            self.logger.info(f"🎯 Hybrid artist+genre follow-up: {target_artist_from_override} with preserved genre filters")
                        else:
                            intent_analysis['intent'] = 'artist_similarity'
                            self.logger.info(f"🎯 Overriding intent to 'artist_similarity' for {intent_override}: {target_artist_from_override}")
                    elif intent_override == 'artist_style_refinement':
                        # 🔧 NEW: Artist-style refinement handling
                        intent_analysis['intent'] = 'hybrid'  # Treat as hybrid with artist + style constraints
                        intent_analysis['hybrid_subtype'] = 'artist_style_refinement'
                        
                        # Add style modifier to intent analysis
                        style_modifier = None
                        if isinstance(context_override, dict):
                            style_modifier = context_override.get('style_modifier')
                        elif hasattr(context_override, 'style_modifier'):
                            style_modifier = getattr(context_override, 'style_modifier', None)
                        
                        if style_modifier:
                            intent_analysis['style_modifier'] = style_modifier
                            self.logger.info(f"🎵 Artist-style refinement: {target_artist_from_override} + {style_modifier}")
                        else:
                            self.logger.warning(f"Artist-style refinement detected but no style_modifier found")
                    else:
                        self.logger.info(f"🔧 Unknown intent override: {intent_override}")
                else:
                    self.logger.info(f"🔧 DEBUG: No followup detected - is_followup={is_followup}, target_entity={target_entity}")
                    
                # No context override found
                if not hasattr(state, 'context_override') or not state.context_override:
                    self.logger.info("🔧 DEBUG: No context_override found in state")
            
            # 🔧 Get intent and adapt parameters accordingly
            query_understanding = state.query_understanding
            detected_intent = 'discovery'  # Default for discovery agent
            
            if query_understanding and hasattr(query_understanding, 'intent'):
                # QueryUnderstanding is an object, get intent value
                intent_from_query = query_understanding.intent.value if hasattr(query_understanding.intent, 'value') else str(query_understanding.intent)
                detected_intent = intent_from_query
                
                # Add intent to intent_analysis if missing
                if not intent_analysis.get('intent') and intent_from_query:
                    intent_analysis['intent'] = intent_from_query
                    self.logger.info(f"🔧 FIXED: Added intent '{intent_from_query}' to intent_analysis from query_understanding")
            else:
                self.logger.warning("No query_understanding or intent found in state, using default discovery parameters")
            
            # 🔧 If context override applied, use the override intent
            if context_override_applied and hasattr(context_override, 'intent_override'):
                if context_override.intent_override == 'artist_deep_dive':
                    detected_intent = 'artist_similarity'
                elif context_override.intent_override == 'artist_style_refinement':
                    detected_intent = 'hybrid'
                elif context_override.intent_override == 'hybrid_artist_genre':
                    detected_intent = 'hybrid_artist_genre'  # 🎯 NEW: Use hybrid_artist_genre for candidate generation
            
            # 🔧 CRITICAL FIX: For similarity-primary hybrids, treat as artist_similarity for candidate generation
            candidate_generation_intent = detected_intent
            if detected_intent == 'hybrid':
                # Check if this is a similarity-primary hybrid or artist-style refinement
                reasoning = intent_analysis.get('reasoning', '')
                hybrid_subtype = intent_analysis.get('hybrid_subtype', '')
                if (hybrid_subtype == 'similarity_primary' or 
                    'similarity_primary' in reasoning or
                    'Hybrid sub-type: similarity_primary' in reasoning):
                    candidate_generation_intent = 'artist_similarity'
                    self.logger.info(f"🔧 SIMILARITY-PRIMARY HYBRID: Using 'artist_similarity' for candidate generation instead of 'hybrid'")
                elif hybrid_subtype == 'artist_style_refinement':
                    candidate_generation_intent = 'artist_similarity'  # Find artist tracks first, then filter by style
                    self.logger.info(f"🎵 ARTIST-STYLE REFINEMENT: Using 'artist_similarity' for candidate generation, will filter by style later")
            
            # 🚀 If we have a target artist from context override, inject it into entities
            if target_artist_from_override:
                musical_entities = entities.get('musical_entities', {})
                artists = musical_entities.get('artists', {})
                if 'primary' not in artists:
                    artists['primary'] = []
                
                # Ensure target artist is in primary artists list
                if target_artist_from_override not in artists['primary']:
                    artists['primary'].insert(0, target_artist_from_override)  # Put at front
                    self.logger.info(f"🎯 Injected target artist '{target_artist_from_override}' into entities for candidate generation")
                
                musical_entities['artists'] = artists
                entities['musical_entities'] = musical_entities
            
            # 🚀 PHASE 2: Adapt agent parameters based on detected intent
            self._adapt_to_intent(detected_intent)
            
            # 🔧 ENHANCED CANDIDATE SCALING: Adapt based on query type and expected duplicate rate
            base_target_candidates = self.target_candidates  # Original 100
            recently_shown_count = len(state.recently_shown_track_ids or [])
            
            # 🎯 ENHANCED STATE LOGGING: Log what we received from workflow state
            self.logger.info(f"🎯 DISCOVERY AGENT STATE: recently_shown={recently_shown_count} tracks")
            if state.recently_shown_track_ids:
                sample_ids = state.recently_shown_track_ids[:3]
                self.logger.info(f"🎯 SAMPLE RECENTLY SHOWN: {sample_ids}")
            
            # Scale up for follow-up queries to avoid candidate bottlenecks
            if recently_shown_count > 0:
                # 🚨 AGGRESSIVE SCALING for artist follow-ups (high duplicate probability)
                if candidate_generation_intent in ['artist_similarity', 'by_artist']:
                    # Calculate expected duplicate rate for same-artist queries
                    if target_artist_from_override:
                        # Same artist follow-up - expect high duplicates, scale aggressively
                        duplicate_rate_estimate = min(0.8, recently_shown_count / 15)  # Cap at 80%
                        needed_multiplier = int(1 / (1 - duplicate_rate_estimate)) + 2  # +2 safety margin
                        scaled_target_candidates = max(300, recently_shown_count * needed_multiplier)
                        self.logger.info(f"🚨 HIGH-DUPLICATE ARTIST SCALING: Expected {duplicate_rate_estimate:.1%} duplicates, scaling to {scaled_target_candidates} candidates")
                    else:
                        scaled_target_candidates = max(200, recently_shown_count * 15) 
                        self.logger.info(f"🔧 ARTIST FOLLOW-UP SCALING: {scaled_target_candidates} candidates for {recently_shown_count} recently shown tracks")
                else:
                    # Style/genre follow-ups - moderate scaling
                    scaled_target_candidates = max(150, recently_shown_count * 10)
                    self.logger.info(f"🔧 STYLE FOLLOW-UP SCALING: {scaled_target_candidates} candidates for {recently_shown_count} recently shown tracks")
                    
                # Apply the scaling
                self.target_candidates = scaled_target_candidates
                self.logger.info(f"🎯 CANDIDATE SCALING: {self.target_candidates} (from {base_target_candidates}) for follow-up query")
            else:
                self.logger.info(f"🎯 NEW QUERY: Using base {self.target_candidates} candidates")

            # Phase 1: Generate candidates using shared generator with discovery strategy
            candidates = await self.candidate_generator.generate_candidate_pool(
                entities=entities,
                intent_analysis=intent_analysis,
                agent_type="discovery",
                target_candidates=self.target_candidates,
                detected_intent=candidate_generation_intent,  # Use the adjusted intent
                # 🚨 CRITICAL FIX: Pass recently shown tracks to avoid duplicates during generation
                recently_shown_track_ids=state.recently_shown_track_ids or []
            )
            
            self.logger.debug(f"Generated {len(candidates)} discovery candidates")
            
            # Phase 3: Generate and persist large candidate pool if recommended by planner
            planning_strategy = getattr(state, 'planning_strategy', {})
            should_generate_large_pool = planning_strategy.get('generate_large_pool', False)
            
            if should_generate_large_pool and recently_shown_count == 0:  # Only for original queries
                self.logger.info("🎯 Phase 3: Generating large candidate pool for follow-up efficiency")
                try:
                    pool_key = await self.candidate_generator.generate_and_persist_large_pool(
                        entities=entities,
                        intent_analysis=intent_analysis,
                        session_id=state.session_id,
                        agent_type="discovery",
                        detected_intent=candidate_generation_intent
                    )
                    if pool_key:
                        self.logger.info(f"🎯 Phase 3: Large pool generated and stored with key: {pool_key}")
                    else:
                        self.logger.info("🎯 Phase 3: Large pool generation skipped (not beneficial for this intent)")
                except Exception as e:
                    self.logger.warning(f"Phase 3: Large pool generation failed: {e}")
            elif recently_shown_count > 0:
                self.logger.info("🎯 Phase 3: Skipping large pool generation for follow-up query")
            else:
                self.logger.info("🎯 Phase 3: Large pool generation not recommended by planner")
            
            # Phase 2: Score candidates with discovery-specific metrics
            scored_candidates = await self._score_discovery_candidates(candidates, entities, intent_analysis)
            
            # Phase 3: Filter for novelty and underground appeal
            filtered_candidates = await self._filter_for_discovery(
                scored_candidates, entities, intent_analysis, context_override if context_override_applied else None
            )
            
            # 🔧 DEBUG: Log filtered candidates with scores for troubleshooting
            target_artists = self._extract_target_artists(entities)
            self.logger.info(f"Filtered candidates with scores:")
            for i, candidate in enumerate(filtered_candidates[:10]):  # Show top 10
                candidate_name = candidate.get('name', 'Unknown')
                candidate_artist = candidate.get('artist', 'Unknown')
                combined_score = candidate.get('combined_score', 0.0)
                is_target = candidate_artist.lower() in [a.lower() for a in target_artists]
                self.logger.info(f"  {i+1}. {candidate_name} by {candidate_artist} - Score: {combined_score:.3f} {'🎯 TARGET' if is_target else ''}")
            
            # 🔧 BOOST: Prioritize target artist tracks by boosting their combined scores
            # Check for both pure artist_similarity and similarity-primary hybrids OR context override
            is_similarity_intent = (
                intent_analysis.get('intent') == 'artist_similarity' or
                (intent_analysis.get('intent') == 'hybrid' and 
                 ('similarity_primary' in intent_analysis.get('reasoning', '') or
                  intent_analysis.get('hybrid_subtype') == 'similarity_primary')) or
                context_override_applied  # 🚀 NEW: Also boost if context override applied
            )
            
            if is_similarity_intent and target_artists:
                self.logger.info(f"🔧 BOOSTING target artist tracks for similarity intent with artists: {target_artists}")
                for candidate in filtered_candidates:
                    candidate_artist = candidate.get('artist', '')
                    if candidate_artist.lower() in [a.lower() for a in target_artists]:
                        # Boost target artist tracks to ensure they rank higher
                        original_score = candidate.get('combined_score', 0.0)
                        boost_amount = 0.4 if context_override_applied else 0.3  # Higher boost for context override
                        candidate['combined_score'] = min(original_score + boost_amount, 1.0)
                        self.logger.info(f"🚀 BOOSTED target artist track: {candidate.get('name')} by {candidate_artist} from {original_score:.3f} to {candidate['combined_score']:.3f}")
                
                # Re-sort by combined score after boosting
                filtered_candidates.sort(key=lambda x: x.get('combined_score', 0.0), reverse=True)
                self.logger.info(f"Re-sorted after target artist boost - new top 5:")
                for i, candidate in enumerate(filtered_candidates[:5]):
                    candidate_name = candidate.get('name', 'Unknown')
                    candidate_artist = candidate.get('artist', 'Unknown')
                    combined_score = candidate.get('combined_score', 0.0)
                    is_target = candidate_artist.lower() in [a.lower() for a in target_artists]
                    self.logger.info(f"  {i+1}. {candidate_name} by {candidate_artist} - Score: {combined_score:.3f} {'🎯 TARGET' if is_target else ''}")
            
            # Phase 4: Create final recommendations with discovery reasoning
            recommendations = await self._create_discovery_recommendations(
                filtered_candidates[:self.final_recommendations],
                entities,
                intent_analysis
            )
            
            # Update state
            state.discovery_recommendations = [rec.model_dump() for rec in recommendations]
            
            self.logger.info(
                "Discovery agent processing completed",
                candidates=len(candidates),
                filtered=len(filtered_candidates),
                recommendations=len(recommendations),
                context_override_applied=context_override_applied
            )
            
            return state
            
        except Exception as e:
            self.logger.error("Discovery agent processing failed", error=str(e))
            state.discovery_recommendations = []
            return state
    
    async def _score_discovery_candidates(
        self,
        candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score candidates with discovery-specific metrics."""
        scored_candidates = []
        
        for candidate in candidates:
            try:
                # Use shared quality scorer
                quality_score = await self.quality_scorer.calculate_quality_score(
                    candidate, entities, intent_analysis
                )
                
                # Ensure quality_score is a number
                if quality_score is None:
                    quality_score = 0.0
                elif not isinstance(quality_score, (int, float)):
                    quality_score = 0.0
                
                # Add discovery-specific scoring
                novelty_score = self._calculate_novelty_score(candidate, entities, intent_analysis)
                underground_score = self._calculate_underground_score(candidate)
                similarity_score = self._calculate_similarity_score(candidate, entities)
                
                # Combined discovery score
                discovery_score = (
                    novelty_score * 0.4 +
                    underground_score * 0.3 +
                    similarity_score * 0.3
                )
                
                candidate['quality_score'] = quality_score
                candidate['novelty_score'] = novelty_score
                candidate['underground_score'] = underground_score
                candidate['similarity_score'] = similarity_score
                candidate['discovery_score'] = discovery_score
                candidate['combined_score'] = (quality_score * 0.4) + (discovery_score * 0.6)
                
                scored_candidates.append(candidate)
                
            except Exception as e:
                self.logger.warning(f"Failed to score discovery candidate: {e}")
                continue
        
        # Sort by combined score
        scored_candidates.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        return scored_candidates
    
    def _calculate_novelty_score(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> float:
        """Calculate novelty score for discovery."""
        score = 0.0
        
        # Get intent to adjust novelty criteria
        intent = intent_analysis.get('intent', '').lower()
        
        # 🔧 FIX: Detect both pure artist_similarity and similarity-primary hybrids
        is_artist_similarity = False
        if intent == 'artist_similarity':
            is_artist_similarity = True
        elif intent == 'hybrid':
            # Check for similarity-primary hybrid
            reasoning = intent_analysis.get('reasoning', '')
            hybrid_subtype = intent_analysis.get('hybrid_subtype', '')
            if (hybrid_subtype == 'similarity_primary' or 
                'similarity_primary' in reasoning):
                is_artist_similarity = True
        
        # Lower listener count = higher novelty
        listeners = candidate.get('listeners', 0)
        # Handle None values and ensure it's a number
        if listeners is None:
            listeners = 0
        elif not isinstance(listeners, (int, float)):
            try:
                listeners = int(listeners)
            except (ValueError, TypeError):
                listeners = 0
        
        # For artist similarity, be more lenient with popularity thresholds
        if is_artist_similarity:
            if listeners == 0:
                score += 0.5
            elif listeners < 100000:  # Raised from 10k
                score += 0.4
            elif listeners < 1000000:  # Raised from 100k
                score += 0.3
            elif listeners < 10000000:  # New tier for artist similarity
                score += 0.2
            else:
                score += 0.1
        else:
            # Original stricter thresholds for other intents
            if listeners == 0:
                score += 0.5
            elif listeners < 10000:
                score += 0.4
            elif listeners < 100000:
                score += 0.3
            elif listeners < 1000000:
                score += 0.2
            else:
                score += 0.1
        
        # Uncommon tags indicate novelty
        tags = candidate.get('tags', [])
        if tags is None:
            tags = []
        
        uncommon_tags = ['experimental', 'underground', 'indie', 'alternative', 'obscure', 'rare']
        for tag in tags:
            if tag and any(uncommon in str(tag).lower() for uncommon in uncommon_tags):
                score += 0.2
        
        # Source diversity (non-mainstream sources)
        source = candidate.get('source', '')
        if source and ('underground' in source or 'serendipitous' in source):
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_underground_score(self, candidate: Dict[str, Any]) -> float:
        """Calculate underground appeal score."""
        score = 0.0
        
        # Low play count indicates underground status
        playcount = candidate.get('playcount', 0)
        listeners = candidate.get('listeners', 0)
        
        # Handle None values and ensure they're numbers
        if playcount is None:
            playcount = 0
        elif not isinstance(playcount, (int, float)):
            try:
                playcount = int(playcount)
            except (ValueError, TypeError):
                playcount = 0
        
        if listeners is None:
            listeners = 0
        elif not isinstance(listeners, (int, float)):
            try:
                listeners = int(listeners)
            except (ValueError, TypeError):
                listeners = 0
        
        if playcount == 0 and listeners == 0:
            score += 0.3
        elif playcount < 50000:
            score += 0.4
        elif playcount < 500000:
            score += 0.3
        elif playcount < 5000000:
            score += 0.2
        else:
            score += 0.1
        
        # Underground indicators in tags
        tags = candidate.get('tags', [])
        if tags is None:
            tags = []
        
        underground_indicators = ['underground', 'hidden gem', 'obscure', 'cult', 'rare']
        for tag in tags:
            if tag and any(indicator in str(tag).lower() for indicator in underground_indicators):
                score += 0.3
        
        # Artist name length (longer names often indicate less mainstream artists)
        artist = candidate.get('artist', '')
        if artist and len(str(artist)) > 15:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_similarity_score(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any]
    ) -> float:
        """Calculate similarity score to user preferences."""
        score = 0.0
        
        # Extract target preferences
        target_artists = self._extract_target_artists(entities)
        target_genres = self._extract_target_genres(entities)
        
        candidate_tags = candidate.get('tags', [])
        candidate_artist = candidate.get('artist', '').lower()
        
        # Artist similarity (if this is a similar artist)
        source_artist = candidate.get('source_artist', '')
        if source_artist and any(target.lower() in source_artist.lower() for target in target_artists):
            score += 0.4
        
        # Genre similarity
        for genre in target_genres:
            if any(genre.lower() in tag.lower() for tag in candidate_tags):
                score += 0.3
        
        # Multi-hop similarity bonus
        if candidate.get('source') == 'multi_hop_similarity':
            score += 0.2
        
        return min(score, 1.0)
    
    async def _filter_for_discovery(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        context_override: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Filter candidates for discovery criteria."""
        # Extract relevant context
        target_artists = self._extract_target_artists(entities)
        
        # 🔧 FIX: Properly detect artist similarity for both pure and hybrid intents
        intent_value = (
            intent_analysis.get('intent') or 
            intent_analysis.get('primary_intent') or 
            intent_analysis.get('detected_intent') or
            'discovery'
        )
        
        # 🔧 NEW: Check for similarity-primary hybrid with genre filtering requirements
        required_genres = self._extract_required_genres(entities)
        is_similarity_primary_hybrid = False
        
        if intent_value == 'hybrid' and target_artists and required_genres:
            # This looks like "Michael Jackson tracks that are R&B" 
            self.logger.info(f"🎯 DETECTED SIMILARITY-PRIMARY HYBRID WITH GENRE FILTER: artists={target_artists}, genres={required_genres}")
            is_similarity_primary_hybrid = True
            
            # Apply strict genre filtering first
            filtered_candidates = await self._filter_for_similarity_primary_hybrid(
                scored_candidates, entities, intent_analysis
            )
            
            if not filtered_candidates:
                self.logger.warning(f"🎯 Strict genre filtering eliminated all candidates! Falling back to relaxed filtering.")
            else:
                self.logger.info(f"🎯 Using strict genre filtering: {len(scored_candidates)} → {len(filtered_candidates)} candidates")
                # Continue with the filtered candidates
                scored_candidates = filtered_candidates
        
        # 🔧 HYBRID SUPPORT: Check for similarity-primary hybrids
        is_artist_similarity = False
        is_artist_style_refinement = False
        style_modifier = None
        
        if intent_value == 'artist_similarity':
            is_artist_similarity = True
        elif intent_value == 'hybrid':
            # Check if this is a similarity-primary hybrid
            reasoning = intent_analysis.get('reasoning', '')
            hybrid_subtype = intent_analysis.get('hybrid_subtype', '')
            
            if hybrid_subtype == 'artist_style_refinement':
                # 🔧 NEW: Artist-style refinement handling
                is_artist_style_refinement = True
                is_artist_similarity = True  # Also use artist similarity logic
                style_modifier = intent_analysis.get('style_modifier', '')
                self.logger.info(f"🎵 ARTIST-STYLE REFINEMENT DETECTED: Looking for {target_artists} tracks with style: {style_modifier}")
            
            # Multiple ways to detect similarity-primary hybrid
            elif (hybrid_subtype == 'similarity_primary' or 
                'similarity_primary' in reasoning or
                'Hybrid sub-type: similarity_primary' in reasoning or
                is_similarity_primary_hybrid):  # Include our new detection
                is_artist_similarity = True
                self.logger.info(f"🔧 DETECTED SIMILARITY-PRIMARY HYBRID: Using artist similarity logic for hybrid query")
            
            # Fallback: If we have target artists and similarity phrases, assume similarity-primary
            elif target_artists:
                original_query = intent_analysis.get('original_query', entities.get('original_query', ''))
                similarity_phrases = ['like', 'similar', 'sounds like', 'reminds me of', 'in the style of']
                if any(phrase in original_query.lower() for phrase in similarity_phrases):
                    is_artist_similarity = True
                    self.logger.info(f"🔧 FALLBACK DETECTION: Query has artists + similarity phrases, treating as artist similarity")
        
        self.logger.info(f"🔧 FILTER DETECTION: intent='{intent_value}', artist_similarity={is_artist_similarity}, artist_style_refinement={is_artist_style_refinement}, style_modifier='{style_modifier}'")
        
        self.logger.info(
            f"Discovery filtering: {len(scored_candidates)} candidates, "
            f"target_artists={target_artists}, is_artist_similarity={is_artist_similarity}"
        )
        
        # 🔧 DEBUG: Check if we have candidates to process
        self.logger.info(f"About to start filtering loop with {len(scored_candidates)} scored candidates")
        
        # 🔧 NEW: Extract required genres for genre filtering
        required_genres = self._extract_required_genres(entities)
        
        # 🎯 NEW: Also check for preserved genres from context overrides (follow-up queries)
        if context_override:
            constraint_overrides = context_override.get('constraint_overrides', {})
            preserved_genres = constraint_overrides.get('required_genres', [])
            if preserved_genres:
                required_genres.extend(preserved_genres)
                required_genres = list(set(required_genres))  # Remove duplicates
                self.logger.info(f"🎯 PRESERVED GENRES from follow-up context: {preserved_genres}")
                self.logger.info(f"🎯 COMBINED required genres: {required_genres}")
        
        filtered = []
        for i, candidate in enumerate(scored_candidates):
            candidate_artist = candidate.get('artist', '')
            candidate_name = candidate.get('name', 'Unknown')
            
            self.logger.info(f"Processing candidate {i+1}/{len(scored_candidates)}: '{candidate_name}' by '{candidate_artist}'")
            
            # 🔧 NEW: Check for genre filtering requirements in by_artist queries
            if intent_value == 'by_artist' and required_genres:
                self.logger.info(f"🎯 GENRE FILTERING: by_artist query with required genres: {required_genres}")
                
                # Apply strict genre filtering for by_artist queries with genre requirements
                genre_match = await self._strict_genre_match(candidate, required_genres)
                if not genre_match:
                    self.logger.debug(f"❌ REJECTED (genre): {candidate_name} by {candidate_artist} - doesn't match required genres {required_genres}")
                    continue
                else:
                    self.logger.debug(f"✅ GENRE MATCH: {candidate_name} by {candidate_artist} matches required genres")
            
            # Skip target artist checks for by_artist intent since ALL tracks should be by target artist
            if intent_value == 'by_artist':
                # For by_artist intent, we expect ALL tracks to be by the target artist
                self.logger.debug(f"✅ ACCEPTED: {candidate_name} by {candidate_artist}")
                filtered.append(candidate)
                continue
            
            # ✅ SPECIAL HANDLING: For artist similarity, prioritize target artist tracks
            if is_artist_similarity and target_artists:
                # If this is a track by the target artist, use relaxed thresholds
                is_target_artist_track = any(
                    target.lower() in candidate_artist.lower() or candidate_artist.lower() in target.lower()
                    for target in target_artists
                )
                
                # 🔧 DEBUG: Log matching results
                if is_target_artist_track:
                    self.logger.info(f"🎯 MATCHED target artist: '{candidate_artist}' matches {target_artists}")
                
                if is_target_artist_track:
                    # Very relaxed thresholds for target artist tracks
                    quality_score = candidate.get('quality_score', 0)
                    self.logger.info(
                        f"Target artist track: {candidate_name} by {candidate_artist}, "
                        f"quality={quality_score}"
                    )
                    if quality_score >= 0.2:  # Much lower threshold for target artist
                        filtered.append(candidate)
                        self.logger.info(f"✅ ACCEPTED target artist track: {candidate_name}")
                        continue
                    else:
                        self.logger.info(
                            f"❌ REJECTED target artist track (low quality): {candidate_name}"
                        )
            
            # Standard discovery filtering for non-target tracks
            # Quality threshold check (lower for discovery)
            quality_score = candidate.get('quality_score', 0)
            if quality_score is None:
                quality_score = 0
            if quality_score < self.quality_threshold:
                self.logger.debug(f"❌ REJECTED (quality): {candidate_name} by {candidate_artist}, quality={quality_score} < {self.quality_threshold}")
                continue
            
            # 🔧 FIX: Relaxed novelty threshold for artist similarity
            novelty_score = candidate.get('novelty_score', 0)
            if novelty_score is None:
                novelty_score = 0
            
            # Use different novelty thresholds based on query type
            if is_artist_similarity:
                # Much more lenient for artist similarity - we want similar artists!
                novelty_threshold = 0.15  # Lowered from 0.4
                self.logger.debug(f"Using relaxed novelty threshold {novelty_threshold} for artist similarity")
            else:
                novelty_threshold = self.novelty_threshold
            
            if novelty_score < novelty_threshold:
                self.logger.debug(f"❌ REJECTED (novelty): {candidate_name} by {candidate_artist}, novelty={novelty_score} < {novelty_threshold}")
                continue
            
            # Underground bias check
            underground_score = candidate.get('underground_score', 0)
            if underground_score is None:
                underground_score = 0
            if underground_score < 0.2:
                self.logger.debug(f"❌ REJECTED (underground): {candidate_name} by {candidate_artist}, underground={underground_score} < 0.2")
                continue
            
            # 🔧 NEW: Style filtering for artist-style refinement
            if is_artist_style_refinement and style_modifier:
                # Check if the track matches the requested style
                style_match = self._check_style_match(candidate, style_modifier)
                if not style_match:
                    self.logger.debug(
                        f"❌ REJECTED (style): {candidate_name} by {candidate_artist}, "
                        f"doesn't match style '{style_modifier}'"
                    )
                    continue
                else:
                    self.logger.info(
                        f"✅ STYLE MATCH: {candidate_name} by {candidate_artist} "
                        f"matches '{style_modifier}'"
                    )
            
            self.logger.debug(f"✅ ACCEPTED: {candidate_name} by {candidate_artist}")
            filtered.append(candidate)
        
        self.logger.info(f"Discovery filtering result: {len(scored_candidates)} -> {len(filtered)} candidates")
        
        # 🔧 CRITICAL DEBUG: Log what we're passing to diversity filtering
        self.logger.info(f"🔧 DEBUG: About to call diversity filtering with {len(filtered)} candidates")
        self.logger.info(f"🔧 DEBUG: First few candidates: {[c.get('name', 'Unknown') for c in filtered[:3]]}")
        
        # Ensure diversity and novelty
        filtered = self._ensure_discovery_diversity(filtered, intent_analysis, context_override)
        
        # 🔧 CRITICAL DEBUG: Log what diversity filtering returned
        self.logger.info(f"🔧 DEBUG: Diversity filtering returned {len(filtered)} candidates")
        
        return filtered
    
    def _ensure_discovery_diversity(self, candidates: List[Dict[str, Any]], intent_analysis: Dict[str, Any], context_override: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Ensure diversity in discovery recommendations with adaptive limits."""
        # Get intent to adjust diversity rules
        intent = intent_analysis.get('intent', '').lower()
        
        # 🚀 Apply context override constraints for followup intents
        target_artist_from_override = None
        candidate_limit = len(candidates)  # Use input candidates, not empty diverse_candidates
        max_tracks_per_artist = 1  # 🔧 FIX: Initialize default value at the start
        
        # 🔧 CRITICAL FIX: For by_artist intents, allow many tracks from the same artist
        if intent == 'by_artist':
            max_tracks_per_artist = 20  # Allow many tracks for by_artist queries
            self.logger.info(f"🔧 BY_ARTIST INTENT: Allowing up to {max_tracks_per_artist} tracks from same artist")
            return self._apply_diversity_filtering(candidates, max_tracks_per_artist, candidate_limit, target_artist_from_override, intent)
        
        if context_override:
            target_artist_from_override = context_override.get('target_artist')
            intent_override = context_override.get('intent')
            
            # 🔧 NEW: Check for constraint_overrides first
            constraint_overrides = context_override.get('constraint_overrides', {})
            diversity_limits = constraint_overrides.get('diversity_limits', {})
            
            if 'same_artist_limit' in diversity_limits:
                max_tracks_per_artist = diversity_limits['same_artist_limit']
                self.logger.info(f"🔧 CONSTRAINT OVERRIDE: Using same_artist_limit={max_tracks_per_artist} from context override")
                return self._apply_diversity_filtering(candidates, max_tracks_per_artist, candidate_limit, target_artist_from_override, intent)
            elif intent_override in ['artist_deep_dive', 'artist_similarity', 'style_continuation', 'artist_style_refinement', 'hybrid_artist_genre']:
                max_tracks_per_artist = 10  # Allow more tracks from same artist for followup
                self.logger.info(f"🔧 INTENT OVERRIDE: Using max_tracks_per_artist={max_tracks_per_artist} for {intent_override}")
                return self._apply_diversity_filtering(candidates, max_tracks_per_artist, candidate_limit, target_artist_from_override, intent)
            
            # 🚀 CRITICAL FIX: For follow-ups, generate many more candidates to account for duplicates
            if intent_override in ['artist_deep_dive', 'artist_similarity', 'style_continuation', 'artist_style_refinement', 'hybrid_artist_genre']:
                candidate_limit = self.final_recommendations * 12  # Increased from 6x to 12x
                self.logger.info(f"🔧 FOLLOW-UP FIX: Generating 12x candidates ({candidate_limit}) to account for heavy duplicate filtering")
            
            # Extract target artist from followup context  
            if target_artist_from_override:
                self.logger.info(f"🚀 Context override: allowing up to {max_tracks_per_artist} tracks from target artist '{target_artist_from_override}'")
            elif context_override.get('target_entity'):
                target_artist_from_override = context_override.get('target_entity')
                self.logger.info(f"🚀 Context override: allowing up to {max_tracks_per_artist} tracks from target entity '{target_artist_from_override}'")
                
        # 🔧 Fallback candidate limit logic
        if candidate_limit > len(candidates):
            candidate_limit = len(candidates)
        
        # 🎯 NEW: ADAPTIVE DIVERSITY SYSTEM
        return self._adaptive_diversity_filtering(candidates, candidate_limit, target_artist_from_override, intent)
    
    def _adaptive_diversity_filtering(self, candidates: List[Dict[str, Any]], candidate_limit: int, target_artist_from_override: str, intent: str) -> List[Dict[str, Any]]:
        """Apply adaptive diversity filtering that relaxes limits if we don't have enough valid candidates."""
        target_recommendations = self.final_recommendations  # Usually 10
        
        # 🎯 Progressive diversity limits to try
        diversity_attempts = [1, 2, 3, 5, 8]  # Start strict, progressively relax
        
        self.logger.info(f"🎯 ADAPTIVE DIVERSITY: Target {target_recommendations} recommendations from {len(candidates)} candidates")
        
        best_result = []
        
        for attempt, max_tracks_per_artist in enumerate(diversity_attempts):
            self.logger.info(f"🎯 ATTEMPT {attempt + 1}: Trying max_tracks_per_artist={max_tracks_per_artist}")
            
            # Apply diversity filtering with current limit
            filtered_candidates = self._apply_diversity_filtering(
                candidates, max_tracks_per_artist, candidate_limit, target_artist_from_override, intent
            )
            
            # Count candidates that would pass quality filtering (exclude fake tracks)
            valid_candidates = self._count_valid_candidates(filtered_candidates)
            
            self.logger.info(f"🎯 ATTEMPT {attempt + 1} RESULT: {len(filtered_candidates)} total, {valid_candidates} valid candidates")
            
            # Update best result if this is better
            if len(filtered_candidates) > len(best_result):
                best_result = filtered_candidates
            
            # Check if we have enough valid candidates
            if valid_candidates >= target_recommendations * 0.8:  # 80% of target (8 out of 10)
                self.logger.info(f"🎯 SUCCESS: Found {valid_candidates} valid candidates (≥{target_recommendations * 0.8}), stopping diversity relaxation")
                break
            elif valid_candidates >= target_recommendations * 0.6:  # 60% of target (6 out of 10)
                self.logger.info(f"🎯 ACCEPTABLE: Found {valid_candidates} valid candidates (≥{target_recommendations * 0.6}), trying one more level")
                # Continue to next attempt but mark this as acceptable
                continue
            else:
                self.logger.info(f"🎯 INSUFFICIENT: Only {valid_candidates} valid candidates (<{target_recommendations * 0.6}), trying more relaxed limits")
                continue
        
        final_count = len(best_result)
        valid_count = self._count_valid_candidates(best_result)
        
        self.logger.info(f"🎯 ADAPTIVE DIVERSITY COMPLETE: {final_count} candidates ({valid_count} estimated valid)")
        
        return best_result
    
    def _count_valid_candidates(self, candidates: List[Dict[str, Any]]) -> int:
        """Count candidates that would likely pass Judge Agent quality filtering (not fake tracks)."""
        valid_count = 0
        
        for candidate in candidates:
            # Check for fake track indicators (what Judge Agent rejects)
            listeners = candidate.get('listeners', 0)
            playcount = candidate.get('playcount', 0)
            
            # Handle None values
            if listeners is None:
                listeners = 0
            if playcount is None:
                playcount = 0
                
            # Convert to numbers if they're strings
            try:
                listeners = int(listeners) if listeners else 0
                playcount = int(playcount) if playcount else 0
            except (ValueError, TypeError):
                listeners = 0
                playcount = 0
            
            # Judge Agent rejects tracks with zero listeners AND zero playcount
            if listeners > 0 or playcount > 0:
                valid_count += 1
        
        return valid_count
    
    def _apply_diversity_filtering(self, candidates: List[Dict[str, Any]], max_tracks_per_artist: int, candidate_limit: int, target_artist_from_override: str, intent: str) -> List[Dict[str, Any]]:
        """Apply diversity filtering with specified artist limits."""
        self.logger.info(f"🔧 DEBUG: Starting diversity filtering with {len(candidates)} candidates")
        self.logger.info(f"🔧 DEBUG: Intent: {intent}, max tracks per artist: {max_tracks_per_artist}")
        self.logger.info(f"🔧 DEBUG: Candidate limit: {candidate_limit}")
        self.logger.info(f"🔧 DEBUG: Target artist from context override: {target_artist_from_override}")
        
        final_candidates = []
        # Change seen_artists to track count per artist
        seen_artists = {}
        seen_genres = set()
        
        # 🔧 Sort candidates by combined_score to prioritize higher quality tracks
        candidates.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        for candidate in candidates:
            if len(final_candidates) >= candidate_limit:
                break
                
            artist = candidate.get('artist', '').lower()
            candidate_genres = candidate.get('genres', [])
            
            # Check if this candidate is from target artist
            is_target_artist = (target_artist_from_override and 
                              artist == target_artist_from_override.lower())
                              
            artist_count = seen_artists.get(artist, 0)
            
            # 🔧 UPDATED LOGIC: Use max_tracks_per_artist for all cases
            if artist_count < max_tracks_per_artist:
                final_candidates.append(candidate)
                seen_artists[artist] = artist_count + 1
            # Only log rejections in debug mode to reduce noise
            # else:
            #     self.logger.debug(f"❌ DEBUG: REJECTED {candidate.get('name', 'Unknown')} - artist limit reached ({artist_count}/{max_tracks_per_artist})")
        
        if target_artist_from_override:
            target_count = sum(1 for c in final_candidates if c.get('artist', '').lower() == target_artist_from_override.lower())
            self.logger.info(f"🚀 Target artist '{target_artist_from_override}' tracks in final candidates: {target_count}")
        
        return final_candidates
    
    async def _create_discovery_recommendations(
        self,
        candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[TrackRecommendation]:
        """Create final discovery recommendations."""
        recommendations = []
        
        for i, candidate in enumerate(candidates):
            try:
                # Generate discovery-focused reasoning
                reasoning = await self._generate_discovery_reasoning(
                    candidate, entities, intent_analysis, i + 1
                )
                
                recommendation = TrackRecommendation(
                    title=candidate.get('name', 'Unknown'),
                    artist=candidate.get('artist', 'Unknown'),
                    id=f"{candidate.get('artist', 'Unknown')}_{candidate.get('name', 'Unknown')}".replace(' ', '_').lower(),
                    source='discovery_agent',
                    track_url=candidate.get('url', ''),
                    album_title=candidate.get('album', ''),
                    genres=self._extract_discovery_genres(candidate, entities),
                    moods=self._extract_discovery_tags(candidate, entities, intent_analysis),
                    confidence=candidate.get('combined_score', 0.5),
                    explanation=reasoning,
                    novelty_score=candidate.get('novelty_score', 0.0),
                    quality_score=candidate.get('quality_score', 0.0),
                    advocate_source_agent='discovery_agent',
                    # 🔧 FIXED: Preserve popularity data for ranking
                    raw_source_data={
                        'playcount': candidate.get('playcount', 0),
                        'listeners': candidate.get('listeners', 0),
                        'popularity': candidate.get('popularity', 0),
                        'tags': candidate.get('tags', []),
                        'novelty_score': candidate.get('novelty_score', 0.0),
                        'underground_score': candidate.get('underground_score', 0.0)
                    },
                    additional_scores={
                        'combined_score': candidate.get('combined_score', 0.5),
                        'novelty_score': candidate.get('novelty_score', 0.0),
                        'quality_score': candidate.get('quality_score', 0.0),
                        'underground_score': candidate.get('underground_score', 0.0)
                    }
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                self.logger.warning(f"Failed to create discovery recommendation: {e}")
                continue
        
        return recommendations
    
    async def _generate_discovery_reasoning(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        rank: int
    ) -> str:
        """Generate reasoning for discovery recommendation."""
        try:
            # For now, skip LLM calls to avoid rate limits - use fallback reasoning
            return self._create_discovery_fallback_reasoning(candidate, entities, intent_analysis, rank)
            
            # # Disabled to prevent excessive API calls
            # # Create comprehensive reasoning prompt
            # target_artists = self._extract_target_artists(entities)
            # target_genres = self._extract_target_genres(entities)
            # 
            # try:
            #     listeners = int(candidate.get('listeners', 0))
            # except (ValueError, TypeError):
            #     listeners = 0
            # 
            # prompt = f"""Explain why "{candidate.get('name')}" by {candidate.get('artist')} is a great discovery.
            # 
            # Target artists: {', '.join(target_artists) if target_artists else 'Open to discovery'}
            # Target genres: {', '.join(target_genres) if target_genres else 'Any'}
            # Track tags: {', '.join(candidate.get('tags', [])[:5])}
            # Novelty score: {candidate.get('novelty_score', 0):.2f}
            # Listeners: {listeners:,}
            # Rank: #{rank}
            # 
            # Provide a brief, engaging explanation (2-3 sentences) focusing on discovery value and uniqueness."""
            # 
            # reasoning = await self.llm_utils.call_llm(prompt)
            # return reasoning.strip()
            
        except Exception as e:
            self.logger.debug(f"LLM reasoning failed, using fallback: {e}")
            return self._create_discovery_fallback_reasoning(candidate, entities, intent_analysis, rank)
    
    def _create_discovery_fallback_reasoning(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        rank: int
    ) -> str:
        """Create fallback discovery reasoning when LLM is unavailable."""
        name = candidate.get('name', 'This track')
        artist = candidate.get('artist', 'the artist')
        listeners = candidate.get('listeners', 0)
        tags = candidate.get('tags', [])[:3]
        
        # Ensure listeners is a valid number
        if listeners is None:
            listeners = 0
        elif not isinstance(listeners, (int, float)):
            try:
                listeners = int(listeners)
            except (ValueError, TypeError):
                listeners = 0
        
        reasoning_parts = [f"#{rank}: {name} by {artist}"]
        
        # Highlight discovery aspects
        if listeners < 10000:
            reasoning_parts.append("Hidden gem with limited exposure")
        elif listeners < 100000:
            reasoning_parts.append("Underground favorite")
        
        if tags:
            reasoning_parts.append(f"Tagged as {', '.join(tags)}")
        
        novelty_score = candidate.get('novelty_score', 0)
        if novelty_score is not None and novelty_score > 0.7:
            reasoning_parts.append("High novelty discovery")
        
        return ". ".join(reasoning_parts) + "."
    
    def _extract_target_artists(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target artists from entities."""
        musical_entities = entities.get('musical_entities', {})
        artists = musical_entities.get('artists', {})
        
        target_artists = []
        primary_artists = artists.get('primary', [])
        similar_artists = artists.get('similar_to', [])
        
        # 🚨 CRITICAL FIX: Filter out invalid artist names like "this", "that", "these"
        invalid_artist_names = {'this', 'that', 'these', 'those', 'it', 'them'}
        
        for artist in primary_artists:
            artist_name = str(artist).strip().lower()
            if artist_name not in invalid_artist_names and len(artist_name) > 1:
                target_artists.append(str(artist).strip())
                
        for artist in similar_artists:
            artist_name = str(artist).strip().lower()
            if artist_name not in invalid_artist_names and len(artist_name) > 1:
                target_artists.append(str(artist).strip())
        
        # 🔧 DEBUG: Log filtering results for Discovery Agent
        if any(str(artist).strip().lower() in invalid_artist_names for artist in primary_artists + similar_artists):
            filtered_out = [str(artist) for artist in primary_artists + similar_artists 
                          if str(artist).strip().lower() in invalid_artist_names]
            remaining = [str(artist) for artist in primary_artists + similar_artists 
                        if str(artist).strip().lower() not in invalid_artist_names]
            self.logger.info(f"🚨 DISCOVERY AGENT: Filtered out invalid artist names: {filtered_out}")
            self.logger.info(f"🎯 DISCOVERY AGENT: Remaining valid artist names: {remaining}")
        
        return list(set(target_artists))  # Remove duplicates
    
    def _extract_target_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract target genres from entities."""
        musical_entities = entities.get('musical_entities', {})
        genres = musical_entities.get('genres', {})
        
        target_genres = []
        target_genres.extend(genres.get('primary', []))
        target_genres.extend(genres.get('secondary', []))
        
        return list(set(target_genres))  # Remove duplicates
    
    def _extract_discovery_genres(self, candidate: Dict[str, Any], entities: Dict[str, Any]) -> List[str]:
        """Extract genres for discovery recommendation."""
        tags = candidate.get('tags', [])
        
        # Filter for genre-like tags, prioritizing unique/underground genres
        genre_tags = []
        for tag in tags[:5]:
            if len(tag) > 2 and not tag.isdigit():
                genre_tags.append(tag)
        
        # Add discovery-specific genre indicators
        if candidate.get('underground_score', 0) > 0.7:
            genre_tags.append('underground')
        if candidate.get('novelty_score', 0) > 0.7:
            genre_tags.append('experimental')
        
        return genre_tags
    
    def _extract_discovery_tags(
        self,
        candidate: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract tags for discovery context."""
        tags = []
        
        # Add candidate tags
        if 'tags' in candidate:
            tags.extend(candidate['tags'])
        
        # Add discovery-specific context
        if candidate.get('source') == 'underground_tags':
            tags.append('underground')
        
        if candidate.get('source') == 'multi_hop_similarity':
            tags.append('discovered')
            
        return tags[:5]  # Limit context
    
    def _check_style_match(self, candidate: Dict[str, Any], style_modifier: str) -> bool:
        """
        Check if a candidate track matches the requested style modifier.
        
        Args:
            candidate: The track candidate to check
            style_modifier: The style constraint (e.g., "electronic", "upbeat", "jazzy")
            
        Returns:
            bool: True if the track matches the style, False otherwise
        """
        if not style_modifier:
            return True
        
        style_lower = style_modifier.lower().strip()
        
        # Get track metadata with defensive programming for None values
        tags = candidate.get('tags', [])
        genres = candidate.get('genres', [])
        album = candidate.get('album') or ''
        track_name = candidate.get('name') or ''
        artist = candidate.get('artist') or ''
        
        # 🔧 DEBUG: Log detailed information for troubleshooting
        self.logger.debug(
            f"🔍 Style match debug for '{track_name}' by '{artist}': "
            f"style='{style_modifier}', tags={tags}, genres={genres}"
        )
        
        # Convert to lowercase safely
        album = album.lower()
        track_name = track_name.lower()
        artist = artist.lower()
        
        # Style matching keywords
        style_keywords = {
            'electronic': [
                'electronic', 'electro', 'synth', 'digital', 'edm', 'techno', 
                'house', 'ambient', 'electronica', 'experimental', 'glitch',
                'hypnagogic pop', 'synthwave', 'computer', 'synthesizer',
                'drum machine', 'sequencer', 'loop', 'sample'
            ],
            'rnb': [
                'rnb', 'r&b', 'r and b', 'rhythm and blues', 'soul', 'neo-soul',
                'contemporary r&b', 'motown', 'funk', 'groove', 'rhythm',
                'blues', 'soulful', 'smooth', 'neo soul'
            ],
            'r&b': [
                'rnb', 'r&b', 'r and b', 'rhythm and blues', 'soul', 'neo-soul',
                'contemporary r&b', 'motown', 'funk', 'groove', 'rhythm',
                'blues', 'soulful', 'smooth', 'neo soul'
            ],
            'upbeat': [
                'upbeat', 'energetic', 'fast', 'dance', 'pop', 'uptempo', 
                'lively', 'party', 'driving', 'pumping'
            ],
            'jazzy': [
                'jazz', 'jazzy', 'swing', 'blues', 'smooth', 'improvisational', 
                'bebop', 'cool jazz', 'fusion', 'saxophone', 'trumpet'
            ],
            'chill': [
                'chill', 'relaxed', 'ambient', 'downtempo', 'lo-fi', 'mellow', 
                'calm', 'peaceful', 'laid-back', 'dreamy'
            ],
            'experimental': [
                'experimental', 'avant-garde', 'noise', 'abstract', 
                'unconventional', 'art rock', 'post-rock', 'krautrock',
                'psychedelic', 'weird'
            ],
            'acoustic': [
                'acoustic', 'folk', 'unplugged', 'live', 'organic', 'natural',
                'guitar', 'piano', 'strings'
            ],
            'heavy': [
                'heavy', 'metal', 'hard', 'aggressive', 'intense', 'brutal',
                'distorted', 'loud'
            ],
            'melodic': [
                'melodic', 'melody', 'harmonic', 'tuneful', 'catchy',
                'singable', 'lyrical'
            ],
            'dark': [
                'dark', 'gothic', 'moody', 'atmospheric', 'melancholy', 
                'brooding', 'noir', 'somber'
            ],
            'funky': [
                'funk', 'funky', 'groove', 'rhythm', 'bass', 'percussive',
                'syncopated', 'danceable'
            ]
        }
        
        # Get matching keywords for the style
        matching_keywords = []
        for key, keywords in style_keywords.items():
            if key in style_lower or style_lower in key:
                matching_keywords.extend(keywords)
        
        # If no predefined keywords, use the style modifier itself
        if not matching_keywords:
            matching_keywords = [style_lower]
        
        # 🔧 DEBUG: Log matching keywords
        self.logger.debug(f"🔍 Style keywords for '{style_modifier}': {matching_keywords}")
        
        # Check tags and genres - be more explicit about conversions
        all_metadata = []
        
        # Add tags (convert each to string and lowercase)
        if tags:
            tag_strings = [str(tag).lower() for tag in tags if tag is not None]
            all_metadata.extend(tag_strings)
            self.logger.debug(f"🔍 Tags converted: {tag_strings}")
        
        # Add genres (convert each to string and lowercase)
        if genres:
            genre_strings = [str(genre).lower() for genre in genres if genre is not None]
            all_metadata.extend(genre_strings)
            self.logger.debug(f"🔍 Genres converted: {genre_strings}")
        
        # Add other metadata
        all_metadata.extend([album, track_name, artist])
        
        # Create searchable text
        metadata_text = ' '.join(str(item) for item in all_metadata if item)
        self.logger.debug(f"🔍 Searchable metadata text: '{metadata_text[:100]}...'")
        
        # Check for keyword matches
        for keyword in matching_keywords:
            if keyword in metadata_text:
                self.logger.info(
                    f"✅ Style match found: '{keyword}' in metadata for "
                    f"{candidate.get('name', 'Unknown')} by {candidate.get('artist', 'Unknown')}"
                )
                return True
        
        # 🔧 ENHANCED: Special R&B detection for Michael Jackson family tracks
        if 'r&b' in style_lower or 'rnb' in style_lower:
            # Michael Jackson family artists are inherently R&B
            mj_family_artists = [
                'michael jackson', 'jackson 5', 'the jackson 5', 'jackson five',
                'the jacksons', 'jacksons', '3t', 'jermaine jackson', 'janet jackson',
                'la toya jackson', 'rebbie jackson', 'tito jackson', 'marlon jackson',
                'randy jackson'
            ]
            
            if any(family_artist in artist for family_artist in mj_family_artists):
                self.logger.info(
                    f"✅ R&B match: Michael Jackson family artist '{artist}' "
                    f"for {candidate.get('name', 'Unknown')}"
                )
                return True
            
            # Additional R&B indicators for broader matching
            rb_indicators = [
                'soul', 'funk', 'groove', 'rhythm', 'blues', 'smooth',
                'motown', 'neo-soul', 'contemporary', 'urban', 'r&b', 'rnb'
            ]
            
            # Check if any R&B indicators are present
            for indicator in rb_indicators:
                if indicator in metadata_text:
                    self.logger.info(
                        f"✅ R&B indicator match: '{indicator}' found for "
                        f"{candidate.get('name', 'Unknown')} by {candidate.get('artist', 'Unknown')}"
                    )
                    return True
        
        # Special handling for common style modifiers
        if 'electronic' in style_lower:
            # Additional electronic indicators
            electronic_indicators = [
                'synth', 'digital', 'computer', 'electronic', 'electro',
                'hypnagogic', 'glitch', 'drum machine', 'sequencer', 'loop',
                'sample', 'synthesizer', 'synthwave'
            ]
            if any(indicator in metadata_text for indicator in electronic_indicators):
                self.logger.info(
                    f"✅ Electronic indicator match found for "
                    f"{candidate.get('name', 'Unknown')}"
                )
                return True
        
        if 'upbeat' in style_lower or 'energetic' in style_lower:
            # Look for tempo and energy indicators
            energy_indicators = [
                'fast', 'quick', 'energy', 'power', 'drive', 'pump'
            ]
            if any(indicator in metadata_text for indicator in energy_indicators):
                self.logger.info(
                    f"✅ Energy indicator match found for "
                    f"{candidate.get('name', 'Unknown')}"
                )
                return True
        
        self.logger.debug(
            f"❌ No style match found for '{style_modifier}' in "
            f"{candidate.get('name', 'Unknown')} by {candidate.get('artist', 'Unknown')}"
        )
        return False
    
    # 🔧 NEW: Enhanced genre filtering methods for similarity-primary hybrid queries
    
    async def _filter_for_similarity_primary_hybrid(
        self,
        scored_candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Strict filtering for similarity-primary hybrid queries."""
        
        target_artists = self._extract_target_artists(entities)
        required_genres = self._extract_required_genres(entities)
        
        if not required_genres:
            self.logger.info("🎯 No genre filtering required - no required genres found")
            return scored_candidates
        
        self.logger.info(f"🎯 GENRE FILTERING: Looking for {required_genres} tracks similar to {target_artists}")
        
        filtered = []
        for candidate in scored_candidates:
            # First pass: Genre filtering
            if not await self._strict_genre_match(candidate, required_genres):
                self.logger.debug(f"❌ Genre mismatch: {candidate.get('name')} by {candidate.get('artist')}")
                continue
                
            # Second pass: Artist similarity (if artist specified)
            if target_artists and not self._artist_similarity_match(candidate, target_artists):
                self.logger.debug(f"❌ Artist mismatch: {candidate.get('name')} by {candidate.get('artist')}")
                continue
                
            filtered.append(candidate)
            
        self.logger.info(f"🎯 Strict filtering: {len(scored_candidates)} → {len(filtered)} candidates")
        return filtered

    def _extract_required_genres(self, entities: Dict[str, Any]) -> List[str]:
        """Extract required genres from the query entities."""
        try:
            musical_entities = entities.get('musical_entities', {})
            genres = musical_entities.get('genres', {})
            
            # Get required genres if any
            required_genres = []
            if genres.get('primary'):
                required_genres.extend(genres['primary'])
            if genres.get('secondary'):
                required_genres.extend(genres['secondary'])
                
            self.logger.debug(f"🎯 Extracted required genres: {required_genres}")
            return required_genres
        
        except Exception as e:
            self.logger.warning(f"Failed to extract required genres: {e}")
            return []

    async def _strict_genre_match(self, candidate: Dict[str, Any], required_genres: List[str]) -> bool:
        """Strict genre matching for hybrid queries using API service."""
        for genre in required_genres:
            if await self._check_genre_match(candidate, genre):
                return True
        return False

    async def _check_genre_match(self, candidate: Dict[str, Any], target_genre: str) -> bool:
        """Enhanced genre matching logic using API service."""
        try:
            # Use API service to check genre match
            from ...services.api_service import get_api_service
            api_service = get_api_service()
            
            # Check track-level genre match
            track_match = await api_service.check_track_genre_match(
                artist=candidate.get('artist', ''),
                track=candidate.get('name', ''),
                target_genre=target_genre,
                include_related_genres=True
            )
            
            if track_match['matches']:
                self.logger.debug(
                    f"✅ Genre match: {candidate.get('artist')} - {candidate.get('name')} matches {target_genre}",
                    match_type=track_match['match_type'],
                    confidence=track_match['confidence'],
                    matched_tags=track_match['matched_tags']
                )
                return True
            
            self.logger.debug(
                f"❌ No genre match: {candidate.get('artist')} - {candidate.get('name')} doesn't match {target_genre}",
                track_tags=track_match.get('track_tags', []),
                artist_tags=track_match.get('artist_match', {}).get('artist_tags', [])
            )
            return False
            
        except Exception as e:
            self.logger.error(f"Genre matching failed: {e}")
            # Fall back to simple tag matching
            return self._fallback_genre_match(candidate, target_genre)
    
    def _fallback_genre_match(self, candidate: Dict[str, Any], target_genre: str) -> bool:
        """Fallback genre matching using simple tag comparison."""
        try:
            target_lower = target_genre.lower()
            
            # Get all text sources for matching
            artist = candidate.get('artist', '').lower()
            name = candidate.get('name', '').lower()
            tags = [str(tag).lower() for tag in candidate.get('tags', []) if tag]
            search_term = candidate.get('search_term', '').lower()
            genres = [str(genre).lower() for genre in candidate.get('genres', []) if genre]
            
            # Simple text matching
            all_text = f"{artist} {name} {' '.join(tags)} {search_term} {' '.join(genres)}"
            return target_lower in all_text
            
        except Exception as e:
            self.logger.error(f"Fallback genre matching failed: {e}")
            return False

    def _artist_similarity_match(self, candidate: Dict[str, Any], target_artists: List[str]) -> bool:
        """Check if candidate is similar to target artists."""
        candidate_artist = candidate.get('artist', '').lower()
        
        # Direct artist match
        for target in target_artists:
            if target.lower() in candidate_artist or candidate_artist in target.lower():
                return True
        
        # For now, accept all candidates that pass genre filtering
        # In a full implementation, this would check artist similarity scores
        return True 