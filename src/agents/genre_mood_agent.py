"""
GenreMoodAgent for BeatDebate Multi-Agent Music Recommendation System

Enhanced advocate agent specializing in genre and mood-based music discovery
using Last.fm API with 100→20 candidate generation and quality scoring.
"""

import asyncio
from typing import Dict, List, Any, Optional
import structlog

from .base_agent import BaseAgent
from ..models.agent_models import (
    MusicRecommenderState, 
    AgentConfig
)
from ..models.recommendation_models import TrackRecommendation
from ..api.lastfm_client import LastFmClient
from .enhanced_candidate_generator import EnhancedCandidateGenerator
from .quality_scorer import ComprehensiveQualityScorer

logger = structlog.get_logger(__name__)


class GenreMoodAgent(BaseAgent):
    """
    Advocate agent for genre and mood-based music recommendations.
    
    Specializes in:
    - Tag-based music discovery using Last.fm
    - Mood and energy level matching
    - Genre-specific search strategies
    - Popular and trending track identification
    """
    
    def __init__(self, config: AgentConfig, lastfm_client: LastFmClient, 
                 gemini_client=None):
        """
        Initialize Enhanced GenreMoodAgent with candidate generation and 
        quality scoring.
        
        Args:
            config: Agent configuration
            lastfm_client: Last.fm API client for music data
            gemini_client: Gemini LLM client for reasoning (optional)
        """
        super().__init__(config)
        # Store client configuration instead of client instance
        self.lastfm_api_key = lastfm_client.api_key
        self.lastfm_rate_limit = lastfm_client.rate_limiter.calls_per_second
        self.llm_client = gemini_client
        
        # Enhanced components for 100→20 pipeline
        self.candidate_generator = EnhancedCandidateGenerator(lastfm_client)
        self.quality_scorer = ComprehensiveQualityScorer()
        
        # Quality filtering thresholds
        self.quality_threshold = 0.6  # Minimum quality score
        self.target_recommendations = 20  # Final recommendation count
        
        # Mood and energy mappings
        self.mood_tag_mappings = self._initialize_mood_mappings()
        self.energy_level_mappings = self._initialize_energy_mappings()
        self.genre_tag_mappings = self._initialize_genre_mappings()
        
        self.logger.info(
            "Enhanced GenreMoodAgent initialized with 100→20 pipeline"
        )
    
    async def process(self, state: MusicRecommenderState) -> MusicRecommenderState:
        """
        Generate enhanced genre and mood-based music recommendations using
        100→20 candidate generation and quality scoring pipeline.
        
        Args:
            state: Current workflow state with entities and intent analysis
            
        Returns:
            Updated state with high-quality genre/mood recommendations
        """
        self.add_reasoning_step(
            "Starting enhanced genre/mood discovery with 100→20 pipeline"
        )
        
        try:
            # Extract entities and intent from PlannerAgent
            entities = state.entities or {}
            intent_analysis = state.intent_analysis or {}
            
            self.add_reasoning_step(
                f"Using entities: {len(entities)} categories, "
                f"intent: {intent_analysis.get('primary_intent', 'discovery')}"
            )
            
            # Step 1: Generate 100 candidate tracks from multiple sources
            candidate_pool = await self.candidate_generator.generate_candidate_pool(
                entities, intent_analysis, agent_type="genre_mood"
            )
            self.add_reasoning_step(
                f"Generated {len(candidate_pool)} candidates from multiple sources"
            )
            
            # Step 2: Apply comprehensive quality scoring to all candidates
            scored_candidates = await self._score_all_candidates(
                candidate_pool, entities, intent_analysis
            )
            self.add_reasoning_step(
                f"Quality scored {len(scored_candidates)} candidates"
            )
            
            # Step 3: Filter by quality threshold and apply advanced filtering
            high_quality_tracks = await self._apply_advanced_filtering(
                scored_candidates, entities, intent_analysis
            )
            self.add_reasoning_step(
                f"Filtered to {len(high_quality_tracks)} high-quality tracks"
            )
            
            # Step 4: Select top 20 with diversity and create recommendations
            final_tracks = high_quality_tracks[:self.target_recommendations]
            recommendations = await self._create_enhanced_recommendations(
                final_tracks, entities, intent_analysis
            )
            
            # Update state with enhanced recommendations
            state.genre_mood_recommendations = [
                rec.model_dump() for rec in recommendations
            ]
            state.reasoning_log.append(
                f"Enhanced GenreMoodAgent: Generated {len(recommendations)} "
                f"high-quality recommendations using 100→20 pipeline"
            )
            
            self.logger.info(
                "Enhanced genre/mood recommendations completed",
                recommendation_count=len(recommendations),
                candidate_pool_size=len(candidate_pool),
                quality_filtered=len(high_quality_tracks),
                primary_intent=intent_analysis.get('primary_intent', 'discovery')
            )
            
            return state
            
        except Exception as e:
            self.logger.error("Enhanced genre/mood recommendation failed", 
                            error=str(e))
            state.reasoning_log.append(f"Enhanced GenreMoodAgent ERROR: {str(e)}")
            return state
    
    async def _analyze_mood_requirements(
        self, user_query: str, strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze mood and energy requirements from query and strategy.
        
        Args:
            user_query: Original user query
            strategy: Strategy from PlannerAgent
            
        Returns:
            Mood analysis with primary mood, energy level, and context
        """
        # Extract from strategy if available
        mood_priority = strategy.get('mood_priority', 'general')
        energy_level = strategy.get('energy_level', 'medium')
        search_tags = strategy.get('search_tags', [])
        
        # Analyze query for mood indicators
        query_lower = user_query.lower()
        detected_moods = []
        
        for mood, keywords in self.mood_tag_mappings.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_moods.append(mood)
        
        # Determine primary mood
        if mood_priority != 'general' and mood_priority in self.mood_tag_mappings:
            primary_mood = mood_priority
        elif detected_moods:
            primary_mood = detected_moods[0]
        else:
            primary_mood = 'chill'  # default
        
        # Analyze energy level
        if energy_level == 'high' or any(word in query_lower for word in ['energetic', 'pump', 'workout']):
            energy = 'high'
        elif energy_level == 'low' or any(word in query_lower for word in ['calm', 'relax', 'chill']):
            energy = 'low'
        else:
            energy = 'medium'
        
        return {
            'primary_mood': primary_mood,
            'secondary_moods': detected_moods[1:3] if len(detected_moods) > 1 else [],
            'energy_level': energy,
            'context_tags': search_tags,
            'mood_confidence': 0.8 if detected_moods else 0.6
        }
    
    def _generate_search_tags(
        self, strategy: Dict[str, Any], mood_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate Last.fm search tags based on strategy and mood analysis.
        
        Args:
            strategy: Agent strategy from planner
            mood_analysis: Mood analysis results
            
        Returns:
            List of search tags for Last.fm API
        """
        tags = []
        
        # Add primary mood tags
        primary_mood = mood_analysis['primary_mood']
        if primary_mood in self.mood_tag_mappings:
            tags.extend(self.mood_tag_mappings[primary_mood][:2])
        
        # Add energy level tags
        energy_level = mood_analysis['energy_level']
        if energy_level in self.energy_level_mappings:
            tags.extend(self.energy_level_mappings[energy_level][:2])
        
        # Add genre focus areas from strategy
        focus_areas = strategy.get('focus_areas', [])
        for area in focus_areas[:2]:  # Limit to 2 genre areas
            if area in self.genre_tag_mappings:
                tags.extend(self.genre_tag_mappings[area][:2])
            else:
                tags.append(area)  # Use as-is if not in mappings
        
        # Add context tags from strategy
        context_tags = strategy.get('search_tags', [])
        tags.extend(context_tags[:2])
        
        # Add secondary moods
        for mood in mood_analysis.get('secondary_moods', [])[:1]:
            if mood in self.mood_tag_mappings:
                tags.extend(self.mood_tag_mappings[mood][:1])
        
        # Remove duplicates while preserving order
        unique_tags = []
        for tag in tags:
            if tag not in unique_tags:
                unique_tags.append(tag)
        
        return unique_tags[:8]  # Limit to 8 tags for focused search
    
    def _track_metadata_to_dict(self, track_metadata) -> Dict[str, Any]:
        """Convert TrackMetadata object to dictionary for compatibility."""
        # Handle both TrackMetadata objects and dictionaries (for testing)
        if isinstance(track_metadata, dict):
            # Already a dictionary, just ensure consistent structure
            return {
                'name': track_metadata.get('name'),
                'artist': track_metadata.get('artist'),
                'url': track_metadata.get('url'),
                'listeners': track_metadata.get('listeners'),
                'playcount': track_metadata.get('playcount'),
                'mbid': track_metadata.get('mbid'),
                'tags': track_metadata.get('tags', []),
                'album': track_metadata.get('album', {'title': None})
            }
        else:
            # TrackMetadata object
            return {
                'name': track_metadata.name,
                'artist': track_metadata.artist,
                'url': track_metadata.url,
                'listeners': track_metadata.listeners,
                'playcount': track_metadata.playcount,
                'mbid': track_metadata.mbid,
                'tags': track_metadata.tags,
                'album': {'title': None}  # LastFm doesn't provide album in search
            }
    
    async def _search_tracks_by_tags(self, search_tags: List[str]) -> List[Dict[str, Any]]:
        """
        Search for tracks using multiple tag combinations.
        
        Args:
            search_tags: List of tags to search with
            
        Returns:
            List of candidate tracks from Last.fm
        """
        from ..api.lastfm_client import LastFmClient
        
        all_tracks = []
        
        # Search with primary tag combinations
        primary_combinations = [
            search_tags[:2],  # First 2 tags
            search_tags[1:3] if len(search_tags) > 2 else search_tags[:2],  # Next 2 tags
            [search_tags[0]] if search_tags else ['indie']  # Single primary tag
        ]
        
        # Create a new Last.fm client for this search session
        async with LastFmClient(
            api_key=self.lastfm_api_key,
            rate_limit=self.lastfm_rate_limit
        ) as client:
            for tag_combo in primary_combinations:
                if not tag_combo:
                    continue
                    
                try:
                    # Search by tag
                    tag_query = ' '.join(tag_combo)
                    track_metadata_list = await client.search_tracks(tag_query, limit=20)
                    
                    if track_metadata_list:
                        # Convert to dictionaries and add search context
                        for track_metadata in track_metadata_list:
                            # Handle mocked track metadata in tests
                            track = self._track_metadata_to_dict(track_metadata)
                            if track:
                                track['search_tags'] = tag_combo.copy()
                                track['search_type'] = 'tag_based'
                                all_tracks.append(track)
                        
                        self.logger.debug(
                            "Tag search completed",
                            tags=tag_combo,
                            track_count=len(track_metadata_list)
                        )
                    
                    # Small delay to respect rate limits
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(
                        "Track search failed",
                        query=tag_query,
                        error=str(e)
                    )
                    continue
        
        # Remove duplicates based on artist + title
        unique_tracks = []
        seen_tracks = set()
        
        for track in all_tracks:
            track_key = f"{track.get('artist', '').lower()}_{track.get('name', '').lower()}"
            if track_key not in seen_tracks:
                seen_tracks.add(track_key)
                unique_tracks.append(track)
        
        return unique_tracks[:50]  # Limit to 50 candidates
    
    async def _score_all_candidates(
        self, 
        candidate_pool: List[Dict[str, Any]], 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply comprehensive quality scoring to all candidate tracks.
        
        Args:
            candidate_pool: List of candidate tracks from generator
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent
            
        Returns:
            List of tracks with quality scores and breakdowns
        """
        scored_candidates = []
        
        for track in candidate_pool:
            try:
                # Calculate comprehensive quality score
                quality_result = await self.quality_scorer.calculate_track_quality(
                    track, entities, intent_analysis
                )
                
                # Add quality information to track
                track['quality_score'] = quality_result['total_quality_score']
                track['quality_breakdown'] = quality_result['quality_breakdown']
                track['quality_tier'] = quality_result['quality_tier']
                
                scored_candidates.append(track)
                
            except Exception as e:
                self.logger.warning(
                    "Quality scoring failed for track",
                    track=f"{track.get('artist', 'Unknown')} - {track.get('name', 'Unknown')}",
                    error=str(e)
                )
                # Add track with default quality score
                track['quality_score'] = 0.5
                track['quality_breakdown'] = {}
                track['quality_tier'] = 'medium'
                scored_candidates.append(track)
        
        return scored_candidates
    
    async def _apply_advanced_filtering(
        self, 
        scored_candidates: List[Dict[str, Any]], 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply advanced filtering to ensure quality and diversity.
        
        Args:
            scored_candidates: Tracks with quality scores
            entities: Extracted entities
            intent_analysis: Intent analysis
            
        Returns:
            Filtered and ranked high-quality tracks
        """
        # Step 1: Filter by quality threshold
        high_quality_tracks = [
            track for track in scored_candidates 
            if track['quality_score'] >= self.quality_threshold
        ]
        
        # Step 2: Sort by quality score
        high_quality_tracks.sort(
            key=lambda x: x['quality_score'], 
            reverse=True
        )
        
        # Step 3: Ensure source diversity
        diverse_tracks = self._ensure_source_diversity(high_quality_tracks)
        
        # Step 4: Remove tracks that are too similar
        final_tracks = self._remove_similar_tracks(diverse_tracks)
        
        return final_tracks
    
    def _ensure_source_diversity(
        self, tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ensure balanced representation from different sources."""
        source_counts = {}
        diverse_tracks = []
        max_per_source = 8  # Maximum tracks per source
        
        for track in tracks:
            source = track.get('source', 'unknown')
            current_count = source_counts.get(source, 0)
            
            if current_count < max_per_source:
                diverse_tracks.append(track)
                source_counts[source] = current_count + 1
        
        return diverse_tracks
    
    def _remove_similar_tracks(
        self, tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove tracks that are too similar to each other."""
        unique_tracks = []
        seen_artists = set()
        
        for track in tracks:
            artist = track.get('artist', '').lower().strip()
            
            # Allow max 2 tracks per artist
            artist_count = sum(1 for t in unique_tracks if t.get('artist', '').lower().strip() == artist)
            
            if artist_count < 2:
                unique_tracks.append(track)
                seen_artists.add(artist)
        
        return unique_tracks
    
    async def _create_enhanced_recommendations(
        self,
        tracks: List[Dict[str, Any]],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> List[TrackRecommendation]:
        """
        Create enhanced TrackRecommendation objects with quality reasoning.
        
        Args:
            tracks: High-quality filtered tracks
            entities: Extracted entities
            intent_analysis: Intent analysis
            
        Returns:
            List of enhanced TrackRecommendation objects
        """
        recommendations = []
        
        for i, track in enumerate(tracks):
            try:
                # Generate enhanced reasoning that includes quality factors
                reasoning = self._generate_enhanced_reasoning(
                    track, entities, intent_analysis, i + 1
                )
                
                # Extract genres and tags
                genres = self._extract_genres_from_entities(track, entities)
                tags = self._extract_tags_from_entities(track, entities, intent_analysis)
                
                # Create recommendation with quality information
                recommendation = TrackRecommendation(
                    rank=i + 1,
                    artist=track.get('artist', 'Unknown Artist'),
                    title=track.get('name', 'Unknown Title'),
                    id=track.get('mbid') or f"enhanced_{track.get('artist', 'unknown')}_{track.get('name', 'unknown')}".replace(' ', '_').lower(),
                    source="enhanced_lastfm",
                    track_url=track.get('url', ''),
                    genres=genres,
                    moods=tags,
                    quality_score=track.get('quality_score', 0.5),
                    novelty_score=0.3,  # GenreMoodAgent focuses on established genres
                    concentration_friendliness_score=self._calculate_enhanced_confidence(track),
                    confidence=self._calculate_enhanced_confidence(track),
                    advocate_source_agent="GenreMoodAgent",  # CRITICAL: Set the source agent
                    additional_scores={
                        'source': track.get('source', 'unknown'),
                        'quality_tier': track.get('quality_tier', 'medium'),
                        'listeners': track.get('listeners', 0),
                        'playcount': track.get('playcount', 0)
                    },
                    raw_source_data={
                        'quality_breakdown': track.get('quality_breakdown', {}),
                        'agent': 'enhanced_genre_mood'
                    }
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                self.logger.warning(
                    "Enhanced recommendation creation failed",
                    track=f"{track.get('artist', 'Unknown')} - {track.get('name', 'Unknown')}",
                    error=str(e)
                )
                continue
        
        return recommendations
    
    def _generate_enhanced_reasoning(
        self,
        track: Dict[str, Any],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        rank: int
    ) -> str:
        """Generate enhanced reasoning that includes quality factors."""
        quality_score = track.get('quality_score', 0.5)
        quality_tier = track.get('quality_tier', 'medium')
        source = track.get('source', 'unknown')
        
        # Base reasoning
        reasoning_parts = [
            f"Ranked #{rank} with {quality_tier} quality (score: {quality_score:.2f})"
        ]
        
        # Add source information
        source_descriptions = {
            'primary_search': 'found through targeted genre/mood search',
            'similar_artists': 'discovered via artist similarity',
            'genre_exploration': 'found through genre exploration',
            'underground_gems': 'discovered as underground gem'
        }
        
        if source in source_descriptions:
            reasoning_parts.append(source_descriptions[source])
        
        # Add quality breakdown if available
        quality_breakdown = track.get('quality_breakdown', {})
        if quality_breakdown:
            strong_factors = [
                factor for factor, score in quality_breakdown.items() 
                if score > 0.7
            ]
            if strong_factors:
                reasoning_parts.append(
                    f"Strong in: {', '.join(strong_factors[:2])}"
                )
        
        # Add intent alignment
        primary_intent = intent_analysis.get('primary_intent', 'discovery')
        reasoning_parts.append(f"Matches {primary_intent} intent")
        
        return ". ".join(reasoning_parts) + "."
    
    def _extract_genres_from_entities(
        self, track: Dict[str, Any], entities: Dict[str, Any]
    ) -> List[str]:
        """Extract genres based on entities and track source."""
        genres = []
        
        # Add genres from entities
        musical_entities = entities.get("musical_entities", {})
        entity_genres = musical_entities.get("genres", {}).get("primary", [])
        genres.extend(entity_genres)
        
        # Add genre from search context
        search_term = track.get('search_term', '')
        exploration_tag = track.get('exploration_tag', '')
        
        if search_term:
            genres.append(search_term)
        if exploration_tag:
            genres.append(exploration_tag)
        
        # Remove duplicates and return
        return list(set(genres))[:3]  # Limit to 3 genres
    
    def _extract_tags_from_entities(
        self, 
        track: Dict[str, Any], 
        entities: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract tags based on entities and intent."""
        tags = []
        
        # Add mood tags
        contextual_entities = entities.get("contextual_entities", {})
        moods = contextual_entities.get("moods", {})
        for mood_category in moods.values():
            tags.extend(mood_category)
        
        # Add activity tags
        activities = contextual_entities.get("activities", {})
        for activity_category in activities.values():
            tags.extend(activity_category)
        
        # Add intent-based tags
        primary_intent = intent_analysis.get('primary_intent', '')
        if primary_intent:
            tags.append(primary_intent)
        
        # Add quality tier as tag
        quality_tier = track.get('quality_tier', '')
        if quality_tier:
            tags.append(f"{quality_tier}_quality")
        
        # Remove duplicates and return
        return list(set(tags))[:5]  # Limit to 5 tags
    
    def _calculate_enhanced_confidence(self, track: Dict[str, Any]) -> float:
        """Calculate confidence based on quality score and metadata."""
        quality_score = track.get('quality_score', 0.5)
        
        # Base confidence from quality score
        confidence = quality_score
        
        # Boost confidence for tracks with good metadata
        if track.get('url'):
            confidence += 0.1
        if track.get('listeners', 0) > 1000:
            confidence += 0.1
        if track.get('source') in ['primary_search', 'similar_artists']:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    async def _filter_and_rank_tracks(
        self, 
        tracks: List[Dict[str, Any]], 
        mood_analysis: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter and rank tracks based on mood/genre fit.
        
        Args:
            tracks: Candidate tracks from search
            mood_analysis: Mood analysis results
            strategy: Agent strategy
            
        Returns:
            Filtered and ranked tracks
        """
        if not tracks:
            return []
        
        scored_tracks = []
        
        for track in tracks:
            try:
                # Calculate mood/genre fit score
                score = self._calculate_track_score(track, mood_analysis, strategy)
                
                if score > 0.3:  # Minimum threshold
                    track['genre_mood_score'] = score
                    track['mood_analysis'] = mood_analysis
                    scored_tracks.append(track)
                    
            except Exception as e:
                self.logger.warning(
                    "Track scoring failed",
                    track=f"{track.get('artist', 'Unknown')} - {track.get('name', 'Unknown')}",
                    error=str(e)
                )
                continue
        
        # Sort by score (descending)
        scored_tracks.sort(key=lambda x: x['genre_mood_score'], reverse=True)
        
        return scored_tracks[:15]  # Return top 15 tracks
    
    def _calculate_track_score(
        self, 
        track: Dict[str, Any], 
        mood_analysis: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> float:
        """
        Calculate relevance score for a track based on mood/genre fit.
        
        Args:
            track: Track data from Last.fm
            mood_analysis: Mood analysis results
            strategy: Agent strategy
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        # Base score for having basic metadata
        if track.get('artist') and track.get('name'):
            score += 0.2
        
        # Search tag relevance (30% weight)
        search_tags = track.get('search_tags', [])
        if search_tags:
            # Higher score if found with primary mood tags
            primary_mood_tags = self.mood_tag_mappings.get(mood_analysis['primary_mood'], [])
            tag_overlap = len(set(search_tags) & set(primary_mood_tags))
            score += (tag_overlap / max(len(primary_mood_tags), 1)) * 0.3
        
        # Listener count (popularity) - 20% weight
        listeners = track.get('listeners', 0)
        if isinstance(listeners, (int, str)):
            try:
                listener_count = int(listeners)
                # Normalize listener count (log scale)
                if listener_count > 0:
                    import math
                    normalized_listeners = min(math.log10(listener_count) / 6, 1.0)  # Max at 1M listeners
                    score += normalized_listeners * 0.2
            except (ValueError, TypeError):
                pass
        
        # URL availability (20% weight)
        if track.get('url'):
            score += 0.2
        
        # Artist name quality (10% weight)
        artist = track.get('artist', '')
        if artist and len(artist) > 1 and not artist.lower().startswith('unknown'):
            score += 0.1
        
        # Track name quality (10% weight)
        name = track.get('name', '')
        if name and len(name) > 1:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _create_recommendations(
        self,
        tracks: List[Dict[str, Any]],
        mood_analysis: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> List[TrackRecommendation]:
        """
        Create TrackRecommendation objects with reasoning.
        
        Args:
            tracks: Filtered and ranked tracks
            mood_analysis: Mood analysis results
            strategy: Agent strategy
            
        Returns:
            List of TrackRecommendation objects
        """
        recommendations = []
        
        for i, track in enumerate(tracks[:5]):  # Top 5 tracks
            try:
                # Generate reasoning for this recommendation
                reasoning = self._generate_track_reasoning(track, mood_analysis, strategy, i + 1)
                
                # Extract genres and tags
                genres = self._extract_genres(track, strategy)
                tags = self._extract_tags(track, mood_analysis)
                
                recommendation = TrackRecommendation(
                    title=track.get('name', 'Unknown Title'),
                    artist=track.get('artist', 'Unknown Artist'),
                    id=track.get('mbid') or f"lastfm_{track.get('artist', 'unknown')}_{track.get('name', 'unknown')}".replace(' ', '_').lower(),
                    source="lastfm",
                    album_title=track.get('album', {}).get('title') if isinstance(track.get('album'), dict) else None,
                    track_url=track.get('url'),
                    genres=genres,
                    moods=tags,
                    # Add scoring fields for Judge Agent
                    quality_score=track.get('genre_mood_score', 0.5),
                    novelty_score=0.3,  # GenreMoodAgent focuses on established genres, lower novelty
                    concentration_friendliness_score=self._calculate_concentration_score(mood_analysis, track),
                    confidence=track.get('genre_mood_score', 0.5),
                    advocate_source_agent="GenreMoodAgent"
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                self.logger.warning(
                    "Failed to create recommendation",
                    track=f"{track.get('artist', 'Unknown')} - {track.get('name', 'Unknown')}",
                    error=str(e)
                )
                continue
        
        return recommendations
    
    def _generate_track_reasoning(
        self,
        track: Dict[str, Any],
        mood_analysis: Dict[str, Any],
        strategy: Dict[str, Any],
        rank: int
    ) -> str:
        """Generate reasoning chain for track recommendation."""
        reasoning_parts = [
            f"Recommendation #{rank} based on genre/mood analysis:",
            f"• Primary mood match: '{mood_analysis['primary_mood']}' with {mood_analysis['energy_level']} energy",
            f"• Found via tags: {', '.join(track.get('search_tags', []))}",
            f"• Artist: {track.get('artist', 'Unknown')}",
            f"• Track: {track.get('name', 'Unknown')}",
            f"• Relevance score: {track.get('genre_mood_score', 0.5):.2f}"
        ]
        
        # Add listener count if available
        listeners = track.get('listeners')
        if listeners:
            reasoning_parts.append(f"• Popularity: {listeners} listeners")
        
        # Add genre focus if available
        focus_areas = strategy.get('focus_areas', [])
        if focus_areas:
            reasoning_parts.append(f"• Genre focus: {', '.join(focus_areas)}")
        
        return '\n'.join(reasoning_parts)
    
    def _extract_genres(self, track: Dict[str, Any], strategy: Dict[str, Any]) -> List[str]:
        """Extract genre information from track and strategy."""
        genres = []
        
        # Add focus areas from strategy
        focus_areas = strategy.get('focus_areas', [])
        genres.extend(focus_areas)
        
        # Add search tags that look like genres
        search_tags = track.get('search_tags', [])
        for tag in search_tags:
            if tag in self.genre_tag_mappings:
                genres.append(tag)
        
        return list(set(genres))[:3]  # Unique genres, max 3
    
    def _extract_tags(self, track: Dict[str, Any], mood_analysis: Dict[str, Any]) -> List[str]:
        """Extract mood and style tags."""
        tags = []
        
        # Add primary mood
        tags.append(mood_analysis['primary_mood'])
        
        # Add energy level
        tags.append(f"{mood_analysis['energy_level']}_energy")
        
        # Add search tags
        search_tags = track.get('search_tags', [])
        tags.extend(search_tags)
        
        return list(set(tags))[:5]  # Unique tags, max 5
    
    def _initialize_mood_mappings(self) -> Dict[str, List[str]]:
        """Initialize mood to Last.fm tag mappings."""
        return {
            'happy': ['happy', 'uplifting', 'cheerful', 'joyful', 'positive'],
            'sad': ['sad', 'melancholy', 'melancholic', 'emotional', 'depressing'],
            'energetic': ['energetic', 'upbeat', 'pump up', 'motivational', 'driving'],
            'chill': ['chill', 'relaxing', 'mellow', 'laid-back', 'easy listening'],
            'focus': ['instrumental', 'ambient', 'concentration', 'study', 'background'],
            'romantic': ['romantic', 'love', 'intimate', 'passionate', 'sensual'],
            'nostalgic': ['nostalgic', 'retro', 'vintage', 'classic', 'throwback'],
            'aggressive': ['aggressive', 'intense', 'heavy', 'hard', 'powerful'],
            'peaceful': ['peaceful', 'calm', 'serene', 'tranquil', 'soothing'],
            'party': ['party', 'dance', 'club', 'celebration', 'fun']
        }
    
    def _initialize_energy_mappings(self) -> Dict[str, List[str]]:
        """Initialize energy level to tag mappings."""
        return {
            'low': ['calm', 'peaceful', 'quiet', 'soft', 'gentle'],
            'medium': ['moderate', 'balanced', 'steady', 'comfortable'],
            'high': ['energetic', 'intense', 'powerful', 'driving', 'upbeat']
        }
    
    def _initialize_genre_mappings(self) -> Dict[str, List[str]]:
        """Initialize genre to Last.fm tag mappings."""
        return {
            'rock': ['rock', 'alternative rock', 'indie rock', 'classic rock'],
            'electronic': ['electronic', 'electronica', 'ambient', 'techno', 'house'],
            'indie': ['indie', 'independent', 'indie pop', 'indie rock'],
            'pop': ['pop', 'pop rock', 'electropop', 'synth-pop'],
            'hip-hop': ['hip hop', 'rap', 'hip-hop', 'urban'],
            'jazz': ['jazz', 'smooth jazz', 'contemporary jazz', 'jazz fusion'],
            'classical': ['classical', 'orchestral', 'symphony', 'chamber music'],
            'folk': ['folk', 'acoustic', 'singer-songwriter', 'americana'],
            'metal': ['metal', 'heavy metal', 'progressive metal', 'alternative metal'],
            'reggae': ['reggae', 'dub', 'ska', 'dancehall'],
            'blues': ['blues', 'electric blues', 'chicago blues', 'delta blues'],
            'country': ['country', 'alt-country', 'bluegrass', 'americana']
        }
    
    async def _make_llm_call(self, prompt: str, system_prompt: str = None) -> str:
        """
        Make LLM call using Gemini client (if available).
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            LLM response
        """
        if not self.llm_client:
            raise RuntimeError("Gemini client not available")
        
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = self.llm_client.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            self.logger.error("Gemini API call failed", error=str(e))
            raise
    
    def _calculate_concentration_score(self, mood_analysis: Dict[str, Any], track: Dict[str, Any]) -> float:
        """Calculate concentration friendliness score for a track."""
        score = 0.5  # Base score
        
        # Higher score for calm/focus moods
        primary_mood = mood_analysis.get('primary_mood', '')
        if primary_mood in ['chill', 'focus', 'peaceful', 'calm']:
            score += 0.3
        elif primary_mood in ['energetic', 'party', 'aggressive']:
            score -= 0.2
            
        # Energy level consideration
        energy_level = mood_analysis.get('energy_level', 'medium')
        if energy_level == 'low':
            score += 0.2
        elif energy_level == 'high':
            score -= 0.1
            
        return max(0.0, min(1.0, score))
    
    def _extract_output_data(self, state: MusicRecommenderState) -> Dict[str, Any]:
        """Extract GenreMoodAgent output data."""
        return {
            "recommendations_generated": len(state.genre_mood_recommendations),
            "agent_type": "genre_mood_specialist"
        }
    
    def _calculate_confidence(self, state: MusicRecommenderState) -> float:
        """Calculate confidence in genre/mood recommendations."""
        rec_count = len(state.genre_mood_recommendations)
        
        if rec_count == 0:
            return 0.0
        elif rec_count >= 3:
            return 0.9
        elif rec_count >= 2:
            return 0.7
        else:
            return 0.5 