"""
Underground Detector for BeatDebate

Implements sophisticated underground music detection using quality metrics,
engagement analysis, and genre-specific thresholds.
"""

import asyncio
import math
from typing import Dict, List, Any, Optional, Tuple
import structlog
from collections import defaultdict
import random

logger = structlog.get_logger(__name__)


class UndergroundQualityAnalyzer:
    """
    Analyzes tracks to determine underground quality and authenticity.
    """
    
    def __init__(self):
        """Initialize underground quality analyzer."""
        self.logger = logger.bind(component="UndergroundQualityAnalyzer")
        
        # Quality thresholds for different underground tiers
        self.underground_tiers = {
            'deep_underground': {
                'max_listeners': 5000,
                'max_playcount': 50000,
                'min_engagement_ratio': 5.0,  # plays per listener
                'quality_weight': 1.0
            },
            'underground': {
                'max_listeners': 25000,
                'max_playcount': 250000,
                'min_engagement_ratio': 3.0,
                'quality_weight': 0.8
            },
            'emerging': {
                'max_listeners': 100000,
                'max_playcount': 1000000,
                'min_engagement_ratio': 2.0,
                'quality_weight': 0.6
            }
        }
        
        # Genre-specific underground indicators
        self.genre_underground_indicators = {
            'experimental': ['experimental', 'avant-garde', 'noise', 'drone'],
            'indie': ['indie', 'independent', 'lo-fi', 'bedroom'],
            'electronic': ['ambient', 'idm', 'breakcore', 'glitch'],
            'folk': ['folk', 'acoustic', 'singer-songwriter', 'americana'],
            'metal': ['black metal', 'doom', 'sludge', 'post-metal'],
            'jazz': ['free jazz', 'contemporary jazz', 'fusion', 'avant-jazz']
        }
        
        self.logger.info("Underground Quality Analyzer initialized")
    
    def analyze_underground_potential(
        self, 
        track_data: Dict[str, Any],
        genre_context: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a track's underground potential and quality.
        
        Args:
            track_data: Track metadata including play counts
            genre_context: Genre context for specialized analysis
            
        Returns:
            Underground analysis with tier, score, and reasoning
        """
        try:
            listeners = int(track_data.get('listeners') or 0)
            playcount = int(track_data.get('playcount') or 0)
            
            # Calculate engagement ratio
            engagement_ratio = playcount / max(listeners, 1)
            
            # Determine underground tier
            underground_tier = self._determine_underground_tier(
                listeners, playcount, engagement_ratio
            )
            
            # Calculate underground score
            underground_score = self._calculate_underground_score(
                listeners, playcount, engagement_ratio, genre_context
            )
            
            # Analyze quality indicators
            quality_indicators = self._analyze_quality_indicators(
                track_data, genre_context
            )
            
            # Generate reasoning
            reasoning = self._generate_underground_reasoning(
                underground_tier, underground_score, quality_indicators,
                listeners, playcount, engagement_ratio
            )
            
            analysis = {
                'underground_tier': underground_tier,
                'underground_score': underground_score,
                'engagement_ratio': engagement_ratio,
                'quality_indicators': quality_indicators,
                'reasoning': reasoning,
                'is_underground': underground_tier is not None,
                'listeners': listeners,
                'playcount': playcount
            }
            
            self.logger.debug(
                "Underground analysis completed",
                track=f"{track_data.get('artist', 'Unknown')} - {track_data.get('name', 'Unknown')}",
                tier=underground_tier,
                score=underground_score,
                engagement_ratio=engagement_ratio
            )
            
            return analysis
            
        except Exception as e:
            self.logger.warning(
                "Underground analysis failed",
                track=f"{track_data.get('artist', 'Unknown')} - {track_data.get('name', 'Unknown')}",
                error=str(e)
            )
            return {
                'underground_tier': None,
                'underground_score': 0.0,
                'engagement_ratio': 0.0,
                'quality_indicators': {},
                'reasoning': 'Analysis failed',
                'is_underground': False,
                'listeners': 0,
                'playcount': 0
            }
    
    def _determine_underground_tier(
        self, 
        listeners: int, 
        playcount: int, 
        engagement_ratio: float
    ) -> Optional[str]:
        """Determine which underground tier a track belongs to."""
        for tier_name, thresholds in self.underground_tiers.items():
            if (listeners <= thresholds['max_listeners'] and
                playcount <= thresholds['max_playcount'] and
                engagement_ratio >= thresholds['min_engagement_ratio']):
                return tier_name
        
        return None  # Not underground
    
    def _calculate_underground_score(
        self,
        listeners: int,
        playcount: int,
        engagement_ratio: float,
        genre_context: List[str] = None
    ) -> float:
        """Calculate a comprehensive underground score."""
        score = 0.0
        
        # Listener count score (inverse relationship)
        if listeners > 0:
            listener_score = max(0, 1.0 - math.log10(listeners) / 6.0)  # 1M listeners = 0
            score += listener_score * 0.4
        else:
            score += 0.4  # No listeners = maximum underground
        
        # Engagement ratio score
        engagement_score = min(1.0, engagement_ratio / 10.0)  # 10 plays per listener = 1.0
        score += engagement_score * 0.3
        
        # Genre-specific underground bonus
        genre_bonus = self._calculate_genre_underground_bonus(genre_context)
        score += genre_bonus * 0.2
        
        # Playcount consistency score
        if listeners > 0 and playcount > 0:
            expected_plays = listeners * 5  # Expected average plays per listener
            consistency = 1.0 - abs(playcount - expected_plays) / max(expected_plays, playcount)
            score += consistency * 0.1
        
        return min(1.0, max(0.0, score))
    
    def _calculate_genre_underground_bonus(self, genre_context: List[str] = None) -> float:
        """Calculate bonus score for underground-associated genres."""
        if not genre_context:
            return 0.0
        
        bonus = 0.0
        genre_context_lower = [genre.lower() for genre in genre_context]
        
        for genre_category, indicators in self.genre_underground_indicators.items():
            for indicator in indicators:
                if any(indicator in genre for genre in genre_context_lower):
                    bonus += 0.2
                    break  # Only count each category once
        
        return min(1.0, bonus)
    
    def _analyze_quality_indicators(
        self, 
        track_data: Dict[str, Any], 
        genre_context: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze various quality indicators for the track."""
        indicators = {}
        
        # URL availability (indicates proper metadata)
        indicators['has_url'] = bool(track_data.get('url'))
        
        # MBID availability (indicates proper cataloging)
        indicators['has_mbid'] = bool(track_data.get('mbid'))
        
        # Tag richness
        tags = track_data.get('tags', [])
        indicators['tag_count'] = len(tags) if isinstance(tags, list) else 0
        indicators['has_rich_tags'] = indicators['tag_count'] >= 3
        
        # Artist name quality (not generic/unknown)
        artist = track_data.get('artist', '').lower()
        indicators['quality_artist_name'] = (
            len(artist) > 2 and 
            not artist.startswith('unknown') and
            not artist.startswith('various')
        )
        
        # Track name quality
        track_name = track_data.get('name', '').lower()
        indicators['quality_track_name'] = (
            len(track_name) > 1 and
            not track_name.startswith('unknown') and
            not track_name.startswith('untitled')
        )
        
        # Calculate overall quality score
        quality_factors = [
            indicators['has_url'],
            indicators['has_mbid'],
            indicators['has_rich_tags'],
            indicators['quality_artist_name'],
            indicators['quality_track_name']
        ]
        indicators['overall_quality'] = sum(quality_factors) / len(quality_factors)
        
        return indicators
    
    def _generate_underground_reasoning(
        self,
        tier: Optional[str],
        score: float,
        quality_indicators: Dict[str, Any],
        listeners: int,
        playcount: int,
        engagement_ratio: float
    ) -> str:
        """Generate human-readable reasoning for underground classification."""
        if not tier:
            return f"Not classified as underground (listeners: {listeners:,}, plays: {playcount:,})"
        
        reasoning_parts = [
            f"Classified as {tier.replace('_', ' ')} (score: {score:.2f})",
            f"Listeners: {listeners:,}, Plays: {playcount:,}",
            f"Engagement: {engagement_ratio:.1f} plays per listener"
        ]
        
        # Add quality insights
        if quality_indicators.get('overall_quality', 0) > 0.7:
            reasoning_parts.append("High metadata quality")
        elif quality_indicators.get('overall_quality', 0) < 0.3:
            reasoning_parts.append("Limited metadata available")
        
        if quality_indicators.get('has_rich_tags'):
            reasoning_parts.append(f"Well-tagged ({quality_indicators['tag_count']} tags)")
        
        return ". ".join(reasoning_parts) + "."


class UndergroundDetector:
    """
    Main underground detection system that combines quality analysis
    with genre-specific search strategies.
    """
    
    def __init__(self, lastfm_client):
        """
        Initialize underground detector.
        
        Args:
            lastfm_client: Last.fm API client for music data
        """
        self.lastfm_api_key = lastfm_client.api_key
        self.lastfm_rate_limit = lastfm_client.rate_limiter.calls_per_second
        self.logger = logger.bind(component="UndergroundDetector")
        
        # Quality analyzer
        self.quality_analyzer = UndergroundQualityAnalyzer()
        
        # Underground search strategies
        self.search_strategies = {
            'genre_deep_dive': self._search_genre_deep_dive,
            'tag_exploration': self._search_tag_exploration,
            'artist_discovery': self._search_artist_discovery,
            'random_exploration': self._search_random_exploration
        }
        
        # Underground search terms by genre
        self.underground_search_terms = {
            'experimental': [
                'experimental', 'avant-garde', 'noise', 'drone', 'ambient',
                'field recording', 'sound art', 'electroacoustic'
            ],
            'indie': [
                'indie', 'independent', 'lo-fi', 'bedroom pop', 'dream pop',
                'shoegaze', 'indie folk', 'indie rock'
            ],
            'electronic': [
                'idm', 'breakcore', 'glitch', 'microsound', 'lowercase',
                'clicks and cuts', 'minimal techno', 'dub techno'
            ],
            'folk': [
                'folk', 'acoustic', 'singer-songwriter', 'americana', 'alt-country',
                'freak folk', 'new weird america', 'psych folk'
            ],
            'metal': [
                'black metal', 'doom metal', 'sludge', 'post-metal', 'drone metal',
                'funeral doom', 'atmospheric black metal'
            ],
            'jazz': [
                'free jazz', 'avant-jazz', 'contemporary jazz', 'nu jazz',
                'jazz fusion', 'spiritual jazz', 'european jazz'
            ]
        }
        
        self.logger.info("Underground Detector initialized")
    
    async def detect_underground_artists(
        self, 
        genres: List[str], 
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        target_candidates: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Detect underground artists and tracks based on genres and intent.
        
        Args:
            genres: Target genres for underground detection
            entities: Extracted entities from PlannerAgent
            intent_analysis: Intent analysis from PlannerAgent
            target_candidates: Target number of underground candidates
            
        Returns:
            List of underground track candidates with analysis
        """
        self.logger.info(
            "Starting underground detection",
            genres=genres,
            target_candidates=target_candidates
        )
        
        all_candidates = []
        
        # Create Last.fm client for this detection session
        from ..api.lastfm_client import LastFmClient
        
        async with LastFmClient(
            api_key=self.lastfm_api_key,
            rate_limit=self.lastfm_rate_limit
        ) as client:
            
            # Apply different search strategies
            for strategy_name, strategy_func in self.search_strategies.items():
                try:
                    strategy_candidates = await strategy_func(
                        genres, entities, intent_analysis, client
                    )
                    
                    # Add strategy information
                    for candidate in strategy_candidates:
                        candidate['detection_strategy'] = strategy_name
                        candidate['source'] = 'underground_detection'
                    
                    all_candidates.extend(strategy_candidates)
                    
                    # Rate limiting between strategies
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    self.logger.warning(
                        "Underground detection strategy failed",
                        strategy=strategy_name,
                        error=str(e)
                    )
                    continue
        
        # Analyze all candidates for underground quality
        analyzed_candidates = []
        for candidate in all_candidates:
            underground_analysis = self.quality_analyzer.analyze_underground_potential(
                candidate, genres
            )
            
            # Only keep tracks that qualify as underground
            if underground_analysis['is_underground']:
                candidate.update(underground_analysis)
                analyzed_candidates.append(candidate)
        
        # Sort by underground score and limit to target
        analyzed_candidates.sort(
            key=lambda x: x.get('underground_score', 0), 
            reverse=True
        )
        
        final_candidates = analyzed_candidates[:target_candidates]
        
        self.logger.info(
            "Underground detection completed",
            total_found=len(all_candidates),
            underground_qualified=len(analyzed_candidates),
            final_candidates=len(final_candidates)
        )
        
        return final_candidates
    
    async def _search_genre_deep_dive(
        self,
        genres: List[str],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        client
    ) -> List[Dict[str, Any]]:
        """Search for underground tracks using genre-specific terms."""
        candidates = []
        
        for genre in genres[:3]:  # Limit to 3 genres
            genre_lower = genre.lower()
            
            # Find matching underground search terms
            search_terms = []
            for genre_category, terms in self.underground_search_terms.items():
                if genre_category in genre_lower or any(term in genre_lower for term in terms):
                    search_terms.extend(terms[:3])  # Take first 3 terms
                    break
            
            # Fallback to genre name if no specific terms found
            if not search_terms:
                search_terms = [genre]
            
            # Search with each term
            for term in search_terms[:5]:  # Limit to 5 terms per genre
                try:
                    tracks = await client.search_tracks(query=term, limit=8)
                    
                    for track_metadata in tracks:
                        track = {
                            'name': track_metadata.name,
                            'artist': track_metadata.artist,
                            'url': track_metadata.url,
                            'listeners': track_metadata.listeners,
                            'playcount': track_metadata.playcount,
                            'mbid': track_metadata.mbid,
                            'tags': track_metadata.tags,
                            'search_term': term,
                            'target_genre': genre
                        }
                        candidates.append(track)
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.warning(
                        "Genre deep dive search failed",
                        term=term,
                        error=str(e)
                    )
                    continue
        
        return candidates
    
    async def _search_tag_exploration(
        self,
        genres: List[str],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        client
    ) -> List[Dict[str, Any]]:
        """Search using tag-based exploration for underground music."""
        candidates = []
        
        # Extract mood and activity tags from entities
        contextual_entities = entities.get("contextual_entities", {})
        mood_tags = []
        for mood_category in contextual_entities.get("moods", {}).values():
            mood_tags.extend(mood_category)
        
        activity_tags = []
        for activity_category in contextual_entities.get("activities", {}).values():
            activity_tags.extend(activity_category)
        
        # Combine with underground-specific modifiers
        underground_modifiers = ['underground', 'hidden', 'obscure', 'rare', 'unknown']
        
        # Create search combinations
        search_combinations = []
        
        # Genre + underground modifier
        for genre in genres[:2]:
            for modifier in underground_modifiers[:2]:
                search_combinations.append(f"{modifier} {genre}")
        
        # Mood + underground modifier
        for mood in mood_tags[:2]:
            for modifier in underground_modifiers[:2]:
                search_combinations.append(f"{modifier} {mood}")
        
        # Search with combinations
        for combination in search_combinations[:10]:  # Limit to 10 combinations
            try:
                tracks = await client.search_tracks(query=combination, limit=5)
                
                for track_metadata in tracks:
                    track = {
                        'name': track_metadata.name,
                        'artist': track_metadata.artist,
                        'url': track_metadata.url,
                        'listeners': track_metadata.listeners,
                        'playcount': track_metadata.playcount,
                        'mbid': track_metadata.mbid,
                        'tags': track_metadata.tags,
                        'search_combination': combination
                    }
                    candidates.append(track)
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.warning(
                    "Tag exploration search failed",
                    combination=combination,
                    error=str(e)
                )
                continue
        
        return candidates
    
    async def _search_artist_discovery(
        self,
        genres: List[str],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        client
    ) -> List[Dict[str, Any]]:
        """Discover underground artists through artist search."""
        candidates = []
        
        # Search for artists with underground-associated terms
        underground_artist_terms = [
            'experimental', 'ambient', 'drone', 'noise', 'lo-fi',
            'bedroom', 'indie', 'underground', 'unknown'
        ]
        
        for term in underground_artist_terms[:5]:  # Limit to 5 terms
            try:
                artists = await client.search_artists(query=term, limit=5)
                
                for artist_metadata in artists:
                    # Get tracks from this artist
                    try:
                        tracks = await client.get_artist_top_tracks(
                            artist=artist_metadata.name, 
                            limit=3
                        )
                        
                        for track_metadata in tracks:
                            track = {
                                'name': track_metadata.name,
                                'artist': track_metadata.artist,
                                'url': track_metadata.url,
                                'listeners': track_metadata.listeners,
                                'playcount': track_metadata.playcount,
                                'mbid': track_metadata.mbid,
                                'tags': track_metadata.tags,
                                'discovery_term': term
                            }
                            candidates.append(track)
                        
                        # Rate limiting
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        self.logger.warning(
                            "Failed to get tracks for underground artist",
                            artist=artist_metadata.name,
                            error=str(e)
                        )
                        continue
                
            except Exception as e:
                self.logger.warning(
                    "Artist discovery search failed",
                    term=term,
                    error=str(e)
                )
                continue
        
        return candidates
    
    async def _search_random_exploration(
        self,
        genres: List[str],
        entities: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        client
    ) -> List[Dict[str, Any]]:
        """Random exploration for serendipitous underground discovery."""
        candidates = []
        
        # Random search terms for exploration
        random_terms = [
            'experimental', 'ambient', 'drone', 'field recording',
            'lo-fi', 'bedroom', 'indie', 'underground', 'hidden',
            'obscure', 'rare', 'unknown', 'new', 'emerging'
        ]
        
        # Randomly select and search
        selected_terms = random.sample(random_terms, min(5, len(random_terms)))
        
        for term in selected_terms:
            try:
                tracks = await client.search_tracks(query=term, limit=4)
                
                for track_metadata in tracks:
                    track = {
                        'name': track_metadata.name,
                        'artist': track_metadata.artist,
                        'url': track_metadata.url,
                        'listeners': track_metadata.listeners,
                        'playcount': track_metadata.playcount,
                        'mbid': track_metadata.mbid,
                        'tags': track_metadata.tags,
                        'random_term': term
                    }
                    candidates.append(track)
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.warning(
                    "Random exploration search failed",
                    term=term,
                    error=str(e)
                )
                continue
        
        return candidates 