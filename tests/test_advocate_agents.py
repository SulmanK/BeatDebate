"""
Tests for Advocate Agents (GenreMoodAgent and DiscoveryAgent)

Tests to verify advocate agent functionality, Last.fm integration,
and recommendation generation capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.models.agent_models import MusicRecommenderState, AgentConfig
from src.agents.genre_mood_agent import GenreMoodAgent
from src.agents.discovery_agent import DiscoveryAgent


class TestGenreMoodAgent:
    """Test suite for GenreMoodAgent"""
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration"""
        return AgentConfig(
            agent_name="GenreMoodAgent",
            agent_type="advocate",
            llm_model="gemini-2.0-flash-exp",
            temperature=0.7,
            timeout_seconds=30
        )
    
    @pytest.fixture
    def mock_lastfm_client(self):
        """Create mock Last.fm client"""
        client = Mock()
        client.search_tracks = AsyncMock()
        client.get_artist_top_tracks = AsyncMock()
        client.search_artists = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Create mock Gemini client"""
        client = Mock()
        client.generate_content = AsyncMock()
        return client
    
    @pytest.fixture
    def genre_mood_agent(self, agent_config, mock_lastfm_client, mock_gemini_client):
        """Create GenreMoodAgent instance for testing"""
        return GenreMoodAgent(agent_config, mock_lastfm_client, mock_gemini_client)
    
    @pytest.fixture
    def test_state_with_strategy(self):
        """Create test state with planning strategy"""
        return MusicRecommenderState(
            user_query="I need chill indie music for studying",
            planning_strategy={
                "coordination_strategy": {
                    "genre_mood_agent": {
                        "focus_areas": ["indie", "alternative"],
                        "energy_level": "low",
                        "search_tags": ["chill", "study", "focus"],
                        "mood_priority": "chill"
                    }
                }
            },
            session_id="test_session_123"
        )
    
    @pytest.fixture
    def mock_track_data(self):
        """Mock track data from Last.fm"""
        return [
            {
                "name": "Holocene",
                "artist": "Bon Iver",
                "url": "https://www.last.fm/music/Bon+Iver/_/Holocene",
                "listeners": "150000",
                "album": {"title": "Bon Iver, Bon Iver"}
            },
            {
                "name": "Two Weeks",
                "artist": "Grizzly Bear",
                "url": "https://www.last.fm/music/Grizzly+Bear/_/Two+Weeks",
                "listeners": "120000",
                "album": {"title": "Veckatimest"}
            },
            {
                "name": "Sleepyhead",
                "artist": "Passion Pit",
                "url": "https://www.last.fm/music/Passion+Pit/_/Sleepyhead",
                "listeners": "200000",
                "album": {"title": "Manners"}
            }
        ]
    
    def test_genre_mood_agent_initialization(self, genre_mood_agent):
        """Test that GenreMoodAgent initializes correctly"""
        assert genre_mood_agent.agent_name == "GenreMoodAgent"
        assert genre_mood_agent.agent_type == "advocate"
        assert genre_mood_agent.lastfm_client is not None
        assert hasattr(genre_mood_agent, 'mood_tag_mappings')
        assert hasattr(genre_mood_agent, 'energy_level_mappings')
        assert hasattr(genre_mood_agent, 'genre_tag_mappings')
    
    def test_mood_mappings_initialization(self, genre_mood_agent):
        """Test that mood mappings are properly initialized"""
        mood_mappings = genre_mood_agent.mood_tag_mappings
        
        assert 'happy' in mood_mappings
        assert 'sad' in mood_mappings
        assert 'chill' in mood_mappings
        assert 'energetic' in mood_mappings
        
        # Verify mapping content
        assert 'uplifting' in mood_mappings['happy']
        assert 'relaxing' in mood_mappings['chill']
        assert 'upbeat' in mood_mappings['energetic']
    
    def test_mood_analysis(self, genre_mood_agent):
        """Test mood analysis from query and strategy"""
        strategy = {
            "mood_priority": "chill",
            "energy_level": "low",
            "search_tags": ["study", "focus"]
        }
        
        analysis = asyncio.run(genre_mood_agent._analyze_mood_requirements(
            "I need chill music for studying", strategy
        ))
        
        assert analysis['primary_mood'] == 'chill'
        assert analysis['energy_level'] == 'low'
        assert analysis['context_tags'] == ["study", "focus"]
        assert analysis['mood_confidence'] > 0.5
    
    def test_search_tags_generation(self, genre_mood_agent):
        """Test search tag generation"""
        strategy = {
            "focus_areas": ["indie", "alternative"],
            "search_tags": ["chill", "study"]
        }
        mood_analysis = {
            "primary_mood": "chill",
            "energy_level": "low",
            "secondary_moods": ["focus"]
        }
        
        tags = genre_mood_agent._generate_search_tags(strategy, mood_analysis)
        
        assert len(tags) <= 8  # Should limit to 8 tags
        assert any(tag in ['chill', 'relaxing', 'mellow'] for tag in tags)  # Mood tags
        assert any(tag in ['indie', 'alternative'] for tag in tags)  # Genre tags
    
    def test_track_scoring(self, genre_mood_agent):
        """Test track scoring algorithm"""
        track = {
            "name": "Test Track",
            "artist": "Test Artist",
            "url": "https://test.url",
            "listeners": "100000",
            "search_tags": ["chill", "indie"]
        }
        mood_analysis = {"primary_mood": "chill"}
        strategy = {}
        
        score = genre_mood_agent._calculate_track_score(track, mood_analysis, strategy)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should score well with good metadata
    
    @pytest.mark.asyncio
    async def test_process_with_mock_data(self, genre_mood_agent, test_state_with_strategy, mock_track_data):
        """Test process method with mocked Last.fm data"""
        # Mock Last.fm search to return test data
        genre_mood_agent.lastfm_client.search_tracks.return_value = mock_track_data
        
        result_state = await genre_mood_agent.process(test_state_with_strategy)
        
        # Verify recommendations were generated
        assert len(result_state.genre_mood_recommendations) > 0
        assert len(result_state.reasoning_log) > 0
        
        # Verify recommendation structure
        first_rec = result_state.genre_mood_recommendations[0]
        assert 'title' in first_rec
        assert 'artist' in first_rec
        assert 'reasoning_chain' in first_rec
        assert 'recommending_agent' in first_rec
        assert first_rec['recommending_agent'] == 'GenreMoodAgent'


class TestDiscoveryAgent:
    """Test suite for DiscoveryAgent"""
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration"""
        return AgentConfig(
            agent_name="DiscoveryAgent",
            agent_type="advocate",
            llm_model="gemini-2.0-flash-exp",
            temperature=0.7,
            timeout_seconds=30
        )
    
    @pytest.fixture
    def mock_lastfm_client(self):
        """Create mock Last.fm client"""
        client = Mock()
        client.search_tracks = AsyncMock()
        client.get_artist_top_tracks = AsyncMock()
        client.search_artists = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Create mock Gemini client"""
        client = Mock()
        client.generate_content = AsyncMock()
        return client
    
    @pytest.fixture
    def discovery_agent(self, agent_config, mock_lastfm_client, mock_gemini_client):
        """Create DiscoveryAgent instance for testing"""
        return DiscoveryAgent(agent_config, mock_lastfm_client, mock_gemini_client)
    
    @pytest.fixture
    def test_state_with_discovery_strategy(self):
        """Create test state with discovery strategy"""
        return MusicRecommenderState(
            user_query="Find me underground indie music similar to Radiohead",
            planning_strategy={
                "coordination_strategy": {
                    "discovery_agent": {
                        "novelty_priority": "high",
                        "underground_bias": 0.8,
                        "discovery_scope": "broad",
                        "similarity_base": "artist_similarity"
                    }
                }
            },
            session_id="test_session_123"
        )
    
    @pytest.fixture
    def mock_underground_tracks(self):
        """Mock underground track data"""
        return [
            {
                "name": "Weird Fishes",
                "artist": "Radiohead",
                "url": "https://www.last.fm/music/Radiohead/_/Weird+Fishes",
                "listeners": "80000",
                "album": {"title": "In Rainbows"}
            },
            {
                "name": "Bloodbuzz Ohio",
                "artist": "The National",
                "url": "https://www.last.fm/music/The+National/_/Bloodbuzz+Ohio",
                "listeners": "45000",
                "album": {"title": "High Violet"}
            },
            {
                "name": "Sprawl II",
                "artist": "Arcade Fire",
                "url": "https://www.last.fm/music/Arcade+Fire/_/Sprawl+II",
                "listeners": "30000",
                "album": {"title": "The Suburbs"}
            }
        ]
    
    def test_discovery_agent_initialization(self, discovery_agent):
        """Test that DiscoveryAgent initializes correctly"""
        assert discovery_agent.agent_name == "DiscoveryAgent"
        assert discovery_agent.agent_type == "advocate"
        assert discovery_agent.lastfm_client is not None
        assert hasattr(discovery_agent, 'discovery_strategies')
        assert hasattr(discovery_agent, 'seed_artists')
        assert hasattr(discovery_agent, 'underground_indicators')
    
    def test_seed_artists_initialization(self, discovery_agent):
        """Test that seed artists are properly initialized"""
        seed_artists = discovery_agent.seed_artists
        
        assert 'rock' in seed_artists
        assert 'electronic' in seed_artists
        assert 'indie' in seed_artists
        assert 'underground' in seed_artists
        
        # Verify content
        assert 'Radiohead' in seed_artists['rock']
        assert 'Bon Iver' in seed_artists['indie']
        assert 'Death Grips' in seed_artists['underground']
    
    def test_discovery_analysis(self, discovery_agent):
        """Test discovery requirements analysis"""
        strategy = {
            "novelty_priority": "high",
            "underground_bias": 0.8,
            "discovery_scope": "broad"
        }
        
        analysis = asyncio.run(discovery_agent._analyze_discovery_requirements(
            "Find underground music similar to Radiohead", strategy
        ))
        
        assert analysis['exploration_type'] == 'underground'
        assert analysis['underground_bias'] >= 0.8
        assert analysis['novelty_priority'] == 'high'
        assert analysis['max_listeners_threshold'] < 50000  # High underground bias
    
    def test_listener_threshold_calculation(self, discovery_agent):
        """Test listener threshold calculation for underground bias"""
        # High underground bias should give low threshold
        high_bias_threshold = discovery_agent._calculate_listener_threshold(0.9)
        assert high_bias_threshold < 20000
        
        # Low underground bias should give high threshold
        low_bias_threshold = discovery_agent._calculate_listener_threshold(0.2)
        assert low_bias_threshold > 50000
        
        # Medium bias
        medium_bias_threshold = discovery_agent._calculate_listener_threshold(0.5)
        assert 20000 < medium_bias_threshold < 80000
    
    def test_seed_artist_finding(self, discovery_agent):
        """Test seed artist finding from query and strategy"""
        strategy = {"focus_areas": ["indie", "rock"]}
        discovery_analysis = {"exploration_type": "underground"}
        
        seed_artists = asyncio.run(discovery_agent._find_seed_artists(
            "Find music like Radiohead", discovery_analysis, strategy
        ))
        
        assert len(seed_artists) <= 5  # Should limit to 5 seeds
        assert len(seed_artists) > 0  # Should find some seeds
        
        # Should include artists from focus areas or query
        seed_artists_lower = [artist.lower() for artist in seed_artists]
        assert any('radiohead' in artist or 'bon iver' in artist for artist in seed_artists_lower)
    
    def test_artist_extraction_from_query(self, discovery_agent):
        """Test artist name extraction from user query"""
        # Test with "like" pattern
        artists = discovery_agent._extract_artists_from_query("Find music like Radiohead")
        assert "Radiohead" in artists
        
        # Test with "similar" pattern
        artists = discovery_agent._extract_artists_from_query("Similar to Bon Iver")
        assert "Bon Iver" in artists
        
        # Test with no artists
        artists = discovery_agent._extract_artists_from_query("Find some chill music")
        assert len(artists) == 0
    
    def test_novelty_scoring(self, discovery_agent):
        """Test novelty score calculation"""
        discovery_analysis = {
            "underground_bias": 0.8,
            "max_listeners_threshold": 50000
        }
        
        # Underground track (low listeners)
        underground_track = {
            "name": "Test Track",
            "artist": "Underground Artist",
            "url": "https://test.url",
            "listeners": "10000",
            "discovery_method": "artist_similarity",
            "seed_artist": "Radiohead",
            "similar_artist": "Test Artist"
        }
        
        score = discovery_agent._calculate_novelty_score(underground_track, discovery_analysis)
        assert score > 0.6  # Should score high for underground track
        
        # Popular track (high listeners)
        popular_track = {
            "name": "Popular Track",
            "artist": "Popular Artist",
            "listeners": "1000000"
        }
        
        score = discovery_agent._calculate_novelty_score(popular_track, discovery_analysis)
        assert score < 0.4  # Should score low for popular track
    
    @pytest.mark.asyncio
    async def test_similar_artists_search(self, discovery_agent):
        """Test similar artists search functionality"""
        # Mock search results
        discovery_agent.lastfm_client.search_artists.return_value = [
            {"name": "Radiohead"},  # Exact match (should be skipped)
            {"name": "Thom Yorke"},
            {"name": "Atoms for Peace"},
            {"name": "Jonny Greenwood"},
            {"name": "The Smile"}
        ]
        
        similar_artists = await discovery_agent._get_similar_artists("Radiohead")
        
        assert len(similar_artists) > 0
        assert "Radiohead" not in similar_artists  # Should exclude exact match
        assert "Thom Yorke" in similar_artists
    
    @pytest.mark.asyncio
    async def test_process_with_mock_data(self, discovery_agent, test_state_with_discovery_strategy, mock_underground_tracks):
        """Test process method with mocked data"""
        # Mock Last.fm responses
        discovery_agent.lastfm_client.search_artists.return_value = [
            {"name": "Radiohead"},
            {"name": "Thom Yorke"},
            {"name": "The National"}
        ]
        discovery_agent.lastfm_client.get_artist_top_tracks.return_value = mock_underground_tracks
        
        result_state = await discovery_agent.process(test_state_with_discovery_strategy)
        
        # Verify recommendations were generated
        assert len(result_state.discovery_recommendations) > 0
        assert len(result_state.reasoning_log) > 0
        
        # Verify recommendation structure
        first_rec = result_state.discovery_recommendations[0]
        assert 'title' in first_rec
        assert 'artist' in first_rec
        assert 'reasoning_chain' in first_rec
        assert 'novelty_score' in first_rec
        assert 'recommending_agent' in first_rec
        assert first_rec['recommending_agent'] == 'DiscoveryAgent'
    
    def test_underground_filtering(self, discovery_agent):
        """Test underground track filtering"""
        discovery_analysis = {
            "underground_bias": 0.8,
            "max_listeners_threshold": 50000,
            "novelty_threshold": 0.5
        }
        
        tracks = [
            {
                "name": "Underground Track",
                "artist": "Indie Artist",
                "listeners": "20000",
                "discovery_method": "artist_similarity"
            },
            {
                "name": "Popular Track",
                "artist": "Mainstream Artist",
                "listeners": "500000",
                "discovery_method": "artist_similarity"
            }
        ]
        
        filtered = asyncio.run(discovery_agent._filter_for_underground(
            tracks, discovery_analysis, {}
        ))
        
        # Should prefer underground track
        assert len(filtered) > 0
        underground_track = next((t for t in filtered if t['name'] == 'Underground Track'), None)
        assert underground_track is not None
        assert underground_track['novelty_score'] > 0.5


class TestAgentIntegration:
    """Test integration between advocate agents"""
    
    @pytest.fixture
    def mock_clients(self):
        """Create mock clients for integration testing"""
        lastfm_client = Mock()
        lastfm_client.search_tracks = AsyncMock()
        lastfm_client.get_artist_top_tracks = AsyncMock()
        lastfm_client.search_artists = AsyncMock()
        
        gemini_client = Mock()
        gemini_client.generate_content = AsyncMock()
        
        return lastfm_client, gemini_client
    
    def test_strategy_extraction_compatibility(self, mock_clients):
        """Test that both agents can extract strategies correctly"""
        lastfm_client, gemini_client = mock_clients
        
        # Create agents
        genre_config = AgentConfig(agent_name="GenreMoodAgent", agent_type="advocate")
        discovery_config = AgentConfig(agent_name="DiscoveryAgent", agent_type="advocate")
        
        genre_agent = GenreMoodAgent(genre_config, lastfm_client, gemini_client)
        discovery_agent = DiscoveryAgent(discovery_config, lastfm_client, gemini_client)
        
        # Test strategy with both agent configurations
        full_strategy = {
            "coordination_strategy": {
                "genre_mood_agent": {
                    "focus_areas": ["indie", "rock"],
                    "energy_level": "medium",
                    "search_tags": ["chill", "alternative"]
                },
                "discovery_agent": {
                    "novelty_priority": "high",
                    "underground_bias": 0.7,
                    "discovery_scope": "broad"
                }
            }
        }
        
        # Both agents should extract their specific strategies
        genre_strategy = genre_agent.extract_strategy_for_agent(full_strategy)
        discovery_strategy = discovery_agent.extract_strategy_for_agent(full_strategy)
        
        assert genre_strategy["focus_areas"] == ["indie", "rock"]
        assert discovery_strategy["novelty_priority"] == "high"
        assert genre_strategy != discovery_strategy  # Should be different


if __name__ == "__main__":
    pytest.main([__file__]) 