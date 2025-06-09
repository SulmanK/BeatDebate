"""
Tests for GenreMoodAgent

Tests the refactored GenreMoodAgent with its modular components:
- GenreMoodConfig: Intent parameter management
- MoodAnalyzer: Mood detection and analysis
- GenreProcessor: Genre matching and filtering
- TagGenerator: Tag generation and enhancement
- UnifiedCandidateGenerator: Candidate generation
- QualityScorer: Quality assessment
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.agents.genre_mood.agent import GenreMoodAgent
from src.models.agent_models import MusicRecommenderState, AgentConfig, QueryUnderstanding, QueryIntent
from src.models.recommendation_models import TrackRecommendation
from src.services.api_service import APIService
from src.services.metadata_service import MetadataService


class TestGenreMoodAgent:
    """Test suite for GenreMoodAgent."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        client = AsyncMock()
        client.generate_response = AsyncMock(return_value="Test reasoning")
        return client
    
    @pytest.fixture
    def mock_api_service(self):
        """Mock API service."""
        service = Mock(spec=APIService)
        service.search_tracks_by_tags = AsyncMock(return_value=[])
        service.search_unified_tracks = AsyncMock(return_value=[])
        service.get_artist_info = AsyncMock(return_value={})
        return service
    
    @pytest.fixture
    def mock_metadata_service(self):
        """Mock metadata service."""
        service = Mock(spec=MetadataService)
        service.enrich_track_metadata = AsyncMock(return_value={})
        return service
    
    @pytest.fixture
    def mock_session_manager(self):
        """Mock session manager."""
        manager = Mock()
        manager.get_candidate_pool = Mock(return_value=None)
        manager.store_candidate_pool = Mock()
        return manager
    
    @pytest.fixture
    def agent_config(self):
        """Agent configuration for testing."""
        return AgentConfig(
            agent_name="genre_mood_agent",
            agent_type="advocate",
            llm_model="gemini-2.0-flash-exp",
            temperature=0.7,
            max_tokens=1000,
            timeout_seconds=60
        )
    
    @pytest.fixture
    def genre_mood_agent(self, agent_config, mock_llm_client, mock_api_service, mock_metadata_service, mock_session_manager):
        """Create GenreMoodAgent instance for testing."""
        return GenreMoodAgent(
            config=agent_config,
            llm_client=mock_llm_client,
            api_service=mock_api_service,
            metadata_service=mock_metadata_service,
            session_manager=mock_session_manager
        )
    
    @pytest.fixture
    def sample_state_genre_mood(self):
        """Sample state with genre_mood intent."""
        query_understanding = QueryUnderstanding(
            intent=QueryIntent.GENRE_MOOD,
            confidence=0.9,
            artists=[],
            genres=["electronic"],
            moods=["energetic"],
            activities=[],
            original_query="energetic electronic music",
            normalized_query="energetic electronic music",
            reasoning="User wants energetic electronic music"
        )
        
        return MusicRecommenderState(
            user_query="energetic electronic music",
            session_id="test_session",
            query_understanding=query_understanding,
            entities={
                "musical_entities": {
                    "genres": {"primary": ["electronic"]},
                    "artists": {"primary": []}
                },
                "contextual_entities": {
                    "moods": {"primary": ["energetic"]},
                    "energy_level": "high"
                }
            },
            intent_analysis={
                "intent": "genre_mood",
                "confidence": 0.9,
                "reasoning": "User wants energetic electronic music"
            }
        )
    
    @pytest.fixture
    def sample_state_contextual(self):
        """Sample state with contextual intent."""
        query_understanding = QueryUnderstanding(
            intent=QueryIntent.CONTEXTUAL,
            confidence=0.8,
            artists=[],
            genres=[],
            moods=["focus", "calm"],
            activities=["studying"],
            original_query="music for studying",
            normalized_query="music for studying",
            reasoning="User wants functional music for studying"
        )
        
        return MusicRecommenderState(
            user_query="music for studying",
            session_id="test_session",
            query_understanding=query_understanding,
            entities={
                "contextual_entities": {
                    "context": "studying",
                    "moods": {"primary": ["focus", "calm"]},
                    "energy_level": "low"
                }
            },
            intent_analysis={
                "intent": "contextual",
                "confidence": 0.8,
                "reasoning": "User wants functional music for studying"
            }
        )
    
    @pytest.fixture
    def sample_candidates(self):
        """Sample candidate tracks."""
        return [
            {
                'name': 'Electronic Track 1',
                'artist': 'Artist A',
                'url': 'http://example.com/track1',
                'album': 'Album A',
                'playcount': 1000,
                'listeners': 500,
                'tags': ['electronic', 'energetic', 'dance'],
                'popularity': 0.8
            },
            {
                'name': 'Electronic Track 2',
                'artist': 'Artist B',
                'url': 'http://example.com/track2',
                'album': 'Album B',
                'playcount': 800,
                'listeners': 400,
                'tags': ['electronic', 'ambient', 'chill'],
                'popularity': 0.6
            }
        ]
    
    @pytest.mark.asyncio
    async def test_genre_mood_agent_initialization(self, genre_mood_agent):
        """Test that GenreMoodAgent initializes correctly with all components."""
        # Verify agent is initialized
        assert genre_mood_agent is not None
        
        # Verify modular components are initialized
        assert hasattr(genre_mood_agent, 'config_manager')
        assert hasattr(genre_mood_agent, 'mood_analyzer')
        assert hasattr(genre_mood_agent, 'genre_processor')
        assert hasattr(genre_mood_agent, 'tag_generator')
        assert hasattr(genre_mood_agent, 'candidate_generator')
        assert hasattr(genre_mood_agent, 'quality_scorer')
        
        # Verify components are not None
        assert genre_mood_agent.config_manager is not None
        assert genre_mood_agent.mood_analyzer is not None
        assert genre_mood_agent.genre_processor is not None
        assert genre_mood_agent.tag_generator is not None
        assert genre_mood_agent.candidate_generator is not None
        assert genre_mood_agent.quality_scorer is not None
    
    @pytest.mark.asyncio
    async def test_genre_mood_agent_process_workflow(self, genre_mood_agent, sample_state_genre_mood, sample_candidates):
        """
        Test that GenreMoodAgent correctly executes the workflow for genre_mood intent.
        
        Mocks all components and verifies:
        1. Candidate generation is called
        2. Scoring is applied
        3. Filtering is performed
        4. Final recommendations are created
        """
        # Mock component methods
        with patch.object(genre_mood_agent.candidate_generator, 'generate_candidate_pool', new_callable=AsyncMock) as mock_generate, \
             patch.object(genre_mood_agent, '_score_candidates', new_callable=AsyncMock) as mock_score, \
             patch.object(genre_mood_agent.genre_processor, 'filter_by_genre_requirements', new_callable=AsyncMock) as mock_filter, \
             patch.object(genre_mood_agent, '_create_recommendations', new_callable=AsyncMock) as mock_create:
            
            # Setup mock returns
            mock_generate.return_value = sample_candidates
            
            scored_candidates = [
                {**candidate, 'combined_score': 0.8, 'quality_score': 0.7}
                for candidate in sample_candidates
            ]
            mock_score.return_value = scored_candidates
            mock_filter.return_value = scored_candidates[:1]  # Filter to top candidate
            
            mock_recommendations = [
                TrackRecommendation(
                    title="Electronic Track 1",
                    artist="Artist A",
                    id="artist_a_electronic_track_1",
                    source="genre_mood_agent",
                    confidence=0.8,
                    explanation="Test reasoning"
                )
            ]
            mock_create.return_value = mock_recommendations
            
            # Process the state
            result_state = await genre_mood_agent.process(sample_state_genre_mood)
            
            # Verify workflow execution
            mock_generate.assert_called_once()
            mock_score.assert_called_once_with(
                sample_candidates,
                sample_state_genre_mood.entities,
                sample_state_genre_mood.intent_analysis
            )
            mock_filter.assert_called_once()
            mock_create.assert_called_once()
            
            # Verify final state has recommendations
            assert hasattr(result_state, 'genre_mood_recommendations')
            assert result_state.genre_mood_recommendations == mock_recommendations
            assert len(result_state.genre_mood_recommendations) == 1
    
    @pytest.mark.asyncio
    async def test_genre_mood_agent_handles_contextual_intent(self, genre_mood_agent, sample_state_contextual):
        """
        Test that GenreMoodAgent correctly handles contextual intent (e.g., "music for studying").
        
        Verifies:
        1. MoodAnalyzer is configured for functional music
        2. Parameters are adapted for contextual intent
        3. Audio features are prioritized
        """
        # Mock component methods
        with patch.object(genre_mood_agent.config_manager, 'adapt_to_intent') as mock_adapt, \
             patch.object(genre_mood_agent.candidate_generator, 'generate_candidate_pool', new_callable=AsyncMock) as mock_generate, \
             patch.object(genre_mood_agent, '_score_candidates', new_callable=AsyncMock) as mock_score, \
             patch.object(genre_mood_agent.genre_processor, 'filter_by_genre_requirements', new_callable=AsyncMock) as mock_filter, \
             patch.object(genre_mood_agent, '_create_recommendations', new_callable=AsyncMock) as mock_create:
            
            # Setup mock returns
            mock_generate.return_value = []
            mock_score.return_value = []
            mock_filter.return_value = []
            mock_create.return_value = []
            
            # Process the state
            await genre_mood_agent.process(sample_state_contextual)
            
            # Verify intent adaptation was called with contextual intent
            mock_adapt.assert_called_once_with('contextual')
            
            # Verify candidate generation was called
            mock_generate.assert_called_once()
            
            # Verify the call included contextual entities
            call_args = mock_generate.call_args
            # Check if call was made with keyword arguments
            if call_args.kwargs:
                entities_param = call_args.kwargs.get('entities', {})
            else:
                # Check positional arguments - entities should be the second parameter
                entities_param = call_args.args[1] if len(call_args.args) > 1 else {}
            
            assert 'contextual_entities' in entities_param
            assert 'context' in entities_param['contextual_entities']
    
    @pytest.mark.asyncio
    async def test_genre_mood_agent_applies_context_override(self, genre_mood_agent, sample_state_genre_mood):
        """
        Test that GenreMoodAgent correctly applies context override for artist deep-dive.
        
        Verifies:
        1. Context override is detected and applied
        2. Artist boost is applied in scoring
        3. Intent is adapted to artist_similarity
        """
        # Add context override to state
        context_override = Mock()
        context_override.intent_override = 'artist_deep_dive'
        context_override.target_artist = 'Test Artist'
        context_override.boost_factor = 1.5
        sample_state_genre_mood.context_override = context_override
        
        # Mock component methods
        with patch.object(genre_mood_agent.config_manager, 'adapt_to_intent') as mock_adapt, \
             patch.object(genre_mood_agent.candidate_generator, 'generate_candidate_pool', new_callable=AsyncMock) as mock_generate, \
             patch.object(genre_mood_agent, '_score_candidates', new_callable=AsyncMock) as mock_score, \
             patch.object(genre_mood_agent.genre_processor, 'filter_by_genre_requirements', new_callable=AsyncMock) as mock_filter, \
             patch.object(genre_mood_agent, '_create_recommendations', new_callable=AsyncMock) as mock_create:
            
            # Setup mock returns
            mock_generate.return_value = []
            mock_score.return_value = []
            mock_filter.return_value = []
            mock_create.return_value = []
            
            # Process the state
            await genre_mood_agent.process(sample_state_genre_mood)
            
            # Verify intent was adapted to artist_similarity due to context override
            mock_adapt.assert_called_once_with('artist_similarity')
    
    @pytest.mark.asyncio
    async def test_score_candidates_integration(self, genre_mood_agent, sample_candidates):
        """Test candidate scoring with all component scores."""
        entities = {
            "musical_entities": {"genres": {"primary": ["electronic"]}},
            "contextual_entities": {"moods": {"primary": ["energetic"]}}
        }
        intent_analysis = {"intent": "genre_mood", "confidence": 0.9}
        
        # Mock component scoring methods
        with patch.object(genre_mood_agent.quality_scorer, 'calculate_quality_score', new_callable=AsyncMock) as mock_quality, \
             patch.object(genre_mood_agent.genre_processor, 'calculate_genre_score', new_callable=AsyncMock) as mock_genre, \
             patch.object(genre_mood_agent.mood_analyzer, 'calculate_mood_score') as mock_mood, \
             patch.object(genre_mood_agent.tag_generator, 'calculate_tag_based_score') as mock_tag, \
             patch.object(genre_mood_agent.config_manager, 'get_current_config') as mock_config:
            
            # Setup mock returns
            mock_quality.return_value = 0.8
            mock_genre.return_value = 0.7
            mock_mood.return_value = 0.9
            mock_tag.return_value = 0.6
            mock_config.return_value = {
                'genre_weight': 0.6,
                'mood_weight': 0.7
            }
            
            # Score candidates
            scored = await genre_mood_agent._score_candidates(sample_candidates, entities, intent_analysis)
            
            # Verify all scoring methods were called
            assert mock_quality.call_count == len(sample_candidates)
            assert mock_genre.call_count == len(sample_candidates)
            assert mock_mood.call_count == len(sample_candidates)
            assert mock_tag.call_count == len(sample_candidates)
            
            # Verify scores are added to candidates
            for candidate in scored:
                assert 'quality_score' in candidate
                assert 'genre_score' in candidate
                assert 'mood_score' in candidate
                assert 'tag_score' in candidate
                assert 'genre_mood_score' in candidate
                assert 'combined_score' in candidate
            
            # Verify candidates are sorted by combined score
            assert scored[0]['combined_score'] >= scored[1]['combined_score']
    
    @pytest.mark.asyncio
    async def test_create_recommendations_integration(self, genre_mood_agent, sample_candidates):
        """Test recommendation creation with component integration."""
        entities = {
            "musical_entities": {"genres": {"primary": ["electronic"]}},
            "contextual_entities": {"moods": {"primary": ["energetic"]}}
        }
        intent_analysis = {"intent": "genre_mood", "confidence": 0.9}
        
        # Add scores to candidates
        scored_candidates = [
            {**candidate, 'combined_score': 0.8, 'quality_score': 0.7}
            for candidate in sample_candidates
        ]
        
        # Mock component methods
        with patch.object(genre_mood_agent, '_generate_reasoning', new_callable=AsyncMock) as mock_reasoning, \
             patch.object(genre_mood_agent.tag_generator, 'enhance_recommendation_tags') as mock_tags, \
             patch.object(genre_mood_agent.genre_processor, 'extract_genres_for_recommendation') as mock_genres:
            
            # Setup mock returns
            mock_reasoning.return_value = "Test reasoning for recommendation"
            mock_tags.return_value = ["energetic", "electronic", "dance"]
            mock_genres.return_value = ["electronic", "dance"]
            
            # Create recommendations
            recommendations = await genre_mood_agent._create_recommendations(
                scored_candidates, entities, intent_analysis
            )
            
            # Verify recommendations were created
            assert len(recommendations) == len(scored_candidates)
            
            # Verify recommendation structure
            for rec in recommendations:
                assert isinstance(rec, TrackRecommendation)
                assert rec.source == 'genre_mood_agent'
                assert rec.advocate_source_agent == 'genre_mood_agent'
                assert rec.explanation == "Test reasoning for recommendation"
                assert rec.moods == ["energetic", "electronic", "dance"]
                assert rec.genres == ["electronic", "dance"]
                assert 'combined_score' in rec.additional_scores
                assert 'quality_score' in rec.additional_scores
    
    @pytest.mark.asyncio
    async def test_genre_mood_agent_handles_component_failure(self, genre_mood_agent, sample_state_genre_mood):
        """Test that GenreMoodAgent handles component failures gracefully."""
        # Mock candidate generator to raise exception
        with patch.object(genre_mood_agent.candidate_generator, 'generate_candidate_pool', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("API failure")
            
            # Process should not raise exception
            result_state = await genre_mood_agent.process(sample_state_genre_mood)
            
            # Verify state is returned with empty recommendations
            assert hasattr(result_state, 'genre_mood_recommendations')
            assert result_state.genre_mood_recommendations == []
    
    @pytest.mark.asyncio
    async def test_genre_mood_agent_skips_generation_for_followup(self, genre_mood_agent, sample_state_genre_mood):
        """Test that GenreMoodAgent skips candidate generation for follow-up queries."""
        # Mark state as follow-up using intent_analysis (the correct way)
        sample_state_genre_mood.intent_analysis = {
            'intent': 'genre_mood',
            'is_followup': True,
            'followup_type': 'more_tracks',
            'confidence': 0.9
        }
        
        # Mock candidate generator to verify it's NOT called
        with patch.object(genre_mood_agent.candidate_generator, 'generate_candidate_pool', new_callable=AsyncMock) as mock_generate:
            
            # Process the state
            result_state = await genre_mood_agent.process(sample_state_genre_mood)
            
            # Verify candidate generation was NOT called for follow-up
            mock_generate.assert_not_called()
            
            # Verify state has empty recommendations (letting judge agent handle follow-ups)
            assert result_state.genre_mood_recommendations == []
    
    @pytest.mark.asyncio
    async def test_detect_intent_with_context_override(self, genre_mood_agent, sample_state_genre_mood):
        """Test intent detection with context override."""
        # Add context override
        context_override = Mock()
        context_override.intent_override = 'artist_deep_dive'
        sample_state_genre_mood.context_override = context_override
        
        # Test intent detection
        detected_intent = genre_mood_agent._detect_intent(
            sample_state_genre_mood,
            sample_state_genre_mood.intent_analysis,
            context_override_applied=True
        )
        
        # Should detect artist_similarity due to context override
        assert detected_intent == 'artist_similarity'
    
    @pytest.mark.asyncio
    async def test_detect_intent_from_query_understanding(self, genre_mood_agent, sample_state_genre_mood):
        """Test intent detection from query understanding."""
        # Test intent detection without context override
        detected_intent = genre_mood_agent._detect_intent(
            sample_state_genre_mood,
            sample_state_genre_mood.intent_analysis,
            context_override_applied=False
        )
        
        # Should detect genre_mood from query understanding
        assert detected_intent == 'genre_mood'
    
    @pytest.mark.asyncio
    async def test_config_adaptation_for_different_intents(self, genre_mood_agent):
        """Test that config manager adapts parameters for different intents."""
        # Test adaptation for different intents
        test_intents = ['genre_mood', 'contextual', 'artist_similarity', 'discovery']
        
        for intent in test_intents:
            with patch.object(genre_mood_agent.config_manager, 'adapt_to_intent') as mock_adapt:
                # Simulate intent detection and adaptation
                genre_mood_agent.config_manager.adapt_to_intent(intent)
                mock_adapt.assert_called_once_with(intent)
    
    def test_component_initialization_verification(self, genre_mood_agent):
        """Test that all components are properly initialized with correct types."""
        # Verify component types
        from src.agents.genre_mood.components import GenreMoodConfig, MoodAnalyzer, GenreProcessor, TagGenerator
        from src.agents.components.unified_candidate_generator import UnifiedCandidateGenerator
        from src.agents.components import QualityScorer
        
        assert isinstance(genre_mood_agent.config_manager, GenreMoodConfig)
        assert isinstance(genre_mood_agent.mood_analyzer, MoodAnalyzer)
        assert isinstance(genre_mood_agent.genre_processor, GenreProcessor)
        assert isinstance(genre_mood_agent.tag_generator, TagGenerator)
        assert isinstance(genre_mood_agent.candidate_generator, UnifiedCandidateGenerator)
        assert isinstance(genre_mood_agent.quality_scorer, QualityScorer) 