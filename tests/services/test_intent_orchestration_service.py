"""
Tests for IntentOrchestrationService.

Validates the complex logic for resolving user intent, especially for follow-up queries.
Tests intent resolution, follow-up detection, and effective intent generation.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from src.services.intent_orchestration_service import (
    IntentOrchestrationService,
    FollowUpType
)
from src.services.session_manager_service import (
    SessionManagerService,
    OriginalQueryContext
)


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    return Mock(spec=SessionManagerService)


@pytest.fixture
def mock_llm_utils():
    """Create a mock LLM utils."""
    mock_llm = Mock()
    mock_llm.call_llm_with_json_response = AsyncMock()
    return mock_llm


@pytest.fixture
def intent_orchestrator(mock_session_manager, mock_llm_utils):
    """Create an IntentOrchestrationService instance."""
    return IntentOrchestrationService(
        session_manager=mock_session_manager,
        llm_utils=mock_llm_utils
    )


@pytest.fixture
def sample_original_context():
    """Create a sample original query context."""
    return OriginalQueryContext(
        query="music like Radiohead",
        intent="artist_similarity",
        entities={
            "artists": ["Radiohead"],
            "genres": ["alternative rock", "experimental"]
        },
        timestamp=datetime.now(),
        confidence=0.9
    )


@pytest.fixture
def sample_by_artist_context():
    """Create a sample by_artist original context."""
    return OriginalQueryContext(
        query="songs by The Beatles",
        intent="by_artist",
        entities={
            "artists": ["The Beatles"],
            "genres": ["rock", "pop"]
        },
        timestamp=datetime.now(),
        confidence=0.95
    )


class TestFreshQueryResolution:
    """Test intent resolution for fresh (non-follow-up) queries."""
    
    @pytest.mark.asyncio
    async def test_resolve_fresh_query(self, intent_orchestrator, mock_session_manager, mock_llm_utils):
        """Test intent resolution for a new, standalone query."""
        # Setup mock LLM response
        mock_llm_response = {
            "intent": "artist_similarity",
            "entities": {
                "artists": ["Radiohead"],
                "genres": ["alternative rock"]
            },
            "confidence": 0.9,
            "reasoning": "User wants music similar to Radiohead"
        }
        mock_llm_utils.call_llm_with_json_response.return_value = mock_llm_response
        
        # Setup session manager to return no original context (fresh query)
        mock_session_manager.get_original_query_context.return_value = None
        
        # Test fresh query resolution
        result = await intent_orchestrator.resolve_effective_intent(
            current_query="music like Radiohead",
            session_id="test-session",
            llm_understanding=None,
            context_override=None
        )
        
        # Verify result
        assert result["intent"] == "artist_similarity"
        assert result["entities"]["artists"] == ["Radiohead"]
        assert result["entities"]["genres"] == ["alternative rock"]
        assert result["is_followup"] is False
        assert result["confidence"] == 0.9
        assert result["reasoning"] == "Fresh query with LLM understanding"
        
        # Verify LLM was called
        mock_llm_utils.call_llm_with_json_response.assert_called_once()


class TestFollowUpDetection:
    """Test follow-up query detection and resolution."""
    
    @pytest.mark.asyncio
    async def test_resolve_followup_more_tracks_by_artist(self, intent_orchestrator, mock_session_manager, sample_by_artist_context):
        """Test 'by_artist' query followed by 'more tracks' maintains by_artist intent."""
        # Setup session manager to return by_artist original context
        mock_session_manager.get_original_query_context.return_value = sample_by_artist_context
        
        # Test with context override indicating follow-up
        context_override = {
            "is_followup": True,
            "intent_override": "by_artist",
            "target_entity": "The Beatles",
            "followup_type": "artist_deep_dive",
            "entities": sample_by_artist_context.entities,
            "confidence": 0.9
        }
        
        result = await intent_orchestrator.resolve_effective_intent(
            current_query="more tracks",
            session_id="test-session",
            llm_understanding=None,
            context_override=context_override
        )
        
        # Verify the effective intent remains by_artist
        assert result["intent"] == "by_artist"
        assert result["is_followup"] is True
        assert result["followup_type"] == "artist_deep_dive"
        assert result["target_entity"] == "The Beatles"
        assert result["confidence"] == 0.9
        assert "The Beatles" in result["reasoning"]
    
    @pytest.mark.asyncio
    async def test_resolve_followup_more_tracks_like_artist(self, intent_orchestrator, mock_session_manager, sample_original_context):
        """Test 'artist_similarity' query followed by 'more like this' maintains artist_similarity intent."""
        # Setup session manager to return artist_similarity original context
        mock_session_manager.get_original_query_context.return_value = sample_original_context
        
        # Test with context override for similarity follow-up
        context_override = {
            "is_followup": True,
            "intent_override": "artist_similarity",
            "target_entity": "Radiohead",
            "followup_type": "style_continuation",
            "entities": sample_original_context.entities,
            "confidence": 0.85
        }
        
        result = await intent_orchestrator.resolve_effective_intent(
            current_query="more like this",
            session_id="test-session",
            llm_understanding=None,
            context_override=context_override
        )
        
        # Verify the effective intent remains artist_similarity
        assert result["intent"] == "artist_similarity"
        assert result["is_followup"] is True
        assert result["followup_type"] == "style_continuation"
        assert result["target_entity"] == "Radiohead"
        assert result["confidence"] == 0.85
    
    @pytest.mark.asyncio
    async def test_resolve_followup_artist_style_refinement(self, intent_orchestrator, mock_session_manager, sample_by_artist_context):
        """Test query like 'more by [Artist] but more electronic' resolves to artist_style_refinement."""
        # Setup session manager
        mock_session_manager.get_original_query_context.return_value = sample_by_artist_context
        
        # Test with style refinement context override
        context_override = {
            "is_followup": True,
            "intent_override": "by_artist",
            "target_entity": "The Beatles",
            "followup_type": "artist_style_refinement",
            "entities": {
                "artists": ["The Beatles"],
                "genres": ["rock", "pop", "electronic"],
                "moods": {"secondary": [{"name": "electronic", "confidence": 0.8}]}
            },
            "confidence": 0.8
        }
        
        result = await intent_orchestrator.resolve_effective_intent(
            current_query="more by The Beatles but more electronic",
            session_id="test-session",
            llm_understanding=None,
            context_override=context_override
        )
        
        # Verify style refinement is detected
        assert result["intent"] == "by_artist"
        assert result["is_followup"] is True
        assert result["followup_type"] == "artist_style_refinement"
        assert result["target_entity"] == "The Beatles"
        assert "electronic" in str(result["entities"])


class TestContextReset:
    """Test context reset scenarios."""
    
    @pytest.mark.asyncio
    async def test_resolve_new_query_resets_context(self, intent_orchestrator, mock_session_manager, mock_llm_utils, sample_by_artist_context):
        """Test that a completely new query is correctly identified as fresh, not a follow-up."""
        # Setup session manager to return previous by_artist context
        mock_session_manager.get_original_query_context.return_value = sample_by_artist_context
        
        # Setup LLM response for new genre_mood query
        mock_llm_response = {
            "intent": "genre_mood",
            "entities": {
                "genres": ["jazz"],
                "moods": ["relaxing"]
            },
            "confidence": 0.9,
            "reasoning": "User wants jazz music for relaxation"
        }
        mock_llm_utils.call_llm_with_json_response.return_value = mock_llm_response
        
        # Test with no context override (fresh query)
        result = await intent_orchestrator.resolve_effective_intent(
            current_query="play some relaxing jazz music",
            session_id="test-session",
            llm_understanding=None,
            context_override=None
        )
        
        # Verify it's treated as a fresh query, not a follow-up
        assert result["intent"] == "genre_mood"
        assert result["is_followup"] is False
        assert result["entities"]["genres"] == ["jazz"]
        assert result["entities"]["moods"] == ["relaxing"]
        assert result["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_resolve_serendipity_followup(self, intent_orchestrator, mock_session_manager):
        """Test that 'more tracks' after a 'surprise me' query correctly resolves to discovering_serendipity."""
        # Setup original serendipity context
        serendipity_context = OriginalQueryContext(
            query="surprise me with something",
            intent="discovering_serendipity",
            entities={},
            timestamp=datetime.now(),
            confidence=0.9
        )
        mock_session_manager.get_original_query_context.return_value = serendipity_context
        
        # Test with serendipity follow-up context override
        context_override = {
            "is_followup": True,
            "intent_override": "discovering_serendipity",
            "target_entity": "serendipity",
            "followup_type": "more_content",
            "entities": {},
            "confidence": 0.9
        }
        
        result = await intent_orchestrator.resolve_effective_intent(
            current_query="more tracks",
            session_id="test-session",
            llm_understanding=None,
            context_override=context_override
        )
        
        # Verify serendipity intent is preserved
        assert result["intent"] == "discovering_serendipity"
        assert result["is_followup"] is True
        assert result["followup_type"] == "more_content"
        assert result["target_entity"] == "serendipity"


class TestFollowUpTypeDetection:
    """Test specific follow-up type detection logic."""
    
    def test_detect_artist_deep_dive(self, intent_orchestrator, sample_by_artist_context):
        """Test detection of artist deep dive follow-ups."""
        # Test generic "more" with by_artist context
        followup_type = intent_orchestrator._detect_followup_type(
            "more tracks", 
            sample_by_artist_context
        )
        assert followup_type == FollowUpType.ARTIST_DEEP_DIVE
        
        # Test explicit artist mention
        followup_type = intent_orchestrator._detect_followup_type(
            "more Beatles songs", 
            sample_by_artist_context
        )
        assert followup_type == FollowUpType.ARTIST_DEEP_DIVE
    
    def test_detect_style_continuation(self, intent_orchestrator, sample_original_context):
        """Test detection of style continuation follow-ups."""
        followup_type = intent_orchestrator._detect_followup_type(
            "more like this", 
            sample_original_context
        )
        assert followup_type == FollowUpType.STYLE_CONTINUATION
        
        followup_type = intent_orchestrator._detect_followup_type(
            "similar to these", 
            sample_original_context
        )
        assert followup_type == FollowUpType.STYLE_CONTINUATION
    
    def test_detect_artist_style_refinement(self, intent_orchestrator, sample_by_artist_context):
        """Test detection of artist style refinement."""
        followup_type = intent_orchestrator._detect_followup_type(
            "more The Beatles but upbeat", 
            sample_by_artist_context
        )
        assert followup_type == FollowUpType.ARTIST_STYLE_REFINEMENT
        
        followup_type = intent_orchestrator._detect_followup_type(
            "The Beatles electronic tracks", 
            sample_by_artist_context
        )
        assert followup_type == FollowUpType.ARTIST_STYLE_REFINEMENT
    
    def test_detect_similarity_exploration(self, intent_orchestrator, sample_original_context):
        """Test detection of similarity exploration."""
        followup_type = intent_orchestrator._detect_followup_type(
            "similar artists", 
            sample_original_context
        )
        assert followup_type == FollowUpType.SIMILARITY_EXPLORATION
        
        # Note: "artists like Radiohead" will be detected as ARTIST_DEEP_DIVE because 
        # the artist detection logic comes before pattern matching
        followup_type = intent_orchestrator._detect_followup_type(
            "artists like someone else", 
            sample_original_context
        )
        assert followup_type == FollowUpType.SIMILARITY_EXPLORATION
    
    def test_detect_variation_request(self, intent_orchestrator, sample_original_context):
        """Test detection of variation requests."""
        followup_type = intent_orchestrator._detect_followup_type(
            "something different", 
            sample_original_context
        )
        assert followup_type == FollowUpType.VARIATION_REQUEST
        
        followup_type = intent_orchestrator._detect_followup_type(
            "mix it up", 
            sample_original_context
        )
        assert followup_type == FollowUpType.VARIATION_REQUEST


class TestEntityExtraction:
    """Test entity extraction and manipulation utilities."""
    
    def test_extract_artist_names_from_entities(self, intent_orchestrator):
        """Test extraction of artist names from different entity structures."""
        # Test simple list format
        entities1 = {"artists": ["Radiohead", "The Beatles"]}
        artists = intent_orchestrator._extract_artist_names_from_entities(entities1)
        assert artists == ["Radiohead", "The Beatles"]
        
        # Test dict format with names
        entities2 = {
            "artists": [
                {"name": "Radiohead", "confidence": 0.9},
                {"name": "The Beatles", "confidence": 0.8}
            ]
        }
        artists = intent_orchestrator._extract_artist_names_from_entities(entities2)
        assert artists == ["Radiohead", "The Beatles"]
        
        # Test musical_entities structure
        entities3 = {
            "musical_entities": {
                "artists": ["Pink Floyd", "Led Zeppelin"]
            }
        }
        artists = intent_orchestrator._extract_artist_names_from_entities(entities3)
        assert artists == ["Pink Floyd", "Led Zeppelin"]
    
    def test_extract_style_modifiers(self, intent_orchestrator):
        """Test extraction of style modifiers from queries."""
        # Test energy modifiers
        modifiers = intent_orchestrator._extract_style_modifiers("more upbeat tracks")
        assert "high_energy" in modifiers
        
        modifiers = intent_orchestrator._extract_style_modifiers("something slow and chill")
        assert "low_energy" in modifiers
        
        # Test mood modifiers
        modifiers = intent_orchestrator._extract_style_modifiers("happy songs")
        assert "positive_mood" in modifiers
        
        modifiers = intent_orchestrator._extract_style_modifiers("sad and melancholy")
        assert "negative_mood" in modifiers
        
        # Test genre modifiers
        modifiers = intent_orchestrator._extract_style_modifiers("more electronic music")
        assert "genre_electronic" in modifiers
        
        modifiers = intent_orchestrator._extract_style_modifiers("acoustic versions")
        assert "genre_acoustic" in modifiers
    
    def test_apply_style_modifiers(self, intent_orchestrator):
        """Test application of style modifiers to entities."""
        original_entities = {
            "artists": ["Radiohead"],
            "genres": {"primary": ["rock"], "secondary": []},
            "moods": {"primary": ["experimental"], "secondary": []}
        }
        
        style_modifiers = ["high_energy", "genre_electronic"]
        
        modified_entities = intent_orchestrator._apply_style_modifiers(
            original_entities, 
            style_modifiers
        )
        
        # Verify electronic genre was added
        assert any(
            genre.get("name") == "electronic" 
            for genre in modified_entities["genres"]["secondary"]
        )
        
        # Verify high energy mood was added
        assert any(
            mood.get("name") == "high energy" 
            for mood in modified_entities["moods"]["secondary"]
        )
    
    def test_create_variation_entities(self, intent_orchestrator):
        """Test creation of variation entities."""
        original_entities = {
            "artists": ["Radiohead"],
            "genres": {
                "primary": ["alternative rock", "experimental", "art rock"],
                "secondary": ["progressive rock"]
            },
            "moods": {
                "primary": ["melancholy", "introspective"],
                "secondary": ["atmospheric"]
            }
        }
        
        variation_entities = intent_orchestrator._create_variation_entities(original_entities)
        
        # Verify artists are preserved but marked for variation
        assert variation_entities["artists"] == ["Radiohead"]
        assert variation_entities["constraint_overrides"]["vary_from_artists"] is True
        
        # Verify only some genres are kept (reduced weight)
        assert len(variation_entities["genres"]["secondary"]) <= 2
        
        # Verify only some moods are kept
        assert len(variation_entities["moods"]["secondary"]) <= 1


class TestIntentSummary:
    """Test intent summary functionality."""
    
    @pytest.mark.asyncio
    async def test_get_intent_summary(self, intent_orchestrator, mock_session_manager, sample_original_context):
        """Test getting intent summary for a session."""
        # Setup mock session context
        mock_session_context = {
            "interaction_history": [
                {"is_original_query": True, "query": "music like Radiohead"},
                {"is_original_query": False, "query": "more tracks"},
                {"is_original_query": False, "query": "something upbeat"}
            ],
            "context_state": "continuing",
            "last_updated": datetime.now()
        }
        
        mock_session_manager.get_session_context.return_value = mock_session_context
        mock_session_manager.get_original_query_context.return_value = sample_original_context
        
        summary = await intent_orchestrator.get_intent_summary("test-session")
        
        # Verify summary content
        assert summary["session_id"] == "test-session"
        assert summary["original_intent"] == "artist_similarity"
        assert summary["original_query"] == "music like Radiohead"
        assert summary["interaction_count"] == 3
        assert summary["followup_count"] == 2
        assert summary["context_state"] == "continuing"
    
    @pytest.mark.asyncio
    async def test_get_intent_summary_no_session(self, intent_orchestrator, mock_session_manager):
        """Test intent summary for non-existent session."""
        mock_session_manager.get_session_context.return_value = None
        
        summary = await intent_orchestrator.get_intent_summary("non-existent")
        
        assert "error" in summary
        assert summary["error"] == "No session found"


class TestLLMAnalysis:
    """Test LLM query analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_query_with_llm_success(self, intent_orchestrator, mock_llm_utils):
        """Test successful LLM query analysis."""
        # Setup mock LLM response
        mock_response = {
            "intent": "by_artist",
            "entities": {
                "artists": ["The Beatles"],
                "genres": ["rock", "pop"]
            },
            "contextual_entities": {
                "activities": {
                    "mental": ["study"]
                }
            },
            "confidence": 0.95,
            "reasoning": "User wants Beatles music for studying"
        }
        mock_llm_utils.call_llm_with_json_response.return_value = mock_response
        
        result = await intent_orchestrator._analyze_query_with_llm("Beatles songs for studying")
        
        # Verify result structure
        assert result["intent"] == "by_artist"
        assert result["entities"]["artists"] == ["The Beatles"]
        assert result["entities"]["genres"] == ["rock", "pop"]
        assert result["entities"]["contextual_entities"]["activities"]["mental"] == ["study"]
        assert result["confidence"] == 0.95
        assert result["reasoning"] == "User wants Beatles music for studying"
        assert result["query"] == "Beatles songs for studying"
    
    @pytest.mark.asyncio
    async def test_analyze_query_with_llm_failure(self, intent_orchestrator, mock_llm_utils):
        """Test LLM query analysis failure handling."""
        # Setup mock to raise exception
        mock_llm_utils.call_llm_with_json_response.side_effect = Exception("LLM service unavailable")
        
        # Should raise the exception
        with pytest.raises(Exception, match="LLM service unavailable"):
            await intent_orchestrator._analyze_query_with_llm("test query")
    
    @pytest.mark.asyncio
    async def test_analyze_query_no_llm_utils(self, mock_session_manager):
        """Test query analysis when LLM utils are not available."""
        # Create orchestrator without LLM utils
        orchestrator = IntentOrchestrationService(
            session_manager=mock_session_manager,
            llm_utils=None
        )
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="LLM utils not available"):
            await orchestrator._analyze_query_with_llm("test query")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_resolve_intent_with_empty_query(self, intent_orchestrator, mock_session_manager):
        """Test intent resolution with empty query."""
        mock_session_manager.get_original_query_context.return_value = None
        
        result = await intent_orchestrator.resolve_effective_intent(
            current_query="",
            session_id="test-session",
            llm_understanding=None,
            context_override=None
        )
        
        # Should default to discovery intent
        assert result["intent"] == "discovery"
        assert result["is_followup"] is False
        assert result["confidence"] == 0.5
    
    @pytest.mark.asyncio
    async def test_resolve_intent_with_malformed_context_override(self, intent_orchestrator):
        """Test intent resolution with malformed context override."""
        # Test with incomplete context override
        malformed_override = {
            "is_followup": True,
            # Missing required fields
        }
        
        result = await intent_orchestrator.resolve_effective_intent(
            current_query="more tracks",
            session_id="test-session",
            llm_understanding=None,
            context_override=malformed_override
        )
        
        # Should handle gracefully
        assert result["intent"] == "discovery"  # Default fallback
        assert result["is_followup"] is True
        assert "confidence" in result