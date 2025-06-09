"""
Tests for SessionManagerService.

Tests the logic for session creation, context storage, and candidate pool management.
Validates session state management, original query context storage, and candidate pool persistence.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.services.session_manager_service import (
    SessionManagerService,
    OriginalQueryContext,
    CandidatePool,
    ContextState
)
from src.models.metadata_models import UnifiedTrackMetadata, MetadataSource


@pytest.fixture
def mock_cache_manager():
    """Create a mock cache manager."""
    return Mock()


@pytest.fixture
def session_manager(mock_cache_manager):
    """Create a SessionManagerService instance with mocked dependencies."""
    return SessionManagerService(cache_manager=mock_cache_manager)


@pytest.fixture
def sample_track():
    """Create a sample track for testing."""
    return UnifiedTrackMetadata(
        name="Paranoid Android",
        artist="Radiohead",
        album="OK Computer",
        duration_ms=383000,
        source=MetadataSource.LASTFM,
        listeners=1500000,
        playcount=5000000,
        tags=["alternative rock", "experimental"],
        underground_score=0.3
    )


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return {
        "artist": "Radiohead",
        "genre": "alternative rock"
    }


@pytest.fixture
def sample_candidates(sample_track):
    """Create a list of sample candidate tracks."""
    return [
        sample_track,
        UnifiedTrackMetadata(
            name="Creep",
            artist="Radiohead",
            album="Pablo Honey",
            duration_ms=238000,
            source=MetadataSource.LASTFM,
            listeners=2000000,
            playcount=8000000,
            tags=["alternative rock", "grunge"],
            underground_score=0.1
        ),
        UnifiedTrackMetadata(
            name="Karma Police",
            artist="Radiohead",
            album="OK Computer",
            duration_ms=261000,
            source=MetadataSource.LASTFM,
            listeners=1800000,
            playcount=6500000,
            tags=["alternative rock", "art rock"],
            underground_score=0.2
        )
    ]


class TestSessionCreation:
    """Test session creation and retrieval."""
    
    @pytest.mark.asyncio
    async def test_create_and_get_session(self, session_manager, sample_entities, sample_track):
        """Test that a new session is created with correct initial state and can be retrieved."""
        session_id = "test-session-123"
        query = "music like Radiohead"
        intent = "artist_similarity"
        recommendations = [sample_track]
        
        # Create session
        session = await session_manager.create_or_update_session(
            session_id=session_id,
            query=query,
            intent=intent,
            entities=sample_entities,
            recommendations=recommendations,
            is_original_query=True
        )
        
        # Verify session structure
        assert session is not None
        assert len(session["interaction_history"]) == 1
        assert session["original_query_context"] is not None
        assert session["context_state"] == ContextState.NEW_SESSION.value
        
        # Verify interaction data
        interaction = session["interaction_history"][0]
        assert interaction["query"] == query
        assert interaction["intent"] == intent
        assert interaction["extracted_entities"] == sample_entities
        assert interaction["is_original_query"] is True
        assert len(interaction["recommendations"]) == 1
        
        # Verify original query context
        original_context = OriginalQueryContext.from_dict(session["original_query_context"])
        assert original_context.query == query
        assert original_context.intent == intent
        assert original_context.entities == sample_entities
        assert original_context.confidence == 1.0
        
        # Test retrieval
        retrieved_session = await session_manager.get_session_context(session_id)
        assert retrieved_session == session
        
        # Test non-existent session
        non_existent = await session_manager.get_session_context("non-existent")
        assert non_existent is None


class TestSessionHistory:
    """Test session history management."""
    
    @pytest.mark.asyncio
    async def test_update_session_history(self, session_manager, sample_entities, sample_track):
        """Test adding multiple interactions to a session and verify history is correctly appended."""
        session_id = "test-session-456"
        
        # First interaction (original query)
        await session_manager.create_or_update_session(
            session_id=session_id,
            query="music like Radiohead",
            intent="artist_similarity",
            entities=sample_entities,
            recommendations=[sample_track],
            is_original_query=True
        )
        
        # Second interaction (follow-up)
        await session_manager.create_or_update_session(
            session_id=session_id,
            query="more tracks like this",
            intent="artist_similarity",
            entities=sample_entities,
            is_original_query=False
        )
        
        # Third interaction (with feedback)
        user_feedback = {"liked": True, "rating": 4}
        await session_manager.create_or_update_session(
            session_id=session_id,
            query="something more upbeat",
            intent="mood_refinement",
            entities={"mood": "upbeat"},
            user_feedback=user_feedback,
            is_original_query=False
        )
        
        # Verify session history
        session = await session_manager.get_session_context(session_id)
        assert len(session["interaction_history"]) == 3
        
        # Verify first interaction
        first_interaction = session["interaction_history"][0]
        assert first_interaction["query"] == "music like Radiohead"
        assert first_interaction["intent"] == "artist_similarity"
        assert first_interaction["is_original_query"] is True
        
        # Verify second interaction
        second_interaction = session["interaction_history"][1]
        assert second_interaction["query"] == "more tracks like this"
        assert second_interaction["is_original_query"] is False
        
        # Verify third interaction with feedback
        third_interaction = session["interaction_history"][2]
        assert third_interaction["query"] == "something more upbeat"
        assert third_interaction["user_feedback"] == user_feedback
        assert third_interaction["extracted_entities"] == {"mood": "upbeat"}


class TestOriginalQueryContext:
    """Test original query context storage and retrieval."""
    
    @pytest.mark.asyncio
    async def test_get_original_query_context(self, session_manager, sample_entities):
        """Test that the context of the first query in a session is stored and retrieved correctly."""
        session_id = "test-session-789"
        original_query = "find me some underground electronic music"
        original_intent = "discovery"
        original_entities = {"genre": "electronic", "style": "underground"}
        
        # Create session with original query
        await session_manager.create_or_update_session(
            session_id=session_id,
            query=original_query,
            intent=original_intent,
            entities=original_entities,
            is_original_query=True
        )
        
        # Add follow-up queries
        await session_manager.create_or_update_session(
            session_id=session_id,
            query="more like this",
            intent="discovery",
            entities={"style": "similar"},
            is_original_query=False
        )
        
        await session_manager.create_or_update_session(
            session_id=session_id,
            query="but more ambient",
            intent="style_refinement",
            entities={"mood": "ambient"},
            is_original_query=False
        )
        
        # Retrieve original query context
        original_context = await session_manager.get_original_query_context(session_id)
        
        # Verify original context is preserved
        assert original_context is not None
        assert original_context.query == original_query
        assert original_context.intent == original_intent
        assert original_context.entities == original_entities
        assert original_context.confidence == 1.0
        assert isinstance(original_context.timestamp, datetime)
        
        # Test non-existent session
        non_existent_context = await session_manager.get_original_query_context("non-existent")
        assert non_existent_context is None


class TestCandidatePoolManagement:
    """Test candidate pool storage and retrieval."""
    
    @pytest.mark.asyncio
    async def test_store_and_get_candidate_pool(self, session_manager, sample_candidates, sample_entities):
        """Test that a candidate pool can be stored and retrieved successfully with usage count incremented."""
        session_id = "test-session-pool"
        intent = "artist_similarity"
        
        # Create session first
        await session_manager.create_or_update_session(
            session_id=session_id,
            query="music like Radiohead",
            intent=intent,
            entities=sample_entities,
            is_original_query=True
        )
        
        # Store candidate pool
        pool_key = await session_manager.store_candidate_pool(
            session_id=session_id,
            candidates=sample_candidates,
            intent=intent,
            entities=sample_entities
        )
        
        # Verify pool key is generated
        assert pool_key is not None
        assert intent in pool_key
        
        # Retrieve candidate pool
        retrieved_pool = await session_manager.get_candidate_pool(
            session_id=session_id,
            intent=intent,
            entities=sample_entities
        )
        
        # Verify pool data
        assert retrieved_pool is not None
        assert len(retrieved_pool.candidates) == len(sample_candidates)
        assert retrieved_pool.generated_for_intent == intent
        assert retrieved_pool.generated_for_entities == sample_entities
        assert retrieved_pool.used_count == 1  # Should be incremented after retrieval
        
        # Retrieve again to test usage count increment
        retrieved_pool_again = await session_manager.get_candidate_pool(
            session_id=session_id,
            intent=intent,
            entities=sample_entities
        )
        
        assert retrieved_pool_again.used_count == 2
        
        # Test with custom pool key
        custom_key = "custom_pool_key"
        custom_pool_key = await session_manager.store_candidate_pool(
            session_id=session_id,
            candidates=sample_candidates[:2],  # Smaller pool
            intent="discovery",
            entities={"genre": "rock"},
            pool_key=custom_key
        )
        
        assert custom_pool_key == custom_key
    
    @pytest.mark.asyncio
    async def test_get_expired_candidate_pool_returns_none(self, session_manager, sample_candidates, sample_entities):
        """Test that a pool older than the TTL is not returned."""
        session_id = "test-session-expired"
        intent = "artist_similarity"
        
        # Create session
        await session_manager.create_or_update_session(
            session_id=session_id,
            query="music like Radiohead",
            intent=intent,
            entities=sample_entities,
            is_original_query=True
        )
        
        # Store candidate pool
        pool_key = await session_manager.store_candidate_pool(
            session_id=session_id,
            candidates=sample_candidates,
            intent=intent,
            entities=sample_entities
        )
        
        # Manually set the pool timestamp to be expired
        session = await session_manager.get_session_context(session_id)
        pool_data = session["candidate_pools"][pool_key]
        
        # Set timestamp to be older than TTL (default is 60 minutes)
        expired_time = datetime.now() - timedelta(minutes=session_manager.candidate_pool_max_age_minutes + 10)
        pool_data["timestamp"] = expired_time.isoformat()
        
        # Try to retrieve expired pool
        retrieved_pool = await session_manager.get_candidate_pool(
            session_id=session_id,
            intent=intent,
            entities=sample_entities
        )
        
        # Should return None for expired pool
        assert retrieved_pool is None
    
    @pytest.mark.asyncio
    async def test_get_exhausted_candidate_pool_returns_none(self, session_manager, sample_candidates, sample_entities):
        """Test that a pool used more than its max_usage is not returned."""
        session_id = "test-session-exhausted"
        intent = "artist_similarity"
        
        # Create session
        await session_manager.create_or_update_session(
            session_id=session_id,
            query="music like Radiohead",
            intent=intent,
            entities=sample_entities,
            is_original_query=True
        )
        
        # Store candidate pool
        pool_key = await session_manager.store_candidate_pool(
            session_id=session_id,
            candidates=sample_candidates,
            intent=intent,
            entities=sample_entities
        )
        
        # Manually set the pool usage count to be exhausted
        session = await session_manager.get_session_context(session_id)
        pool_data = session["candidate_pools"][pool_key]
        
        # Set usage count to exceed max_usage (default is 3)
        pool_data["used_count"] = 5  # Exceeds max_usage
        
        # Try to retrieve exhausted pool
        retrieved_pool = await session_manager.get_candidate_pool(
            session_id=session_id,
            intent=intent,
            entities=sample_entities
        )
        
        # Should return None for exhausted pool
        assert retrieved_pool is None


class TestSessionCleanup:
    """Test session cleanup and management."""
    
    @pytest.mark.asyncio
    async def test_clear_session(self, session_manager, sample_entities):
        """Test that a session's data is completely removed."""
        session_id = "test-session-clear"
        
        # Create session with data
        await session_manager.create_or_update_session(
            session_id=session_id,
            query="test query",
            intent="test_intent",
            entities=sample_entities,
            is_original_query=True
        )
        
        # Verify session exists
        session = await session_manager.get_session_context(session_id)
        assert session is not None
        
        # Clear session
        await session_manager.clear_session(session_id)
        
        # Verify session is removed
        cleared_session = await session_manager.get_session_context(session_id)
        assert cleared_session is None
        
        # Test clearing non-existent session (should not raise error)
        await session_manager.clear_session("non-existent-session")


class TestContextDecisionAnalysis:
    """Test context decision analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_context_decision_new_session(self, session_manager):
        """Test context decision for a completely new session."""
        result = await session_manager.analyze_context_decision(
            current_query="find me some jazz music",
            session_id="new-session-123"
        )
        
        assert result["decision"] == ContextState.NEW_SESSION.value
        assert result["action"] == "create_new_context"
        assert result["confidence"] == 1.0
        assert result["reasoning"] == "No existing session context found"
        assert result["context_to_use"] is None
        assert result["reset_context"] is False
        assert result["is_followup"] is False
    
    @pytest.mark.asyncio
    async def test_analyze_context_decision_with_reset_trigger(self, session_manager, sample_entities):
        """Test context decision when query contains reset triggers."""
        session_id = "test-session-reset"
        
        # Create existing session
        await session_manager.create_or_update_session(
            session_id=session_id,
            query="music like Radiohead",
            intent="artist_similarity",
            entities=sample_entities,
            is_original_query=True
        )
        
        # Analyze query with reset trigger
        result = await session_manager.analyze_context_decision(
            current_query="actually, never mind that, find me some jazz instead",
            session_id=session_id
        )
        
        assert result["reset_context"] is True
        assert "actually" in result["reasoning"]


class TestSessionDataStructures:
    """Test session data structure functionality."""
    
    def test_original_query_context_serialization(self, sample_entities):
        """Test OriginalQueryContext to_dict and from_dict methods."""
        original_context = OriginalQueryContext(
            query="test query",
            intent="test_intent",
            entities=sample_entities,
            timestamp=datetime.now(),
            confidence=0.95
        )
        
        # Test serialization
        context_dict = original_context.to_dict()
        assert context_dict["query"] == "test query"
        assert context_dict["intent"] == "test_intent"
        assert context_dict["entities"] == sample_entities
        assert context_dict["confidence"] == 0.95
        assert isinstance(context_dict["timestamp"], str)
        
        # Test deserialization
        restored_context = OriginalQueryContext.from_dict(context_dict)
        assert restored_context.query == original_context.query
        assert restored_context.intent == original_context.intent
        assert restored_context.entities == original_context.entities
        assert restored_context.confidence == original_context.confidence
        assert isinstance(restored_context.timestamp, datetime)
    
    def test_candidate_pool_serialization(self, sample_candidates, sample_entities):
        """Test CandidatePool to_dict and from_dict methods."""
        candidate_pool = CandidatePool(
            candidates=sample_candidates,
            generated_for_intent="artist_similarity",
            generated_for_entities=sample_entities,
            timestamp=datetime.now(),
            used_count=2,
            max_usage=5
        )
        
        # Test serialization
        pool_dict = candidate_pool.to_dict()
        assert pool_dict["generated_for_intent"] == "artist_similarity"
        assert pool_dict["generated_for_entities"] == sample_entities
        assert pool_dict["used_count"] == 2
        assert pool_dict["max_usage"] == 5
        assert isinstance(pool_dict["timestamp"], str)
        assert len(pool_dict["candidates"]) == len(sample_candidates)
        
        # Test deserialization
        restored_pool = CandidatePool.from_dict(pool_dict)
        assert restored_pool.generated_for_intent == candidate_pool.generated_for_intent
        assert restored_pool.generated_for_entities == candidate_pool.generated_for_entities
        assert restored_pool.used_count == candidate_pool.used_count
        assert restored_pool.max_usage == candidate_pool.max_usage
        assert isinstance(restored_pool.timestamp, datetime)
        assert len(restored_pool.candidates) == len(sample_candidates)


class TestSessionManagerConfiguration:
    """Test session manager configuration and edge cases."""
    
    def test_session_manager_initialization(self, mock_cache_manager):
        """Test SessionManagerService initialization with custom configuration."""
        session_manager = SessionManagerService(cache_manager=mock_cache_manager)
        
        assert session_manager.cache_manager == mock_cache_manager
        assert session_manager.context_decay_minutes == 30
        assert session_manager.candidate_pool_max_age_minutes == 60
        assert session_manager.max_interactions_per_session == 100
        assert len(session_manager.reset_triggers) > 0
        assert "never mind" in session_manager.reset_triggers
    
    @pytest.mark.asyncio
    async def test_session_cleanup_on_large_history(self, session_manager, sample_entities):
        """Test that session history is cleaned up when it becomes too large."""
        session_id = "test-session-large"
        
        # Set a small max_interactions for testing
        session_manager.max_interactions_per_session = 5
        
        # Create many interactions
        for i in range(8):
            await session_manager.create_or_update_session(
                session_id=session_id,
                query=f"query {i}",
                intent="test_intent",
                entities=sample_entities,
                is_original_query=(i == 0)
            )
        
        # Verify cleanup occurred
        session = await session_manager.get_session_context(session_id)
        # The cleanup happens when we exceed max_interactions, but then we add more
        # So we should have fewer than the original 8 interactions
        assert len(session["interaction_history"]) < 8
        # And it should be close to the expected cleanup size
        assert len(session["interaction_history"]) <= 5 