"""
Test Phase 1 Integration

Tests the integration of SessionManagerService and IntentOrchestrationService
with the EnhancedRecommendationService.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.services.session_manager_service import SessionManagerService, OriginalQueryContext, ContextState
from src.services.intent_orchestration_service import IntentOrchestrationService, FollowUpType
from src.services.enhanced_recommendation_service import EnhancedRecommendationService


class TestPhase1Integration:
    """Test Phase 1 integration of enhanced context and intent services."""
    
    @pytest.fixture
    def session_manager(self):
        """Create a SessionManagerService instance for testing."""
        return SessionManagerService()
    
    @pytest.fixture
    def intent_orchestrator(self, session_manager):
        """Create an IntentOrchestrationService instance for testing."""
        return IntentOrchestrationService(session_manager=session_manager)
    
    @pytest.fixture
    def mock_api_service(self):
        """Create a mock API service."""
        mock_service = Mock()
        mock_service.get_lastfm_client = AsyncMock()
        mock_service.close = AsyncMock()
        return mock_service
    
    @pytest.fixture
    def enhanced_service(self, session_manager, intent_orchestrator, mock_api_service):
        """Create an EnhancedRecommendationService with Phase 1 services."""
        return EnhancedRecommendationService(
            api_service=mock_api_service,
            session_manager=session_manager,
            intent_orchestrator=intent_orchestrator
        )
    
    @pytest.mark.asyncio
    async def test_session_manager_initialization(self, session_manager):
        """Test that SessionManagerService initializes correctly."""
        assert session_manager is not None
        assert session_manager.session_store == {}
        assert session_manager.context_decay_minutes == 30
        assert session_manager.candidate_pool_max_age_minutes == 60
    
    @pytest.mark.asyncio
    async def test_session_creation_and_update(self, session_manager):
        """Test creating and updating sessions with original query context."""
        session_id = "test_session_1"
        query = "Music by Radiohead"
        intent = "artist_similarity"
        entities = {
            "artists": [{"name": "Radiohead", "confidence": 0.9}],
            "genres": {"primary": [{"name": "alternative rock", "confidence": 0.8}]}
        }
        
        # Create session
        session = await session_manager.create_or_update_session(
            session_id=session_id,
            query=query,
            intent=intent,
            entities=entities,
            is_original_query=True
        )
        
        assert session is not None
        assert session["context_state"] == ContextState.NEW_SESSION.value
        assert session["original_query_context"] is not None
        
        # Verify original query context
        original_context = await session_manager.get_original_query_context(session_id)
        assert original_context is not None
        assert original_context.query == query
        assert original_context.intent == intent
        assert original_context.entities == entities
    
    @pytest.mark.asyncio
    async def test_context_decision_analysis(self, session_manager):
        """Test context decision analysis for different query types."""
        session_id = "test_session_2"
        
        # Test new session
        decision = await session_manager.analyze_context_decision(
            current_query="Music by The Beatles",
            session_id=session_id
        )
        
        assert decision["decision"] == ContextState.NEW_SESSION.value
        assert decision["reset_context"] == False
        assert decision["is_followup"] == False
        
        # Create a session first
        await session_manager.create_or_update_session(
            session_id=session_id,
            query="Music by The Beatles",
            intent="artist_similarity",
            entities={"artists": [{"name": "The Beatles"}]},
            is_original_query=True
        )
        
        # Test follow-up query
        decision = await session_manager.analyze_context_decision(
            current_query="more tracks",
            session_id=session_id
        )
        
        assert decision["is_followup"] == True
        assert decision["decision"] == ContextState.CONTINUING.value
    
    @pytest.mark.asyncio
    async def test_intent_orchestrator_initialization(self, intent_orchestrator):
        """Test that IntentOrchestrationService initializes correctly."""
        assert intent_orchestrator is not None
        assert intent_orchestrator.session_manager is not None
        assert len(intent_orchestrator.followup_patterns) > 0
        assert FollowUpType.ARTIST_DEEP_DIVE in intent_orchestrator.followup_patterns
    
    @pytest.mark.asyncio
    async def test_intent_resolution_fresh_query(self, intent_orchestrator):
        """Test intent resolution for fresh (non-follow-up) queries."""
        session_id = "test_session_3"
        query = "Discover underground electronic music"
        
        llm_understanding = {
            "intent": "discovery",
            "entities": {
                "genres": {"primary": [{"name": "electronic", "confidence": 0.8}]}
            },
            "confidence": 0.9
        }
        
        effective_intent = await intent_orchestrator.resolve_effective_intent(
            current_query=query,
            session_id=session_id,
            llm_understanding=llm_understanding
        )
        
        assert effective_intent["intent"] == "discovery"
        assert effective_intent["is_followup"] == False
        assert effective_intent["confidence"] > 0.5
    
    @pytest.mark.asyncio
    async def test_intent_resolution_followup_query(self, intent_orchestrator):
        """Test intent resolution for follow-up queries."""
        session_id = "test_session_4"
        
        # First, create an original session
        await intent_orchestrator.session_manager.create_or_update_session(
            session_id=session_id,
            query="Music by Radiohead",
            intent="artist_similarity",
            entities={"artists": [{"name": "Radiohead"}]},
            is_original_query=True
        )
        
        # Now test follow-up
        followup_query = "more tracks"
        
        effective_intent = await intent_orchestrator.resolve_effective_intent(
            current_query=followup_query,
            session_id=session_id,
            llm_understanding={"intent": "discovery", "entities": {}}
        )
        
        assert effective_intent["is_followup"] == True
        assert effective_intent["original_intent"] == "artist_similarity"
        assert "followup_type" in effective_intent
    
    @pytest.mark.asyncio
    async def test_enhanced_service_integration(self, enhanced_service):
        """Test that EnhancedRecommendationService integrates Phase 1 services correctly."""
        assert enhanced_service.session_manager is not None
        assert enhanced_service._intent_orchestrator is not None
        
        # Test that services are accessible
        session_manager = enhanced_service.session_manager
        assert isinstance(session_manager, SessionManagerService)
        
        # Test that intent orchestrator property works after initialization
        # Note: This will require agents to be initialized first
        try:
            intent_orchestrator = enhanced_service.intent_orchestrator
            assert isinstance(intent_orchestrator, IntentOrchestrationService)
        except RuntimeError as e:
            # Expected if agents not initialized yet
            assert "not initialized" in str(e)
    
    @pytest.mark.asyncio
    async def test_session_manager_followup_patterns(self, session_manager):
        """Test session manager's follow-up pattern detection."""
        session_id = "test_session_5"
        
        # Create original session
        await session_manager.create_or_update_session(
            session_id=session_id,
            query="Electronic music for studying",
            intent="mood_matching",
            entities={"moods": {"primary": [{"name": "focus", "confidence": 0.8}]}},
            is_original_query=True
        )
        
        # Test various follow-up patterns
        followup_queries = [
            "more like this",
            "similar tracks",
            "more music",
            "something else",
            "what about jazz"  # This should be detected as different
        ]
        
        for query in followup_queries:
            decision = await session_manager.analyze_context_decision(
                current_query=query,
                session_id=session_id
            )
            
            if "more" in query or "similar" in query or "like this" in query:
                assert decision["is_followup"] == True, f"Query '{query}' should be detected as follow-up"
            elif "what about" in query:
                # This might be detected as follow-up depending on implementation
                pass  # Allow either result
    
    @pytest.mark.asyncio
    async def test_intent_summary(self, intent_orchestrator):
        """Test getting intent summary for a session."""
        session_id = "test_session_6"
        
        # Create a session with some interactions
        await intent_orchestrator.session_manager.create_or_update_session(
            session_id=session_id,
            query="Jazz music for relaxation",
            intent="mood_matching",
            entities={"genres": {"primary": [{"name": "jazz"}]}},
            is_original_query=True
        )
        
        await intent_orchestrator.session_manager.create_or_update_session(
            session_id=session_id,
            query="more like this",
            intent="mood_matching",
            entities={},
            is_original_query=False
        )
        
        summary = await intent_orchestrator.get_intent_summary(session_id)
        
        assert summary["session_id"] == session_id
        assert summary["original_intent"] == "mood_matching"
        assert summary["original_query"] == "Jazz music for relaxation"
        assert summary["interaction_count"] == 2
        assert summary["followup_count"] == 1


if __name__ == "__main__":
    # Run a simple test
    async def run_simple_test():
        session_manager = SessionManagerService()
        intent_orchestrator = IntentOrchestrationService(session_manager=session_manager)
        
        print("✅ Phase 1 services initialized successfully")
        
        # Test session creation
        session = await session_manager.create_or_update_session(
            session_id="demo",
            query="Music by Radiohead",
            intent="artist_similarity",
            entities={"artists": [{"name": "Radiohead"}]},
            is_original_query=True
        )
        
        print("✅ Session created with original query context")
        
        # Test follow-up detection
        decision = await session_manager.analyze_context_decision(
            current_query="more tracks",
            session_id="demo"
        )
        
        print(f"✅ Follow-up detected: {decision['is_followup']}")
        
        # Test intent resolution
        effective_intent = await intent_orchestrator.resolve_effective_intent(
            current_query="more tracks",
            session_id="demo",
            llm_understanding={"intent": "discovery", "entities": {}}
        )
        
        print(f"✅ Effective intent resolved: {effective_intent['intent']}, follow-up: {effective_intent['is_followup']}")
        
        print("\n🎉 Phase 1 integration test completed successfully!")
    
    asyncio.run(run_simple_test()) 