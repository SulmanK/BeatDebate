import pytest
import pytest_asyncio  # Add import for pytest_asyncio
from typing import List, Dict, Any, Optional

# Assuming JudgeAgent and TrackRecommendation are accessible for import
# Adjust path as necessary based on actual project structure for tests
from src.agents.judge_agent import EnhancedJudgeAgent, JudgeAgent, MusicRecommenderState # MusicRecommenderState might be mocked or imported from elsewhere too
from src.models.recommendation_models import TrackRecommendation

# Helper function to create a basic TrackRecommendation model or dict for tests
def create_track_dict(
    id: str, 
    title: str = "Test Track", 
    artist: str = "Test Artist", 
    source: str = "test_source",
    scores: Optional[Dict[str, float]] = None, 
    attributes: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    track_data = {
        "id": id,
        "title": title,
        "artist": artist,
        "source": source,
        "genres": [], # Default, can be overridden by attributes
        "era": None,    # Default, can be overridden by attributes
        # Initialize standard score fields to None or a default if not in scores
        "concentration_friendliness_score": None,
        "novelty_score": None,
        "quality_score": None,
        "additional_scores": {}
    }
    if scores:
        for key, value in scores.items():
            if key in track_data: # e.g. novelty_score
                track_data[key] = value
            else: # Put into additional_scores
                track_data["additional_scores"][key] = value
    if attributes:
        track_data.update(attributes)
    return track_data

@pytest.fixture
def judge_agent() -> JudgeAgent:
    """Pytest fixture to create a JudgeAgent instance for testing."""
    return JudgeAgent()

@pytest.fixture
def enhanced_judge_agent() -> EnhancedJudgeAgent:
    """Pytest fixture to create an EnhancedJudgeAgent instance for testing."""
    return EnhancedJudgeAgent()

@pytest.fixture
def mock_music_recommender_state() -> MusicRecommenderState:
    """Pytest fixture for a mock MusicRecommenderState."""
    state = MusicRecommenderState(user_query="I need focus music for coding")
    # Pre-populate with some defaults or leave empty based on test needs
    state.planning_strategy = {
        "evaluation_framework": {
            "primary_weights": {},
            "diversity_targets": {}
        }
    }
    state.genre_mood_recommendations = []
    state.discovery_recommendations = []
    state.conversation_context = {}
    return state

@pytest.mark.asyncio
class TestEnhancedJudgeAgentPromptAnalysis:
    """Test class for prompt analysis functionality."""

    async def test_prompt_analysis_concentration_intent(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test prompt analysis for concentration intent."""
        prompt = "I need focus music for coding"
        analysis = enhanced_judge_agent.prompt_analyzer.analyze_prompt(prompt)
        
        assert analysis["primary_intent"] == "concentration"
        assert analysis["activity_context"] == "coding"
        assert isinstance(analysis["exploration_openness"], float)
        assert 0.0 <= analysis["exploration_openness"] <= 1.0

    async def test_prompt_analysis_energy_intent(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test prompt analysis for energy/workout intent."""
        prompt = "I need energetic music for my workout"
        analysis = enhanced_judge_agent.prompt_analyzer.analyze_prompt(prompt)
        
        assert analysis["primary_intent"] == "energy"
        assert analysis["activity_context"] == "workout"
        assert analysis["energy_level"] in ["low", "medium", "high"]

    async def test_prompt_analysis_discovery_intent(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test prompt analysis for discovery intent."""
        prompt = "Surprise me with something new and different"
        analysis = enhanced_judge_agent.prompt_analyzer.analyze_prompt(prompt)
        
        assert analysis["primary_intent"] == "discovery"
        assert analysis["exploration_openness"] > 0.5

    async def test_prompt_analysis_genre_mentions(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test genre mention extraction."""
        prompt = "I want some jazz and electronic music"
        analysis = enhanced_judge_agent.prompt_analyzer.analyze_prompt(prompt)
        
        assert "jazz" in analysis["genre_preferences"]
        assert "electronic" in analysis["genre_preferences"]

    async def test_prompt_analysis_specificity_level(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test specificity level measurement."""
        specific_prompt = "I need exactly this specific type of music"
        open_prompt = "Surprise me with anything"
        
        specific_analysis = enhanced_judge_agent.prompt_analyzer.analyze_prompt(specific_prompt)
        open_analysis = enhanced_judge_agent.prompt_analyzer.analyze_prompt(open_prompt)
        
        assert specific_analysis["specificity_level"] > open_analysis["specificity_level"]


@pytest.mark.asyncio
class TestEnhancedJudgeAgentContextualScoring:
    """Test class for contextual relevance scoring."""

    async def test_contextual_relevance_coding_activity(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test contextual relevance for coding activity."""
        track = TrackRecommendation(
            id="test1",
            title="Ambient Focus",
            artist="Test Artist",
            source="test",
            genres=["ambient", "instrumental"],
            instrumental=True,
            additional_scores={"energy": 0.4}
        )
        
        prompt_analysis = {
            "activity_context": "coding",
            "energy_level": "medium"
        }
        
        score = enhanced_judge_agent.contextual_scorer.calculate_contextual_relevance(track, prompt_analysis)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should score well for coding

    async def test_contextual_relevance_workout_activity(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test contextual relevance for workout activity."""
        track = TrackRecommendation(
            id="test1",
            title="High Energy",
            artist="Test Artist",
            source="test",
            genres=["electronic", "dance"],
            additional_scores={"energy": 0.9}
        )
        
        prompt_analysis = {
            "activity_context": "workout",
            "energy_level": "high"
        }
        
        score = enhanced_judge_agent.contextual_scorer.calculate_contextual_relevance(track, prompt_analysis)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should score well for workout

    async def test_contextual_relevance_mood_matching(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test mood matching in contextual relevance."""
        track = TrackRecommendation(
            id="test1",
            title="Upbeat Track",
            artist="Test Artist",
            source="test",
            moods=["upbeat", "energetic"]
        )
        
        prompt_analysis = {
            "mood_request": ["upbeat"]
        }
        
        score = enhanced_judge_agent.contextual_scorer.calculate_contextual_relevance(track, prompt_analysis)
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
class TestEnhancedJudgeAgentIntentAlignment:
    """Test class for intent alignment scoring."""

    async def test_intent_alignment_concentration(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test intent alignment for concentration."""
        track = TrackRecommendation(
            id="test1",
            title="Focus Track",
            artist="Test Artist",
            source="test",
            genres=["ambient"],
            instrumental=True,
            concentration_friendliness_score=0.9
        )
        
        prompt_analysis = {"primary_intent": "concentration"}
        
        score = enhanced_judge_agent._calculate_intent_alignment(track, prompt_analysis)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should score well for concentration

    async def test_intent_alignment_energy(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test intent alignment for energy."""
        track = TrackRecommendation(
            id="test1",
            title="Energy Track",
            artist="Test Artist",
            source="test",
            genres=["electronic", "rock"],
            additional_scores={"energy": 0.8}
        )
        
        prompt_analysis = {"primary_intent": "energy"}
        
        score = enhanced_judge_agent._calculate_intent_alignment(track, prompt_analysis)
        assert 0.0 <= score <= 1.0

    async def test_intent_alignment_discovery(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test intent alignment for discovery."""
        track = TrackRecommendation(
            id="test1",
            title="Underground Track",
            artist="Test Artist",
            source="test",
            genres=["experimental"],
            novelty_score=0.8
        )
        
        prompt_analysis = {"primary_intent": "discovery"}
        
        score = enhanced_judge_agent._calculate_intent_alignment(track, prompt_analysis)
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
class TestEnhancedJudgeAgentDiscoveryScoring:
    """Test class for discovery appropriateness scoring."""

    async def test_discovery_score_high_openness(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test discovery scoring with high exploration openness."""
        track = TrackRecommendation(
            id="test1",
            title="Underground Track",
            artist="Test Artist",
            source="test",
            novelty_score=0.8
        )
        
        prompt_analysis = {
            "exploration_openness": 0.8,
            "specificity_level": 0.2
        }
        
        score = enhanced_judge_agent.discovery_scorer.calculate_discovery_score(track, prompt_analysis)
        assert 0.0 <= score <= 1.0

    async def test_discovery_score_low_openness(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test discovery scoring with low exploration openness."""
        track = TrackRecommendation(
            id="test1",
            title="Mainstream Track",
            artist="Test Artist",
            source="test",
            novelty_score=0.2
        )
        
        prompt_analysis = {
            "exploration_openness": 0.2,
            "specificity_level": 0.8
        }
        
        score = enhanced_judge_agent.discovery_scorer.calculate_discovery_score(track, prompt_analysis)
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
class TestEnhancedJudgeAgentPromptDrivenRanking:
    """Test class for the complete prompt-driven ranking system."""

    async def test_prompt_driven_ranking_basic(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test basic prompt-driven ranking."""
        candidates = [
            TrackRecommendation(
                id="test1",
                title="Focus Track",
                artist="Test Artist",
                source="test",
                genres=["ambient"],
                instrumental=True,
                concentration_friendliness_score=0.9
            ),
            TrackRecommendation(
                id="test2",
                title="Energy Track",
                artist="Test Artist",
                source="test",
                genres=["rock"],
                additional_scores={"energy": 0.8}
            )
        ]
        
        prompt_analysis = {
            "primary_intent": "concentration",
            "activity_context": "coding",
            "exploration_openness": 0.5,
            "specificity_level": 0.6,
            "conversation_continuity": {"builds_on_context": False}
        }
        
        ranked = await enhanced_judge_agent._apply_prompt_driven_ranking(candidates, prompt_analysis)
        
        assert len(ranked) == 2
        assert all(len(item) == 3 for item in ranked)  # (track, score, factors)
        assert all(0.0 <= item[1] <= 1.0 for item in ranked)  # Valid scores
        
        # First track should score higher for concentration intent
        focus_track_score = next(item[1] for item in ranked if item[0].title == "Focus Track")
        energy_track_score = next(item[1] for item in ranked if item[0].title == "Energy Track")
        assert focus_track_score > energy_track_score

    async def test_prompt_driven_ranking_empty_candidates(self, enhanced_judge_agent: EnhancedJudgeAgent):
        """Test prompt-driven ranking with empty candidates."""
        candidates = []
        prompt_analysis = {"primary_intent": "concentration"}
        
        ranked = await enhanced_judge_agent._apply_prompt_driven_ranking(candidates, prompt_analysis)
        assert ranked == []


# Keep some of the original diversity tests since that functionality is preserved
@pytest.mark.asyncio
class TestEnhancedJudgeAgentEvaluateAndSelect:
    """Test class for the main evaluate_and_select method."""

    async def test_evaluate_and_select_basic_flow(self, enhanced_judge_agent: EnhancedJudgeAgent, mock_music_recommender_state: MusicRecommenderState):
        """Test the basic flow of evaluate_and_select with enhanced agent."""
        # Add some test recommendations
        mock_music_recommender_state.genre_mood_recommendations = [
            create_track_dict("gm1", "Focus Track 1", "Artist 1", scores={"concentration_friendliness_score": 0.9}),
            create_track_dict("gm2", "Focus Track 2", "Artist 2", scores={"concentration_friendliness_score": 0.7})
        ]
        mock_music_recommender_state.discovery_recommendations = [
            create_track_dict("d1", "Discovery Track 1", "Artist 3", scores={"novelty_score": 0.8})
        ]
        
        result_state = await enhanced_judge_agent.evaluate_and_select(mock_music_recommender_state)
        
        assert isinstance(result_state, MusicRecommenderState)
        assert len(result_state.final_recommendations) > 0
        assert len(result_state.reasoning_log) > 0
        
        # Check that tracks have explanations
        for rec in result_state.final_recommendations:
            assert "explanation" in rec
            assert rec["explanation"] is not None

    async def test_evaluate_and_select_empty_candidates(self, enhanced_judge_agent: EnhancedJudgeAgent, mock_music_recommender_state: MusicRecommenderState):
        """Test evaluate_and_select with no candidate tracks."""
        # Leave recommendations empty
        result_state = await enhanced_judge_agent.evaluate_and_select(mock_music_recommender_state)
        
        assert isinstance(result_state, MusicRecommenderState)
        assert len(result_state.final_recommendations) == 0
        assert any("No candidate tracks" in log for log in result_state.reasoning_log)

    async def test_evaluate_and_select_prompt_analysis_integration(self, enhanced_judge_agent: EnhancedJudgeAgent, mock_music_recommender_state: MusicRecommenderState):
        """Test that prompt analysis is properly integrated."""
        mock_music_recommender_state.user_query = "I need energetic music for my workout"
        mock_music_recommender_state.genre_mood_recommendations = [
            create_track_dict("gm1", "High Energy Track", "Artist 1", 
                            attributes={"genres": ["electronic"], "additional_scores": {"energy": 0.9}})
        ]
        
        result_state = await enhanced_judge_agent.evaluate_and_select(mock_music_recommender_state)
        
        # Check that prompt analysis was logged
        assert any("Analyzed prompt" in log for log in result_state.reasoning_log)
        assert any("Intent: energy" in log for log in result_state.reasoning_log)


# Backward compatibility tests for the original JudgeAgent methods that are still used
@pytest.mark.asyncio
class TestJudgeAgentBackwardCompatibility:
    """Test class for backward compatibility with original JudgeAgent methods."""

    async def test_parse_candidates_basic(self, judge_agent: JudgeAgent):
        """Test the _parse_candidates method."""
        candidates = [
            create_track_dict("t1", "Track 1", "Artist 1"),
            create_track_dict("t2", "Track 2", "Artist 2")
        ]
        
        parsed = judge_agent._parse_candidates(candidates)
        assert len(parsed) == 2
        assert all(isinstance(track, TrackRecommendation) for track in parsed)

    async def test_parse_candidates_invalid_data(self, judge_agent: JudgeAgent):
        """Test _parse_candidates with invalid data."""
        candidates = [
            {"invalid": "data"},  # Missing required fields
            create_track_dict("t1", "Valid Track", "Artist 1")
        ]
        
        parsed = judge_agent._parse_candidates(candidates)
        assert len(parsed) == 1  # Only valid track should be parsed
        assert parsed[0].title == "Valid Track"

    def test_ensure_diversity_basic(self, judge_agent: JudgeAgent):
        """Test the _ensure_diversity method."""
        candidates = [
            create_track_recommendation_obj("t1", 0.9, genres=["rock"]),
            create_track_recommendation_obj("t2", 0.8, genres=["jazz"]),
            create_track_recommendation_obj("t3", 0.7, genres=["electronic"])
        ]
        
        diversity_targets = {
            "attributes": ["genres"],
            "genres": 2
        }
        
        result = judge_agent._ensure_diversity(candidates, diversity_targets, num_recommendations=2)
        assert len(result) == 2
        
        # Should select tracks with different genres
        selected_genres = set()
        for track in result:
            selected_genres.update(track.genres)
        assert len(selected_genres) >= 2


def create_track_recommendation_obj(
    id: str,
    judge_score: float,
    title: str = "Test Track",
    artist: str = "Test Artist",
    source: str = "test_source",
    genres: Optional[List[str]] = None,
    era: Optional[str] = None,
    moods: Optional[List[str]] = None,
    # Add other relevant diversity attributes as Optional params
    **kwargs: Any  # For any other TrackRecommendation fields
) -> TrackRecommendation:
    """Helper function to create TrackRecommendation objects for testing."""
    track_data = {
        "id": id,
        "title": title,
        "artist": artist,
        "source": source,
        "judge_score": judge_score,
        "genres": genres or [],
        "era": era,
        "moods": moods or [],
        **kwargs
    }
    return TrackRecommendation(**track_data)

