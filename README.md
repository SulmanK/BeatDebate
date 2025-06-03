# 🎵 BeatDebate

**Multi-Agent Music Recommendation System for AgentX Competition**

BeatDebate is a sophisticated music recommendation system that uses 4 specialized AI agents to discover under-the-radar tracks through intelligent debate and strategic planning. Built for the AgentX competition, it demonstrates advanced agentic planning behavior in a real-world application.

## 🎯 Key Features

- **Strategic Planning**: `PlannerAgent` analyzes queries and orchestrates a multi-step recommendation strategy using Gemini.
- **Multi-Agent System**: Four specialized agents (`Planner`, `GenreMood`, `Discovery`, `Judge`) collaborate within a LangGraph workflow.
- **Intent-Aware Recommendations**: The system adapts its scoring and diversity logic based on the user's detected intent (e.g., artist similarity, pure discovery, contextual needs).
- **Underground Discovery**: `DiscoveryAgent` focuses on indie, lesser-known tracks, and serendipitous finds.
- **Explainable AI**: `JudgeAgent` provides transparent reasoning for each recommendation, linking back to the planning strategy.
- **Conversational Interface**: A Gradio-based UI allows for natural language interaction and displays rich track information.
- **Contextual Conversations**: `SmartContextManager` and `ContextAwareIntentAnalyzer` enable multi-turn dialogues.
- **$0 Cost Goal**: Built leveraging free tiers (Gemini, Last.fm, Spotify API access, HuggingFace for hosting).

## 🏗️ Architecture

The core workflow follows this sequence, orchestrated by LangGraph:

```
User Query → PlannerAgent (Strategy & Intent Analysis) → [GenreMoodAgent || DiscoveryAgent] (Candidate Generation) → JudgeAgent (Ranking & Explanation) → Formatted Response
```

### Agent Roles:
- **🧠 PlannerAgent**:
    -   Analyzes user queries using its `QueryUnderstandingEngine` (powered by Gemini and pattern matching).
    -   Extracts entities, detects intent (including hybrid intents and context overrides).
    -   Generates a comprehensive `planning_strategy` dict detailing task analysis, advocate agent coordination parameters, and the `evaluation_framework` for the Judge.
- **🎸 GenreMoodAgent**:
    -   Executes the planner's strategy for genre and mood-based discovery.
    -   Uses `UnifiedCandidateGenerator` and `ComprehensiveQualityScorer` (from `src/agents/components/`) to fetch and score tracks.
    -   Employs `MoodLogic` and `TagGenerator` for nuanced style matching.
- **🔍 DiscoveryAgent**:
    -   Focuses on similarity, novelty, and underground tracks as per the planner's strategy.
    -   Also uses `UnifiedCandidateGenerator` and `ComprehensiveQualityScorer`.
    -   Internal components like `SimilarityExplorer` and `UndergroundDetector` aid in finding unique recommendations.
- **⚖️ JudgeAgent**:
    -   Evaluates candidates from both advocate agents against the planner's `evaluation_framework`.
    -   Applies intent-aware `RankingLogic` to score and select the final tracks.
    -   Uses `ConversationalExplainer` (potentially with LLM) to generate explanations.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- API keys for Gemini, Last.fm, and Spotify (see `.env.example`)

### Installation

1.  **Clone and setup environment**
    ```bash
    git clone https://github.com/SulmanK/BeatDebate.git
    cd BeatDebate
    ```

2.  **Install `uv` dependency manager**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Setup project and install dependencies**
    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    uv sync --dev
    ```

4.  **Configure environment variables**
    ```bash
    cp env.example .env
    # Edit .env with your API keys:
    # GEMINI_API_KEY=your_gemini_api_key
    # LASTFM_API_KEY=your_lastfm_api_key
    # SPOTIFY_CLIENT_ID=your_spotify_client_id
    # SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
    # Optional: LASTFM_SHARED_SECRET
    ```

5.  **Run the application**
    ```bash
    uv run python -m src.main
    ```
    The Gradio interface will typically be available at `http://localhost:7860` and the FastAPI backend at `http://localhost:8000`.

## 🧪 Development

### Data Validation
(Scripts to test API responses and data quality)
```bash
uv run python scripts/validate_lastfm.py
# uv run python scripts/validate_spotify.py # (If you create this)
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run a specific test file
uv run pytest tests/agents/test_planner_agent.py
```

### Code Quality
```bash
# Format code
uv run black src/ tests/
uv run isort src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

## 📁 Project Structure

The project is organized into distinct layers and components:

```
beatDebate/
├── Design/                  # Design documents, plans, and refactoring notes
├── scripts/                 # Utility and validation scripts (e.g., validate_lastfm.py)
├── src/                     # Main source code
│   ├── agents/              # Core AI agent implementations
│   │   ├── base_agent.py    # Abstract base class for all agents
│   │   ├── components/      # Shared utilities for agents
│   │   │   ├── llm_utils.py
│   │   │   ├── entity_extraction_utils.py
│   │   │   ├── query_analysis_utils.py
│   │   │   ├── unified_candidate_generator.py
│   │   │   └── quality_scorer.py
│   │   ├── planner/         # PlannerAgent and its specific modules
│   │   │   ├── agent.py     # PlannerAgent logic
│   │   │   └── query_understanding_engine.py
│   │   ├── genre_mood/      # GenreMoodAgent and its specific modules
│   │   │   ├── agent.py
│   │   │   └── mood_logic.py, tag_generator.py
│   │   ├── discovery/       # DiscoveryAgent and its specific modules
│   │   │   ├── agent.py
│   │   │   └── similarity_explorer.py, underground_detector.py
│   │   └── judge/           # JudgeAgent and its specific modules
│   │       ├── agent.py
│   │       └── ranking_logic.py, explainer.py
│   ├── api/                 # FastAPI backend and external API clients
│   │   ├── backend.py       # FastAPI application exposing endpoints
│   │   ├── base_client.py   # Base class for Last.fm/Spotify clients
│   │   ├── lastfm_client.py
│   │   ├── spotify_client.py
│   │   ├── rate_limiter.py  # UnifiedRateLimiter
│   │   ├── client_factory.py# For creating API client instances
│   │   └── logging_middleware.py # For API request/response logging
│   ├── models/              # Pydantic data models and schemas
│   │   ├── agent_models.py  # MusicRecommenderState, AgentConfig, etc.
│   │   ├── metadata_models.py # UnifiedTrackMetadata, etc.
│   │   └── recommendation_models.py # TrackRecommendation
│   ├── services/            # Business logic and service orchestration
│   │   ├── enhanced_recommendation_service.py # Main service, orchestrates LangGraph
│   │   ├── api_service.py   # Centralized access to external API clients
│   │   ├── metadata_service.py# Unified metadata operations
│   │   ├── conversation_context_service.py # Manages session data
│   │   ├── smart_context_manager.py # Decides context handling strategy
│   │   └── cache_manager.py # Caching for API responses (using diskcache)
│   ├── ui/                  # Gradio user interface components
│   │   ├── chat_interface.py# Main Gradio UI layout and logic
│   │   ├── response_formatter.py # Formats recommendations for display
│   │   └── planning_display.py # (For visualizing planner strategy - if integrated)
│   ├── utils/               # Shared utility functions
│   │   └── logging_config.py# Centralized logging setup (structlog)
│   └── main.py              # Application entry point, launches FastAPI & Gradio
├── tests/                   # Unit and integration tests
├── data/                    # Local data, cache, validation outputs (managed by .gitignore)
├── .env.example             # Example environment file
├── pyproject.toml           # Project dependencies and build configuration
└── README.md
```

## 🎵 Usage Examples

### Basic Music Discovery
```
You: "I need focus music for coding"

🧠 PlannerAgent: "Analyzing coding music requirements, setting intent to 'contextual' with activity 'coding'..."
🎸 GenreMoodAgent: "Fetching instrumental, ambient, post-rock based on plan..."
🔍 DiscoveryAgent: "Searching for lesser-known artists suitable for study/focus..."
⚖️ JudgeAgent: "Evaluating candidates. Prioritizing 'concentration_friendliness_score' and quality. Selecting optimal tracks..."

🎵 Results: 3 tracks like "Ambient Focus" by Concentration Master, with explanations referencing coding and focus.
```

### Artist Similarity with Contextual Refinement
```
You: "Music like Mk.gee"
🤖 BeatDebate: (Recommends some Mk.gee-like tracks)
You: "More Mk.gee tracks, but make them more electronic"

🧠 PlannerAgent: "Context override detected: 'artist_style_refinement' for Mk.gee with 'electronic' style. Updating coordination strategy."
🎸 GenreMoodAgent: "Focusing on Mk.gee's discography, filtering for electronic elements and related tags..."
🔍 DiscoveryAgent: "Looking for Mk.gee tracks or very close collaborators with strong electronic tags..."
⚖️ JudgeAgent: "Prioritizing Mk.gee tracks matching 'electronic'. Evaluating based on similarity to Mk.gee's core style AND electronic fit..."

🎵 Results: Mk.gee tracks that lean electronic, or similar artists known for that specific fusion.
```

## 🏆 AgentX Competition

BeatDebate demonstrates sophisticated **agentic planning behavior** for the AgentX competition:

- **Strategic Planning**: `PlannerAgent` creates comprehensive, LLM-driven recommendation strategies.
- **Agent Coordination**: Structured communication via `MusicRecommenderState` and targeted strategies.
- **Reasoning Transparency**: `reasoning_log` in `MusicRecommenderState` and explanations from `JudgeAgent`.
- **Technical Innovation**: Novel application of multi-agent planning to music recommendation, including intent-aware logic and context management.

## 📊 Technical Details

### Core Technologies
- **Backend**: FastAPI, Python 3.11
- **Agent Orchestration**: LangGraph
- **LLM**: Google Gemini (via `langchain-google-genai`)
- **Data Models**: Pydantic
- **Frontend**: Gradio
- **Dependency Management**: `uv`
- **Logging**: `structlog`

### Rate Limiting Strategy
- Implemented via `UnifiedRateLimiter` (`src/api/rate_limiter.py`) and configured per service (Gemini, Last.fm, Spotify) in `APIClientFactory`.
- **Gemini**: Default 15 calls/minute (configurable, e.g., 8-12 for safety).
- **Last.fm**: Default 3 calls/second.
- **Spotify**: Default 50 calls/hour.

### Caching & Performance
- **`CacheManager` (`src/services/cache_manager.py`):** Uses `diskcache` for file-based caching of API responses and track metadata with configurable TTLs.
- **Async Processing**: FastAPI and `aiohttp` (in `BaseAPIClient`) ensure non-blocking I/O for external API calls. LangGraph orchestrates agents asynchronously.
- Request optimization and careful LLM use aim to keep costs low and performance acceptable.

### Data Sources
- **Last.fm API**: Primary source for track/artist metadata, tags, and similarity information. Accessed via `LastFmClient`.
- **Spotify Web API**: Secondary source for audio previews and potentially audio features (though full audio feature integration for scoring is a future enhancement). Accessed via `SpotifyClient`.
- **Text Embeddings (Future Enhancement)**: Design allows for future integration of sentence transformers for semantic search (e.g., with ChromaDB), but current MVP focuses on API-driven metadata and LLM reasoning.

## 🚀 Deployment

### HuggingFace Spaces
The application is structured for easy deployment to HuggingFace Spaces. The `src.main:create_gradio_app()` function is the entry point for Spaces.

### Local Development
```bash
# Start the backend and frontend (FastAPI runs on port 8000, Gradio on 7860 by default)
uv run python -m src.main
```
Alternatively, for hot-reloading of the FastAPI backend during development:
```bash
# Terminal 1: Start FastAPI backend
uv run uvicorn src.api.backend:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start Gradio UI (pointing to the backend)
# (Ensure your chat_interface.py is configured to use http://127.0.0.1:8000 if run separately)
# Or simply run the main `uv run python -m src.main` which handles both.
```

## 🤝 Contributing

1.  **Fork the repository.**
2.  **Create feature branch**: `git checkout -b feature/your-new-feature`
3.  **Install dependencies**: `uv sync --dev`
4.  **Make your changes.**
5.  **Follow code standards**: Run `uv run black .`, `uv run isort .`, `uv run ruff check .`
6.  **Add tests**: Ensure new functionality is covered by tests.
7.  **Run tests**: `uv run pytest`
8.  **Update docs**: If applicable, update README and relevant design documents.
9.  **Create Pull Request**: Submit for review.

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- **Primary Design Document**: `Design/Plans/beatdebate-design-doc.md`
- **AgentX Course**: [LLM Agents Learning @ Stanford](https://llmagents-learning.org/sp25)
- **HuggingFace Space**: [Coming Soon]
```