# 🎵 BeatDebate

**Multi-Agent Music Recommendation System for AgentX Competition**

BeatDebate is a sophisticated music recommendation system that uses 4 specialized AI agents to discover under-the-radar tracks through intelligent debate and strategic planning. Built for the AgentX competition, it demonstrates advanced agentic planning behavior in a real-world application.

## 🎯 Key Features

- **Strategic Planning**: PlannerAgent coordinates recommendation strategy
- **Multi-Agent Debate**: 4 specialized agents collaborate and compete
- **Underground Discovery**: Focus on indie and lesser-known tracks  
- **Explainable AI**: Transparent reasoning for every recommendation
- **Conversational Interface**: ChatGPT-style music discovery experience
- **$0 Cost**: Built on free tiers (Gemini, Last.fm, Spotify, HuggingFace)

## 🏗️ Architecture

```
User Query → PlannerAgent (Strategy) → [GenreMoodAgent || DiscoveryAgent] → JudgeAgent → Response
```

### Agent Roles
- **🧠 PlannerAgent**: Strategic coordination and planning (AgentX focus)
- **🎸 GenreMoodAgent**: Genre and mood specialist 
- **🔍 DiscoveryAgent**: Similarity and underground discovery specialist
- **⚖️ JudgeAgent**: Multi-criteria decision making with explanations

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- API keys for Gemini, Last.fm, and Spotify

### Installation

1. **Clone and setup environment**
```bash
git clone <your-repo-url>
cd beatdebate
```

2. **Install uv dependency manager**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Setup project**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --dev
```

4. **Configure environment**
```bash
cp env.example .env
# Edit .env with your API keys
```

5. **Run the application**
```bash
uv run python src/main.py
```

### API Keys Setup

You'll need:
- **Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Last.fm API Key**: Get from [Last.fm API](https://www.last.fm/api/account/create)
- **Spotify Client ID/Secret**: Get from [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)

## 🧪 Development

### Data Validation
Before building the full system, validate data sources:
```bash
uv run python scripts/validate_lastfm.py
uv run python scripts/validate_spotify.py
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test
uv run pytest tests/test_agents.py
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

```
beatdebate/
├── src/
│   ├── agents/          # Agent implementations
│   │   ├── base_agent.py
│   │   ├── planner_agent.py      # Strategic coordinator
│   │   ├── genre_mood_agent.py   # Genre/mood specialist
│   │   ├── discovery_agent.py    # Similarity specialist
│   │   └── judge_agent.py        # Decision maker
│   ├── api/             # External API clients
│   │   ├── lastfm_client.py
│   │   └── spotify_client.py
│   ├── models/          # Data models and schemas
│   ├── services/        # Business logic
│   │   ├── recommendation_engine.py  # LangGraph workflow
│   │   ├── cache_manager.py
│   │   └── vector_store.py
│   ├── ui/              # Frontend components
│   │   └── chat_interface.py
│   └── main.py          # Application entry point
├── tests/               # Test files
├── data/                # Local data and cache
├── scripts/             # Utility scripts
├── docs/                # Documentation
└── Design/              # Design documents
```

## 🎵 Usage Examples

### Basic Music Discovery
```
You: "I need focus music for coding"

🧠 PlannerAgent: "Analyzing coding music requirements..."
🎸 GenreMoodAgent: "Searching instrumental post-rock..."
🔍 DiscoveryAgent: "Finding underground study music..."
⚖️ JudgeAgent: "Selecting optimal recommendations..."

🎵 Results: 3 tracks with detailed explanations and previews
```

### Mood-Based Search
```
You: "Something upbeat but not mainstream"

🧠 PlannerAgent: "Planning discovery for high-energy indie tracks..."
[Agent coordination and reasoning...]
🎵 Results: Fresh indie tracks with energy explanations
```

## 🏆 AgentX Competition

BeatDebate demonstrates sophisticated **agentic planning behavior** for the AgentX competition:

- **Strategic Planning**: PlannerAgent creates comprehensive recommendation strategies
- **Agent Coordination**: Sophisticated inter-agent communication protocols  
- **Reasoning Transparency**: All agent decisions are explainable and traceable
- **Technical Innovation**: Novel approach to music recommendation through multi-agent planning

### Competition Positioning
- **Track**: Entrepreneurship (consumer music application)
- **Key Innovation**: Planning-driven multi-agent recommendation system
- **Market Opportunity**: $25B music streaming + AI personalization trends
- **Competitive Advantage**: Only explainable multi-agent music recommender

## 📊 Technical Details

### Rate Limiting Strategy
- **Gemini**: 12 calls/minute (4 agents × 3 calls each)
- **Last.fm**: 3 calls/second with aggressive caching
- **Spotify**: 50 calls/hour with batch requests

### Caching & Performance
- **File-based cache** with TTL for API responses
- **ChromaDB** for vector similarity search
- **Async processing** for agent coordination
- **Request optimization** to stay within free tiers

### Data Sources
- **Last.fm**: 15M+ tracks with rich metadata and indie focus
- **Spotify**: Audio features and 30-second previews
- **Text embeddings**: Sentence transformers for semantic search

## 🚀 Deployment

### HuggingFace Spaces
The application is designed for HuggingFace Spaces deployment:
```bash
# Production build
uv run python -m src.main
```

### Local Development
```bash
# Development server with hot reload
uv run uvicorn src.main:app --reload --port 7860
```

## 🤝 Contributing

1. **Create feature branch**: `git checkout -b feature/new-feature`
2. **Follow code standards**: Run `uv run black` and `uv run ruff`
3. **Add tests**: Include tests for new functionality
4. **Update docs**: Update README and docstrings
5. **Create PR**: Submit for review

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- **Design Document**: `Design/Plans/beatdebate-design-doc.md`
- **AgentX Course**: [LLM Agents Learning](https://llmagents-learning.org/sp25)
- **HuggingFace Spaces**: [Coming Soon]
- **Demo Video**: [Coming Soon]

---

**Built for AgentX Competition 2025** | **Demonstrates Advanced Agentic Planning** 