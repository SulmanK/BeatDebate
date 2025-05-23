# Phase 2: PlannerAgent & 4-Agent MVP System - Design Document

**Date**: January 2025  
**Author**: BeatDebate Team  
**Status**: Implementation Ready  
**Review Status**: Approved  

---

## 1. Problem Statement

**Objective**: Implement a 4-agent MVP system with sophisticated planning capabilities to demonstrate agentic behavior for the AgentX competition.

**Current State**: We have validated Last.fm data sources and a working API client. Now we need to build the core agent system that showcases strategic planning and coordination.

**Value Proposition**: 
- Demonstrate advanced agentic planning behavior (AgentX core requirement)
- Show sophisticated multi-agent coordination
- Provide transparent reasoning chains for all decisions
- Create a working MVP that can be extended in later phases

---

## 2. Goals & Non-Goals

### ✅ In Scope (Phase 2)
- **PlannerAgent**: Strategic query analysis and coordination planning
- **GenreMoodAgent**: Strategy-guided genre/mood search implementation  
- **DiscoveryAgent**: Strategy-guided similarity/discovery search implementation
- **JudgeAgent**: Strategy-informed multi-criteria decision making
- **LangGraph Workflow**: Complete orchestration of 4-agent process
- **State Management**: Shared state across agents
- **Basic Testing**: Unit tests for each agent

### ❌ Out of Scope (Phase 2)
- Frontend implementation (Phase 3)
- Complex UI/UX features (Phase 3)  
- Advanced caching optimization (Phase 3)
- Performance monitoring (Phase 3)
- User authentication (Post-MVP)
- Full playlist generation (Post-MVP)

---

## 3. Technical Architecture

### 3.1 Agent Architecture Overview

```
User Query: "I need focus music for coding"
     ↓
🧠 PlannerAgent: Strategic Analysis & Coordination Planning
├─ Task Analysis: "Concentration-optimized music discovery"
├─ Agent Strategies: Defines search parameters for advocates
├─ Success Criteria: Sets evaluation framework for judge
└─ Execution Plan: Monitors and adapts coordination
     ↓
Coordinated Advocate Execution (Parallel):
├─ 🎸 GenreMoodAgent: Genre/mood-based search with strategy
└─ 🔍 DiscoveryAgent: Similarity/discovery search with strategy
     ↓
⚖️ JudgeAgent: Strategy-Informed Decision Making
├─ Applies PlannerAgent's evaluation criteria
├─ Multi-criteria analysis of candidate tracks
└─ Generates explanations with reasoning transparency
     ↓
🎵 Response: 3 strategically selected tracks with full reasoning chains
```

### 3.2 LangGraph State Management

```python
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

class MusicRecommenderState(BaseModel):
    """Shared state across all agents in the workflow"""
    user_query: str
    user_profile: Optional[Dict[str, Any]] = None
    
    # Planning phase
    planning_strategy: Optional[Dict[str, Any]] = None
    execution_plan: Optional[Dict[str, Any]] = None
    
    # Advocate phase  
    genre_mood_recommendations: List[Dict] = []
    discovery_recommendations: List[Dict] = []
    
    # Judge phase
    final_recommendations: List[Dict] = []
    
    # Reasoning transparency
    reasoning_log: List[str] = []
    agent_deliberations: List[Dict] = []
    
    # Metadata
    processing_start_time: Optional[float] = None
    total_processing_time: Optional[float] = None
```

### 3.3 Agent Specifications

#### 3.3.1 PlannerAgent
**Responsibility**: Strategic coordination and planning engine

**Key Methods**:
```python
class PlannerAgent:
    async def create_music_discovery_strategy(self, state: MusicRecommenderState) -> Dict
    async def _extract_primary_intent(self, query: str) -> str
    async def _assess_query_complexity(self, query: str) -> str  
    async def _plan_genre_strategy(self, query: str) -> Dict
    async def _plan_discovery_strategy(self, query: str) -> Dict
    async def _calculate_criteria_weights(self, query: str) -> Dict
```

**Strategy Output Format**:
```python
{
    "task_analysis": {
        "primary_goal": "concentration_music",
        "complexity_level": "medium", 
        "context_factors": ["work", "focus", "instrumental_preference"]
    },
    "coordination_strategy": {
        "genre_mood_agent": {
            "focus_areas": ["instrumental", "ambient", "post-rock"],
            "energy_level": "medium-low",
            "search_tags": ["focus", "study", "instrumental"]
        },
        "discovery_agent": {
            "novelty_priority": "high",
            "similarity_base": "coding_music_archetypes", 
            "underground_bias": 0.7
        }
    },
    "evaluation_framework": {
        "primary_weights": {
            "concentration_friendliness": 0.4,
            "novelty": 0.3,
            "quality": 0.3
        },
        "diversity_targets": {"genre": 2, "era": 2, "energy": 1}
    }
}
```

#### 3.3.2 GenreMoodAgent  
**Responsibility**: Strategy-guided genre and mood-based search

**Key Methods**:
```python
class GenreMoodAgent:
    async def execute_strategy(self, state: MusicRecommenderState) -> List[Dict]
    async def _search_by_mood_tags(self, strategy: Dict) -> List[Dict]
    async def _apply_genre_filters(self, tracks: List[Dict], strategy: Dict) -> List[Dict]
    async def _generate_reasoning_chain(self, recommendation: Dict, strategy: Dict) -> str
```

#### 3.3.3 DiscoveryAgent
**Responsibility**: Strategy-guided similarity and underground discovery

**Key Methods**:
```python
class DiscoveryAgent:
    async def execute_strategy(self, state: MusicRecommenderState) -> List[Dict]
    async def _find_similar_underground_tracks(self, strategy: Dict) -> List[Dict]
    async def _apply_novelty_scoring(self, tracks: List[Dict]) -> List[Dict]
    async def _generate_reasoning_chain(self, recommendation: Dict, strategy: Dict) -> str
```

#### 3.3.4 JudgeAgent
**Responsibility**: Strategy-informed multi-criteria decision making

**Key Methods**:
```python
class JudgeAgent:
    async def evaluate_and_select(self, state: MusicRecommenderState) -> List[Dict]
    async def _apply_evaluation_framework(self, candidates: List[Dict], strategy: Dict) -> List[Dict]
    async def _generate_final_explanations(self, selections: List[Dict], strategy: Dict) -> List[Dict]
    async def _ensure_diversity(self, candidates: List[Dict], targets: Dict) -> List[Dict]
```

---

## 4. Implementation Plan

### 4.1 Development Phases

#### Week 2 Phase 2.1: Core Agent Infrastructure (Days 1-2)
- [ ] Create base agent class with common functionality
- [ ] Implement PlannerAgent with strategic planning logic
- [ ] Set up LangGraph state management
- [ ] Create agent factory and configuration system
- [ ] Basic unit tests for PlannerAgent

#### Week 2 Phase 2.2: Advocate Agents (Days 3-4)  
- [ ] Implement GenreMoodAgent with strategy execution
- [ ] Implement DiscoveryAgent with novelty scoring
- [ ] Integration with Last.fm API client
- [ ] ChromaDB integration for similarity search
- [ ] Unit tests for advocate agents

#### Week 2 Phase 2.3: Judge Agent & Workflow (Days 5-7)
- [ ] Implement JudgeAgent with multi-criteria evaluation
- [ ] Complete LangGraph workflow orchestration
- [ ] End-to-end integration testing
- [ ] Error handling and recovery mechanisms
- [ ] Performance optimization and logging

### 4.2 File Structure
```
src/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py           # Common agent functionality
│   ├── planner_agent.py        # Strategic planning and coordination
│   ├── genre_mood_agent.py     # Genre/mood-based search
│   ├── discovery_agent.py      # Similarity/discovery search  
│   └── judge_agent.py          # Multi-criteria decision making
├── services/
│   ├── recommendation_engine.py # LangGraph workflow orchestration
│   ├── state_manager.py        # State management utilities
│   └── reasoning_chain.py      # Reasoning transparency utilities
└── models/
    ├── agent_models.py         # Pydantic models for agents
    └── recommendation_models.py # Response format models
```

### 4.3 Testing Strategy
- **Unit Tests**: Each agent class with mocked dependencies
- **Integration Tests**: Full workflow with test data
- **Performance Tests**: Response time and resource usage
- **Demo Scenarios**: Curated examples for AgentX presentation

---

## 5. Success Criteria

### 5.1 Functional Requirements
- [ ] PlannerAgent creates sophisticated strategy objects
- [ ] Advocate agents execute strategies with reasoning chains
- [ ] JudgeAgent applies evaluation frameworks effectively
- [ ] Complete workflow processes queries end-to-end
- [ ] All agent interactions are logged for transparency

### 5.2 Performance Requirements  
- [ ] End-to-end response time < 30 seconds
- [ ] Strategy generation < 5 seconds
- [ ] Each advocate search < 10 seconds
- [ ] Judge evaluation < 5 seconds
- [ ] 95% success rate on test queries

### 5.3 AgentX Competition Requirements
- [ ] Demonstrates sophisticated agentic planning behavior
- [ ] Shows clear agent coordination and communication
- [ ] Provides transparent reasoning chains for all decisions
- [ ] Handles complex queries with strategic decomposition
- [ ] Showcases innovation in multi-agent orchestration

---

## 6. Risk Mitigation

### 6.1 Technical Risks
**Risk**: API rate limiting during agent coordination  
**Mitigation**: Implement request queuing and caching

**Risk**: LangGraph state management complexity  
**Mitigation**: Start with simple state, iterate incrementally

**Risk**: Agent reasoning quality inconsistency  
**Mitigation**: Comprehensive prompt engineering and testing

### 6.2 Timeline Risks
**Risk**: Agent coordination more complex than expected  
**Mitigation**: Implement incremental workflow (2-agent → 3-agent → 4-agent)

**Risk**: LLM response latency impacting demo  
**Mitigation**: Implement response caching and async optimization

---

## 7. Next Steps

1. **Immediate**: Begin Phase 2.1 implementation with base agent infrastructure
2. **Week 2 End**: Complete 4-agent MVP system with basic testing
3. **Week 3 Start**: Move to Phase 3 frontend implementation
4. **Ongoing**: Document all design decisions for AgentX submission

This design positions us to showcase sophisticated agentic planning behavior while maintaining a clear implementation timeline for the AgentX competition deadline. 