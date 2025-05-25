# LangGraph Workflow Orchestration - Design Document

**Date**: January 2025
**Author**: BeatDebate Team
**Status**: Draft
**Review Status**: Pending

---

## 0. References

*   Overall Phase 2 Design: [phase2_planner_agent_design.md](phase2_planner_agent_design.md)
*   Main Project Design: [Plans/beatdebate-design-doc.md](Plans/beatdebate-design-doc.md)
*   PlannerAgent Design: [phase2_planner_agent_design.md](phase2_planner_agent_design.md) (contains PlannerAgent details)
*   JudgeAgent Design: [phase2_judge_agent_design.md](phase2_judge_agent_design.md)
*   Agent Implementations: `src/agents/`
*   State Model: `src/models/agent_models.py` (expecting `MusicRecommenderState` or similar)

---

## 1. Problem Statement

The primary objective is to orchestrate the four core agents of the BeatDebate system (`PlannerAgent`, `GenreMoodAgent`, `DiscoveryAgent`, and `JudgeAgent`) into a cohesive, sequential, and partially parallel workflow using LangGraph. This orchestration is essential for processing a user's music query from initial planning through to final recommendation selection and explanation.

The current state is that individual agents (`PlannerAgent`, `GenreMoodAgent`, `DiscoveryAgent`, `JudgeAgent`) have been designed and implemented (as per `src/agents/`). The API clients (`src/api/`) and data models (`src/models/`) are also in place. What's missing is the central nervous system that connects these components, manages the flow of data (state), and handles the execution sequence.

The value of this orchestration includes:
-   Demonstrating a functional multi-agent system.
-   Enabling end-to-end processing of user queries.
-   Providing a clear structure for managing complex agent interactions.
-   Facilitating transparency and debugging through LangGraph's inherent state tracking.

---

## 2. Goals & Non-Goals

### ✅ In Scope
-   Defining the LangGraph graph structure, including nodes and edges, to represent the 4-agent workflow.
-   Implementing the graph in `src/services/recommendation_engine.py`.
-   Utilizing the `MusicRecommenderState` (likely defined in `src/models/agent_models.py` or a dedicated state management file like `src/services/state_manager.py` as hinted in `Design/phase2_planner_agent_design.md`) as the shared state object that flows through the graph.
-   Defining nodes for each agent:
    -   `PlannerAgentNode`: Executes the `PlannerAgent`'s `create_music_discovery_strategy` (or similar method as per `src/agents/planner_agent.py`).
    -   `GenreMoodAgentNode`: Executes the `GenreMoodAgent`'s strategy execution method.
    -   `DiscoveryAgentNode`: Executes the `DiscoveryAgent`'s strategy execution method.
    -   `JudgeAgentNode`: Executes the `JudgeAgent`'s `evaluate_and_select` method.
-   Implementing parallel execution for `GenreMoodAgentNode` and `DiscoveryAgentNode`.
-   Defining conditional edges if necessary (e.g., handling cases where the Planner fails or no candidates are found by advocates).
-   Ensuring the final output is a populated `MusicRecommenderState` with `final_recommendations` and `reasoning_log`.
-   Basic error handling within graph nodes to update the state with error information.
-   Logging of graph execution steps and state transitions.

### ❌ Out of Scope
-   Advanced error recovery strategies beyond logging and basic state updates (e.g., automated retries with modified parameters).
-   Complex dynamic routing or agent selection beyond the defined 4-agent sequence.
-   User interface integration for visualizing the graph execution (this is a UI concern).
-   Detailed performance optimization of individual agent calls (focus is on orchestration).
-   Implementation of the `MusicRecommenderState` itself if not already fully defined (this design assumes it exists or will be created based on `Design/phase2_planner_agent_design.md`).

---

## 3. Technical Architecture

### 3.1 System Overview

The LangGraph workflow will be the core of the `recommendation_engine.py` service. It will take an initial `MusicRecommenderState` (primarily with `user_query` populated) and pass it through a sequence of agent nodes.

### 3.2 LangGraph State

The shared state for the graph will be an instance of `MusicRecommenderState`. Based on `Design/phase2_planner_agent_design.md`, this state should include at least:

```python
# Expected structure from src/models/agent_models.py or similar
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field # Assuming Pydantic is used

class MusicRecommenderState(BaseModel):
    user_query: str
    user_profile: Optional[Dict[str, Any]] = None # Not used in MVP workflow directly but good to have
    
    # Planning phase output
    planning_strategy: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Advocate phase outputs (as lists of dictionaries, to be parsed by Judge or if models are used directly)
    genre_mood_recommendations: List[Dict] = Field(default_factory=list) # Or List[TrackRecommendationDict]
    discovery_recommendations: List[Dict] = Field(default_factory=list) # Or List[TrackRecommendationDict]
    
    # Judge phase output
    final_recommendations: List[Dict] = Field(default_factory=list) # Or List[TrackRecommendationDict]
    
    # Reasoning & Logging
    reasoning_log: List[str] = Field(default_factory=list)
    agent_deliberations: List[Dict] = Field(default_factory=list) # For more structured logs
    
    # Error Handling
    error_info: Optional[Dict[str, str]] = None # e.g., {"agent": "PlannerAgent", "message": "API timeout"}

    # Metadata
    processing_start_time: Optional[float] = None
    total_processing_time: Optional[float] = None
```
*Actual Pydantic model for `TrackRecommendation` would be used if advocates return model instances, but design docs lean towards List[Dict] passed between agents in the state.*

### 3.3 Graph Definition

The graph will be constructed using `langgraph.StateGraph`.

**Nodes**:

1.  **`planner_node`**:
    *   **Function**: Wraps `PlannerAgent.create_music_discovery_strategy()` (or its equivalent async method from `src/agents/planner_agent.py`).
    *   **Input**: `MusicRecommenderState` (uses `user_query`).
    *   **Output**: Updates `state.planning_strategy` and `state.reasoning_log`.
    *   **Error Handling**: If planning fails, updates `state.error_info` and `state.reasoning_log`.

2.  **`genre_mood_advocate_node`**:
    *   **Function**: Wraps `GenreMoodAgent.execute_strategy()` (or its equivalent from `src/agents/genre_mood_agent.py`).
    *   **Input**: `MusicRecommenderState` (uses `planning_strategy`).
    *   **Output**: Updates `state.genre_mood_recommendations` and `state.reasoning_log`.
    *   **Error Handling**: Updates `state.error_info` if it fails.

3.  **`discovery_advocate_node`**:
    *   **Function**: Wraps `DiscoveryAgent.execute_strategy()` (or its equivalent from `src/agents/discovery_agent.py`).
    *   **Input**: `MusicRecommenderState` (uses `planning_strategy`).
    *   **Output**: Updates `state.discovery_recommendations` and `state.reasoning_log`.
    *   **Error Handling**: Updates `state.error_info` if it fails.

4.  **`judge_node`**:
    *   **Function**: Wraps `JudgeAgent.evaluate_and_select()`.
    *   **Input**: `MusicRecommenderState` (uses `planning_strategy`, `genre_mood_recommendations`, `discovery_recommendations`).
    *   **Output**: Updates `state.final_recommendations` and `state.reasoning_log`.
    *   **Error Handling**: Updates `state.error_info` if it fails.

**Edges**:

*   **START** → `planner_node`
*   `planner_node` → **`should_proceed_after_planning` (Conditional Edge)**
    *   If `state.error_info` is set by `planner_node` OR `state.planning_strategy` is empty/invalid: → **END**
    *   Else: → `execute_advocates_parallel` (a conceptual branch point)

*   **`execute_advocates_parallel`** (This isn't a node itself, but LangGraph supports fanning out to multiple nodes that can run in parallel if their inputs are ready. We'll add edges from `planner_node` (or the conditional node success branch) to both advocate nodes.)
    *   `planner_node` (on success) → `genre_mood_advocate_node`
    *   `planner_node` (on success) → `discovery_advocate_node`
    (LangGraph will execute these concurrently as they don't depend on each other's direct output, only on the `planner_node`'s output in the state.)

*   After both `genre_mood_advocate_node` AND `discovery_advocate_node` complete, they will converge to the `judge_node`. LangGraph handles this implicitly by defining `judge_node` as the next step after both. We need a mechanism to gather results, which LangGraph handles via the shared state object. The `judge_node` will be added after both advocate nodes.
    *   `genre_mood_advocate_node` → `judge_node`
    *   `discovery_advocate_node` → `judge_node`
    *(LangGraph's compilation will create a join point before the judge if these are set as direct next steps for the parallel branches.)*

*   `judge_node` → **END**

**Workflow Diagram**:
```
[START]
   ↓
┌────────────────┐
│  PlannerAgent  │
│ (planner_node) │
└────────────────┘
   ↓
┌───────────────────────────┐
│ should_proceed_after_plan │ (Conditional)
└───────────────────────────┘
   ├── (Error/No Plan) → [END]
   ↓ (Success)
┌────────────────┐   ┌──────────────────┐
│ GenreMoodAgent │   │ DiscoveryAgent   │
│ (gm_node)      │   │ (discovery_node) │
└────────────────┘   └──────────────────┘
   │                 │
   └───────▼─────────┘
           ↓ (Implicit Join)
┌────────────────┐
│   JudgeAgent   │
│  (judge_node)  │
└────────────────┘
   ↓
[END]
```

### 3.4 `RecommendationEngine` Service

A class or a set of functions in `src/services/recommendation_engine.py` will be responsible for:
1.  Initializing agent instances.
2.  Defining the LangGraph `StateGraph`.
3.  Adding nodes and edges as described above.
4.  Compiling the graph.
5.  Providing a main method to invoke the graph with an initial `user_query`.

```python
# Example structure in src/services/recommendation_engine.py
from langgraph.graph import StateGraph, END
from src.models.agent_models import MusicRecommenderState # Assuming this path
from src.agents.planner_agent import PlannerAgent
from src.agents.genre_mood_agent import GenreMoodAgent
from src.agents.discovery_agent import DiscoveryAgent
from src.agents.judge_agent import JudgeAgent
import time # For processing_start_time

class RecommendationEngine:
    def __init__(self):
        self.planner_agent = PlannerAgent()
        self.genre_mood_agent = GenreMoodAgent()
        self.discovery_agent = DiscoveryAgent()
        self.judge_agent = JudgeAgent()
        self.graph = self._build_graph()

    async def _planner_node_func(self, state: MusicRecommenderState):
        # self.logger.info("Executing Planner Node")
        try:
            # Assuming PlannerAgent has an async method like 'process' or 'run'
            # that takes the state and updates it.
            # For example, if it's 'create_music_discovery_strategy'
            updated_planning_strategy = await self.planner_agent.create_music_discovery_strategy(state) # Or equivalent
            state.planning_strategy = updated_planning_strategy
            state.reasoning_log.append("PlannerAgent: Strategy created.")
        except Exception as e:
            # self.logger.error(f"Planner Node Error: {e}")
            state.error_info = {"agent": "PlannerAgent", "message": str(e)}
            state.reasoning_log.append(f"PlannerAgent: Error - {str(e)}")
        return state

    async def _genre_mood_advocate_node_func(self, state: MusicRecommenderState):
        # self.logger.info("Executing GenreMood Advocate Node")
        try:
            # Assuming agent has a method that takes state (esp. planning_strategy)
            # and returns recommendations or updates state directly
            # Example: recommendations = await self.genre_mood_agent.execute_strategy(state.planning_strategy)
            # For now, let's assume agents modify state directly for simplicity in graph nodes
            await self.genre_mood_agent.execute_strategy(state) # Modifies state.genre_mood_recommendations
            state.reasoning_log.append("GenreMoodAgent: Recommendations generated.")
        except Exception as e:
            # self.logger.error(f"GenreMoodAdvocate Node Error: {e}")
            state.error_info = {"agent": "GenreMoodAgent", "message": str(e)} # Potentially overwrite, needs careful thought
            state.reasoning_log.append(f"GenreMoodAgent: Error - {str(e)}")
        return state
        
    async def _discovery_advocate_node_func(self, state: MusicRecommenderState):
        # self.logger.info("Executing Discovery Advocate Node")
        try:
            await self.discovery_agent.execute_strategy(state) # Modifies state.discovery_recommendations
            state.reasoning_log.append("DiscoveryAgent: Recommendations generated.")
        except Exception as e:
            # self.logger.error(f"DiscoveryAdvocate Node Error: {e}")
            state.error_info = {"agent": "DiscoveryAgent", "message": str(e)}
            state.reasoning_log.append(f"DiscoveryAgent: Error - {str(e)}")
        return state

    async def _judge_node_func(self, state: MusicRecommenderState):
        # self.logger.info("Executing Judge Node")
        try:
            # JudgeAgent's evaluate_and_select updates state.final_recommendations
            await self.judge_agent.evaluate_and_select(state)
            state.reasoning_log.append("JudgeAgent: Final selection complete.")
        except Exception as e:
            # self.logger.error(f"Judge Node Error: {e}")
            state.error_info = {"agent": "JudgeAgent", "message": str(e)}
            state.reasoning_log.append(f"JudgeAgent: Error - {str(e)}")
        return state

    def _should_proceed_after_planning(self, state: MusicRecommenderState):
        # self.logger.debug(f"Conditional Edge: Checking planner output. Error: {state.error_info}")
        if state.error_info and state.error_info.get("agent") == "PlannerAgent":
            return "end_workflow"
        if not state.planning_strategy or not state.planning_strategy.get("evaluation_framework"): # Basic check
            # self.logger.warning("Planner did not produce a valid strategy. Ending workflow.")
            state.reasoning_log.append("Workflow: Planner failed to produce strategy, ending.")
            return "end_workflow"
        return "execute_advocates"

    def _build_graph(self):
        workflow = StateGraph(MusicRecommenderState)

        workflow.add_node("planner", self._planner_node_func)
        workflow.add_node("genre_mood_advocate", self._genre_mood_advocate_node_func)
        workflow.add_node("discovery_advocate", self._discovery_advocate_node_func)
        workflow.add_node("judge", self._judge_node_func)

        workflow.set_entry_point("planner")

        workflow.add_conditional_edges(
            "planner",
            self._should_proceed_after_planning,
            {
                "execute_advocates": "genre_mood_advocate", # Start of one parallel branch
                "end_workflow": END
            }
        )
        # The other parallel branch also starts after successful planning
        workflow.add_edge("planner", "discovery_advocate") # This needs careful handling with conditional edges.
                                                           # Langchain typically routes from conditional output.

        # Revised approach for parallel execution after conditional check:
        # The conditional edge from 'planner' will route to a dummy "fan_out_advocates" node if successful,
        # or directly to END. This dummy node then has unconditional edges to both advocates.
        # However, LangGraph allows multiple targets from a conditional branch.
        # A simpler way is for `_should_proceed_after_planning` to return the name of the
        # *first* node in the parallel execution if successful, or END.
        # Let's assume `genre_mood_advocate` is one, and we add another edge for `discovery_advocate`.
        # For true parallelism, they should not depend on each other.

        # If planning succeeds, it goes to 'genre_mood_advocate'.
        # We also need an edge from 'planner' (on success) to 'discovery_advocate'.
        # This is where conditional logic becomes tricky for multiple next steps.
        # A better pattern for conditional branching to parallel execution:
        # planner -> conditional_router
        # conditional_router:
        #   if success: "fan_out_node" (dummy node)
        #   if failure: END
        # fan_out_node -> genre_mood_advocate
        # fan_out_node -> discovery_advocate
        #
        # Or, if the conditional function can return a list of next nodes, that's ideal.
        # LangGraph's `add_conditional_edges` maps a string return to a single next node.
        # For fan-out after condition:
        # 1. `planner`
        # 2. `conditional_planner_gate` (routes to `run_advocates` or `END`)
        # 3. `run_advocates` (dummy node, or just ensure edges are set)
        # From `run_advocates` (or directly from `conditional_planner_gate`'s "success" branch):
        # Edge to `genre_mood_advocate`
        # Edge to `discovery_advocate`

        # For now, let's try to make the conditional router point to one, and add another edge.
        # If `_should_proceed_after_planning` returns "execute_advocates", we go to `genre_mood_advocate`.
        # We also need `discovery_advocate` to run.
        # LangGraph requires that if a node is specified as a target in `add_conditional_edges`,
        # other direct `add_edge` calls from the source of the conditional edge might be ignored or cause conflict.

        # A cleaner pattern:
        # workflow.add_node("fan_out_advocates_anchor", lambda x: x) # Dummy node
        # workflow.add_conditional_edges(
        #     "planner",
        #     self._should_proceed_after_planning,
        #     {
        #         "execute_advocates": "fan_out_advocates_anchor",
        #         "end_workflow": END
        #     }
        # )
        # workflow.add_edge("fan_out_advocates_anchor", "genre_mood_advocate")
        # workflow.add_edge("fan_out_advocates_anchor", "discovery_advocate")

        # Simpler: The conditional edge path for "execute_advocates" will go to "genre_mood_advocate".
        # An additional edge will also be triggered to "discovery_advocate" if the condition isn't "end_workflow".
        # This needs to be expressed carefully.
        # The most straightforward way for parallel branches after a condition is to route to *one* of the parallel
        # tasks and ensure the other parallel task is also set to start *after the same condition point*.
        # This is usually handled by LangGraph if the tasks truly can run in parallel.

        # Edges for advocate to judge (these create an implicit join)
        workflow.add_edge("genre_mood_advocate", "judge")
        workflow.add_edge("discovery_advocate", "judge")
        
        workflow.add_edge("judge", END)
        
        return workflow.compile()

    async def process_query(self, user_query: str) -> MusicRecommenderState:
        initial_state = MusicRecommenderState(user_query=user_query)
        initial_state.processing_start_time = time.time()
        
        final_state = await self.graph.ainvoke(initial_state)
        
        if final_state.processing_start_time: # Should always be true
            final_state.total_processing_time = time.time() - final_state.processing_start_time
        
        return final_state

```
*The parallel execution setup with conditional edges needs careful handling in LangGraph. The above `_build_graph` provides a conceptual sketch. It might require a dummy node or ensuring the conditional routing correctly triggers all parallel starts.*

---

## 4. Implementation Plan

1.  **Verify/Define `MusicRecommenderState`**: Ensure `MusicRecommenderState` is fully defined in `src/models/agent_models.py` (or `src/services/state_manager.py`) covering all necessary fields for the 4-agent workflow, including error handling fields.
2.  **Implement Agent Invocation Wrappers**: Create the `async def _<agent_name>_node_func(self, state: MusicRecommenderState)` methods within `RecommendationEngine`. These methods will:
    *   Call the respective agent's main processing method (e.g., `planner_agent.create_music_discovery_strategy(state)`).
    *   Ensure agents update the passed-in `state` object or return the necessary parts to update the state.
    *   Implement try-except blocks to catch errors, update `state.error_info` and `state.reasoning_log`.
3.  **Implement Conditional Logic**: Implement `_should_proceed_after_planning` to check `state.error_info` and the validity of `state.planning_strategy`.
4.  **Build Graph in `RecommendationEngine`**:
    *   Instantiate `StateGraph` with `MusicRecommenderState`.
    *   Add all agent nodes and the conditional logic node.
    *   Define the entry point.
    *   Add edges, carefully configuring the parallel execution of advocate agents and the subsequent join at the judge agent. This might involve an anchor/dummy node if direct conditional fanning out is complex.
    *   Compile the graph.
5.  **Implement `process_query` method**: This public method on `RecommendationEngine` will initialize the state with the user query, invoke the compiled graph, and return the final state.
6.  **Basic Logging**: Add `logging` calls within node functions and the `RecommendationEngine` for key events and state changes.
7.  **Initial Unit Tests**: Write basic tests for:
    *   `RecommendationEngine` initialization.
    *   Each node function wrapper (mocking the underlying agent) to ensure it correctly updates state and handles errors.
    *   The conditional logic function `_should_proceed_after_planning`.

---

## 5. Testing Strategy

*   **Unit Tests (`tests/services/test_recommendation_engine.py`)**:
    *   Test `RecommendationEngine` initialization.
    *   Test each node function (`_planner_node_func`, `_genre_mood_advocate_node_func`, etc.) by mocking the actual agent calls:
        *   Verify correct state updates on successful agent execution.
        *   Verify correct `state.error_info` and `reasoning_log` updates on agent failure.
    *   Test the `_should_proceed_after_planning` conditional logic with various states.
    *   Test the `process_query` method by mocking the compiled graph's `ainvoke` to ensure state initialization and timing logic works.
*   **Integration Tests (Covered in overall Phase 2.3)**:
    *   Once implemented, this engine will be the core of end-to-end tests.
    *   Test the full graph execution with mocked agents to verify flow and parallel execution.
    *   Then, test with actual agent instances (potentially mocking their external API calls) to ensure data compatibility and correct state flow through the real agents via the graph. Scenarios:
        *   Successful end-to-end run.
        *   Planner failure.
        *   One advocate agent fails, the other succeeds.
        *   Both advocate agents fail or return no candidates.
        *   Judge agent failure.

---

## 6. Success Criteria

*   The LangGraph workflow is successfully implemented in `src/services/recommendation_engine.py`.
*   The system can process a user query from start to finish using the 4-agent pipeline.
*   `GenreMoodAgent` and `DiscoveryAgent` execute in parallel after the `PlannerAgent`.
*   The `MusicRecommenderState` is correctly passed and updated between agents.
*   Basic error conditions (e.g., planner failure) are handled by routing to END or updating state appropriately.
*   The final state contains `final_recommendations` and a `reasoning_log` detailing the process.
*   Unit tests for the orchestration logic pass.

---

## 7. Risk Mitigation

*   **Risk**: Complexity in configuring parallel execution with conditional logic in LangGraph.
    *   **Mitigation**: Refer to LangGraph documentation for best practices on parallel execution and conditional branching. Start with a simpler sequential flow if necessary and incrementally add parallelism. Consider using an explicit "fan-out" dummy node if direct conditional branching to multiple parallel starts is problematic.
*   **Risk**: Agents might not correctly update the shared `MusicRecommenderState` object, leading to data inconsistencies.
    *   **Mitigation**: Clearly define the responsibility of each agent node function to update specific parts of the state. Implement thorough unit tests for node functions verifying state modifications. Ensure agents either modify the state object by reference or return specific Pydantic models that the node function then uses to update the state.
*   **Risk**: Error handling logic in one node might incorrectly halt the entire graph or mask errors.
    *   **Mitigation**: Design error handling in nodes to update `state.error_info` clearly, allowing downstream conditional logic or the final result to reflect partial failures. Ensure `reasoning_log` captures all errors.
*   **Risk**: Difficulty in debugging the state flow through the compiled graph.
    *   **Mitigation**: Utilize LangGraph's built-in debugging capabilities (e.g., `get_graph().print_ascii()`). Implement comprehensive logging within each node and in the `RecommendationEngine` to trace state changes.

---

## 8. Open Questions & Discussion Points

*   **State Modification**: Should agent methods directly modify the passed-in `state` object, or should they return their results, with the node wrapper function responsible for updating the state?
    *   **Proposed Resolution**: For cleaner separation, agent methods (e.g., `planner_agent.create_music_discovery_strategy`) should ideally return their primary output (e.g., a `PlanningStrategy` object or dict). The LangGraph node wrapper function (`_planner_node_func`) will then be responsible for updating the `MusicRecommenderState` based on this return value and handling any errors. This keeps agent logic more self-contained. However, some agents like the JudgeAgent are designed to take the full state and modify it. This needs to be consistent or clearly documented for each agent node. For MVP, direct state modification within agent methods (if they are designed that way) is acceptable for the node functions to call, as long as it's clear. The current agent designs seem to imply they update the state given to them.
*   **Detailed Parallelism Configuration**: The exact LangGraph syntax for ensuring the advocate agents run in parallel *after* a conditional check on the planner's output needs to be finalized.
    *   **Proposed Resolution**: Consult LangGraph examples for conditional fan-out. If `add_conditional_edges` path mapping only supports a single string target, an anchor/dummy node post-condition for fanning out is a standard pattern. E.g., `planner` -> `planner_gate` (conditional) -> (if success) `advocate_fanout_anchor` -> (parallel edges to) `genre_mood_advocate` AND `discovery_advocate`.
*   **Error Overwriting**: If multiple parallel nodes fail, how should `state.error_info` be handled? Last error wins, or a list of errors?
    *   **Proposed Resolution**: For MVP, `state.error_info` can store the first critical error that halts a main path or the last one encountered. More sophisticated error aggregation can be a future enhancement. The `reasoning_log` should capture all errors encountered.

--- 