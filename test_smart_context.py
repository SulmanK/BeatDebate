"""
Test script to demonstrate Smart Context Manager functionality
"""

import asyncio
import json
from src.services.smart_context_manager import SmartContextManager, ContextState


async def test_smart_context_scenarios():
    """Test different conversation scenarios with smart context management."""
    
    print("🧠 Testing Smart Context Manager\n")
    print("="*50)
    
    context_manager = SmartContextManager()
    session_id = "test_session_123"
    
    # Test scenarios
    scenarios = [
        {
            "name": "🎵 Scenario 1: Initial Artist Similarity Query",
            "query": "Music like Mk.gee",
            "expected_decision": ContextState.NEW_SESSION.value
        },
        {
            "name": "🔄 Scenario 2: Follow-up for Same Artist",
            "query": "More tracks like that",
            "expected_decision": ContextState.CONTINUING.value
        },
        {
            "name": "🎭 Scenario 3: Different Artist (Intent Switch)",
            "query": "Music like Chief Keef",
            "expected_decision": ContextState.INTENT_SWITCH.value
        },
        {
            "name": "🎯 Scenario 4: Continuation Signal",
            "query": "Also something similar to Chief Keef",
            "expected_decision": ContextState.PREFERENCE_REFINEMENT.value
        },
        {
            "name": "🔄 Scenario 5: Reset Trigger",
            "query": "Actually, never mind. Something completely different",
            "expected_decision": ContextState.RESET_NEEDED.value
        },
        {
            "name": "🎧 Scenario 6: New Activity Context",
            "query": "Music for working out",
            "expected_decision": ContextState.NEW_SESSION.value  # After reset
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{scenario['name']}")
        print("-" * 60)
        
        # Simulate LLM understanding (you can replace with actual LLM calls)
        llm_understanding = simulate_llm_understanding(scenario["query"])
        
        # Get context decision
        context_decision = await context_manager.analyze_context_decision(
            current_query=scenario["query"],
            session_id=session_id,
            llm_understanding=llm_understanding
        )
        
        # Print results
        print(f"Query: '{scenario['query']}'")
        print(f"Decision: {context_decision['decision']}")
        print(f"Action: {context_decision['action']}")
        print(f"Confidence: {context_decision['confidence']:.2f}")
        print(f"Reasoning: {context_decision['reasoning']}")
        
        # Check if matches expected
        expected = scenario["expected_decision"]
        actual = context_decision["decision"]
        status = "✅ EXPECTED" if actual == expected else f"❌ UNEXPECTED (expected: {expected})"
        print(f"Status: {status}")
        
        # Simulate updating context after recommendation
        mock_recommendations = [
            {
                "title": f"Mock Track {i}",
                "artist": f"Mock Artist {i}",
                "id": f"mock_{i}",
                "source": "test"
            }
        ]
        
        await context_manager.update_context_after_recommendation(
            session_id=session_id,
            query=scenario["query"],
            llm_understanding=llm_understanding,
            recommendations=mock_recommendations,
            context_decision=context_decision
        )
        
        # Show session analysis details
        if "session_analysis" in context_decision:
            analysis = context_decision["session_analysis"]
            print(f"\n📊 Analysis Details:")
            print(f"  Intent Analysis: {analysis.get('intent_analysis', {}).get('intent_changed', 'N/A')}")
            print(f"  Continuity Score: {analysis.get('continuity_analysis', {}).get('continuity_score', 'N/A'):.2f}")
            print(f"  Temporal Relevance: {analysis.get('temporal_analysis', {}).get('relevance_score', 'N/A'):.2f}")
    
    # Final context summary
    print(f"\n{'='*50}")
    print("📋 Final Session Summary")
    print("="*50)
    
    context_summary = await context_manager.get_context_summary(session_id)
    print(json.dumps(context_summary, indent=2, default=str))


def simulate_llm_understanding(query: str) -> dict:
    """Simulate LLM understanding for testing purposes."""
    
    # Simple pattern matching for demo
    query_lower = query.lower()
    
    llm_data = {
        "intent": {"value": "discovery"},
        "artists": [],
        "genres": [],
        "moods": [],
        "activities": [],
        "confidence": 0.8
    }
    
    # Artist detection
    if "mk.gee" in query_lower:
        llm_data["intent"]["value"] = "artist_similarity"
        llm_data["artists"] = ["Mk.gee"]
    elif "chief keef" in query_lower:
        llm_data["intent"]["value"] = "artist_similarity"
        llm_data["artists"] = ["Chief Keef"]
    
    # Activity detection
    if "workout" in query_lower or "working out" in query_lower:
        llm_data["intent"]["value"] = "activity_context"
        llm_data["activities"] = ["workout"]
    
    # Continuation detection
    if any(word in query_lower for word in ["more", "also", "similar", "like that"]):
        llm_data["intent"]["value"] = "conversation_continuation"
    
    return llm_data


def print_context_decision_explanation():
    """Print explanation of context decision types."""
    
    print("🧠 Smart Context Manager - Decision Types")
    print("="*50)
    print()
    
    explanations = {
        "NEW_SESSION": "🆕 No existing context - start fresh",
        "CONTINUING": "➡️  Maintain current context - user is continuing the same thread",
        "INTENT_SWITCH": "🔄 Major intent change detected - context needs updating",
        "PREFERENCE_REFINEMENT": "🎯 User is refining preferences within same context",
        "RESET_NEEDED": "🔄 Explicit reset trigger or stale context - start over"
    }
    
    for decision_type, explanation in explanations.items():
        print(f"{decision_type}: {explanation}")
    
    print()
    print("📊 Key Analysis Factors:")
    print("• Intent Change: Does the query target a different artist/genre/activity?")
    print("• Temporal Relevance: How recent was the last interaction?")
    print("• Continuity Signals: Words like 'more', 'also', 'similar', references")
    print("• Reset Triggers: Phrases like 'actually', 'never mind', 'different'")
    print()


if __name__ == "__main__":
    print_context_decision_explanation()
    asyncio.run(test_smart_context_scenarios()) 