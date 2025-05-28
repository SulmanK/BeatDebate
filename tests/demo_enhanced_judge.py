#!/usr/bin/env python3
"""
Demonstration of Enhanced JudgeAgent with Prompt-Driven Ranking

This script shows how the enhanced JudgeAgent analyzes user prompts and 
applies sophisticated ranking based on intent, context, and conversational appropriateness.
"""

import asyncio
import logging
from typing import Dict, List

from src.agents.judge_agent import EnhancedJudgeAgent
from src.models.agent_models import MusicRecommenderState
from src.models.recommendation_models import TrackRecommendation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_tracks() -> List[Dict]:
    """Create sample tracks for demonstration"""
    return [
        {
            "id": "track1",
            "title": "Ambient Focus",
            "artist": "Concentration Master",
            "source": "demo",
            "genres": ["ambient", "instrumental"],
            "moods": ["calm", "focused"],
            "instrumental": True,
            "concentration_friendliness_score": 0.9,
            "quality_score": 0.8,
            "novelty_score": 0.4,
            "additional_scores": {"energy": 0.3}
        },
        {
            "id": "track2", 
            "title": "High Energy Workout",
            "artist": "Pump It Up",
            "source": "demo",
            "genres": ["electronic", "dance"],
            "moods": ["energetic", "motivational"],
            "instrumental": False,
            "quality_score": 0.7,
            "novelty_score": 0.3,
            "additional_scores": {"energy": 0.9}
        },
        {
            "id": "track3",
            "title": "Underground Experimental",
            "artist": "Hidden Gem",
            "source": "demo", 
            "genres": ["experimental", "indie"],
            "moods": ["mysterious", "unique"],
            "instrumental": True,
            "quality_score": 0.6,
            "novelty_score": 0.9,
            "additional_scores": {"energy": 0.5}
        },
        {
            "id": "track4",
            "title": "Chill Jazz Evening",
            "artist": "Smooth Operator",
            "source": "demo",
            "genres": ["jazz", "lounge"],
            "moods": ["relaxing", "smooth"],
            "instrumental": False,
            "quality_score": 0.8,
            "novelty_score": 0.5,
            "additional_scores": {"energy": 0.2}
        },
        {
            "id": "track5",
            "title": "Post-Rock Journey",
            "artist": "Skyward Dreams",
            "source": "demo",
            "genres": ["post-rock", "instrumental"],
            "moods": ["epic", "emotional"],
            "instrumental": True,
            "concentration_friendliness_score": 0.7,
            "quality_score": 0.9,
            "novelty_score": 0.6,
            "additional_scores": {"energy": 0.6}
        }
    ]


async def demonstrate_prompt_analysis(judge_agent: EnhancedJudgeAgent, prompts: List[str]):
    """Demonstrate prompt analysis capabilities"""
    print("\n" + "="*60)
    print("PROMPT ANALYSIS DEMONSTRATION")
    print("="*60)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        analysis = judge_agent.prompt_analyzer.analyze_prompt(prompt)
        
        print(f"Primary Intent: {analysis['primary_intent']}")
        print(f"Activity Context: {analysis['activity_context']}")
        print(f"Mood Request: {analysis['mood_request']}")
        print(f"Genre Preferences: {analysis['genre_preferences']}")
        print(f"Exploration Openness: {analysis['exploration_openness']:.2f}")
        print(f"Specificity Level: {analysis['specificity_level']:.2f}")
        print(f"Energy Level: {analysis['energy_level']}")


async def demonstrate_ranking(judge_agent: EnhancedJudgeAgent, prompt: str, tracks: List[Dict]):
    """Demonstrate the complete ranking process"""
    print("\n" + "="*60)
    print("PROMPT-DRIVEN RANKING DEMONSTRATION")
    print("="*60)
    
    print(f"\nUser Prompt: '{prompt}'")
    print("-" * 40)
    
    # Analyze prompt
    prompt_analysis = judge_agent.prompt_analyzer.analyze_prompt(prompt)
    
    # Parse tracks
    parsed_tracks = judge_agent._parse_candidates(tracks)
    
    # Apply ranking
    ranked_results = await judge_agent._apply_prompt_driven_ranking(parsed_tracks, prompt_analysis)
    
    print(f"\nRanking Results (Intent: {prompt_analysis['primary_intent']}):")
    print("-" * 40)
    
    for i, (track, final_score, factor_scores) in enumerate(ranked_results, 1):
        print(f"\n{i}. {track.title} by {track.artist}")
        print(f"   Final Score: {final_score:.3f}")
        print(f"   Intent Alignment: {factor_scores['intent_alignment']:.3f}")
        print(f"   Contextual Relevance: {factor_scores['contextual_relevance']:.3f}")
        print(f"   Discovery Appropriateness: {factor_scores['discovery_appropriateness']:.3f}")
        print(f"   Quality Score: {factor_scores['quality_score']:.3f}")
        print(f"   Conversational Fit: {factor_scores['conversational_fit']:.3f}")


async def demonstrate_full_workflow(judge_agent: EnhancedJudgeAgent, prompt: str, tracks: List[Dict]):
    """Demonstrate the complete enhanced workflow"""
    print("\n" + "="*60)
    print("COMPLETE ENHANCED WORKFLOW DEMONSTRATION")
    print("="*60)
    
    # Create state
    state = MusicRecommenderState(user_query=prompt)
    state.genre_mood_recommendations = tracks[:3]  # First 3 tracks from GenreMoodAgent
    state.discovery_recommendations = tracks[3:]   # Remaining tracks from DiscoveryAgent
    state.conversation_context = {}
    state.planning_strategy = {
        "evaluation_framework": {
            "diversity_targets": {
                "attributes": ["genres"],
                "genres": 2
            }
        }
    }
    
    print(f"\nUser Query: '{prompt}'")
    print(f"Candidate Tracks: {len(tracks)} total")
    print(f"  - GenreMoodAgent: {len(state.genre_mood_recommendations)} tracks")
    print(f"  - DiscoveryAgent: {len(state.discovery_recommendations)} tracks")
    
    # Process with enhanced agent
    result_state = await judge_agent.evaluate_and_select(state)
    
    print(f"\nFinal Recommendations: {len(result_state.final_recommendations)}")
    print("-" * 40)
    
    for i, rec in enumerate(result_state.final_recommendations, 1):
        print(f"\n{i}. {rec['title']} by {rec['artist']}")
        print(f"   Judge Score: {rec.get('judge_score', 'N/A'):.3f}")
        print(f"   Genres: {', '.join(rec.get('genres', []))}")
        print(f"   Explanation: {rec.get('explanation', 'No explanation')}")
    
    print(f"\nReasoning Log:")
    print("-" * 20)
    for log_entry in result_state.reasoning_log:
        print(f"  • {log_entry}")


async def main():
    """Main demonstration function"""
    print("Enhanced JudgeAgent Demonstration")
    print("Prompt-Driven Music Recommendation Ranking")
    
    # Initialize enhanced agent
    judge_agent = EnhancedJudgeAgent()
    
    # Sample prompts for different scenarios
    test_prompts = [
        "I need focus music for coding",
        "Give me energetic music for my workout",
        "Surprise me with something new and experimental", 
        "I want relaxing jazz for the evening",
        "Find me some underground indie rock"
    ]
    
    # Sample tracks
    sample_tracks = create_sample_tracks()
    
    # Demonstrate prompt analysis
    await demonstrate_prompt_analysis(judge_agent, test_prompts)
    
    # Demonstrate ranking for specific scenarios
    for prompt in test_prompts[:3]:  # First 3 prompts
        await demonstrate_ranking(judge_agent, prompt, sample_tracks)
    
    # Demonstrate complete workflow
    await demonstrate_full_workflow(
        judge_agent, 
        "I need focus music for coding", 
        sample_tracks
    )
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("• Prompt Analysis Engine - Extracts intent, context, and preferences")
    print("• Contextual Relevance Scoring - Matches tracks to activities and moods")
    print("• Discovery Appropriateness - Balances exploration vs familiarity")
    print("• Intent-Weighted Ranking - Prioritizes based on user intent")
    print("• Conversational Explanations - Generates prompt-referencing explanations")
    print("• Enhanced Workflow - Complete 100→20 candidate filtering pipeline")


if __name__ == "__main__":
    asyncio.run(main()) 