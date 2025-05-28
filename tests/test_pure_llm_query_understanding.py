"""
Test script for Pure LLM Query Understanding System

This script tests the new query understanding engine with various music queries
to verify it correctly identifies intent, extracts entities, and provides
high confidence scores.
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.query_understanding import QueryUnderstandingEngine, QueryIntent
from src.services.recommendation_engine import create_gemini_client

async def test_query_understanding():
    """Test the Pure LLM Query Understanding system."""
    
    # Get API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY environment variable not set.")
        print("Please set your Gemini API key in the .env file or environment.")
        return
    
    # Initialize the system
    gemini_client = create_gemini_client(gemini_api_key)
    
    if not gemini_client:
        print("❌ Failed to create Gemini client. Check your API key.")
        return
    
    query_engine = QueryUnderstandingEngine(gemini_client)
    
    # Test queries
    test_queries = [
        "Music like Mk.gee",
        "Happy workout music",
        "Some good jazz for studying", 
        "Surprise me with something new",
        "Songs similar to Radiohead",
        "Chill electronic music for coding",
        "Upbeat pop songs for running",
        "Discover underground hip hop artists"
    ]
    
    print("🎵 Testing Pure LLM Query Understanding System\n")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing Query: '{query}'")
        print("-" * 50)
        
        try:
            # Get understanding
            understanding = await query_engine.understand_query(query)
            
            # Display results
            print(f"Intent: {understanding.intent.value}")
            print(f"Confidence: {understanding.confidence:.2f}")
            print(f"Primary Agent: {understanding.primary_agent}")
            
            if understanding.artists:
                print(f"Artists: {', '.join(understanding.artists)}")
            if understanding.genres:
                print(f"Genres: {', '.join(understanding.genres)}")
            if understanding.moods:
                print(f"Moods: {', '.join(understanding.moods)}")
            if understanding.activities:
                print(f"Activities: {', '.join(understanding.activities)}")
            
            if understanding.similarity_type:
                print(f"Similarity Type: {understanding.similarity_type.value}")
            
            print(f"Exploration Level: {understanding.exploration_level}")
            
            if understanding.agent_weights:
                print("Agent Weights:")
                for agent, weight in understanding.agent_weights.items():
                    print(f"  {agent}: {weight:.1f}")
            
            print(f"Reasoning: {understanding.reasoning}")
            
            # Confidence assessment
            if understanding.confidence >= 0.8:
                confidence_level = "🟢 HIGH"
            elif understanding.confidence >= 0.6:
                confidence_level = "🟡 MEDIUM"
            else:
                confidence_level = "🔴 LOW"
            
            print(f"Confidence Assessment: {confidence_level}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Pure LLM Query Understanding Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_query_understanding()) 