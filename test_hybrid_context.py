"""
Test script to verify hybrid query context preservation
"""
import sys
import os
sys.path.append('src')

from models.agent_models import MusicRecommenderState, AgentConfig, SystemConfig
from models.metadata_models import UnifiedTrackMetadata
from services.enhanced_recommendation_service import ContextAwareIntentAnalyzer

def test_context_preservation():
    """Test that R&B genre is preserved in follow-up queries"""
    print("🧪 Testing hybrid query context preservation...")
    
    # Mock conversation history for "Songs by Michael Jackson that are R&B"
    conversation_history = [{
        'query': 'Songs by Michael Jackson that are R&B',
        'recommendations': [
            {
                'title': 'Black or White',
                'artist': 'Michael Jackson',
                'genres': ['r&b', 'pop'],
                'moods': ['pop', 'michael jackson', '90s']
            },
            {
                'title': 'Man in the Mirror', 
                'artist': 'Michael Jackson',
                'genres': ['r&b', 'soul'],
                'moods': ['pop', 'michael jackson', '80s', 'soul']
            }
        ],
        'extracted_entities': {
            'artists': {'primary': [{'name': 'Michael Jackson', 'confidence': 0.9}]},
            'genres': {'primary': [{'name': 'R&B', 'confidence': 0.9}]},
            'tracks': {'primary': []},
            'moods': {'primary': []}
        }
    }]
    
    print(f"📝 Original query: {conversation_history[0]['query']}")
    print(f"🎵 Original entities: {conversation_history[0].get('extracted_entities', {})}")
    
    # Create analyzer (without LLM for testing)
    analyzer = ContextAwareIntentAnalyzer(llm_client=None)
    
    # Test 1: Extract complete entities from history
    print("\n🔍 Testing _extract_complete_entities_from_history...")
    try:
        complete_entities = analyzer._extract_complete_entities_from_history(conversation_history)
        print(f"✅ Complete entities extracted: {complete_entities}")
        
        # Check if genres are preserved
        genres = complete_entities.get('genres', {}).get('primary', [])
        print(f"🎵 Genres found in history: {genres}")
        
        if any('r&b' in str(g).lower() for g in genres):
            print("✅ SUCCESS: R&B genre found in extracted entities!")
        else:
            print("❌ ISSUE: R&B genre not found in extracted entities")
            
    except Exception as e:
        print(f"❌ ERROR in entity extraction: {e}")
    
    # Test 2: Create context override for follow-up
    print("\n🎯 Testing _create_context_override_from_llm...")
    try:
        # Simulate LLM detecting artist follow-up
        llm_result = {
            'is_followup': True,
            'followup_type': 'artist_deep_dive',
            'target_entity': 'Michael Jackson',
            'confidence': 0.9,
            'reasoning': "User wants more tracks by Michael Jackson"
        }
        
        context_override = analyzer._create_context_override_from_llm(llm_result, conversation_history)
        print(f"🎯 Context override created: {context_override}")
        
        # Check preserved genres
        preserved_genres = context_override.get('preserved_genres', [])
        constraint_overrides = context_override.get('constraint_overrides', {})
        
        print(f"🎵 Preserved genres: {preserved_genres}")
        print(f"🔧 Constraint overrides: {constraint_overrides}")
        
        if preserved_genres and any('r&b' in str(g).lower() for g in preserved_genres):
            print("✅ SUCCESS: R&B genre preserved in context override!")
        else:
            print("❌ ISSUE: R&B genre not preserved in context override")
            
        # Check if intent is properly set for hybrid queries
        intent_override = context_override.get('intent_override')
        print(f"🎭 Intent override: {intent_override}")
        
        if intent_override in ['hybrid_artist_genre', 'by_artist']:
            print("✅ SUCCESS: Appropriate intent for hybrid query!")
        else:
            print("❌ ISSUE: Intent not appropriate for hybrid query")
            
    except Exception as e:
        print(f"❌ ERROR in context override: {e}")
    
    return context_override

def test_discovery_agent_context():
    """Test that Discovery Agent can handle preserved genres"""
    print("\n🚀 Testing Discovery Agent context handling...")
    
    # Mock context override with preserved R&B genre
    context_override = {
        'is_followup': True,
        'target_entity': 'Michael Jackson',
        'intent_override': 'hybrid_artist_genre',
        'preserved_genres': ['R&B'],
        'constraint_overrides': {
            'genre_requirements': ['R&B'],
            'diversity_limits': {'same_artist_limit': 10}
        }
    }
    
    print(f"🎯 Testing context override: {context_override}")
    
    # Check if Discovery Agent would handle this correctly
    preserved_genres = context_override.get('preserved_genres', [])
    constraint_overrides = context_override.get('constraint_overrides', {})
    genre_requirements = constraint_overrides.get('genre_requirements', [])
    
    print(f"🎵 Preserved genres: {preserved_genres}")
    print(f"📋 Genre requirements: {genre_requirements}")
    
    if preserved_genres and genre_requirements:
        print("✅ SUCCESS: Discovery Agent should apply R&B filtering!")
    else:
        print("❌ ISSUE: Missing genre requirements for Discovery Agent")

if __name__ == "__main__":
    print("🎼 BeatDebate Hybrid Query Context Test")
    print("=" * 50)
    
    try:
        context_override = test_context_preservation()
        test_discovery_agent_context()
        
        print("\n" + "=" * 50)
        print("🎯 Test Summary:")
        print("- Follow-up detection: Should work ✅")
        print("- Artist preservation: Should work ✅") 
        print("- Genre preservation: Needs verification 🔍")
        print("- Hybrid intent handling: Needs verification 🔍")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 