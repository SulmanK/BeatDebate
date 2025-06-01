#!/usr/bin/env python3
"""
Debug script to test entity structure and genre detection.
"""

# Test entities structures from the logs
entities_from_log = {
    'musical_entities': {
        'artists': {
            'primary': [{'name': 'Michael Jackson', 'confidence': 0.9}], 
            'secondary': [], 
            'similar_to': []
        }, 
        'genres': {
            'primary': [{'name': 'R&B', 'confidence': 0.9}], 
            'secondary': [], 
            'similar_to': []
        }, 
        'tracks': {
            'primary': [], 
            'secondary': [], 
            'similar_to': []
        }, 
        'moods': {
            'primary': [], 
            'secondary': [], 
            'similar_to': []
        }
    }
}

def _has_genre_requirements(entities) -> bool:
    """Check if entities contain genre requirements."""
    print("Testing _has_genre_requirements...")
    
    # Try nested format first (musical_entities wrapper)
    print(f"entities.get('musical_entities'): {entities.get('musical_entities')}")
    musical_entities = entities.get('musical_entities', {})
    print(f"musical_entities.get('genres'): {musical_entities.get('genres')}")
    genres_dict = musical_entities.get('genres', {})
    print(f"genres_dict.get('primary'): {genres_dict.get('primary')}")
    
    if genres_dict.get('primary'):
        print("✅ Found genres in musical_entities.genres.primary")
        return True
    
    # Try direct format
    elif entities.get('genres', {}).get('primary'):
        print("✅ Found genres in direct entities.genres.primary")
        return True
    
    # Try simple list format
    elif entities.get('genres') and isinstance(entities['genres'], list):
        print("✅ Found genres in simple list format")
        return len(entities['genres']) > 0
    
    print("❌ No genres found in any expected format")
    return False

def _extract_required_genres_for_filtering(entities):
    """Extract required genres for strict filtering."""
    required_genres = []
    
    # Try nested format first (musical_entities wrapper)
    musical_entities = entities.get('musical_entities', {})
    if musical_entities.get('genres', {}).get('primary'):
        required_genres = musical_entities['genres']['primary']
        print(f"Found genres in musical_entities: {required_genres}")
    # Try direct format
    elif entities.get('genres', {}).get('primary'):
        required_genres = entities['genres']['primary']
        print(f"Found genres in direct format: {required_genres}")
    # Try simple list format
    elif entities.get('genres') and isinstance(entities['genres'], list):
        required_genres = entities['genres']
        print(f"Found genres in list format: {required_genres}")
    
    # Convert to strings and clean up
    if required_genres and isinstance(required_genres[0], dict):
        # Handle structured format with name/confidence
        required_genres = [genre['name'].lower().strip() for genre in required_genres if genre.get('name')]
        print(f"Converted structured genres to strings: {required_genres}")
    else:
        required_genres = [str(genre).lower().strip() for genre in required_genres if genre]
        print(f"Converted to strings: {required_genres}")
    
    return required_genres

print("🧪 Testing entity structure and genre detection...")
print("\nTesting with entities from log:")
print(f"Has genre requirements: {_has_genre_requirements(entities_from_log)}")
print(f"Extracted genres: {_extract_required_genres_for_filtering(entities_from_log)}")

# Test simple format too
simple_entities = {
    'genres': {'primary': ['r&b']}
}

print(f"\nTesting with simple format:")
print(f"Has genre requirements: {_has_genre_requirements(simple_entities)}")
print(f"Extracted genres: {_extract_required_genres_for_filtering(simple_entities)}")

print("\n�� Debug completed!") 