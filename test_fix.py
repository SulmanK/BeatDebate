import requests
import json

def test_api():
    url = 'http://127.0.0.1:8000/recommendations'
    data = {
        'query': 'Music like Mk.gee',
        'session_id': 'test-session-fix-2',
        'max_recommendations': 3
    }
    
    try:
        print("Testing API endpoint...")
        response = requests.post(url, json=data, timeout=60)
        result = response.json()
        
        print(f'Status Code: {response.status_code}')
        print(f'Number of recommendations: {len(result.get("recommendations", []))}')
        
        if result.get('recommendations'):
            print('\nFirst recommendation:')
            first_rec = result['recommendations'][0]
            print(f'  Title: {first_rec.get("title", "N/A")}')
            print(f'  Artist: {first_rec.get("artist", "N/A")}')
            print(f'  Explanation: {first_rec.get("explanation", "N/A")}')
        
        print('\nReasoning Log (first 3):')
        for i, reason in enumerate(result.get('reasoning_log', [])[:3]):
            print(f'  {i+1}. {reason}')
            
        return response.status_code == 200 and len(result.get('recommendations', [])) > 0
            
    except Exception as e:
        print(f'Error: {e}')
        return False

if __name__ == "__main__":
    success = test_api()
    print(f'\nTest {"PASSED" if success else "FAILED"}') 