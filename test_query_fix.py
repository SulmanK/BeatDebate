import asyncio
from src.services.api_service import get_api_service
from src.agents.planner.query_understanding_engine import QueryUnderstandingEngine
from src.services.enhanced_recommendation_service import EnhancedRecommendationService

async def test_query_understanding():
    try:
        rec_service = EnhancedRecommendationService()
        await rec_service.initialize()
        print('Testing query understanding with Music like Mk.gee')
        
        # Test the query understanding directly
        planner = rec_service.planner_agent
        understanding = await planner._understand_user_query('Music like Mk.gee')
        print(f'Understanding: {understanding.intent.value}, confidence: {understanding.confidence}')
        print('Query understanding test passed!')
        
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'rec_service' in locals():
            await rec_service.close()

if __name__ == "__main__":
    asyncio.run(test_query_understanding()) 