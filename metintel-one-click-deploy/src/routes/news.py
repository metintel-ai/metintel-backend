"""
News routes for PreciousAI Market News feature
"""

from flask import Blueprint, request, jsonify
from src.services.market_news_service import MarketNewsService
import logging

logger = logging.getLogger(__name__)

# Create blueprint
news_bp = Blueprint('news', __name__)

# Initialize service
news_service = MarketNewsService()

@news_bp.route('/latest', methods=['GET'])
def get_latest_news():
    """Get latest precious metals news"""
    try:
        # Get query parameters
        metal = request.args.get('metal')  # gold, silver, platinum, palladium
        limit = int(request.args.get('limit', 10))
        
        # Validate limit
        if limit > 50:
            limit = 50
        elif limit < 1:
            limit = 10
        
        # Get news data
        news_data = news_service.get_latest_news(metal=metal, limit=limit)
        
        return jsonify({
            'success': True,
            'data': news_data,
            'timestamp': news_data.get('timestamp')
        })
        
    except Exception as e:
        logger.error(f"Error getting latest news: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch latest news',
            'message': str(e)
        }), 500

@news_bp.route('/category/<category>', methods=['GET'])
def get_news_by_category(category):
    """Get news by specific category"""
    try:
        # Validate category
        valid_categories = ['gold', 'silver', 'platinum', 'palladium', 'general']
        if category.lower() not in valid_categories:
            return jsonify({
                'success': False,
                'error': 'Invalid category',
                'valid_categories': valid_categories
            }), 400
        
        limit = int(request.args.get('limit', 5))
        if limit > 20:
            limit = 20
        
        # Get category news
        news_data = news_service.get_news_by_category(category=category.lower(), limit=limit)
        
        return jsonify({
            'success': True,
            'data': news_data,
            'category': category.lower()
        })
        
    except Exception as e:
        logger.error(f"Error getting news by category: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch category news',
            'message': str(e)
        }), 500

@news_bp.route('/sentiment', methods=['GET'])
def get_market_sentiment():
    """Get overall market sentiment analysis"""
    try:
        sentiment_data = news_service.get_market_sentiment()
        
        return jsonify({
            'success': True,
            'data': sentiment_data,
            'timestamp': sentiment_data.get('last_updated')
        })
        
    except Exception as e:
        logger.error(f"Error getting market sentiment: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch market sentiment',
            'message': str(e)
        }), 500

@news_bp.route('/search', methods=['GET'])
def search_news():
    """Search for specific news topics"""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({
                'success': False,
                'error': 'Search query is required',
                'message': 'Please provide a search query using the "q" parameter'
            }), 400
        
        limit = int(request.args.get('limit', 10))
        if limit > 20:
            limit = 20
        
        # Search news
        search_results = news_service.search_news(query=query, limit=limit)
        
        return jsonify({
            'success': True,
            'data': search_results,
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Error searching news: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to search news',
            'message': str(e)
        }), 500

@news_bp.route('/metals', methods=['GET'])
def get_all_metals_news():
    """Get news for all precious metals with breakdown"""
    try:
        metals = ['gold', 'silver', 'platinum', 'palladium']
        limit_per_metal = int(request.args.get('limit', 3))
        
        all_metals_news = {}
        
        for metal in metals:
            try:
                metal_news = news_service.get_latest_news(metal=metal, limit=limit_per_metal)
                all_metals_news[metal] = metal_news
            except Exception as e:
                logger.warning(f"Error getting news for {metal}: {e}")
                all_metals_news[metal] = {
                    'error': f'Failed to fetch {metal} news',
                    'timestamp': None,
                    'news_articles': []
                }
        
        # Get general market summary
        try:
            general_news = news_service.get_latest_news(metal=None, limit=5)
            market_summary = general_news.get('market_summary', {})
        except:
            market_summary = {}
        
        return jsonify({
            'success': True,
            'data': {
                'market_summary': market_summary,
                'metals_breakdown': all_metals_news,
                'timestamp': general_news.get('timestamp') if 'general_news' in locals() else None
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting all metals news: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch metals news',
            'message': str(e)
        }), 500

@news_bp.route('/health', methods=['GET'])
def news_health_check():
    """Health check for news service"""
    try:
        # Test basic functionality
        test_news = news_service._get_demo_news('gold', 1)
        
        return jsonify({
            'success': True,
            'service': 'Market News Service',
            'status': 'operational',
            'features': [
                'Latest news',
                'Category filtering',
                'Market sentiment',
                'News search',
                'AI analysis'
            ],
            'timestamp': test_news.get('timestamp')
        })
        
    except Exception as e:
        logger.error(f"News service health check failed: {e}")
        return jsonify({
            'success': False,
            'service': 'Market News Service',
            'status': 'error',
            'error': str(e)
        }), 500

