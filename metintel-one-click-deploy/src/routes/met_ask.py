"""
Met-Ask routes for PreciousAI Q&A feature
"""

from flask import Blueprint, request, jsonify
from src.services.met_ask_service import MetAskService
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Create blueprint
met_ask_bp = Blueprint('met_ask', __name__)

# Initialize service
met_ask_service = MetAskService()

@met_ask_bp.route('/ask', methods=['POST'])
def ask_question():
    """Ask a question about precious metals"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required',
                'message': 'Please provide a JSON body with your question'
            }), 400
        
        question = data.get('question', '').strip()
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question is required',
                'message': 'Please provide a question in the request body'
            }), 400
        
        # Optional parameters
        user_id = data.get('user_id')
        context = data.get('context', {})
        
        # Validate question length
        if len(question) > 1000:
            return jsonify({
                'success': False,
                'error': 'Question too long',
                'message': 'Please limit your question to 1000 characters'
            }), 400
        
        # Process the question
        result = met_ask_service.ask_question(
            question=question,
            user_id=user_id,
            context=context
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to process question',
            'message': str(e)
        }), 500

@met_ask_bp.route('/popular', methods=['GET'])
def get_popular_questions():
    """Get popular/common questions about precious metals"""
    try:
        popular_questions = met_ask_service.get_popular_questions()
        
        return jsonify({
            'success': True,
            'data': {
                'popular_questions': popular_questions,
                'total_count': len(popular_questions)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting popular questions: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch popular questions',
            'message': str(e)
        }), 500

@met_ask_bp.route('/suggest/<topic>', methods=['GET'])
def suggest_questions(topic):
    """Suggest questions based on a topic"""
    try:
        # Validate topic
        valid_topics = ['gold', 'silver', 'platinum', 'palladium', 'investment', 'market', 'storage', 'trading']
        if topic.lower() not in valid_topics:
            return jsonify({
                'success': False,
                'error': 'Invalid topic',
                'valid_topics': valid_topics,
                'message': f'Topic "{topic}" is not supported'
            }), 400
        
        suggested_questions = met_ask_service.suggest_questions(topic)
        
        return jsonify({
            'success': True,
            'data': {
                'topic': topic.lower(),
                'suggested_questions': suggested_questions,
                'count': len(suggested_questions)
            }
        })
        
    except Exception as e:
        logger.error(f"Error suggesting questions for topic {topic}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to suggest questions',
            'message': str(e)
        }), 500

@met_ask_bp.route('/history/<user_id>', methods=['GET'])
def get_conversation_history(user_id):
    """Get conversation history for a user"""
    try:
        if not user_id or len(user_id) < 3:
            return jsonify({
                'success': False,
                'error': 'Invalid user ID',
                'message': 'User ID must be at least 3 characters long'
            }), 400
        
        limit = int(request.args.get('limit', 5))
        if limit > 20:
            limit = 20
        elif limit < 1:
            limit = 5
        
        history = met_ask_service.get_conversation_history(user_id, limit)
        
        return jsonify({
            'success': True,
            'data': {
                'user_id': user_id,
                'conversation_history': history,
                'count': len(history)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch conversation history',
            'message': str(e)
        }), 500

@met_ask_bp.route('/quick-ask', methods=['GET'])
def quick_ask():
    """Quick ask via GET request for simple questions"""
    try:
        question = request.args.get('q', '').strip()
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question parameter "q" is required',
                'message': 'Use ?q=your-question to ask a quick question'
            }), 400
        
        if len(question) > 500:
            return jsonify({
                'success': False,
                'error': 'Question too long for quick ask',
                'message': 'Use POST /ask for longer questions'
            }), 400
        
        # Process the question (no user_id for quick ask)
        result = met_ask_service.ask_question(question=question)
        
        # Simplify response for quick ask
        if result.get('success'):
            return jsonify({
                'success': True,
                'question': question,
                'answer': result['answer']['main_answer'],
                'follow_up_questions': result['answer'].get('follow_up_questions', []),
                'timestamp': result['timestamp']
            })
        else:
            return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in quick ask: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to process quick question',
            'message': str(e)
        }), 500

@met_ask_bp.route('/categories', methods=['GET'])
def get_question_categories():
    """Get available question categories"""
    try:
        categories = {
            'pricing': {
                'name': 'Pricing & Valuation',
                'description': 'Questions about prices, costs, and market values',
                'examples': ['What affects gold prices?', 'How is silver valued?']
            },
            'investment': {
                'name': 'Investment & Portfolio',
                'description': 'Investment strategies and portfolio management',
                'examples': ['How much should I invest?', 'Physical vs ETFs?']
            },
            'technical': {
                'name': 'Technical Properties',
                'description': 'Metal properties, purity, and specifications',
                'examples': ['What is 24k gold?', 'How to test purity?']
            },
            'market': {
                'name': 'Market Analysis',
                'description': 'Market trends, forecasts, and analysis',
                'examples': ['Market outlook?', 'Economic impact?']
            },
            'storage': {
                'name': 'Storage & Security',
                'description': 'Safe storage and security considerations',
                'examples': ['How to store safely?', 'Insurance options?']
            },
            'trading': {
                'name': 'Trading & Transactions',
                'description': 'Buying, selling, and trading precious metals',
                'examples': ['Where to buy?', 'Trading costs?']
            },
            'history': {
                'name': 'History & Background',
                'description': 'Historical context and background information',
                'examples': ['Gold standard history?', 'Price history?']
            },
            'industrial': {
                'name': 'Industrial Applications',
                'description': 'Industrial uses and applications',
                'examples': ['Industrial demand?', 'Technology uses?']
            }
        }
        
        return jsonify({
            'success': True,
            'data': {
                'categories': categories,
                'total_categories': len(categories)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch categories',
            'message': str(e)
        }), 500

@met_ask_bp.route('/health', methods=['GET'])
def met_ask_health_check():
    """Health check for Met-Ask service"""
    try:
        # Test basic functionality
        test_result = met_ask_service.ask_question("What is gold?")
        
        return jsonify({
            'success': True,
            'service': 'Met-Ask Q&A Service',
            'status': 'operational',
            'features': [
                'Intelligent Q&A',
                'Real-time context',
                'Conversation history',
                'Popular questions',
                'Topic suggestions',
                'Multiple categories'
            ],
            'test_successful': test_result.get('success', False),
            'timestamp': test_result.get('timestamp')
        })
        
    except Exception as e:
        logger.error(f"Met-Ask service health check failed: {e}")
        return jsonify({
            'success': False,
            'service': 'Met-Ask Q&A Service',
            'status': 'error',
            'error': str(e)
        }), 500

@met_ask_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback on an answer"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        question = data.get('question')
        answer_helpful = data.get('helpful')  # boolean
        feedback_text = data.get('feedback', '').strip()
        user_id = data.get('user_id')
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question is required for feedback'
            }), 400
        
        if answer_helpful is None:
            return jsonify({
                'success': False,
                'error': 'Helpful rating is required (true/false)'
            }), 400
        
        # Store feedback (in production, this would go to a database)
        feedback_data = {
            'question': question,
            'helpful': answer_helpful,
            'feedback': feedback_text,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Feedback received: {feedback_data}")
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your feedback!',
            'data': {
                'feedback_id': f"fb_{int(datetime.utcnow().timestamp())}",
                'status': 'received'
            }
        })
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to submit feedback',
            'message': str(e)
        }), 500

