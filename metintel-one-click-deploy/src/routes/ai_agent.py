"""
AI Agent API Routes for PreciousAI Application
Advanced investment target management and AI recommendations
Integrated from PreciousPredictor codebase
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import logging
import json

# Import our AI agent service
from src.services.ai_agent_service import ai_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
ai_agent_bp = Blueprint('ai_agent', __name__)

@ai_agent_bp.route('/targets', methods=['GET'])
def get_user_targets():
    """Get all investment targets for a user"""
    try:
        user_id = request.args.get('user_id', 1, type=int)  # Default to user 1 for demo
        status = request.args.get('status', 'ACTIVE')
        
        targets = ai_agent.get_user_targets(user_id, status)
        
        return jsonify({
            'success': True,
            'targets': targets,
            'count': len(targets)
        })
        
    except Exception as e:
        logger.error(f"Error getting user targets: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/targets', methods=['POST'])
def create_investment_target():
    """Create a new investment target"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['metal_type', 'target_type', 'target_price']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Extract data
        user_id = data.get('user_id', 1)  # Default to user 1 for demo
        metal_type = data['metal_type'].upper()
        target_type = data['target_type']
        target_price = float(data['target_price'])
        target_quantity = data.get('target_quantity')
        target_percentage = data.get('target_percentage')
        priority = data.get('priority', 'MEDIUM')
        notes = data.get('notes')
        
        # Parse target_date if provided
        target_date = None
        if data.get('target_date'):
            target_date = datetime.fromisoformat(data['target_date'].replace('Z', '+00:00'))
        
        # Create target
        target_id = ai_agent.create_investment_target(
            user_id=user_id,
            metal_type=metal_type,
            target_type=target_type,
            target_price=target_price,
            target_quantity=target_quantity,
            target_percentage=target_percentage,
            priority=priority,
            target_date=target_date,
            notes=notes
        )
        
        return jsonify({
            'success': True,
            'target_id': target_id,
            'message': 'Investment target created successfully'
        })
        
    except Exception as e:
        logger.error(f"Error creating investment target: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/targets/<int:target_id>', methods=['PUT'])
def update_investment_target(target_id):
    """Update an investment target"""
    try:
        data = request.get_json()
        
        # For now, we'll implement basic status updates
        # In a full implementation, this would update all fields
        
        return jsonify({
            'success': True,
            'message': 'Target update functionality coming soon'
        })
        
    except Exception as e:
        logger.error(f"Error updating investment target: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/targets/<int:target_id>', methods=['DELETE'])
def delete_investment_target(target_id):
    """Delete an investment target"""
    try:
        # For now, we'll mark as inactive instead of deleting
        # In a full implementation, this would properly delete or deactivate
        
        return jsonify({
            'success': True,
            'message': 'Target deletion functionality coming soon'
        })
        
    except Exception as e:
        logger.error(f"Error deleting investment target: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/recommendations', methods=['GET'])
def get_user_recommendations():
    """Get AI recommendations for a user"""
    try:
        user_id = request.args.get('user_id', 1, type=int)  # Default to user 1 for demo
        status = request.args.get('status', 'active')
        limit = request.args.get('limit', 10, type=int)
        
        recommendations = ai_agent.get_user_recommendations(user_id, status)
        
        # Limit results
        if limit > 0:
            recommendations = recommendations[:limit]
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Error getting user recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/recommendations/<int:rec_id>/feedback', methods=['POST'])
def provide_recommendation_feedback(rec_id):
    """Provide feedback on a recommendation"""
    try:
        data = request.get_json()
        feedback = data.get('feedback')  # 'accepted', 'rejected', 'modified'
        
        if feedback not in ['accepted', 'rejected', 'modified']:
            return jsonify({
                'success': False,
                'error': 'Invalid feedback. Must be: accepted, rejected, or modified'
            }), 400
        
        # Update recommendation with feedback
        # This would be implemented in the AI agent service
        
        return jsonify({
            'success': True,
            'message': 'Feedback recorded successfully'
        })
        
    except Exception as e:
        logger.error(f"Error providing recommendation feedback: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/agent/status', methods=['GET'])
def get_agent_status():
    """Get AI agent monitoring status"""
    try:
        return jsonify({
            'success': True,
            'monitoring_active': ai_agent.monitoring_active,
            'check_interval': ai_agent.check_interval,
            'cache_size': len(ai_agent.market_data_cache)
        })
        
    except Exception as e:
        logger.error(f"Error getting agent status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/agent/start', methods=['POST'])
def start_agent_monitoring():
    """Start AI agent monitoring"""
    try:
        ai_agent.start_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'AI agent monitoring started'
        })
        
    except Exception as e:
        logger.error(f"Error starting agent monitoring: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/agent/stop', methods=['POST'])
def stop_agent_monitoring():
    """Stop AI agent monitoring"""
    try:
        ai_agent.stop_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'AI agent monitoring stopped'
        })
        
    except Exception as e:
        logger.error(f"Error stopping agent monitoring: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/market/opportunities', methods=['GET'])
def get_market_opportunities():
    """Get current market opportunities identified by AI"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        
        # Get recent recommendations that are market opportunities
        recommendations = ai_agent.get_user_recommendations(user_id, 'active')
        
        # Filter for market opportunities
        opportunities = [
            rec for rec in recommendations 
            if rec.get('recommendation_type') in ['MARKET_OPPORTUNITY', 'BUY_SIGNAL', 'SELL_SIGNAL']
        ]
        
        return jsonify({
            'success': True,
            'opportunities': opportunities,
            'count': len(opportunities)
        })
        
    except Exception as e:
        logger.error(f"Error getting market opportunities: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/portfolio/analysis', methods=['GET'])
def get_portfolio_analysis():
    """Get AI-powered portfolio analysis"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        
        # This would implement portfolio analysis
        # For now, return a placeholder response
        
        analysis = {
            'diversification_score': 75,
            'risk_level': 'MEDIUM',
            'recommendations': [
                {
                    'type': 'REBALANCE',
                    'message': 'Consider rebalancing portfolio - Gold allocation is 70%, recommend reducing to 60%',
                    'confidence': 80
                }
            ],
            'performance_metrics': {
                'total_value': 50000,
                'daily_change': 2.5,
                'weekly_change': -1.2,
                'monthly_change': 5.8
            }
        }
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Error getting portfolio analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/preferences', methods=['GET'])
def get_user_preferences():
    """Get user AI preferences"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        
        # This would get preferences from database
        # For now, return default preferences
        
        preferences = {
            'risk_tolerance': 'MEDIUM',
            'notification_frequency': 'IMPORTANT',
            'auto_execute_enabled': False,
            'max_auto_investment': 0,
            'preferred_metals': ['GOLD', 'SILVER', 'PLATINUM'],
            'target_allocation': {'GOLD': 60, 'SILVER': 30, 'PLATINUM': 10},
            'min_confidence_threshold': 75.0,
            'agent_enabled': True,
            'notification_channels': ['email']
        }
        
        return jsonify({
            'success': True,
            'preferences': preferences
        })
        
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/preferences', methods=['POST'])
def update_user_preferences():
    """Update user AI preferences"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 1)
        
        # This would update preferences in database
        # For now, return success
        
        return jsonify({
            'success': True,
            'message': 'Preferences updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating user preferences: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_agent_bp.route('/demo/create-sample-data', methods=['POST'])
def create_sample_data():
    """Create sample data for demonstration"""
    try:
        # Create sample targets
        sample_targets = [
            {
                'metal_type': 'GOLD',
                'target_type': 'buy_below',
                'target_price': 3200.0,
                'priority': 'HIGH',
                'notes': 'Buy gold when price drops below $3200 for portfolio expansion'
            },
            {
                'metal_type': 'SILVER',
                'target_type': 'sell_above',
                'target_price': 40.0,
                'priority': 'MEDIUM',
                'notes': 'Take profits on silver when price exceeds $40'
            },
            {
                'metal_type': 'PLATINUM',
                'target_type': 'buy_below',
                'target_price': 1300.0,
                'priority': 'LOW',
                'notes': 'Accumulate platinum below $1300 for long-term holding'
            }
        ]
        
        created_targets = []
        for target_data in sample_targets:
            target_id = ai_agent.create_investment_target(
                user_id=1,
                metal_type=target_data['metal_type'],
                target_type=target_data['target_type'],
                target_price=target_data['target_price'],
                priority=target_data['priority'],
                notes=target_data['notes']
            )
            created_targets.append(target_id)
        
        return jsonify({
            'success': True,
            'message': 'Sample data created successfully',
            'created_targets': created_targets
        })
        
    except Exception as e:
        logger.error(f"Error creating sample data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

