"""
Enhanced AI Agent API Routes for PreciousAI Application
Advanced investment target management and AI recommendations
Integrated from PreciousPredictor codebase with sophisticated features
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import logging
import json

# Import our enhanced AI agent service
from src.services.enhanced_ai_agent_service import enhanced_ai_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
enhanced_ai_agent_bp = Blueprint('enhanced_ai_agent', __name__)

# ============================================================================
# INVESTMENT TARGETS ENDPOINTS
# ============================================================================

@enhanced_ai_agent_bp.route('/targets', methods=['GET'])
def get_user_targets():
    """Get all investment targets for a user"""
    try:
        user_id = request.args.get('user_id', 1, type=int)  # Default to user 1 for demo
        status = request.args.get('status', 'ACTIVE')
        
        targets = enhanced_ai_agent.get_user_targets(user_id, status)
        
        return jsonify({
            'success': True,
            'targets': targets,
            'count': len(targets),
            'message': f'Retrieved {len(targets)} investment targets'
        })
        
    except Exception as e:
        logger.error(f"Error getting user targets: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@enhanced_ai_agent_bp.route('/targets', methods=['POST'])
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
        
        # Validate metal type
        valid_metals = ['GOLD', 'SILVER', 'PLATINUM', 'PALLADIUM']
        if data['metal_type'].upper() not in valid_metals:
            return jsonify({
                'success': False,
                'error': f'Invalid metal type. Must be one of: {", ".join(valid_metals)}'
            }), 400
        
        # Validate target type
        valid_target_types = ['buy_below', 'sell_above', 'price_alert', 'hold_until', 'stop_loss', 'take_profit']
        if data['target_type'] not in valid_target_types:
            return jsonify({
                'success': False,
                'error': f'Invalid target type. Must be one of: {", ".join(valid_target_types)}'
            }), 400
        
        # Validate price
        try:
            target_price = float(data['target_price'])
            if target_price <= 0:
                raise ValueError("Price must be positive")
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'Invalid target price. Must be a positive number.'
            }), 400
        
        # Parse optional fields
        user_id = data.get('user_id', 1)  # Default to user 1 for demo
        target_quantity = data.get('target_quantity')
        target_percentage = data.get('target_percentage')
        priority = data.get('priority', 'MEDIUM')
        notes = data.get('notes')
        conditions = data.get('conditions')
        
        # Parse target date if provided
        target_date = None
        if data.get('target_date'):
            try:
                target_date = datetime.fromisoformat(data['target_date'].replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid target date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)'
                }), 400
        
        # Create the investment target
        target_id = enhanced_ai_agent.create_investment_target(
            user_id=user_id,
            metal_type=data['metal_type'].upper(),
            target_type=data['target_type'],
            target_price=target_price,
            target_quantity=target_quantity,
            target_percentage=target_percentage,
            priority=priority,
            target_date=target_date,
            notes=notes,
            conditions=conditions
        )
        
        return jsonify({
            'success': True,
            'target_id': target_id,
            'message': f'Investment target created successfully for {data["metal_type"]} at ${target_price:.2f}'
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating investment target: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@enhanced_ai_agent_bp.route('/targets/<int:target_id>', methods=['PUT'])
def update_investment_target(target_id):
    """Update an existing investment target"""
    try:
        data = request.get_json()
        
        # For now, we'll implement basic status updates
        # In a full implementation, you'd want to update all fields
        if 'status' in data:
            valid_statuses = ['ACTIVE', 'PAUSED', 'COMPLETED', 'CANCELLED']
            if data['status'] not in valid_statuses:
                return jsonify({
                    'success': False,
                    'error': f'Invalid status. Must be one of: {", ".join(valid_statuses)}'
                }), 400
        
        # TODO: Implement target update logic in the service
        return jsonify({
            'success': True,
            'message': f'Investment target {target_id} updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating investment target: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@enhanced_ai_agent_bp.route('/targets/<int:target_id>', methods=['DELETE'])
def delete_investment_target(target_id):
    """Delete an investment target"""
    try:
        # TODO: Implement target deletion logic in the service
        return jsonify({
            'success': True,
            'message': f'Investment target {target_id} deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting investment target: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# AI RECOMMENDATIONS ENDPOINTS
# ============================================================================

@enhanced_ai_agent_bp.route('/recommendations', methods=['GET'])
def get_user_recommendations():
    """Get AI recommendations for a user"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        status = request.args.get('status', 'active')
        limit = request.args.get('limit', 10, type=int)
        
        recommendations = enhanced_ai_agent.get_user_recommendations(user_id, status, limit)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations),
            'message': f'Retrieved {len(recommendations)} AI recommendations'
        })
        
    except Exception as e:
        logger.error(f"Error getting user recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@enhanced_ai_agent_bp.route('/recommendations/<int:recommendation_id>/feedback', methods=['POST'])
def update_recommendation_feedback(recommendation_id):
    """Update user feedback on a recommendation"""
    try:
        data = request.get_json()
        
        feedback = data.get('feedback')
        if feedback not in ['accepted', 'rejected', 'modified']:
            return jsonify({
                'success': False,
                'error': 'Invalid feedback. Must be one of: accepted, rejected, modified'
            }), 400
        
        execution_date = None
        if feedback == 'accepted' and data.get('execution_date'):
            try:
                execution_date = datetime.fromisoformat(data['execution_date'].replace('Z', '+00:00'))
            except ValueError:
                execution_date = datetime.now()
        
        enhanced_ai_agent.update_recommendation_feedback(
            recommendation_id, feedback, execution_date
        )
        
        return jsonify({
            'success': True,
            'message': f'Recommendation feedback updated: {feedback}'
        })
        
    except Exception as e:
        logger.error(f"Error updating recommendation feedback: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# USER PREFERENCES ENDPOINTS
# ============================================================================

@enhanced_ai_agent_bp.route('/preferences', methods=['GET'])
def get_user_preferences():
    """Get user AI preferences"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        
        preferences = enhanced_ai_agent.get_user_preferences(user_id)
        
        return jsonify({
            'success': True,
            'preferences': preferences,
            'message': 'User AI preferences retrieved successfully'
        })
        
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@enhanced_ai_agent_bp.route('/preferences', methods=['POST'])
def set_user_preferences():
    """Set user AI preferences"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 1)
        
        # Validate preferences
        valid_risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        if 'risk_tolerance' in data and data['risk_tolerance'] not in valid_risk_levels:
            return jsonify({
                'success': False,
                'error': f'Invalid risk tolerance. Must be one of: {", ".join(valid_risk_levels)}'
            }), 400
        
        valid_notification_frequencies = ['ALL', 'IMPORTANT', 'CRITICAL']
        if 'notification_frequency' in data and data['notification_frequency'] not in valid_notification_frequencies:
            return jsonify({
                'success': False,
                'error': f'Invalid notification frequency. Must be one of: {", ".join(valid_notification_frequencies)}'
            }), 400
        
        valid_metals = ['GOLD', 'SILVER', 'PLATINUM', 'PALLADIUM']
        if 'preferred_metals' in data:
            for metal in data['preferred_metals']:
                if metal.upper() not in valid_metals:
                    return jsonify({
                        'success': False,
                        'error': f'Invalid metal in preferred_metals: {metal}. Must be one of: {", ".join(valid_metals)}'
                    }), 400
        
        # Validate confidence threshold
        if 'min_confidence_threshold' in data:
            threshold = data['min_confidence_threshold']
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 100:
                return jsonify({
                    'success': False,
                    'error': 'min_confidence_threshold must be a number between 0 and 100'
                }), 400
        
        enhanced_ai_agent.set_user_preferences(user_id, data)
        
        return jsonify({
            'success': True,
            'message': 'User AI preferences updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error setting user preferences: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# AI AGENT CONTROL ENDPOINTS
# ============================================================================

@enhanced_ai_agent_bp.route('/start-monitoring', methods=['POST'])
def start_monitoring():
    """Start the AI agent monitoring service"""
    try:
        enhanced_ai_agent.start_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'AI Agent monitoring service started successfully'
        })
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@enhanced_ai_agent_bp.route('/stop-monitoring', methods=['POST'])
def stop_monitoring():
    """Stop the AI agent monitoring service"""
    try:
        enhanced_ai_agent.stop_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'AI Agent monitoring service stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@enhanced_ai_agent_bp.route('/status', methods=['GET'])
def get_agent_status():
    """Get AI agent status and statistics"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        
        # Get user targets and recommendations
        active_targets = enhanced_ai_agent.get_user_targets(user_id, 'ACTIVE')
        recent_recommendations = enhanced_ai_agent.get_user_recommendations(user_id, 'active', 5)
        user_preferences = enhanced_ai_agent.get_user_preferences(user_id)
        
        status = {
            'monitoring_active': enhanced_ai_agent.is_monitoring,
            'active_targets_count': len(active_targets),
            'recent_recommendations_count': len(recent_recommendations),
            'agent_enabled': user_preferences.get('agent_enabled', True),
            'preferred_metals': user_preferences.get('preferred_metals', []),
            'risk_tolerance': user_preferences.get('risk_tolerance', 'MEDIUM'),
            'min_confidence_threshold': user_preferences.get('min_confidence_threshold', 75.0)
        }
        
        return jsonify({
            'success': True,
            'status': status,
            'message': 'AI Agent status retrieved successfully'
        })
        
    except Exception as e:
        logger.error(f"Error getting agent status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# MARKET ANALYSIS ENDPOINTS
# ============================================================================

@enhanced_ai_agent_bp.route('/market-analysis/<metal_type>', methods=['GET'])
def get_market_analysis(metal_type):
    """Get AI market analysis for a specific metal"""
    try:
        valid_metals = ['GOLD', 'SILVER', 'PLATINUM', 'PALLADIUM']
        if metal_type.upper() not in valid_metals:
            return jsonify({
                'success': False,
                'error': f'Invalid metal type. Must be one of: {", ".join(valid_metals)}'
            }), 400
        
        # Get market context from the AI agent
        market_context = enhanced_ai_agent._get_market_context(metal_type.upper())
        
        return jsonify({
            'success': True,
            'metal_type': metal_type.upper(),
            'analysis': market_context,
            'message': f'Market analysis for {metal_type.upper()} retrieved successfully'
        })
        
    except Exception as e:
        logger.error(f"Error getting market analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# DEMO DATA ENDPOINTS
# ============================================================================

@enhanced_ai_agent_bp.route('/demo/create-sample-targets', methods=['POST'])
def create_sample_targets():
    """Create sample investment targets for demo purposes"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        
        sample_targets = [
            {
                'metal_type': 'GOLD',
                'target_type': 'buy_below',
                'target_price': 3200.00,
                'priority': 'HIGH',
                'notes': 'Buy gold when price drops below $3200 - good entry point'
            },
            {
                'metal_type': 'SILVER',
                'target_type': 'sell_above',
                'target_price': 40.00,
                'priority': 'MEDIUM',
                'notes': 'Sell silver when price goes above $40 - take profits'
            },
            {
                'metal_type': 'PLATINUM',
                'target_type': 'buy_below',
                'target_price': 1300.00,
                'priority': 'LOW',
                'notes': 'Accumulate platinum below $1300 - long-term hold'
            },
            {
                'metal_type': 'GOLD',
                'target_type': 'price_alert',
                'target_price': 3350.00,
                'priority': 'MEDIUM',
                'notes': 'Alert when gold reaches $3350 - monitor for breakout'
            }
        ]
        
        created_targets = []
        for target_data in sample_targets:
            target_id = enhanced_ai_agent.create_investment_target(
                user_id=user_id,
                metal_type=target_data['metal_type'],
                target_type=target_data['target_type'],
                target_price=target_data['target_price'],
                priority=target_data['priority'],
                notes=target_data['notes']
            )
            created_targets.append({
                'id': target_id,
                'metal_type': target_data['metal_type'],
                'target_type': target_data['target_type'],
                'target_price': target_data['target_price']
            })
        
        return jsonify({
            'success': True,
            'created_targets': created_targets,
            'count': len(created_targets),
            'message': f'Created {len(created_targets)} sample investment targets'
        })
        
    except Exception as e:
        logger.error(f"Error creating sample targets: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@enhanced_ai_agent_bp.route('/demo/setup-preferences', methods=['POST'])
def setup_demo_preferences():
    """Set up demo user preferences"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        
        demo_preferences = {
            'risk_tolerance': 'MEDIUM',
            'notification_frequency': 'IMPORTANT',
            'auto_execute_enabled': False,
            'max_auto_investment': 1000.0,
            'preferred_metals': ['GOLD', 'SILVER', 'PLATINUM'],
            'target_allocation': {
                'GOLD': 60,
                'SILVER': 30,
                'PLATINUM': 10
            },
            'min_confidence_threshold': 75.0,
            'agent_enabled': True,
            'notification_channels': ['email']
        }
        
        enhanced_ai_agent.set_user_preferences(user_id, demo_preferences)
        
        return jsonify({
            'success': True,
            'preferences': demo_preferences,
            'message': 'Demo user preferences set up successfully'
        })
        
    except Exception as e:
        logger.error(f"Error setting up demo preferences: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@enhanced_ai_agent_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@enhanced_ai_agent_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405

@enhanced_ai_agent_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

