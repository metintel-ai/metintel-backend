from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import uuid
import logging
from src.services.ai_prediction_service import AIPredictionService
from src.services.metals_price_service import MetalsPriceService
from src.models.metals import Metal, Prediction, PredictionFactor
from src.models.user import db, User

logger = logging.getLogger(__name__)

predictions_bp = Blueprint('predictions', __name__)
ai_service = AIPredictionService()
price_service = MetalsPriceService()

@predictions_bp.route('/generate', methods=['POST'])
def generate_prediction():
    """Generate new price predictions using AI"""
    try:
        data = request.get_json()
        
        # Validate request data
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
        
        metals = data.get('metals', [])
        timeframe_mode = data.get('timeframe_mode', 'preset')
        horizons = data.get('horizons', [])
        custom_date_range = data.get('custom_date_range', {})
        parameters = data.get('parameters', {})
        include_factors = data.get('includeFactors', True)
        user_id = data.get('user_id', 1)  # Default user for demo
        
        # Validate metals
        valid_metals = ['gold', 'silver', 'platinum', 'palladium']
        metals = [m.lower() for m in metals if m.lower() in valid_metals]
        
        if not metals:
            return jsonify({'error': 'At least one valid metal must be specified'}), 400
        
        # Validate timeframe based on mode
        prediction_timeframes = []
        
        if timeframe_mode == 'preset':
            # Validate preset horizons
            valid_horizons = ['1 week', '2 weeks', '1 month', '3 months', '6 months', '12 months', '18 months', '24 months']
            horizons = [h for h in horizons if h in valid_horizons]
            
            if not horizons:
                return jsonify({'error': 'At least one valid horizon must be specified'}), 400
            
            prediction_timeframes = [{'type': 'preset', 'horizon': h} for h in horizons]
            
        elif timeframe_mode == 'custom':
            # Validate custom date range
            if not custom_date_range or not custom_date_range.get('startDate') or not custom_date_range.get('endDate'):
                return jsonify({'error': 'Custom date range requires both start and end dates'}), 400
            
            try:
                start_date = datetime.fromisoformat(custom_date_range['startDate'])
                end_date = datetime.fromisoformat(custom_date_range['endDate'])
                
                # Validate date range
                now = datetime.utcnow()
                if start_date < now:
                    return jsonify({'error': 'Start date cannot be in the past'}), 400
                
                if end_date <= start_date:
                    return jsonify({'error': 'End date must be after start date'}), 400
                
                # Check maximum range (24 months)
                max_date = now.replace(year=now.year + 2)
                if end_date > max_date:
                    return jsonify({'error': 'End date cannot be more than 24 months in the future'}), 400
                
                prediction_timeframes = [{
                    'type': 'custom',
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration': custom_date_range.get('duration', (end_date - start_date).days),
                    'label': custom_date_range.get('label', f"Custom ({(end_date - start_date).days} days)")
                }]
                
            except ValueError as e:
                return jsonify({'error': f'Invalid date format: {str(e)}'}), 400
        
        else:
            return jsonify({'error': 'Invalid timeframe_mode. Must be "preset" or "custom"'}), 400
        
        # Check user exists (create demo user if needed)
        user = User.query.get(user_id)
        if not user:
            user = User(
                email='demo@example.com',
                password_hash='demo',
                first_name='Demo',
                last_name='User',
                account_type='individual',
                subscription_tier='free'
            )
            db.session.add(user)
            db.session.commit()
            user_id = user.id
        
        # Get current prices for the metals
        current_prices = price_service.get_current_prices(metals)
        
        # Generate predictions for each metal and horizon combination
        predictions = []
        total_cost = 0
        
        for metal in metals:
            # Convert metal name to symbol
            metal_symbol = {
                'gold': 'USDXAU',
                'silver': 'USDXAG',
                'platinum': 'USDXPT',
                'palladium': 'USDXPD'
            }.get(metal)
            
            if not metal_symbol:
                continue
            
            # Find current price for this metal
            current_price = None
            for price_info in current_prices.get('prices', []):
                if price_info['metal'] == metal_symbol:
                    current_price = price_info['price']
                    break
            
            if current_price is None:
                logger.warning(f"No current price found for {metal}")
                continue
            
            # Get historical data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=90)  # Last 90 days
            historical_data = price_service.get_historical_prices(metal, start_date, end_date)
            
            for timeframe in prediction_timeframes:
                try:
                    # Determine horizon and target date based on timeframe type
                    if timeframe['type'] == 'preset':
                        horizon = timeframe['horizon']
                        # Calculate target date for preset horizon
                        horizon_days = {
                            '1 week': 7, '2 weeks': 14, '1 month': 30, '3 months': 90,
                            '6 months': 180, '12 months': 365, '18 months': 547, '24 months': 730
                        }
                        target_date = datetime.utcnow() + timedelta(days=horizon_days.get(horizon, 30))
                    else:  # custom
                        horizon = timeframe['label']
                        target_date = timeframe['end_date']
                    
                    # Generate prediction
                    prediction_result = ai_service.generate_prediction(
                        metal_symbol, horizon, historical_data, current_price, parameters
                    )
                    
                    # Override target date for custom timeframes
                    if timeframe['type'] == 'custom':
                        prediction_result['target_date'] = target_date.isoformat()
                        prediction_result['horizon'] = horizon
                        prediction_result['custom_timeframe'] = {
                            'start_date': timeframe['start_date'].isoformat(),
                            'end_date': timeframe['end_date'].isoformat(),
                            'duration_days': timeframe['duration']
                        }
                    
                    # Store prediction in database
                    prediction_id = prediction_result['id']
                    
                    # Get or create metal record
                    metal_record = Metal.query.filter_by(symbol=metal_symbol).first()
                    if not metal_record:
                        metal_record = Metal(
                            symbol=metal_symbol,
                            name=prediction_result.get('metal_name', metal.title()),
                            unit='troy_ounce'
                        )
                        db.session.add(metal_record)
                        db.session.flush()
                    
                    # Create prediction record
                    prediction = Prediction(
                        id=prediction_id,
                        user_id=user_id,
                        metal_id=metal_record.id,
                        prediction_horizon=horizon,
                        target_date=target_date,
                        predicted_price=prediction_result['predicted_price'],
                        confidence_interval_lower=prediction_result['confidence_interval_lower'],
                        confidence_interval_upper=prediction_result['confidence_interval_upper'],
                        confidence_score=prediction_result['confidence_score'],
                        model_version=prediction_result['model_version'],
                        input_parameters=parameters,
                        market_context=prediction_result['market_context']
                    )
                    
                    db.session.add(prediction)
                    
                    # Store prediction factors if requested
                    if include_factors and 'factors' in prediction_result:
                        for factor_data in prediction_result['factors']:
                            factor = PredictionFactor(
                                id=factor_data['id'],
                                prediction_id=prediction_id,
                                factor_type=factor_data['type'],
                                factor_name=factor_data['name'],
                                impact_weight=factor_data['impact_weight'],
                                description=factor_data['description']
                            )
                            db.session.add(factor)
                    
                    predictions.append(prediction_result)
                    total_cost += 0.50  # Estimated cost per prediction
                    
                except Exception as e:
                    logger.error(f"Error generating prediction for {metal} {horizon}: {e}")
                    continue
        
        # Commit all predictions to database
        db.session.commit()
        
        if not predictions:
            return jsonify({'error': 'Failed to generate any predictions'}), 500
        
        return jsonify({
            'success': True,
            'data': {
                'predictions': predictions,
                'total_predictions': len(predictions),
                'estimated_cost': round(total_cost, 2),
                'processing_time': '2-5 minutes',
                'user_id': user_id
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to generate predictions'}), 500

@predictions_bp.route('/<prediction_id>', methods=['GET'])
def get_prediction(prediction_id):
    """Get specific prediction results"""
    try:
        # Validate prediction ID format
        try:
            uuid.UUID(prediction_id)
        except ValueError:
            return jsonify({'error': 'Invalid prediction ID format'}), 400
        
        # Fetch prediction from database
        prediction = Prediction.query.get(prediction_id)
        
        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404
        
        # Get prediction factors
        factors = PredictionFactor.query.filter_by(prediction_id=prediction_id).all()
        
        # Build response
        result = prediction.to_dict()
        result['factors'] = [factor.to_dict() for factor in factors]
        
        # Add current price for comparison
        try:
            metal_name = {
                'USDXAU': 'gold',
                'USDXAG': 'silver',
                'USDXPT': 'platinum',
                'USDXPD': 'palladium'
            }.get(prediction.metal.symbol)
            
            if metal_name:
                current_prices = price_service.get_current_prices([metal_name])
                for price_info in current_prices.get('prices', []):
                    if price_info['metal'] == prediction.metal.symbol:
                        result['current_price'] = price_info['price']
                        result['price_change'] = price_info['price'] - float(prediction.predicted_price)
                        result['price_change_percent'] = (result['price_change'] / float(prediction.predicted_price)) * 100
                        break
        except Exception as e:
            logger.warning(f"Could not fetch current price for comparison: {e}")
        
        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching prediction {prediction_id}: {e}")
        return jsonify({'error': 'Failed to fetch prediction'}), 500

@predictions_bp.route('/user/<int:user_id>', methods=['GET'])
def get_user_predictions(user_id):
    """Get user's prediction history"""
    try:
        # Parse query parameters
        page = int(request.args.get('page', 1))
        limit = min(int(request.args.get('limit', 20)), 100)  # Max 100 per page
        metal = request.args.get('metal', '').lower()
        status = request.args.get('status', '')
        sort_by = request.args.get('sortBy', 'created_at')
        
        # Build query
        query = Prediction.query.filter_by(user_id=user_id)
        
        # Apply filters
        if metal and metal in ['gold', 'silver', 'platinum', 'palladium']:
            metal_symbol = {
                'gold': 'USDXAU',
                'silver': 'USDXAG',
                'platinum': 'USDXPT',
                'palladium': 'USDXPD'
            }.get(metal)
            
            metal_record = Metal.query.filter_by(symbol=metal_symbol).first()
            if metal_record:
                query = query.filter_by(metal_id=metal_record.id)
        
        if status:
            # Filter by prediction status (active, expired, etc.)
            now = datetime.utcnow()
            if status == 'active':
                query = query.filter(Prediction.target_date > now)
            elif status == 'expired':
                query = query.filter(Prediction.target_date <= now)
        
        # Apply sorting
        if sort_by == 'created_at':
            query = query.order_by(Prediction.created_at.desc())
        elif sort_by == 'target_date':
            query = query.order_by(Prediction.target_date.desc())
        elif sort_by == 'confidence':
            query = query.order_by(Prediction.confidence_score.desc())
        
        # Paginate results
        offset = (page - 1) * limit
        predictions = query.offset(offset).limit(limit).all()
        total_count = query.count()
        
        # Build response
        prediction_list = []
        for prediction in predictions:
            pred_dict = prediction.to_dict()
            
            # Add status information
            now = datetime.utcnow()
            if prediction.target_date <= now:
                pred_dict['status'] = 'expired'
            else:
                pred_dict['status'] = 'active'
                days_remaining = (prediction.target_date - now).days
                pred_dict['days_remaining'] = days_remaining
            
            prediction_list.append(pred_dict)
        
        # Calculate pagination info
        total_pages = (total_count + limit - 1) // limit
        has_next = page < total_pages
        has_prev = page > 1
        
        return jsonify({
            'success': True,
            'data': {
                'predictions': prediction_list,
                'pagination': {
                    'page': page,
                    'limit': limit,
                    'total_count': total_count,
                    'total_pages': total_pages,
                    'has_next': has_next,
                    'has_prev': has_prev
                },
                'summary': {
                    'total_predictions': total_count,
                    'active_predictions': len([p for p in prediction_list if p.get('status') == 'active']),
                    'expired_predictions': len([p for p in prediction_list if p.get('status') == 'expired'])
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching user predictions: {e}")
        return jsonify({'error': 'Failed to fetch user predictions'}), 500

@predictions_bp.route('/accuracy', methods=['GET'])
def get_prediction_accuracy():
    """Get prediction accuracy statistics"""
    try:
        # Get predictions that have reached their target date
        now = datetime.utcnow()
        completed_predictions = Prediction.query.filter(
            Prediction.target_date <= now,
            Prediction.actual_price.isnot(None)
        ).all()
        
        if not completed_predictions:
            return jsonify({
                'success': True,
                'data': {
                    'message': 'No completed predictions available for accuracy analysis',
                    'total_completed': 0
                },
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Calculate accuracy statistics
        total_predictions = len(completed_predictions)
        accurate_predictions = 0
        total_error = 0
        
        accuracy_by_metal = {}
        accuracy_by_horizon = {}
        
        for prediction in completed_predictions:
            actual_price = float(prediction.actual_price)
            predicted_price = float(prediction.predicted_price)
            
            # Calculate error
            error = abs(actual_price - predicted_price) / actual_price
            total_error += error
            
            # Check if prediction was within confidence interval
            if (prediction.confidence_interval_lower <= actual_price <= prediction.confidence_interval_upper):
                accurate_predictions += 1
            
            # Track by metal
            metal_symbol = prediction.metal.symbol
            if metal_symbol not in accuracy_by_metal:
                accuracy_by_metal[metal_symbol] = {'total': 0, 'accurate': 0, 'total_error': 0}
            
            accuracy_by_metal[metal_symbol]['total'] += 1
            accuracy_by_metal[metal_symbol]['total_error'] += error
            if (prediction.confidence_interval_lower <= actual_price <= prediction.confidence_interval_upper):
                accuracy_by_metal[metal_symbol]['accurate'] += 1
            
            # Track by horizon
            horizon = prediction.prediction_horizon
            if horizon not in accuracy_by_horizon:
                accuracy_by_horizon[horizon] = {'total': 0, 'accurate': 0, 'total_error': 0}
            
            accuracy_by_horizon[horizon]['total'] += 1
            accuracy_by_horizon[horizon]['total_error'] += error
            if (prediction.confidence_interval_lower <= actual_price <= prediction.confidence_interval_upper):
                accuracy_by_horizon[horizon]['accurate'] += 1
        
        # Calculate overall statistics
        overall_accuracy = (accurate_predictions / total_predictions) * 100
        average_error = (total_error / total_predictions) * 100
        
        # Calculate statistics by category
        metal_stats = {}
        for metal, stats in accuracy_by_metal.items():
            metal_stats[metal] = {
                'accuracy_rate': (stats['accurate'] / stats['total']) * 100,
                'average_error': (stats['total_error'] / stats['total']) * 100,
                'total_predictions': stats['total']
            }
        
        horizon_stats = {}
        for horizon, stats in accuracy_by_horizon.items():
            horizon_stats[horizon] = {
                'accuracy_rate': (stats['accurate'] / stats['total']) * 100,
                'average_error': (stats['total_error'] / stats['total']) * 100,
                'total_predictions': stats['total']
            }
        
        return jsonify({
            'success': True,
            'data': {
                'overall': {
                    'total_predictions': total_predictions,
                    'accurate_predictions': accurate_predictions,
                    'accuracy_rate': round(overall_accuracy, 2),
                    'average_error': round(average_error, 2)
                },
                'by_metal': metal_stats,
                'by_horizon': horizon_stats,
                'last_updated': now.isoformat()
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error calculating prediction accuracy: {e}")
        return jsonify({'error': 'Failed to calculate prediction accuracy'}), 500

@predictions_bp.route('/status', methods=['GET'])
def get_prediction_service_status():
    """Get status of AI prediction services"""
    try:
        # Validate AI service API keys
        ai_status = ai_service.validate_api_keys()
        
        # Get recent prediction statistics
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        
        recent_predictions = Prediction.query.filter(
            Prediction.created_at >= last_24h
        ).count()
        
        active_predictions = Prediction.query.filter(
            Prediction.target_date > now
        ).count()
        
        return jsonify({
            'success': True,
            'data': {
                'ai_services': ai_status,
                'statistics': {
                    'predictions_last_24h': recent_predictions,
                    'active_predictions': active_predictions
                },
                'status': 'operational' if any(ai_status.values()) else 'degraded'
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error checking prediction service status: {e}")
        return jsonify({'error': 'Failed to check prediction service status'}), 500

