from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import logging
from src.services.metals_price_service import MetalsPriceService
from src.models.metals import Metal, PriceData
from src.models.user import db
from src.utils.unit_conversion import convert_price, validate_unit, get_all_units

logger = logging.getLogger(__name__)

prices_bp = Blueprint('prices', __name__)
price_service = MetalsPriceService()

@prices_bp.route('/current', methods=['GET'])
def get_current_prices():
    """Get current spot prices for precious metals"""
    try:
        # Parse query parameters
        metals = request.args.getlist('metals[]') or request.args.getlist('metals')
        currency = request.args.get('currency', 'USD')
        source = request.args.get('source', 'auto')
        unit = request.args.get('unit', 'troy_ounce')
        
        # Validate unit parameter
        if not validate_unit(unit):
            return jsonify({'error': f'Invalid unit: {unit}. Valid units: {list(get_all_units().keys())}'}), 400
        
        # Validate metals parameter
        valid_metals = ['gold', 'silver', 'platinum', 'palladium']
        if metals:
            metals = [m.lower() for m in metals if m.lower() in valid_metals]
        else:
            metals = valid_metals
        
        if not metals:
            return jsonify({'error': 'No valid metals specified'}), 400
        
        # Fetch current prices (in troy ounce - base unit)
        price_data = price_service.get_current_prices(metals, currency)
        
        # Convert prices to requested unit if not troy ounce
        if unit != 'troy_ounce' and price_data and 'prices' in price_data:
            for price_item in price_data['prices']:
                if 'price' in price_item:
                    # Convert price from troy ounce to requested unit
                    converted_price = convert_price(price_item['price'], 'troy_ounce', unit)
                    price_item['price'] = converted_price
                    price_item['unit'] = unit
                    
                    # Also convert change amounts if present
                    if 'change_24h' in price_item:
                        converted_change = convert_price(price_item['change_24h'], 'troy_ounce', unit)
                        price_item['change_24h'] = converted_change
        
        # Add unit information to response
        if price_data:
            price_data['unit'] = unit
            price_data['unit_info'] = get_all_units().get(unit, {})
        
        # Store in database for historical tracking (always store in troy ounce)
        _store_price_data(price_data)
        
        return jsonify({
            'success': True,
            'data': price_data,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching current prices: {e}")
        return jsonify({'error': 'Failed to fetch current prices'}), 500

@prices_bp.route('/historical', methods=['GET'])
def get_historical_prices():
    """Get historical price data for a specific metal"""
    try:
        # Parse query parameters
        metal = request.args.get('metal', '').lower()
        start_date_str = request.args.get('startDate')
        end_date_str = request.args.get('endDate')
        interval = request.args.get('interval', 'daily')
        currency = request.args.get('currency', 'USD')
        
        # Validate parameters
        if not metal:
            return jsonify({'error': 'Metal parameter is required'}), 400
        
        if metal not in ['gold', 'silver', 'platinum', 'palladium']:
            return jsonify({'error': 'Invalid metal specified'}), 400
        
        # Parse dates
        try:
            if start_date_str:
                start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            else:
                start_date = datetime.utcnow() - timedelta(days=30)  # Default to last 30 days
            
            if end_date_str:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            else:
                end_date = datetime.utcnow()
                
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use ISO format (YYYY-MM-DD)'}), 400
        
        # Validate date range
        if start_date >= end_date:
            return jsonify({'error': 'Start date must be before end date'}), 400
        
        if (end_date - start_date).days > 365:
            return jsonify({'error': 'Date range cannot exceed 365 days'}), 400
        
        # Fetch historical data
        historical_data = price_service.get_historical_prices(metal, start_date, end_date, interval)
        
        return jsonify({
            'success': True,
            'data': {
                'metal': metal,
                'currency': currency,
                'interval': interval,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'data_points': len(historical_data),
                'data': historical_data
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching historical prices: {e}")
        return jsonify({'error': 'Failed to fetch historical prices'}), 500

@prices_bp.route('/charts', methods=['GET'])
def get_chart_data():
    """Get optimized chart data with aggregation options"""
    try:
        # Parse query parameters
        metals = request.args.getlist('metals[]') or request.args.getlist('metals')
        period = request.args.get('period', '1M')  # 1D, 1W, 1M, 3M, 6M, 1Y
        aggregation = request.args.get('aggregation', 'daily')
        indicators = request.args.getlist('indicators[]') or []
        
        # Validate metals
        valid_metals = ['gold', 'silver', 'platinum', 'palladium']
        if metals:
            metals = [m.lower() for m in metals if m.lower() in valid_metals]
        else:
            metals = ['gold']  # Default to gold
        
        # Parse period to date range
        end_date = datetime.utcnow()
        period_map = {
            '1D': timedelta(days=1),
            '1W': timedelta(weeks=1),
            '1M': timedelta(days=30),
            '3M': timedelta(days=90),
            '6M': timedelta(days=180),
            '1Y': timedelta(days=365),
            '2Y': timedelta(days=730)
        }
        
        if period not in period_map:
            return jsonify({'error': 'Invalid period specified'}), 400
        
        start_date = end_date - period_map[period]
        
        # Fetch chart data for each metal
        charts = []
        for metal in metals:
            try:
                historical_data = price_service.get_historical_prices(metal, start_date, end_date, aggregation)
                
                # Calculate technical indicators if requested
                chart_indicators = {}
                if 'sma_20' in indicators:
                    chart_indicators['sma_20'] = _calculate_sma(historical_data, 20)
                if 'sma_50' in indicators:
                    chart_indicators['sma_50'] = _calculate_sma(historical_data, 50)
                if 'rsi' in indicators:
                    chart_indicators['rsi'] = _calculate_rsi(historical_data, 14)
                
                charts.append({
                    'metal': metal,
                    'data': historical_data,
                    'indicators': chart_indicators,
                    'metadata': {
                        'period': period,
                        'aggregation': aggregation,
                        'data_points': len(historical_data),
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat()
                    }
                })
                
            except Exception as e:
                logger.warning(f"Error fetching chart data for {metal}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'data': {
                'charts': charts,
                'period': period,
                'aggregation': aggregation,
                'indicators': indicators
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching chart data: {e}")
        return jsonify({'error': 'Failed to fetch chart data'}), 500

@prices_bp.route('/metals', methods=['GET'])
def get_available_metals():
    """Get list of available metals for tracking"""
    try:
        metals = Metal.query.filter_by(is_active=True).all()
        
        return jsonify({
            'success': True,
            'data': {
                'metals': [metal.to_dict() for metal in metals],
                'count': len(metals)
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching metals: {e}")
        return jsonify({'error': 'Failed to fetch available metals'}), 500

@prices_bp.route('/status', methods=['GET'])
def get_service_status():
    """Get status of price data services"""
    try:
        # Validate API keys
        api_status = price_service.validate_api_keys()
        
        # Check database connectivity
        try:
            db.session.execute('SELECT 1')
            db_status = True
        except:
            db_status = False
        
        # Get latest price data timestamp
        latest_price = PriceData.query.order_by(PriceData.timestamp.desc()).first()
        latest_timestamp = latest_price.timestamp.isoformat() if latest_price else None
        
        return jsonify({
            'success': True,
            'data': {
                'api_services': api_status,
                'database': db_status,
                'latest_data': latest_timestamp,
                'status': 'operational' if all(api_status.values()) and db_status else 'degraded'
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error checking service status: {e}")
        return jsonify({'error': 'Failed to check service status'}), 500

def _store_price_data(price_data):
    """Store price data in database for historical tracking"""
    try:
        for price_info in price_data.get('prices', []):
            # Get or create metal record
            metal_symbol = price_info['metal']
            metal = Metal.query.filter_by(symbol=metal_symbol).first()
            
            if not metal:
                # Create new metal record
                metal = Metal(
                    symbol=metal_symbol,
                    name=price_info['metal_name'],
                    unit='troy_ounce'
                )
                db.session.add(metal)
                db.session.flush()  # Get the ID
            
            # Create price data record
            price_record = PriceData(
                metal_id=metal.id,
                timestamp=datetime.fromisoformat(price_info['timestamp'].replace('Z', '+00:00')),
                price=price_info['price'],
                currency=price_info['currency'],
                source=price_info['source']
            )
            
            db.session.add(price_record)
        
        db.session.commit()
        
    except Exception as e:
        logger.error(f"Error storing price data: {e}")
        db.session.rollback()

def _calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    if len(data) < period:
        return []
    
    sma_values = []
    prices = [float(d['price']) for d in data]
    
    for i in range(period - 1, len(prices)):
        sma = sum(prices[i - period + 1:i + 1]) / period
        sma_values.append({
            'timestamp': data[i]['timestamp'],
            'value': round(sma, 2)
        })
    
    return sma_values

def _calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    if len(data) < period + 1:
        return []
    
    prices = [float(d['price']) for d in data]
    gains = []
    losses = []
    
    # Calculate price changes
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    
    if len(gains) < period:
        return []
    
    # Calculate RSI
    rsi_values = []
    
    for i in range(period - 1, len(gains)):
        avg_gain = sum(gains[i - period + 1:i + 1]) / period
        avg_loss = sum(losses[i - period + 1:i + 1]) / period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append({
            'timestamp': data[i + 1]['timestamp'],
            'value': round(rsi, 2)
        })
    
    return rsi_values



@prices_bp.route('/units', methods=['GET'])
def get_available_units():
    """Get all available measurement units for precious metals"""
    try:
        from src.utils.unit_conversion import get_all_units, get_unit_categories, get_popular_units
        
        units = get_all_units()
        categories = get_unit_categories()
        popular = get_popular_units()
        
        return jsonify({
            'success': True,
            'data': {
                'units': units,
                'categories': categories,
                'popular': popular,
                'default': 'troy_ounce'
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching available units: {e}")
        return jsonify({'error': 'Failed to fetch available units'}), 500

@prices_bp.route('/convert', methods=['POST'])
def convert_price_endpoint():
    """Convert price between different units"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        price = data.get('price')
        from_unit = data.get('from_unit', 'troy_ounce')
        to_unit = data.get('to_unit', 'troy_ounce')
        
        if price is None:
            return jsonify({'error': 'Price is required'}), 400
        
        if not validate_unit(from_unit):
            return jsonify({'error': f'Invalid from_unit: {from_unit}'}), 400
        
        if not validate_unit(to_unit):
            return jsonify({'error': f'Invalid to_unit: {to_unit}'}), 400
        
        converted_price = convert_price(price, from_unit, to_unit)
        
        return jsonify({
            'success': True,
            'data': {
                'original_price': price,
                'from_unit': from_unit,
                'to_unit': to_unit,
                'converted_price': converted_price,
                'conversion_rate': converted_price / price if price != 0 else 0
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error converting price: {e}")
        return jsonify({'error': 'Failed to convert price'}), 500

