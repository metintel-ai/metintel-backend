"""
Unit conversion utility for precious metals
Base unit is troy ounce for precious metals
"""

PRECIOUS_METALS_UNITS = {
    'troy_ounce': {
        'id': 'troy_ounce',
        'name': 'Troy Ounce',
        'symbol': 'oz t',
        'multiplier': 1.0,  # Base unit
        'category': 'imperial',
        'description': 'Standard unit for precious metals trading'
    },
    'gram': {
        'id': 'gram',
        'name': 'Gram',
        'symbol': 'g',
        'multiplier': 31.1034768,  # 1 troy ounce = 31.1034768 grams
        'category': 'metric',
        'description': 'Metric unit commonly used worldwide'
    },
    'kilogram': {
        'id': 'kilogram',
        'name': 'Kilogram',
        'symbol': 'kg',
        'multiplier': 0.0311034768,  # 1 troy ounce = 0.0311034768 kg
        'category': 'metric',
        'description': 'Metric unit for larger quantities'
    },
    'ounce': {
        'id': 'ounce',
        'name': 'Ounce (Avoirdupois)',
        'symbol': 'oz',
        'multiplier': 1.09714286,  # 1 troy ounce = 1.09714286 avoirdupois ounces
        'category': 'imperial',
        'description': 'Standard ounce (different from troy ounce)'
    },
    'pound': {
        'id': 'pound',
        'name': 'Pound',
        'symbol': 'lb',
        'multiplier': 0.0685714286,  # 1 troy ounce = 0.0685714286 pounds
        'category': 'imperial',
        'description': 'Imperial unit for larger quantities'
    },
    'tola': {
        'id': 'tola',
        'name': 'Tola',
        'symbol': 'tola',
        'multiplier': 2.66666667,  # 1 troy ounce = 2.66666667 tolas (1 tola = 11.664 grams)
        'category': 'traditional',
        'description': 'Traditional unit used in South Asia'
    },
    'baht': {
        'id': 'baht',
        'name': 'Baht',
        'symbol': 'baht',
        'multiplier': 2.03318,  # 1 troy ounce = 2.03318 bahts (1 baht = 15.244 grams)
        'category': 'traditional',
        'description': 'Traditional unit used in Thailand'
    }
}

def convert_price(price_in_troy_ounce, from_unit='troy_ounce', to_unit='troy_ounce'):
    """
    Convert price from one unit to another
    
    Args:
        price_in_troy_ounce (float): Price in troy ounce (base unit)
        from_unit (str): Source unit (default: troy_ounce)
        to_unit (str): Target unit (default: troy_ounce)
    
    Returns:
        float: Converted price
    """
    if not price_in_troy_ounce or to_unit not in PRECIOUS_METALS_UNITS:
        return price_in_troy_ounce
    
    # If converting from a different base unit, first convert to troy ounce
    price_in_troy_ounce_base = price_in_troy_ounce
    if from_unit != 'troy_ounce' and from_unit in PRECIOUS_METALS_UNITS:
        price_in_troy_ounce_base = price_in_troy_ounce * PRECIOUS_METALS_UNITS[from_unit]['multiplier']
    
    # Convert from troy ounce to target unit
    target_multiplier = PRECIOUS_METALS_UNITS[to_unit]['multiplier']
    return price_in_troy_ounce_base / target_multiplier

def format_price(price, unit='troy_ounce', currency='USD'):
    """
    Format price with appropriate decimal places based on unit
    
    Args:
        price (float): Price to format
        unit (str): Unit for formatting
        currency (str): Currency symbol
    
    Returns:
        dict: Formatted price information
    """
    if not price or unit not in PRECIOUS_METALS_UNITS:
        return {
            'formatted': f'{currency} 0.00',
            'price': '0.00',
            'unit': 'oz t',
            'currency': currency
        }
    
    # Determine decimal places based on unit and price magnitude
    decimal_places = 2
    
    if unit in ['gram', 'tola', 'baht']:
        decimal_places = 3 if price < 10 else 2
    elif unit == 'kilogram':
        decimal_places = 0
    elif unit in ['troy_ounce', 'ounce']:
        decimal_places = 2
    
    formatted_price = f"{price:.{decimal_places}f}"
    unit_symbol = PRECIOUS_METALS_UNITS[unit]['symbol']
    
    return {
        'formatted': f'{currency} {formatted_price}',
        'price': formatted_price,
        'unit': unit_symbol,
        'currency': currency
    }

def get_unit_categories():
    """
    Get unit categories for filtering
    
    Returns:
        dict: Categories with their units
    """
    categories = {}
    for unit in PRECIOUS_METALS_UNITS.values():
        category = unit['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(unit)
    return categories

def get_popular_units():
    """
    Get popular units for quick selection
    
    Returns:
        list: Popular units
    """
    return [
        PRECIOUS_METALS_UNITS['troy_ounce'],
        PRECIOUS_METALS_UNITS['gram'],
        PRECIOUS_METALS_UNITS['kilogram'],
        PRECIOUS_METALS_UNITS['ounce']
    ]

def convert_weight(weight, from_unit, to_unit):
    """
    Convert weight/quantity between units
    
    Args:
        weight (float): Weight to convert
        from_unit (str): Source unit
        to_unit (str): Target unit
    
    Returns:
        float: Converted weight
    """
    if not weight or from_unit == to_unit:
        return weight
    
    from_unit_data = PRECIOUS_METALS_UNITS.get(from_unit)
    to_unit_data = PRECIOUS_METALS_UNITS.get(to_unit)
    
    if not from_unit_data or not to_unit_data:
        return weight
    
    # Convert to troy ounce first, then to target unit
    weight_in_troy_ounce = weight / from_unit_data['multiplier']
    return weight_in_troy_ounce * to_unit_data['multiplier']

def get_unit_display_info(unit_id):
    """
    Get unit display information
    
    Args:
        unit_id (str): Unit identifier
    
    Returns:
        dict: Unit display information or None
    """
    unit = PRECIOUS_METALS_UNITS.get(unit_id)
    if not unit:
        return None
    
    return {
        'name': unit['name'],
        'symbol': unit['symbol'],
        'category': unit['category'],
        'description': unit['description']
    }

def validate_unit(unit_id):
    """
    Validate if unit ID is supported
    
    Args:
        unit_id (str): Unit identifier to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    return unit_id in PRECIOUS_METALS_UNITS

def get_all_units():
    """
    Get all available units
    
    Returns:
        dict: All available units
    """
    return PRECIOUS_METALS_UNITS.copy()

