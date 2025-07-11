"""
Currency conversion utility for precious metals prices
Supports major global currencies with real-time exchange rates
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Major global currencies supported
SUPPORTED_CURRENCIES = {
    'USD': {
        'id': 'USD',
        'name': 'US Dollar',
        'symbol': '$',
        'country': 'United States',
        'region': 'North America',
        'is_base': True  # USD is the base currency for precious metals
    },
    'EUR': {
        'id': 'EUR',
        'name': 'Euro',
        'symbol': '€',
        'country': 'European Union',
        'region': 'Europe',
        'is_base': False
    },
    'GBP': {
        'id': 'GBP',
        'name': 'British Pound',
        'symbol': '£',
        'country': 'United Kingdom',
        'region': 'Europe',
        'is_base': False
    },
    'JPY': {
        'id': 'JPY',
        'name': 'Japanese Yen',
        'symbol': '¥',
        'country': 'Japan',
        'region': 'Asia',
        'is_base': False
    },
    'CAD': {
        'id': 'CAD',
        'name': 'Canadian Dollar',
        'symbol': 'C$',
        'country': 'Canada',
        'region': 'North America',
        'is_base': False
    },
    'AUD': {
        'id': 'AUD',
        'name': 'Australian Dollar',
        'symbol': 'A$',
        'country': 'Australia',
        'region': 'Oceania',
        'is_base': False
    },
    'CHF': {
        'id': 'CHF',
        'name': 'Swiss Franc',
        'symbol': 'CHF',
        'country': 'Switzerland',
        'region': 'Europe',
        'is_base': False
    },
    'CNY': {
        'id': 'CNY',
        'name': 'Chinese Yuan',
        'symbol': '¥',
        'country': 'China',
        'region': 'Asia',
        'is_base': False
    },
    'INR': {
        'id': 'INR',
        'name': 'Indian Rupee',
        'symbol': '₹',
        'country': 'India',
        'region': 'Asia',
        'is_base': False
    },
    'SGD': {
        'id': 'SGD',
        'name': 'Singapore Dollar',
        'symbol': 'S$',
        'country': 'Singapore',
        'region': 'Asia',
        'is_base': False
    }
}

# Cache for exchange rates (in-memory cache for demo)
_exchange_rate_cache = {}
_cache_timestamp = None
_cache_duration = timedelta(hours=1)  # Cache for 1 hour

def get_exchange_rates(base_currency='USD') -> Dict[str, float]:
    """
    Get current exchange rates from USD to other currencies
    Uses a free exchange rate API with caching
    
    Args:
        base_currency (str): Base currency (default: USD)
    
    Returns:
        dict: Exchange rates from base currency to other currencies
    """
    global _exchange_rate_cache, _cache_timestamp
    
    # Check if cache is valid
    if (_cache_timestamp and 
        datetime.now() - _cache_timestamp < _cache_duration and 
        _exchange_rate_cache):
        return _exchange_rate_cache
    
    try:
        # Use a free exchange rate API (exchangerate-api.com)
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            rates = data.get('rates', {})
            
            # Filter to only supported currencies
            filtered_rates = {}
            for currency_code in SUPPORTED_CURRENCIES.keys():
                if currency_code in rates:
                    filtered_rates[currency_code] = rates[currency_code]
                elif currency_code == base_currency:
                    filtered_rates[currency_code] = 1.0
            
            # Update cache
            _exchange_rate_cache = filtered_rates
            _cache_timestamp = datetime.now()
            
            logger.info(f"Successfully fetched exchange rates for {len(filtered_rates)} currencies")
            return filtered_rates
            
    except Exception as e:
        logger.error(f"Error fetching exchange rates: {e}")
    
    # Fallback to demo rates if API fails
    return get_demo_exchange_rates()

def get_demo_exchange_rates() -> Dict[str, float]:
    """
    Get demo exchange rates for development/fallback
    
    Returns:
        dict: Demo exchange rates from USD
    """
    return {
        'USD': 1.0,
        'EUR': 0.85,
        'GBP': 0.73,
        'JPY': 110.0,
        'CAD': 1.25,
        'AUD': 1.35,
        'CHF': 0.92,
        'CNY': 6.45,
        'INR': 74.5,
        'SGD': 1.35
    }

def convert_price(price_usd: float, from_currency='USD', to_currency='USD') -> float:
    """
    Convert price from one currency to another
    
    Args:
        price_usd (float): Price in USD (base currency for precious metals)
        from_currency (str): Source currency
        to_currency (str): Target currency
    
    Returns:
        float: Converted price
    """
    if not price_usd or from_currency == to_currency:
        return price_usd
    
    if from_currency not in SUPPORTED_CURRENCIES or to_currency not in SUPPORTED_CURRENCIES:
        logger.warning(f"Unsupported currency conversion: {from_currency} to {to_currency}")
        return price_usd
    
    try:
        exchange_rates = get_exchange_rates()
        
        # Convert from source currency to USD first (if needed)
        price_in_usd = price_usd
        if from_currency != 'USD':
            from_rate = exchange_rates.get(from_currency, 1.0)
            price_in_usd = price_usd / from_rate
        
        # Convert from USD to target currency
        if to_currency != 'USD':
            to_rate = exchange_rates.get(to_currency, 1.0)
            return price_in_usd * to_rate
        
        return price_in_usd
        
    except Exception as e:
        logger.error(f"Error converting currency: {e}")
        return price_usd

def format_price_with_currency(price: float, currency='USD') -> Dict[str, str]:
    """
    Format price with appropriate currency symbol and decimal places
    
    Args:
        price (float): Price to format
        currency (str): Currency code
    
    Returns:
        dict: Formatted price information
    """
    if currency not in SUPPORTED_CURRENCIES:
        currency = 'USD'
    
    currency_info = SUPPORTED_CURRENCIES[currency]
    symbol = currency_info['symbol']
    
    # Determine decimal places based on currency
    if currency == 'JPY':
        # Japanese Yen typically doesn't use decimal places
        decimal_places = 0
    elif currency in ['INR', 'CNY']:
        # Some Asian currencies use 2 decimal places but can show more for precision
        decimal_places = 2
    else:
        # Most major currencies use 2 decimal places
        decimal_places = 2
    
    formatted_price = f"{price:.{decimal_places}f}"
    
    return {
        'formatted': f"{symbol}{formatted_price}",
        'price': formatted_price,
        'symbol': symbol,
        'currency': currency,
        'currency_name': currency_info['name']
    }

def get_currency_categories() -> Dict[str, List[Dict]]:
    """
    Get currencies organized by region for UI filtering
    
    Returns:
        dict: Currencies grouped by region
    """
    categories = {}
    for currency in SUPPORTED_CURRENCIES.values():
        region = currency['region']
        if region not in categories:
            categories[region] = []
        categories[region].append(currency)
    
    return categories

def get_popular_currencies() -> List[Dict]:
    """
    Get most commonly used currencies for quick selection
    
    Returns:
        list: Popular currencies
    """
    popular_codes = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD']
    return [SUPPORTED_CURRENCIES[code] for code in popular_codes if code in SUPPORTED_CURRENCIES]

def validate_currency(currency_code: str) -> bool:
    """
    Validate if currency code is supported
    
    Args:
        currency_code (str): Currency code to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    return currency_code in SUPPORTED_CURRENCIES

def get_currency_info(currency_code: str) -> Optional[Dict]:
    """
    Get detailed information about a currency
    
    Args:
        currency_code (str): Currency code
    
    Returns:
        dict: Currency information or None if not found
    """
    return SUPPORTED_CURRENCIES.get(currency_code)

def get_all_currencies() -> Dict[str, Dict]:
    """
    Get all supported currencies
    
    Returns:
        dict: All supported currencies
    """
    return SUPPORTED_CURRENCIES.copy()

def refresh_exchange_rates() -> bool:
    """
    Force refresh of exchange rate cache
    
    Returns:
        bool: True if successful, False otherwise
    """
    global _exchange_rate_cache, _cache_timestamp
    
    try:
        _exchange_rate_cache = {}
        _cache_timestamp = None
        
        # Fetch fresh rates
        rates = get_exchange_rates()
        return len(rates) > 0
        
    except Exception as e:
        logger.error(f"Error refreshing exchange rates: {e}")
        return False

