import requests
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class MetalsPriceService:
    """Service for fetching precious metals price data from external APIs"""
    
    def __init__(self):
        # MetalpriceAPI configuration
        self.metalpriceapi_base_url = "https://api.metalpriceapi.com/v1"
        self.metalpriceapi_key = os.getenv('METALS_API_KEY', 'demo')  # Use METALS_API_KEY from .env
        
        # Metals-API configuration (backup)
        self.metals_api_base_url = "https://metals-api.com/api"
        self.metals_api_key = os.getenv('METALS_API_KEY')  # Same key for both services
        
        # Metal symbol mappings for MetalpriceAPI
        self.metalpriceapi_symbols = {
            'gold': 'XAU',
            'silver': 'XAG', 
            'platinum': 'XPT',
            'palladium': 'XPD'
        }
        
        # Metal symbol mappings for Metals-API (backup)
        self.metal_symbols = {
            'gold': 'USDXAU',
            'silver': 'USDXAG', 
            'platinum': 'USDXPT',
            'palladium': 'USDXPD'
        }
        
        self.symbol_to_name = {
            'XAU': 'Gold',
            'XAG': 'Silver',
            'XPT': 'Platinum',
            'XPD': 'Palladium',
            'USDXAU': 'Gold',
            'USDXAG': 'Silver',
            'USDXPT': 'Platinum',
            'USDXPD': 'Palladium'
        }
    
    def get_current_prices(self, metals: Optional[List[str]] = None, currency: str = 'USD') -> Dict:
        """
        Fetch current spot prices for specified metals
        
        Args:
            metals: List of metal names (gold, silver, platinum, palladium)
            currency: Target currency (default: USD)
            
        Returns:
            Dict containing current prices and metadata
        """
        try:
            # Use all metals if none specified
            if metals is None:
                metals = list(self.metalpriceapi_symbols.keys())
            
            # Try MetalpriceAPI first
            try:
                # Convert metal names to MetalpriceAPI symbols
                metalpriceapi_symbols = [self.metalpriceapi_symbols.get(metal.lower()) for metal in metals if metal.lower() in self.metalpriceapi_symbols]
                if metalpriceapi_symbols:
                    return self._fetch_from_metalpriceapi(metalpriceapi_symbols, currency)
            except Exception as e:
                logger.warning(f"MetalpriceAPI failed: {e}")
                
                # Fallback to Metals-API if available
                if self.metals_api_key:
                    # Convert metal names to Metals-API symbols
                    metals_api_symbols = [self.metal_symbols.get(metal.lower()) for metal in metals if metal.lower() in self.metal_symbols]
                    if metals_api_symbols:
                        return self._fetch_from_metals_api(metals_api_symbols, currency)
                
                # Return demo data for development
                demo_symbols = [self.metalpriceapi_symbols.get(metal.lower()) for metal in metals if metal.lower() in self.metalpriceapi_symbols]
                return self._get_demo_prices(demo_symbols)
                    
        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
            raise
    
    def get_historical_prices(self, metal: str, start_date: datetime, end_date: datetime, 
                            interval: str = 'daily') -> List[Dict]:
        """
        Fetch historical price data for a specific metal
        
        Args:
            metal: Metal name (gold, silver, platinum, palladium)
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval (daily, hourly)
            
        Returns:
            List of historical price records
        """
        try:
            symbol = self.metal_symbols.get(metal.lower())
            if not symbol:
                raise ValueError(f"Invalid metal: {metal}")
            
            # For demo purposes, generate sample historical data
            return self._generate_sample_historical_data(symbol, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error fetching historical prices: {e}")
            raise
    
    def _fetch_from_metalpriceapi(self, symbols: List[str], currency: str) -> Dict:
        """Fetch prices from MetalpriceAPI"""
        url = f"{self.metalpriceapi_base_url}/latest"
        params = {
            'api_key': self.metalpriceapi_key,
            'base': currency,
            'currencies': ','.join(symbols)
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('success', False):
            raise Exception(f"API error: {data.get('error', 'Unknown error')}")
        
        # Transform response to our format
        prices = []
        rates = data.get('rates', {})
        timestamp = datetime.utcnow()
        
        for symbol in symbols:
            if symbol in rates:
                # MetalpriceAPI returns price per ounce, we need to convert to price per unit
                price = 1 / rates[symbol] if rates[symbol] > 0 else 0
                
                prices.append({
                    'metal': symbol,
                    'metal_name': self.symbol_to_name.get(symbol, symbol),
                    'price': round(price, 2),
                    'currency': currency,
                    'timestamp': timestamp.isoformat(),
                    'source': 'metalpriceapi',
                    'change_24h': 0,  # Would need additional API call for this
                    'change_24h_percent': 0
                })
        
        return {
            'timestamp': timestamp.isoformat(),
            'source': 'metalpriceapi',
            'currency': currency,
            'prices': prices
        }
    
    def _fetch_from_metals_api(self, symbols: List[str], currency: str) -> Dict:
        """Fetch prices from Metals-API (backup)"""
        url = f"{self.metals_api_base_url}/latest"
        params = {
            'access_key': self.metals_api_key,
            'base': currency,
            'symbols': ','.join(symbols)
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('success', False):
            raise Exception(f"Metals-API error: {data.get('error', 'Unknown error')}")
        
        # Transform response to our format
        prices = []
        rates = data.get('rates', {})
        timestamp = datetime.utcnow()
        
        for symbol in symbols:
            if symbol in rates:
                prices.append({
                    'metal': symbol,
                    'metal_name': self.symbol_to_name.get(symbol, symbol),
                    'price': rates[symbol],
                    'currency': currency,
                    'timestamp': timestamp.isoformat(),
                    'source': 'metals-api',
                    'change_24h': 0,
                    'change_24h_percent': 0
                })
        
        return {
            'timestamp': timestamp.isoformat(),
            'source': 'metals-api',
            'currency': currency,
            'prices': prices
        }
    
    def _get_demo_prices(self, symbols: List[str]) -> Dict:
        """Generate demo prices for development/testing"""
        demo_prices = {
            'USDXAU': 2850.00,  # Gold
            'USDXAG': 32.10,    # Silver
            'USDXPT': 1200.00,  # Platinum
            'USDXPD': 950.00    # Palladium
        }
        
        prices = []
        timestamp = datetime.utcnow()
        
        for symbol in symbols:
            if symbol in demo_prices:
                # Add some random variation for demo
                import random
                base_price = demo_prices[symbol]
                variation = random.uniform(-0.02, 0.02)  # ±2% variation
                price = base_price * (1 + variation)
                
                prices.append({
                    'metal': symbol,
                    'metal_name': self.symbol_to_name.get(symbol, symbol),
                    'price': round(price, 2),
                    'currency': 'USD',
                    'timestamp': timestamp.isoformat(),
                    'source': 'demo',
                    'change_24h': round(base_price * variation, 2),
                    'change_24h_percent': round(variation * 100, 2)
                })
        
        return {
            'timestamp': timestamp.isoformat(),
            'source': 'demo',
            'currency': 'USD',
            'prices': prices
        }
    
    def _generate_sample_historical_data(self, symbol: str, start_date: datetime, 
                                       end_date: datetime) -> List[Dict]:
        """Generate sample historical data for development"""
        base_prices = {
            'USDXAU': 2850.00,
            'USDXAG': 32.10,
            'USDXPT': 1200.00,
            'USDXPD': 950.00
        }
        
        if symbol not in base_prices:
            return []
        
        base_price = base_prices[symbol]
        data = []
        current_date = start_date
        current_price = base_price
        
        import random
        
        while current_date <= end_date:
            # Simulate price movement
            daily_change = random.uniform(-0.03, 0.03)  # ±3% daily variation
            current_price *= (1 + daily_change)
            
            data.append({
                'timestamp': current_date.isoformat(),
                'price': round(current_price, 2),
                'volume': random.randint(1000, 10000),
                'high': round(current_price * 1.02, 2),
                'low': round(current_price * 0.98, 2),
                'open': round(current_price * 0.99, 2),
                'close': round(current_price, 2)
            })
            
            current_date += timedelta(days=1)
        
        return data
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate API keys for external services"""
        results = {}
        
        # Test MetalpriceAPI
        try:
            response = requests.get(
                f"{self.metalpriceapi_base_url}/latest",
                params={'api_key': self.metalpriceapi_key, 'base': 'USD', 'currencies': 'USDXAU'},
                timeout=5
            )
            results['metalpriceapi'] = response.status_code == 200
        except:
            results['metalpriceapi'] = False
        
        # Test Metals-API if key is available
        if self.metals_api_key:
            try:
                response = requests.get(
                    f"{self.metals_api_base_url}/latest",
                    params={'access_key': self.metals_api_key, 'base': 'USD', 'symbols': 'USDXAU'},
                    timeout=5
                )
                results['metals_api'] = response.status_code == 200
            except:
                results['metals_api'] = False
        else:
            results['metals_api'] = False
        
        return results

