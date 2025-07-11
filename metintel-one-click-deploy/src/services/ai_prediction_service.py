import os
import json
import uuid
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

class AIPredictionService:
    """Service for AI-powered precious metals price predictions"""
    
    def __init__(self):
        # OpenAI configuration
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not found. AI predictions will use demo mode.")
        
        # Perplexity configuration
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        self.perplexity_base_url = "https://api.perplexity.ai"
        
        # Model configurations
        self.openai_model = "gpt-4"
        self.perplexity_model = "sonar-pro"
        
        # Prediction parameters
        self.max_prediction_horizon_days = 730  # 24 months
        self.min_prediction_horizon_days = 7    # 1 week
    
    def generate_prediction(self, metal: str, horizon: str, historical_data: List[Dict], 
                          current_price: float, user_parameters: Optional[Dict] = None) -> Dict:
        """
        Generate AI-powered price prediction for a precious metal
        
        Args:
            metal: Metal symbol (USDXAU, USDXAG, etc.)
            horizon: Prediction horizon (e.g., "1 week", "3 months")
            historical_data: Historical price data
            current_price: Current spot price
            user_parameters: Additional user-specified parameters
            
        Returns:
            Dict containing prediction results and analysis
        """
        try:
            # Parse prediction horizon
            target_date = self._parse_horizon_to_date(horizon)
            
            # Get market context from Perplexity
            market_context = self._get_market_context(metal)
            
            # Generate prediction using OpenAI
            prediction_result = self._generate_openai_prediction(
                metal, historical_data, current_price, target_date, market_context
            )
            
            # Generate prediction factors
            factors = self._analyze_prediction_factors(
                metal, historical_data, market_context, prediction_result
            )
            
            # Create prediction record
            prediction_id = str(uuid.uuid4())
            
            return {
                'id': prediction_id,
                'metal': metal,
                'horizon': horizon,
                'target_date': target_date.isoformat(),
                'current_price': current_price,
                'predicted_price': prediction_result['predicted_price'],
                'confidence_interval_lower': prediction_result['confidence_lower'],
                'confidence_interval_upper': prediction_result['confidence_upper'],
                'confidence_score': prediction_result['confidence_score'],
                'model_version': f"openai-{self.openai_model}",
                'market_context': market_context,
                'factors': factors,
                'reasoning': prediction_result['reasoning'],
                'created_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            raise
    
    def _parse_horizon_to_date(self, horizon: str) -> datetime:
        """Parse horizon string to target date"""
        horizon_lower = horizon.lower().strip()
        current_date = datetime.utcnow()
        
        # Parse common horizon formats
        if 'week' in horizon_lower:
            if '1 week' in horizon_lower:
                return current_date + timedelta(weeks=1)
            elif '2 week' in horizon_lower:
                return current_date + timedelta(weeks=2)
            elif '4 week' in horizon_lower:
                return current_date + timedelta(weeks=4)
        
        elif 'month' in horizon_lower:
            if '1 month' in horizon_lower:
                return current_date + timedelta(days=30)
            elif '3 month' in horizon_lower:
                return current_date + timedelta(days=90)
            elif '6 month' in horizon_lower:
                return current_date + timedelta(days=180)
            elif '12 month' in horizon_lower:
                return current_date + timedelta(days=365)
            elif '18 month' in horizon_lower:
                return current_date + timedelta(days=547)
            elif '24 month' in horizon_lower:
                return current_date + timedelta(days=730)
        
        elif 'year' in horizon_lower:
            if '1 year' in horizon_lower:
                return current_date + timedelta(days=365)
            elif '2 year' in horizon_lower:
                return current_date + timedelta(days=730)
        
        # Default to 1 month if parsing fails
        logger.warning(f"Could not parse horizon '{horizon}', defaulting to 1 month")
        return current_date + timedelta(days=30)
    
    def _get_market_context(self, metal: str) -> Dict:
        """Get current market context using Perplexity AI"""
        try:
            metal_name = {
                'USDXAU': 'gold',
                'USDXAG': 'silver', 
                'USDXPT': 'platinum',
                'USDXPD': 'palladium'
            }.get(metal, metal)
            
            # Query for current market sentiment and news
            query = f"Current {metal_name} market analysis, recent price trends, economic factors affecting {metal_name} prices, central bank policies, inflation impact on precious metals"
            
            if self.perplexity_api_key:
                context = self._query_perplexity(query)
            else:
                # Fallback to demo context
                context = self._get_demo_market_context(metal_name)
            
            return {
                'query': query,
                'analysis': context,
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'perplexity' if self.perplexity_api_key else 'demo'
            }
            
        except Exception as e:
            logger.warning(f"Error getting market context: {e}")
            return {
                'analysis': f"Unable to fetch current market context for {metal}",
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'error'
            }
    
    def _query_perplexity(self, query: str) -> str:
        """Query Perplexity AI for market analysis"""
        headers = {
            'Authorization': f'Bearer {self.perplexity_api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.perplexity_model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a financial analyst specializing in precious metals markets. Provide concise, factual analysis based on current market conditions.'
                },
                {
                    'role': 'user',
                    'content': query
                }
            ],
            'max_tokens': 500,
            'temperature': 0.2
        }
        
        response = requests.post(
            f"{self.perplexity_base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
    
    def _get_demo_market_context(self, metal_name: str) -> str:
        """Generate demo market context for development"""
        demo_contexts = {
            'gold': "Current gold market shows strong institutional demand driven by inflation hedging and central bank purchases. Recent geopolitical tensions have increased safe-haven demand. Federal Reserve policy uncertainty continues to support gold prices.",
            'silver': "Silver market experiencing industrial demand growth from renewable energy sector. Supply constraints from major mining regions affecting availability. Gold-silver ratio remains elevated, suggesting potential silver outperformance.",
            'platinum': "Platinum facing mixed signals with automotive industry transition to electric vehicles reducing catalytic converter demand, while hydrogen fuel cell development provides new growth opportunities. Supply disruptions from South African mines continue.",
            'palladium': "Palladium market remains tight due to Russian supply concerns and strong automotive demand for gasoline engine catalytic converters. Inventory levels at multi-year lows supporting higher prices."
        }
        
        return demo_contexts.get(metal_name, f"Market analysis for {metal_name} not available in demo mode.")
    
    def _generate_openai_prediction(self, metal: str, historical_data: List[Dict], 
                                  current_price: float, target_date: datetime, 
                                  market_context: Dict) -> Dict:
        """Generate prediction using OpenAI GPT model"""
        try:
            # Prepare historical data summary
            if historical_data:
                recent_data = historical_data[-30:]  # Last 30 data points
                price_trend = self._calculate_price_trend(recent_data)
                volatility = self._calculate_volatility(recent_data)
            else:
                price_trend = 0
                volatility = 0.02  # Default 2% volatility
            
            # Create prediction prompt
            prompt = self._create_prediction_prompt(
                metal, current_price, target_date, price_trend, volatility, market_context
            )
            
            # Query OpenAI
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert financial analyst specializing in precious metals price prediction. Provide detailed, data-driven analysis with specific price targets and confidence intervals."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                analysis = response.choices[0].message.content
                return self._parse_openai_response(analysis, current_price)
            else:
                # Demo prediction
                return self._generate_demo_prediction(current_price, target_date)
                
        except Exception as e:
            logger.error(f"Error in OpenAI prediction: {e}")
            return self._generate_demo_prediction(current_price, target_date)
    
    def _create_prediction_prompt(self, metal: str, current_price: float, target_date: datetime,
                                price_trend: float, volatility: float, market_context: Dict) -> str:
        """Create detailed prompt for OpenAI prediction"""
        metal_name = {
            'USDXAU': 'Gold',
            'USDXAG': 'Silver',
            'USDXPT': 'Platinum', 
            'USDXPD': 'Palladium'
        }.get(metal, metal)
        
        days_ahead = (target_date - datetime.utcnow()).days
        
        prompt = f"""
        Analyze and predict the price of {metal_name} for {days_ahead} days from now (target date: {target_date.strftime('%Y-%m-%d')}).

        Current Market Data:
        - Current Price: ${current_price:.2f}
        - Recent Price Trend: {price_trend:.2%}
        - Historical Volatility: {volatility:.2%}
        
        Market Context:
        {market_context.get('analysis', 'No market context available')}
        
        Please provide:
        1. Predicted price target with reasoning
        2. Confidence interval (lower and upper bounds)
        3. Confidence score (0-1)
        4. Key factors influencing the prediction
        5. Risk assessment
        
        Format your response as JSON with the following structure:
        {{
            "predicted_price": <number>,
            "confidence_lower": <number>,
            "confidence_upper": <number>, 
            "confidence_score": <number between 0 and 1>,
            "reasoning": "<detailed explanation>",
            "key_factors": ["<factor1>", "<factor2>", ...],
            "risks": ["<risk1>", "<risk2>", ...]
        }}
        """
        
        return prompt
    
    def _parse_openai_response(self, response: str, current_price: float) -> Dict:
        """Parse OpenAI response and extract prediction data"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                return {
                    'predicted_price': float(data.get('predicted_price', current_price)),
                    'confidence_lower': float(data.get('confidence_lower', current_price * 0.95)),
                    'confidence_upper': float(data.get('confidence_upper', current_price * 1.05)),
                    'confidence_score': float(data.get('confidence_score', 0.7)),
                    'reasoning': data.get('reasoning', response),
                    'key_factors': data.get('key_factors', []),
                    'risks': data.get('risks', [])
                }
            else:
                # Fallback parsing
                return self._generate_demo_prediction(current_price, datetime.utcnow())
                
        except Exception as e:
            logger.warning(f"Error parsing OpenAI response: {e}")
            return self._generate_demo_prediction(current_price, datetime.utcnow())
    
    def _generate_demo_prediction(self, current_price: float, target_date: datetime) -> Dict:
        """Generate demo prediction for development"""
        import random
        
        # Simple prediction logic for demo
        days_ahead = (target_date - datetime.utcnow()).days
        trend_factor = 1 + (random.uniform(-0.1, 0.1) * days_ahead / 365)  # ±10% annual trend
        predicted_price = current_price * trend_factor
        
        confidence_range = current_price * 0.1  # ±10% confidence interval
        
        return {
            'predicted_price': round(predicted_price, 2),
            'confidence_lower': round(predicted_price - confidence_range, 2),
            'confidence_upper': round(predicted_price + confidence_range, 2),
            'confidence_score': 0.75,
            'reasoning': f"Demo prediction based on current price of ${current_price:.2f} with simulated market analysis for {days_ahead} days ahead.",
            'key_factors': ["Market sentiment", "Economic indicators", "Supply/demand dynamics"],
            'risks': ["Market volatility", "Economic uncertainty", "Geopolitical events"]
        }
    
    def _calculate_price_trend(self, data: List[Dict]) -> float:
        """Calculate recent price trend from historical data"""
        if len(data) < 2:
            return 0
        
        prices = [float(d['price']) for d in data]
        start_price = prices[0]
        end_price = prices[-1]
        
        return (end_price - start_price) / start_price
    
    def _calculate_volatility(self, data: List[Dict]) -> float:
        """Calculate price volatility from historical data"""
        if len(data) < 2:
            return 0.02  # Default 2%
        
        prices = [float(d['price']) for d in data]
        returns = []
        
        for i in range(1, len(prices)):
            daily_return = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(daily_return)
        
        if not returns:
            return 0.02
        
        # Calculate standard deviation of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        return volatility
    
    def _analyze_prediction_factors(self, metal: str, historical_data: List[Dict],
                                  market_context: Dict, prediction_result: Dict) -> List[Dict]:
        """Analyze factors contributing to the prediction"""
        factors = []
        
        # Technical factors
        if historical_data:
            trend = self._calculate_price_trend(historical_data[-30:])
            factors.append({
                'id': str(uuid.uuid4()),
                'type': 'technical',
                'name': 'Price Trend',
                'impact_weight': min(max(trend, -1), 1),  # Clamp between -1 and 1
                'description': f"Recent 30-day price trend: {trend:.2%}"
            })
        
        # Market sentiment factors
        factors.append({
            'id': str(uuid.uuid4()),
            'type': 'sentiment',
            'name': 'Market Sentiment',
            'impact_weight': 0.6,  # Positive sentiment
            'description': "Current market sentiment based on news and analysis"
        })
        
        # Economic factors
        factors.append({
            'id': str(uuid.uuid4()),
            'type': 'fundamental',
            'name': 'Economic Environment',
            'impact_weight': 0.4,
            'description': "Macroeconomic conditions affecting precious metals"
        })
        
        return factors
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate AI service API keys"""
        results = {}
        
        # Test OpenAI
        try:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=5
                )
                results['openai'] = True
            else:
                results['openai'] = False
        except:
            results['openai'] = False
        
        # Test Perplexity
        try:
            if self.perplexity_api_key:
                headers = {'Authorization': f'Bearer {self.perplexity_api_key}'}
                response = requests.get(
                    f"{self.perplexity_base_url}/models",
                    headers=headers,
                    timeout=5
                )
                results['perplexity'] = response.status_code == 200
            else:
                results['perplexity'] = False
        except:
            results['perplexity'] = False
        
        return results

