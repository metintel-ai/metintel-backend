"""
AI Agent Service for PreciousAI Application
Advanced autonomous AI agent for investment target monitoring and recommendations
Integrated from PreciousPredictor codebase
"""

import os
import sqlite3
import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

from openai import OpenAI
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAgentService:
    """
    Advanced AI Agent service providing:
    - Intelligent target monitoring
    - Automated buy/sell signal generation
    - Portfolio optimization recommendations
    - Market opportunity identification
    """
    
    def __init__(self):
        """Initialize the AI agent service"""
        # API Configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        self.metals_api_key = os.getenv('METALS_API_KEY')
        
        # Initialize OpenAI client if available
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not found - AI recommendations will be limited")
        
        # Database configuration
        self.db_path = "/home/ubuntu/precious-metals-api/ai_agent.db"
        
        # Agent configuration
        self.monitoring_active = False
        self.monitoring_thread = None
        self.check_interval = 300  # 5 minutes
        
        # AI model configurations
        self.analysis_model = "gpt-4"
        self.confidence_threshold = 70.0
        
        # Market data cache
        self.market_data_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("AI Agent Service initialized successfully")
    
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def start_monitoring(self):
        """Start the AI agent monitoring service"""
        if self.monitoring_active:
            logger.warning("Agent monitoring already running")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("AI Agent monitoring service started")
    
    def stop_monitoring(self):
        """Stop the AI agent monitoring service"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("AI Agent monitoring service stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for AI agent"""
        while self.monitoring_active:
            try:
                # Check all active targets
                self._check_investment_targets()
                
                # Generate new recommendations based on market conditions
                self._generate_market_opportunities()
                
                # Clean up expired recommendations
                self._cleanup_expired_recommendations()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in agent monitoring loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def create_investment_target(self, user_id: int, metal_type: str, target_type: str,
                               target_price: float, target_quantity: Optional[float] = None,
                               target_percentage: Optional[float] = None, priority: str = "MEDIUM",
                               target_date: Optional[datetime] = None, notes: Optional[str] = None) -> int:
        """Create a new investment target"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get current price for distance calculation
                current_price = self._get_current_metal_price(metal_type)
                distance_to_target = abs(target_price - current_price) if current_price else None
                
                cursor.execute("""
                    INSERT INTO investment_targets 
                    (user_id, metal_type, target_type, target_price, target_quantity, 
                     target_percentage, current_price, distance_to_target, priority, 
                     target_date, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, metal_type, target_type, target_price, target_quantity,
                      target_percentage, current_price, distance_to_target, priority,
                      target_date, notes))
                
                target_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Created investment target {target_id} for user {user_id}")
                return target_id
                
        except Exception as e:
            logger.error(f"Error creating investment target: {str(e)}")
            raise
    
    def get_user_targets(self, user_id: int, status: str = "ACTIVE") -> List[Dict]:
        """Get all investment targets for a user"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM investment_targets 
                    WHERE user_id = ? AND status = ?
                    ORDER BY created_date DESC
                """, (user_id, status))
                
                columns = [description[0] for description in cursor.description]
                targets = []
                
                for row in cursor.fetchall():
                    target = dict(zip(columns, row))
                    targets.append(target)
                
                return targets
                
        except Exception as e:
            logger.error(f"Error getting user targets: {str(e)}")
            return []
    
    def _check_investment_targets(self):
        """Check all active investment targets for triggers"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get all active targets
                cursor.execute("""
                    SELECT * FROM investment_targets 
                    WHERE status = 'ACTIVE'
                """)
                
                columns = [description[0] for description in cursor.description]
                targets = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                for target in targets:
                    self._check_single_target(target)
                    
        except Exception as e:
            logger.error(f"Error checking investment targets: {str(e)}")
    
    def _check_single_target(self, target: Dict):
        """Check a single investment target for trigger conditions"""
        try:
            metal_type = target['metal_type']
            current_price = self._get_current_metal_price(metal_type)
            
            if not current_price:
                return
            
            target_price = target['target_price']
            target_type = target['target_type']
            triggered = False
            
            # Check trigger conditions based on target type
            if target_type == 'buy_below' and current_price <= target_price:
                triggered = True
            elif target_type == 'sell_above' and current_price >= target_price:
                triggered = True
            elif target_type == 'stop_loss' and current_price <= target_price:
                triggered = True
            elif target_type == 'take_profit' and current_price >= target_price:
                triggered = True
            
            if triggered:
                self._trigger_target(target, current_price)
                
            # Update target with current price and distance
            self._update_target_status(target['id'], current_price)
            
        except Exception as e:
            logger.error(f"Error checking single target: {str(e)}")
    
    def _trigger_target(self, target: Dict, current_price: float):
        """Trigger an investment target and generate recommendation"""
        try:
            # Generate AI recommendation
            recommendation = self._generate_target_recommendation(target, current_price)
            
            if recommendation:
                # Save recommendation to database
                self._save_recommendation(recommendation)
                
                # Update target status
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE investment_targets 
                        SET status = 'TRIGGERED', times_triggered = times_triggered + 1,
                            last_checked = ?
                        WHERE id = ?
                    """, (datetime.now(), target['id']))
                    conn.commit()
                
                logger.info(f"Target {target['id']} triggered at price {current_price}")
                
        except Exception as e:
            logger.error(f"Error triggering target: {str(e)}")
    
    def _generate_target_recommendation(self, target: Dict, current_price: float) -> Optional[Dict]:
        """Generate AI recommendation based on triggered target"""
        try:
            if not self.openai_client:
                # Generate basic recommendation without AI
                return self._generate_basic_recommendation(target, current_price)
            
            # Get market context
            market_context = self._get_market_context(target['metal_type'])
            
            # Create AI prompt
            prompt = f"""
            As an expert precious metals investment advisor, analyze this triggered investment target:
            
            Target Details:
            - Metal: {target['metal_type']}
            - Target Type: {target['target_type']}
            - Target Price: ${target['target_price']}
            - Current Price: ${current_price}
            - User Notes: {target.get('notes', 'None')}
            
            Market Context:
            {market_context}
            
            Provide a recommendation with:
            1. Action (BUY/SELL/HOLD)
            2. Confidence score (0-100)
            3. Reasoning (2-3 sentences)
            4. Suggested quantity or percentage
            5. Urgency level (LOW/MEDIUM/HIGH/CRITICAL)
            
            Format as JSON:
            {{
                "action": "BUY/SELL/HOLD",
                "confidence_score": 85,
                "reasoning": "Clear explanation of recommendation",
                "suggested_quantity": 10.5,
                "urgency": "MEDIUM"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.analysis_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content
            recommendation_data = json.loads(ai_response)
            
            # Create recommendation object
            recommendation = {
                'user_id': target['user_id'],
                'target_id': target['id'],
                'recommendation_type': 'PRICE_TARGET_MONITOR',
                'metal_type': target['metal_type'],
                'action': recommendation_data['action'],
                'confidence_score': recommendation_data['confidence_score'],
                'reasoning': recommendation_data['reasoning'],
                'suggested_quantity': recommendation_data.get('suggested_quantity'),
                'suggested_price': current_price,
                'urgency': recommendation_data['urgency'],
                'expires_date': datetime.now() + timedelta(hours=24),
                'market_data': {'current_price': current_price, 'market_context': market_context}
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating AI recommendation: {str(e)}")
            return self._generate_basic_recommendation(target, current_price)
    
    def _generate_basic_recommendation(self, target: Dict, current_price: float) -> Dict:
        """Generate basic recommendation without AI"""
        action = "BUY" if target['target_type'] in ['buy_below'] else "SELL"
        
        return {
            'user_id': target['user_id'],
            'target_id': target['id'],
            'recommendation_type': 'PRICE_TARGET_MONITOR',
            'metal_type': target['metal_type'],
            'action': action,
            'confidence_score': 75.0,
            'reasoning': f"Target price of ${target['target_price']} reached. Current price: ${current_price}",
            'suggested_quantity': target.get('target_quantity'),
            'suggested_price': current_price,
            'urgency': 'MEDIUM',
            'expires_date': datetime.now() + timedelta(hours=24),
            'market_data': {'current_price': current_price}
        }
    
    def _save_recommendation(self, recommendation: Dict):
        """Save recommendation to database"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO ai_recommendations 
                    (user_id, target_id, recommendation_type, metal_type, action, 
                     confidence_score, reasoning, suggested_quantity, suggested_price, 
                     urgency, expires_date, market_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    recommendation['user_id'], recommendation.get('target_id'),
                    recommendation['recommendation_type'], recommendation['metal_type'],
                    recommendation['action'], recommendation['confidence_score'],
                    recommendation['reasoning'], recommendation.get('suggested_quantity'),
                    recommendation.get('suggested_price'), recommendation['urgency'],
                    recommendation['expires_date'], json.dumps(recommendation.get('market_data', {}))
                ))
                
                conn.commit()
                logger.info(f"Saved recommendation for user {recommendation['user_id']}")
                
        except Exception as e:
            logger.error(f"Error saving recommendation: {str(e)}")
    
    def _get_current_metal_price(self, metal_type: str) -> Optional[float]:
        """Get current price for a metal"""
        try:
            # Check cache first
            cache_key = f"price_{metal_type}"
            if cache_key in self.market_data_cache:
                cached_data = self.market_data_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < timedelta(seconds=self.cache_ttl):
                    return cached_data['price']
            
            # Fetch from MetalpriceAPI
            if self.metals_api_key:
                url = f"https://api.metalpriceapi.com/v1/latest?api_key={self.metals_api_key}&base=USD&currencies={metal_type}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'rates' in data and metal_type in data['rates']:
                        price = 1 / data['rates'][metal_type]  # Convert to price per ounce
                        
                        # Cache the result
                        self.market_data_cache[cache_key] = {
                            'price': price,
                            'timestamp': datetime.now()
                        }
                        
                        return price
            
            # Fallback to demo prices
            demo_prices = {
                'GOLD': 3300.0,
                'SILVER': 36.5,
                'PLATINUM': 1375.0,
                'PALLADIUM': 1110.0
            }
            
            return demo_prices.get(metal_type)
            
        except Exception as e:
            logger.error(f"Error getting metal price: {str(e)}")
            return None
    
    def _get_market_context(self, metal_type: str) -> str:
        """Get market context for AI analysis"""
        try:
            # This would integrate with news APIs and market data
            # For now, return basic context
            current_price = self._get_current_metal_price(metal_type)
            return f"Current {metal_type} price: ${current_price}. Market conditions: Normal trading."
            
        except Exception as e:
            logger.error(f"Error getting market context: {str(e)}")
            return "Market context unavailable."
    
    def _update_target_status(self, target_id: int, current_price: float):
        """Update target with current price and distance"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get target price
                cursor.execute("SELECT target_price FROM investment_targets WHERE id = ?", (target_id,))
                result = cursor.fetchone()
                
                if result:
                    target_price = result[0]
                    distance_to_target = abs(target_price - current_price)
                    
                    cursor.execute("""
                        UPDATE investment_targets 
                        SET current_price = ?, distance_to_target = ?, last_checked = ?
                        WHERE id = ?
                    """, (current_price, distance_to_target, datetime.now(), target_id))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating target status: {str(e)}")
    
    def _generate_market_opportunities(self):
        """Generate new recommendations based on market conditions"""
        try:
            # This would analyze market conditions and generate proactive recommendations
            # For now, we'll implement basic logic
            pass
            
        except Exception as e:
            logger.error(f"Error generating market opportunities: {str(e)}")
    
    def _cleanup_expired_recommendations(self):
        """Clean up expired recommendations"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE ai_recommendations 
                    SET status = 'expired' 
                    WHERE expires_date < ? AND status = 'active'
                """, (datetime.now(),))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error cleaning up expired recommendations: {str(e)}")
    
    def get_user_recommendations(self, user_id: int, status: str = "active") -> List[Dict]:
        """Get recommendations for a user"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM ai_recommendations 
                    WHERE user_id = ? AND status = ?
                    ORDER BY created_date DESC
                """, (user_id, status))
                
                columns = [description[0] for description in cursor.description]
                recommendations = []
                
                for row in cursor.fetchall():
                    rec = dict(zip(columns, row))
                    # Parse JSON fields
                    if rec.get('market_data'):
                        rec['market_data'] = json.loads(rec['market_data'])
                    recommendations.append(rec)
                
                return recommendations
                
        except Exception as e:
            logger.error(f"Error getting user recommendations: {str(e)}")
            return []

# Global instance
ai_agent = AIAgentService()

