"""
Enhanced AI Agent Service for PreciousAI Application
Integrating sophisticated features from PreciousPredictor codebase
Advanced autonomous AI agent for investment target monitoring and recommendations
"""

import os
import sqlite3
import logging
import threading
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

from openai import OpenAI
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAIAgentService:
    """
    Enhanced AI Agent service providing:
    - Intelligent target monitoring with 24/7 automation
    - Automated buy/sell signal generation with confidence scoring
    - Portfolio optimization recommendations
    - Market opportunity identification
    - Risk assessment and management
    - Performance tracking and learning
    """
    
    def __init__(self):
        """Initialize the enhanced AI agent service"""
        # API Configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        self.metals_api_key = os.getenv('METALS_API_KEY')
        self.exchange_rate_api_key = os.getenv('EXCHANGE_RATE_API_KEY')
        
        # Initialize OpenAI client
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not found - AI recommendations will be limited")
        
        # Database configuration
        self.db_path = "/home/ubuntu/precious-metals-api/ai_agent.db"
        
        # Agent configuration
        self.is_monitoring = False
        self.monitoring_thread = None
        self.check_interval = 300  # 5 minutes
        self.confidence_threshold = 70.0
        
        # AI model configurations
        self.analysis_model = "gpt-4"
        self.perplexity_model = "sonar-pro"
        
        # Cache for market analysis
        self.market_analysis_cache = {}
        self.market_analysis_ttl = 1800  # 30 minutes
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database with enhanced tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Investment targets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS investment_targets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        metal_type TEXT NOT NULL,
                        target_type TEXT NOT NULL,
                        target_price REAL NOT NULL,
                        target_quantity REAL,
                        target_percentage REAL,
                        current_price REAL,
                        distance_to_target REAL,
                        priority TEXT DEFAULT 'MEDIUM',
                        status TEXT DEFAULT 'ACTIVE',
                        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        target_date TIMESTAMP,
                        last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        times_triggered INTEGER DEFAULT 0,
                        notes TEXT,
                        conditions TEXT
                    )
                """)
                
                # AI recommendations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ai_recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        target_id INTEGER,
                        recommendation_type TEXT NOT NULL,
                        metal_type TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        reasoning TEXT,
                        suggested_quantity REAL,
                        suggested_price REAL,
                        urgency TEXT DEFAULT 'MEDIUM',
                        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_date TIMESTAMP,
                        status TEXT DEFAULT 'active',
                        user_feedback TEXT,
                        execution_date TIMESTAMP,
                        market_data TEXT,
                        FOREIGN KEY (target_id) REFERENCES investment_targets (id)
                    )
                """)
                
                # Agent performance tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agent_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        recommendation_id INTEGER,
                        accuracy_score REAL,
                        profit_loss REAL,
                        execution_date TIMESTAMP,
                        evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        market_conditions TEXT,
                        FOREIGN KEY (recommendation_id) REFERENCES ai_recommendations (id)
                    )
                """)
                
                # User AI preferences
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_ai_preferences (
                        user_id INTEGER PRIMARY KEY,
                        risk_tolerance TEXT DEFAULT 'MEDIUM',
                        notification_frequency TEXT DEFAULT 'IMPORTANT',
                        auto_execute_enabled BOOLEAN DEFAULT FALSE,
                        max_auto_investment REAL DEFAULT 0,
                        preferred_metals TEXT DEFAULT '["GOLD", "SILVER", "PLATINUM"]',
                        target_allocation TEXT,
                        min_confidence_threshold REAL DEFAULT 75.0,
                        agent_enabled BOOLEAN DEFAULT TRUE,
                        notification_channels TEXT DEFAULT '["email"]',
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # User portfolios table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_portfolios (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        metal_type TEXT NOT NULL,
                        quantity REAL NOT NULL DEFAULT 0,
                        average_cost REAL,
                        current_value REAL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, metal_type)
                    )
                """)
                
                # Portfolio transactions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        metal_type TEXT NOT NULL,
                        transaction_type TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        total_amount REAL NOT NULL,
                        transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        recommendation_id INTEGER,
                        notes TEXT,
                        FOREIGN KEY (recommendation_id) REFERENCES ai_recommendations (id)
                    )
                """)
                
                conn.commit()
                logger.info("Enhanced AI Agent database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def start_monitoring(self):
        """Start the AI agent monitoring service"""
        if self.is_monitoring:
            logger.warning("Agent monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Enhanced AI Agent monitoring service started")
    
    def stop_monitoring(self):
        """Stop the AI agent monitoring service"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("Enhanced AI Agent monitoring service stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for AI agent"""
        while self.is_monitoring:
            try:
                # Check all active targets
                self._check_investment_targets()
                
                # Generate market opportunities
                self._generate_market_opportunities()
                
                # Evaluate portfolio health
                self._evaluate_portfolio_health()
                
                # Clean up expired recommendations
                self._cleanup_expired_recommendations()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def create_investment_target(self, user_id: int, metal_type: str, target_type: str,
                               target_price: float, target_quantity: Optional[float] = None,
                               target_percentage: Optional[float] = None, priority: str = "MEDIUM",
                               target_date: Optional[datetime] = None, notes: Optional[str] = None,
                               conditions: Optional[Dict[str, Any]] = None) -> int:
        """Create a new investment target for monitoring"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current price for the metal
                current_price = self._get_current_price(metal_type)
                distance_to_target = abs(target_price - current_price) if current_price else None
                
                cursor.execute("""
                    INSERT INTO investment_targets 
                    (user_id, metal_type, target_type, target_price, target_quantity, 
                     target_percentage, current_price, distance_to_target, priority, 
                     target_date, notes, conditions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, metal_type, target_type, target_price, target_quantity,
                      target_percentage, current_price, distance_to_target, priority,
                      target_date, notes, json.dumps(conditions) if conditions else None))
                
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, user_id, metal_type, target_type, target_price, target_quantity,
                           target_percentage, current_price, distance_to_target, priority, status,
                           created_date, target_date, notes, conditions, times_triggered
                    FROM investment_targets 
                    WHERE user_id = ? AND status = ?
                    ORDER BY created_date DESC
                """, (user_id, status))
                
                columns = [desc[0] for desc in cursor.description]
                targets = []
                
                for row in cursor.fetchall():
                    target = dict(zip(columns, row))
                    if target['conditions']:
                        target['conditions'] = json.loads(target['conditions'])
                    targets.append(target)
                
                return targets
                
        except Exception as e:
            logger.error(f"Error getting user targets: {str(e)}")
            return []
    
    def _check_investment_targets(self):
        """Check all active investment targets and generate alerts/recommendations"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all active targets
                cursor.execute("""
                    SELECT id, user_id, metal_type, target_type, target_price, target_quantity,
                           priority, notes, conditions
                    FROM investment_targets 
                    WHERE status = 'ACTIVE'
                """)
                
                for row in cursor.fetchall():
                    target_id, user_id, metal_type, target_type, target_price, target_quantity, priority, notes, conditions = row
                    
                    # Get current price
                    current_price = self._get_current_price(metal_type)
                    if not current_price:
                        continue
                    
                    # Update current price and distance
                    distance_to_target = abs(target_price - current_price)
                    cursor.execute("""
                        UPDATE investment_targets 
                        SET current_price = ?, distance_to_target = ?, last_checked = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (current_price, distance_to_target, target_id))
                    
                    # Check if target is triggered
                    triggered = self._is_target_triggered(target_type, target_price, current_price)
                    
                    if triggered:
                        # Generate recommendation
                        recommendation = self._generate_target_recommendation(
                            user_id, target_id, metal_type, target_type, target_price, 
                            current_price, target_quantity, priority
                        )
                        
                        if recommendation:
                            # Save recommendation
                            self._save_recommendation(recommendation)
                            
                            # Update target trigger count
                            cursor.execute("""
                                UPDATE investment_targets 
                                SET times_triggered = times_triggered + 1
                                WHERE id = ?
                            """, (target_id,))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error checking investment targets: {str(e)}")
    
    def _is_target_triggered(self, target_type: str, target_price: float, current_price: float) -> bool:
        """Check if a target condition is met"""
        if target_type == "buy_below":
            return current_price <= target_price
        elif target_type == "sell_above":
            return current_price >= target_price
        elif target_type == "price_alert":
            # For price alerts, trigger when price crosses the target (within 1% tolerance)
            tolerance = target_price * 0.01
            return abs(current_price - target_price) <= tolerance
        return False
    
    def _generate_target_recommendation(self, user_id: int, target_id: int, metal_type: str,
                                      target_type: str, target_price: float, current_price: float,
                                      target_quantity: Optional[float], priority: str) -> Optional[Dict]:
        """Generate AI recommendation when target is triggered"""
        try:
            # Get market context
            market_context = self._get_market_context(metal_type)
            
            # Determine action based on target type
            if target_type == "buy_below":
                action = "BUY"
                recommendation_type = "buy_signal"
            elif target_type == "sell_above":
                action = "SELL"
                recommendation_type = "sell_signal"
            else:
                action = "ALERT"
                recommendation_type = "price_target_monitor"
            
            # Generate AI reasoning
            reasoning = self._generate_ai_reasoning(
                metal_type, action, target_price, current_price, market_context
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                metal_type, action, target_price, current_price, market_context
            )
            
            # Only proceed if confidence is above threshold
            if confidence_score < self.confidence_threshold:
                return None
            
            # Create recommendation
            recommendation = {
                'user_id': user_id,
                'target_id': target_id,
                'recommendation_type': recommendation_type,
                'metal_type': metal_type,
                'action': action,
                'confidence_score': confidence_score,
                'reasoning': reasoning,
                'suggested_quantity': target_quantity,
                'suggested_price': current_price,
                'urgency': priority,
                'expires_date': datetime.now() + timedelta(hours=24),
                'market_data': json.dumps(market_context)
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating target recommendation: {str(e)}")
            return None
    
    def _get_market_context(self, metal_type: str) -> Dict[str, Any]:
        """Get current market context using AI analysis"""
        try:
            # Check cache first
            cache_key = f"{metal_type}_{int(time.time() // self.market_analysis_ttl)}"
            if cache_key in self.market_analysis_cache:
                return self.market_analysis_cache[cache_key]
            
            # Get market analysis using Perplexity AI
            if self.perplexity_api_key:
                context = self._get_perplexity_analysis(metal_type)
            else:
                context = self._get_openai_analysis(metal_type)
            
            # Cache the result
            self.market_analysis_cache[cache_key] = context
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting market context: {str(e)}")
            return {"analysis": "Market context unavailable", "timestamp": datetime.now().isoformat()}
    
    def _get_perplexity_analysis(self, metal_type: str) -> Dict[str, Any]:
        """Get market analysis using Perplexity AI"""
        try:
            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Analyze the current market conditions for {metal_type} precious metal. 
            Provide insights on:
            1. Recent price trends and movements
            2. Key market factors affecting price
            3. Market sentiment and outlook
            4. Any significant news or events
            
            Keep the analysis concise and focused on actionable insights."""
            
            data = {
                "model": self.perplexity_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            analysis = result['choices'][0]['message']['content']
            
            return {
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
                "source": "perplexity_ai"
            }
            
        except Exception as e:
            logger.error(f"Error getting Perplexity analysis: {str(e)}")
            return self._get_openai_analysis(metal_type)
    
    def _get_openai_analysis(self, metal_type: str) -> Dict[str, Any]:
        """Get market analysis using OpenAI"""
        try:
            if not self.openai_client:
                return {"analysis": "AI analysis unavailable", "timestamp": datetime.now().isoformat()}
            
            prompt = f"""As a precious metals market analyst, provide a brief analysis of current {metal_type} market conditions.
            Include:
            1. Recent price trends
            2. Key market factors
            3. Market sentiment
            4. Investment outlook
            
            Keep it concise and actionable."""
            
            response = self.openai_client.chat.completions.create(
                model=self.analysis_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
                "source": "openai"
            }
            
        except Exception as e:
            logger.error(f"Error getting OpenAI analysis: {str(e)}")
            return {"analysis": "AI analysis unavailable", "timestamp": datetime.now().isoformat()}
    
    def _generate_ai_reasoning(self, metal_type: str, action: str, target_price: float,
                             current_price: float, market_context: Dict[str, Any]) -> str:
        """Generate AI reasoning for recommendation"""
        try:
            price_change_pct = ((current_price - target_price) / target_price) * 100
            
            reasoning = f"""AI Analysis for {metal_type} {action} Signal:

Target Price: ${target_price:.2f}
Current Price: ${current_price:.2f}
Price Change: {price_change_pct:+.2f}%

Market Context: {market_context.get('analysis', 'N/A')[:300]}...

Recommendation: {action} signal triggered based on price target achievement and current market conditions."""
            
            return reasoning.strip()
            
        except Exception as e:
            logger.error(f"Error generating AI reasoning: {str(e)}")
            return f"AI recommendation for {metal_type} {action} at ${current_price:.2f}"
    
    def _calculate_confidence_score(self, metal_type: str, action: str, target_price: float,
                                  current_price: float, market_context: Dict[str, Any]) -> float:
        """Calculate confidence score for recommendation"""
        try:
            base_confidence = 75.0
            
            # Adjust based on price deviation from target
            price_deviation = abs(current_price - target_price) / target_price
            if price_deviation < 0.02:  # Within 2%
                base_confidence += 10
            elif price_deviation > 0.1:  # More than 10%
                base_confidence -= 15
            
            # Adjust based on market context availability
            if market_context.get('analysis') and len(market_context['analysis']) > 100:
                base_confidence += 5
            
            # Ensure confidence is within bounds
            return max(0, min(100, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 70.0
    
    def _save_recommendation(self, recommendation: Dict) -> int:
        """Save recommendation to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO ai_recommendations 
                    (user_id, target_id, recommendation_type, metal_type, action, 
                     confidence_score, reasoning, suggested_quantity, suggested_price, 
                     urgency, expires_date, market_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    recommendation['user_id'], recommendation['target_id'],
                    recommendation['recommendation_type'], recommendation['metal_type'],
                    recommendation['action'], recommendation['confidence_score'],
                    recommendation['reasoning'], recommendation['suggested_quantity'],
                    recommendation['suggested_price'], recommendation['urgency'],
                    recommendation['expires_date'], recommendation['market_data']
                ))
                
                recommendation_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Saved recommendation {recommendation_id}")
                return recommendation_id
                
        except Exception as e:
            logger.error(f"Error saving recommendation: {str(e)}")
            raise
    
    def _get_current_price(self, metal_type: str) -> Optional[float]:
        """Get current price for a metal"""
        try:
            if not self.metals_api_key:
                # Return mock prices for testing
                mock_prices = {
                    'GOLD': 3300.0,
                    'SILVER': 36.5,
                    'PLATINUM': 1375.0,
                    'PALLADIUM': 1112.0
                }
                return mock_prices.get(metal_type.upper())
            
            # Use MetalpriceAPI
            url = f"https://api.metalpriceapi.com/v1/latest?api_key={self.metals_api_key}&base=USD&currencies={metal_type}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success') and data.get('rates'):
                # MetalpriceAPI returns rates as 1/price, so we need to invert
                rate = data['rates'].get(metal_type)
                if rate:
                    return 1 / rate
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {metal_type}: {str(e)}")
            return None
    
    def _generate_market_opportunities(self):
        """Generate proactive market opportunity recommendations"""
        try:
            # Get active users with AI preferences
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, preferred_metals, min_confidence_threshold, agent_enabled
                    FROM user_ai_preferences 
                    WHERE agent_enabled = TRUE
                """)
                
                for row in cursor.fetchall():
                    user_id, preferred_metals_json, min_confidence, agent_enabled = row
                    
                    if not agent_enabled:
                        continue
                    
                    preferred_metals = json.loads(preferred_metals_json) if preferred_metals_json else ['GOLD', 'SILVER']
                    
                    for metal_type in preferred_metals:
                        opportunity = self._analyze_market_opportunity(user_id, metal_type, min_confidence)
                        if opportunity:
                            self._save_recommendation(opportunity)
                            
        except Exception as e:
            logger.error(f"Error generating market opportunities: {str(e)}")
    
    def _analyze_market_opportunity(self, user_id: int, metal_type: str, min_confidence: float) -> Optional[Dict]:
        """Analyze market for opportunities"""
        try:
            current_price = self._get_current_price(metal_type)
            if not current_price:
                return None
            
            # Simple opportunity detection based on price volatility
            # In a real implementation, this would use more sophisticated algorithms
            market_context = self._get_market_context(metal_type)
            
            # Generate opportunity based on market analysis
            if "bullish" in market_context.get('analysis', '').lower() or "buy" in market_context.get('analysis', '').lower():
                action = "BUY"
                confidence_score = 78.0
            elif "bearish" in market_context.get('analysis', '').lower() or "sell" in market_context.get('analysis', '').lower():
                action = "SELL"
                confidence_score = 76.0
            else:
                return None  # No clear opportunity
            
            if confidence_score < min_confidence:
                return None
            
            reasoning = f"""Market Opportunity Detected for {metal_type}:

Current Price: ${current_price:.2f}

Market Analysis: {market_context.get('analysis', 'N/A')[:300]}...

Recommendation: Consider {action} position based on current market conditions and analysis."""
            
            return {
                'user_id': user_id,
                'target_id': None,
                'recommendation_type': 'market_opportunity',
                'metal_type': metal_type,
                'action': action,
                'confidence_score': confidence_score,
                'reasoning': reasoning,
                'suggested_quantity': None,
                'suggested_price': current_price,
                'urgency': 'MEDIUM',
                'expires_date': datetime.now() + timedelta(hours=12),
                'market_data': json.dumps(market_context)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market opportunity: {str(e)}")
            return None
    
    def _evaluate_portfolio_health(self):
        """Evaluate portfolio health and suggest rebalancing"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get users with target allocations
                cursor.execute("""
                    SELECT user_id, target_allocation 
                    FROM user_ai_preferences 
                    WHERE target_allocation IS NOT NULL AND agent_enabled = TRUE
                """)
                
                for row in cursor.fetchall():
                    user_id, target_allocation_json = row
                    
                    if not target_allocation_json:
                        continue
                    
                    target_allocation = json.loads(target_allocation_json)
                    
                    # Get current portfolio
                    cursor.execute("""
                        SELECT metal_type, quantity, current_value 
                        FROM user_portfolios 
                        WHERE user_id = ?
                    """, (user_id,))
                    
                    portfolio = cursor.fetchall()
                    if not portfolio:
                        continue
                    
                    # Calculate current allocation
                    total_value = sum(row[2] for row in portfolio if row[2])
                    if total_value == 0:
                        continue
                    
                    current_allocation = {}
                    for metal_type, quantity, current_value in portfolio:
                        if current_value:
                            current_allocation[metal_type] = (current_value / total_value) * 100
                    
                    # Check for significant deviations
                    rebalance_needed = False
                    for metal_type, target_pct in target_allocation.items():
                        current_pct = current_allocation.get(metal_type, 0)
                        deviation = abs(current_pct - target_pct)
                        
                        if deviation > 10:  # More than 10% deviation
                            rebalance_needed = True
                            break
                    
                    if rebalance_needed:
                        # Generate rebalancing recommendation
                        rebalance_rec = self._generate_rebalance_recommendation(
                            user_id, target_allocation, current_allocation, total_value
                        )
                        
                        if rebalance_rec:
                            self._save_recommendation(rebalance_rec)
                            
        except Exception as e:
            logger.error(f"Error evaluating portfolio health: {str(e)}")
    
    def _generate_rebalance_recommendation(self, user_id: int, target_allocation: Dict,
                                         current_allocation: Dict, total_value: float) -> Optional[Dict]:
        """Generate portfolio rebalancing recommendation"""
        try:
            reasoning = f"Portfolio Rebalancing Recommendation:\n\nTotal Portfolio Value: ${total_value:,.2f}\n\n"
            
            for metal_type, target_pct in target_allocation.items():
                current_pct = current_allocation.get(metal_type, 0)
                deviation = abs(current_pct - target_pct)
                
                if deviation > 10:
                    action = "REDUCE" if current_pct > target_pct else "INCREASE"
                    reasoning += f"{metal_type}: {action} (Current: {current_pct:.1f}%, Target: {target_pct:.1f}%)\n"
            
            reasoning += "\nRecommendation: Rebalance portfolio to maintain target allocation."
            
            return {
                'user_id': user_id,
                'target_id': None,
                'recommendation_type': 'portfolio_rebalance',
                'metal_type': 'PORTFOLIO',
                'action': 'REBALANCE',
                'confidence_score': 80.0,
                'reasoning': reasoning,
                'suggested_quantity': None,
                'suggested_price': None,
                'urgency': 'MEDIUM',
                'expires_date': datetime.now() + timedelta(days=7),
                'market_data': json.dumps({'target_allocation': target_allocation, 'current_allocation': current_allocation})
            }
            
        except Exception as e:
            logger.error(f"Error generating rebalance recommendation: {str(e)}")
            return None
    
    def _cleanup_expired_recommendations(self):
        """Clean up expired recommendations"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE ai_recommendations 
                    SET status = 'expired'
                    WHERE expires_date < datetime('now') 
                    AND status = 'active'
                """)
                
                expired_count = cursor.rowcount
                conn.commit()
                
                if expired_count > 0:
                    logger.info(f"Expired {expired_count} recommendations")
                    
        except Exception as e:
            logger.error(f"Error cleaning up expired recommendations: {str(e)}")
    
    def get_user_recommendations(self, user_id: int, status: str = "active", limit: int = 10) -> List[Dict]:
        """Get user recommendations"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, user_id, target_id, recommendation_type, metal_type, action,
                           confidence_score, reasoning, suggested_quantity, suggested_price,
                           urgency, created_date, expires_date, status, user_feedback, execution_date
                    FROM ai_recommendations 
                    WHERE user_id = ? AND status = ?
                    ORDER BY created_date DESC
                    LIMIT ?
                """, (user_id, status, limit))
                
                columns = [desc[0] for desc in cursor.description]
                recommendations = []
                
                for row in cursor.fetchall():
                    recommendation = dict(zip(columns, row))
                    recommendations.append(recommendation)
                
                return recommendations
                
        except Exception as e:
            logger.error(f"Error getting user recommendations: {str(e)}")
            return []
    
    def update_recommendation_feedback(self, recommendation_id: int, feedback: str,
                                     execution_date: Optional[datetime] = None):
        """Update user feedback on recommendation"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                status = 'executed' if feedback == 'accepted' else 'active'
                
                cursor.execute("""
                    UPDATE ai_recommendations 
                    SET user_feedback = ?, execution_date = ?, status = ?
                    WHERE id = ?
                """, (feedback, execution_date, status, recommendation_id))
                
                conn.commit()
                logger.info(f"Updated recommendation {recommendation_id} feedback: {feedback}")
                
        except Exception as e:
            logger.error(f"Error updating recommendation feedback: {str(e)}")
    
    def set_user_preferences(self, user_id: int, preferences: Dict[str, Any]):
        """Set user AI preferences"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO user_ai_preferences 
                    (user_id, risk_tolerance, notification_frequency, auto_execute_enabled,
                     max_auto_investment, preferred_metals, target_allocation, 
                     min_confidence_threshold, agent_enabled, notification_channels)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    preferences.get('risk_tolerance', 'MEDIUM'),
                    preferences.get('notification_frequency', 'IMPORTANT'),
                    preferences.get('auto_execute_enabled', False),
                    preferences.get('max_auto_investment', 0.0),
                    json.dumps(preferences.get('preferred_metals', ['GOLD', 'SILVER'])),
                    json.dumps(preferences.get('target_allocation')) if preferences.get('target_allocation') else None,
                    preferences.get('min_confidence_threshold', 75.0),
                    preferences.get('agent_enabled', True),
                    json.dumps(preferences.get('notification_channels', ['email']))
                ))
                
                conn.commit()
                logger.info(f"Updated AI preferences for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error setting user preferences: {str(e)}")
            raise
    
    def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get user AI preferences"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT risk_tolerance, notification_frequency, auto_execute_enabled,
                           max_auto_investment, preferred_metals, target_allocation,
                           min_confidence_threshold, agent_enabled, notification_channels
                    FROM user_ai_preferences 
                    WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'risk_tolerance': row[0],
                        'notification_frequency': row[1],
                        'auto_execute_enabled': row[2],
                        'max_auto_investment': row[3],
                        'preferred_metals': json.loads(row[4]) if row[4] else ['GOLD', 'SILVER'],
                        'target_allocation': json.loads(row[5]) if row[5] else {},
                        'min_confidence_threshold': row[6],
                        'agent_enabled': row[7],
                        'notification_channels': json.loads(row[8]) if row[8] else ['email']
                    }
                else:
                    # Return default preferences
                    return {
                        'risk_tolerance': 'MEDIUM',
                        'notification_frequency': 'IMPORTANT',
                        'auto_execute_enabled': False,
                        'max_auto_investment': 0.0,
                        'preferred_metals': ['GOLD', 'SILVER', 'PLATINUM'],
                        'target_allocation': {},
                        'min_confidence_threshold': 75.0,
                        'agent_enabled': True,
                        'notification_channels': ['email']
                    }
                    
        except Exception as e:
            logger.error(f"Error getting user preferences: {str(e)}")
            return {}

# Global instance
enhanced_ai_agent = EnhancedAIAgentService()

