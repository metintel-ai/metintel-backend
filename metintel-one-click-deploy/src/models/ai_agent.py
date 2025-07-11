"""
AI Agent Models for PreciousAI Application
Integrating advanced features from PreciousPredictor codebase
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

class AgentTaskType(Enum):
    """Types of tasks the AI agent can perform"""
    PRICE_TARGET_MONITOR = "price_target_monitor"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    MARKET_OPPORTUNITY = "market_opportunity"
    RISK_ASSESSMENT = "risk_assessment"
    PROFIT_TAKING = "profit_taking"
    BUY_SIGNAL = "buy_signal"
    SELL_SIGNAL = "sell_signal"
    USER_ASSISTANCE = "user_assistance"
    MARKET_ANALYSIS = "market_analysis"

class AgentPriority(Enum):
    """Priority levels for agent tasks"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AgentStatus(Enum):
    """Status of agent tasks"""
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    COMPLETED = "COMPLETED"
    EXPIRED = "EXPIRED"
    PAUSED = "PAUSED"

class TargetType(Enum):
    """Types of investment targets"""
    BUY_BELOW = "buy_below"
    SELL_ABOVE = "sell_above"
    HOLD_UNTIL = "hold_until"
    REBALANCE_AT = "rebalance_at"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@dataclass
class InvestmentTarget:
    """Structure for user investment targets"""
    id: Optional[int] = None
    user_id: int = None
    metal_type: str = None  # "GOLD", "SILVER", "PLATINUM", "PALLADIUM"
    target_type: str = None  # TargetType enum value
    target_price: float = None
    target_quantity: Optional[float] = None
    target_percentage: Optional[float] = None  # For portfolio allocation targets
    current_price: Optional[float] = None
    distance_to_target: Optional[float] = None
    priority: str = "MEDIUM"
    status: str = "ACTIVE"
    created_date: Optional[datetime] = None
    target_date: Optional[datetime] = None
    last_checked: Optional[datetime] = None
    times_triggered: int = 0
    notes: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None  # Additional conditions

@dataclass
class AIRecommendation:
    """AI-generated recommendations"""
    id: Optional[int] = None
    user_id: int = None
    target_id: Optional[int] = None
    recommendation_type: str = None  # AgentTaskType enum value
    metal_type: str = None
    action: str = None  # "BUY", "SELL", "HOLD", "REBALANCE"
    confidence_score: float = None  # 0-100
    reasoning: str = None
    suggested_quantity: Optional[float] = None
    suggested_price: Optional[float] = None
    urgency: str = "MEDIUM"
    created_date: Optional[datetime] = None
    expires_date: Optional[datetime] = None
    status: str = "active"  # "active", "executed", "expired", "dismissed"
    user_feedback: Optional[str] = None  # "accepted", "rejected", "modified"
    execution_date: Optional[datetime] = None
    market_data: Optional[Dict[str, Any]] = None

@dataclass
class UserAIPreferences:
    """User preferences for AI agent behavior"""
    user_id: int = None
    risk_tolerance: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    notification_frequency: str = "IMPORTANT"  # ALL, IMPORTANT, CRITICAL
    auto_execute_enabled: bool = False
    max_auto_investment: float = 0.0
    preferred_metals: List[str] = None  # ["GOLD", "SILVER", "PLATINUM"]
    target_allocation: Optional[Dict[str, float]] = None  # {"GOLD": 60, "SILVER": 30, "PLATINUM": 10}
    min_confidence_threshold: float = 75.0
    agent_enabled: bool = True
    notification_channels: List[str] = None  # ["email", "sms", "push"]
    last_updated: Optional[datetime] = None

@dataclass
class AgentPerformance:
    """Track AI agent performance metrics"""
    id: Optional[int] = None
    user_id: int = None
    recommendation_id: int = None
    accuracy_score: Optional[float] = None
    profit_loss: Optional[float] = None
    execution_date: Optional[datetime] = None
    evaluation_date: Optional[datetime] = None
    market_conditions: Optional[Dict[str, Any]] = None

class AIAgentDatabase:
    """Database operations for AI Agent functionality"""
    
    @staticmethod
    def get_create_tables_sql():
        """Return SQL statements to create all AI agent tables"""
        return """
        -- Investment targets table
        CREATE TABLE IF NOT EXISTS investment_targets (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            metal_type VARCHAR(20) NOT NULL,
            target_type VARCHAR(50) NOT NULL,
            target_price DECIMAL(10,2) NOT NULL,
            target_quantity DECIMAL(15,6),
            target_percentage DECIMAL(5,2),
            current_price DECIMAL(10,2),
            distance_to_target DECIMAL(10,2),
            priority VARCHAR(20) DEFAULT 'MEDIUM',
            status VARCHAR(20) DEFAULT 'ACTIVE',
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            target_date TIMESTAMP,
            last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            times_triggered INTEGER DEFAULT 0,
            notes TEXT,
            conditions JSONB
        );

        -- AI recommendations table
        CREATE TABLE IF NOT EXISTS ai_recommendations (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            target_id INTEGER REFERENCES investment_targets(id) ON DELETE CASCADE,
            recommendation_type VARCHAR(50) NOT NULL,
            metal_type VARCHAR(20) NOT NULL,
            action VARCHAR(20) NOT NULL,
            confidence_score DECIMAL(5,2) NOT NULL,
            reasoning TEXT,
            suggested_quantity DECIMAL(15,6),
            suggested_price DECIMAL(10,2),
            urgency VARCHAR(20) DEFAULT 'MEDIUM',
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_date TIMESTAMP,
            status VARCHAR(20) DEFAULT 'active',
            user_feedback VARCHAR(20), -- 'accepted', 'rejected', 'modified'
            execution_date TIMESTAMP,
            market_data JSONB
        );

        -- Agent performance tracking
        CREATE TABLE IF NOT EXISTS agent_performance (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            recommendation_id INTEGER REFERENCES ai_recommendations(id),
            accuracy_score DECIMAL(5,2),
            profit_loss DECIMAL(15,2),
            execution_date TIMESTAMP,
            evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            market_conditions JSONB
        );

        -- User AI preferences
        CREATE TABLE IF NOT EXISTS user_ai_preferences (
            user_id INTEGER PRIMARY KEY,
            risk_tolerance VARCHAR(20) DEFAULT 'MEDIUM', -- LOW, MEDIUM, HIGH
            notification_frequency VARCHAR(20) DEFAULT 'IMPORTANT', -- ALL, IMPORTANT, CRITICAL
            auto_execute_enabled BOOLEAN DEFAULT FALSE,
            max_auto_investment DECIMAL(15,2) DEFAULT 0,
            preferred_metals JSONB DEFAULT '["GOLD", "SILVER", "PLATINUM"]',
            target_allocation JSONB, -- {"GOLD": 60, "SILVER": 30, "PLATINUM": 10}
            min_confidence_threshold DECIMAL(5,2) DEFAULT 75.0,
            agent_enabled BOOLEAN DEFAULT TRUE,
            notification_channels JSONB DEFAULT '["email"]',
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- User portfolios table
        CREATE TABLE IF NOT EXISTS user_portfolios (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            metal_type VARCHAR(20) NOT NULL,
            quantity DECIMAL(15,6) NOT NULL DEFAULT 0,
            average_cost DECIMAL(10,2),
            current_value DECIMAL(15,2),
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, metal_type)
        );

        -- Portfolio transactions table
        CREATE TABLE IF NOT EXISTS portfolio_transactions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            metal_type VARCHAR(20) NOT NULL,
            transaction_type VARCHAR(10) NOT NULL, -- 'BUY', 'SELL'
            quantity DECIMAL(15,6) NOT NULL,
            price DECIMAL(10,2) NOT NULL,
            total_amount DECIMAL(15,2) NOT NULL,
            transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            recommendation_id INTEGER REFERENCES ai_recommendations(id),
            notes TEXT
        );

        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_investment_targets_user_id ON investment_targets(user_id);
        CREATE INDEX IF NOT EXISTS idx_investment_targets_status ON investment_targets(status);
        CREATE INDEX IF NOT EXISTS idx_ai_recommendations_user_id ON ai_recommendations(user_id);
        CREATE INDEX IF NOT EXISTS idx_ai_recommendations_status ON ai_recommendations(status);
        CREATE INDEX IF NOT EXISTS idx_user_portfolios_user_id ON user_portfolios(user_id);
        CREATE INDEX IF NOT EXISTS idx_portfolio_transactions_user_id ON portfolio_transactions(user_id);
        """

    @staticmethod
    def get_sample_data_sql():
        """Return SQL statements to insert sample data for testing"""
        return """
        -- Sample user AI preferences
        INSERT INTO user_ai_preferences (user_id, risk_tolerance, preferred_metals, target_allocation) 
        VALUES (1, 'MEDIUM', '["GOLD", "SILVER", "PLATINUM"]', '{"GOLD": 60, "SILVER": 30, "PLATINUM": 10}')
        ON CONFLICT (user_id) DO NOTHING;

        -- Sample investment targets
        INSERT INTO investment_targets (user_id, metal_type, target_type, target_price, priority, notes)
        VALUES 
        (1, 'GOLD', 'buy_below', 3200.00, 'HIGH', 'Buy gold when price drops below $3200'),
        (1, 'SILVER', 'sell_above', 40.00, 'MEDIUM', 'Sell silver when price goes above $40'),
        (1, 'PLATINUM', 'buy_below', 1300.00, 'LOW', 'Accumulate platinum below $1300')
        ON CONFLICT DO NOTHING;
        """

