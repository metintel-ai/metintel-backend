from datetime import datetime
from src.models.user import db

class Metal(db.Model):
    __tablename__ = 'metals'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)  # XAU, XAG, XPT, XPD
    name = db.Column(db.String(50), nullable=False)  # Gold, Silver, Platinum, Palladium
    unit = db.Column(db.String(20), default='troy_ounce')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to price data
    price_data = db.relationship('PriceData', backref='metal', lazy=True, cascade='all, delete-orphan')
    predictions = db.relationship('Prediction', backref='metal', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Metal {self.symbol}: {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'name': self.name,
            'unit': self.unit,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class PriceData(db.Model):
    __tablename__ = 'price_data'
    
    id = db.Column(db.BigInteger, primary_key=True)
    metal_id = db.Column(db.Integer, db.ForeignKey('metals.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    price = db.Column(db.Numeric(12, 4), nullable=False)
    currency = db.Column(db.String(3), default='USD')
    source = db.Column(db.String(50), nullable=False)  # API source identifier
    bid_price = db.Column(db.Numeric(12, 4))
    ask_price = db.Column(db.Numeric(12, 4))
    volume = db.Column(db.BigInteger)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Indexes for efficient querying
    __table_args__ = (
        db.Index('idx_price_data_metal_timestamp', 'metal_id', 'timestamp'),
        db.Index('idx_price_data_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f'<PriceData {self.metal.symbol if self.metal else "Unknown"}: ${self.price} at {self.timestamp}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'metal_id': self.metal_id,
            'metal_symbol': self.metal.symbol if self.metal else None,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'price': float(self.price) if self.price else None,
            'currency': self.currency,
            'source': self.source,
            'bid_price': float(self.bid_price) if self.bid_price else None,
            'ask_price': float(self.ask_price) if self.ask_price else None,
            'volume': self.volume,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.String(36), primary_key=True)  # UUID
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    metal_id = db.Column(db.Integer, db.ForeignKey('metals.id'), nullable=False)
    prediction_horizon = db.Column(db.String(50), nullable=False)  # e.g., "1 week", "3 months"
    target_date = db.Column(db.DateTime, nullable=False)
    predicted_price = db.Column(db.Numeric(12, 4), nullable=False)
    confidence_interval_lower = db.Column(db.Numeric(12, 4))
    confidence_interval_upper = db.Column(db.Numeric(12, 4))
    confidence_score = db.Column(db.Numeric(5, 4))  # 0.0 to 1.0
    model_version = db.Column(db.String(50))
    input_parameters = db.Column(db.JSON)
    market_context = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)
    actual_price = db.Column(db.Numeric(12, 4))  # Filled when target_date is reached
    accuracy_score = db.Column(db.Numeric(5, 4))  # Calculated post-prediction
    
    # Relationships
    factors = db.relationship('PredictionFactor', backref='prediction', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.metal.symbol if self.metal else "Unknown"} -> ${self.predicted_price}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'metal_id': self.metal_id,
            'metal_symbol': self.metal.symbol if self.metal else None,
            'metal_name': self.metal.name if self.metal else None,
            'prediction_horizon': self.prediction_horizon,
            'target_date': self.target_date.isoformat() if self.target_date else None,
            'predicted_price': float(self.predicted_price) if self.predicted_price else None,
            'confidence_interval_lower': float(self.confidence_interval_lower) if self.confidence_interval_lower else None,
            'confidence_interval_upper': float(self.confidence_interval_upper) if self.confidence_interval_upper else None,
            'confidence_score': float(self.confidence_score) if self.confidence_score else None,
            'model_version': self.model_version,
            'input_parameters': self.input_parameters,
            'market_context': self.market_context,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'actual_price': float(self.actual_price) if self.actual_price else None,
            'accuracy_score': float(self.accuracy_score) if self.accuracy_score else None
        }

class PredictionFactor(db.Model):
    __tablename__ = 'prediction_factors'
    
    id = db.Column(db.String(36), primary_key=True)  # UUID
    prediction_id = db.Column(db.String(36), db.ForeignKey('predictions.id'), nullable=False)
    factor_type = db.Column(db.Enum('technical', 'fundamental', 'sentiment', 'external', name='factor_types'), nullable=False)
    factor_name = db.Column(db.String(100), nullable=False)
    impact_weight = db.Column(db.Numeric(5, 4))  # -1.0 to 1.0
    description = db.Column(db.Text)
    source_data = db.Column(db.JSON)
    
    def __repr__(self):
        return f'<PredictionFactor {self.factor_name}: {self.impact_weight}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'prediction_id': self.prediction_id,
            'factor_type': self.factor_type,
            'factor_name': self.factor_name,
            'impact_weight': float(self.impact_weight) if self.impact_weight else None,
            'description': self.description,
            'source_data': self.source_data
        }

