from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    account_type = db.Column(db.Enum('individual', 'institutional', name='account_types'), default='individual')
    subscription_tier = db.Column(db.Enum('free', 'basic', 'professional', 'enterprise', name='subscription_tiers'), default='free')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    email_verified = db.Column(db.Boolean, default=False)
    two_factor_enabled = db.Column(db.Boolean, default=False)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    reports = db.relationship('Report', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.email}>'

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'account_type': self.account_type,
            'subscription_tier': self.subscription_tier,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'email_verified': self.email_verified,
            'two_factor_enabled': self.two_factor_enabled
        }

class Report(db.Model):
    __tablename__ = 'reports'
    
    id = db.Column(db.String(36), primary_key=True)  # UUID
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    report_type = db.Column(db.Enum('prediction', 'analysis', 'portfolio', 'market_summary', name='report_types'), nullable=False)
    parameters = db.Column(db.JSON, nullable=False)
    content = db.Column(db.JSON)
    file_path = db.Column(db.String(500))  # For generated PDF/Excel files
    status = db.Column(db.Enum('pending', 'generating', 'completed', 'failed', name='report_status'), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    expires_at = db.Column(db.DateTime)
    is_shared = db.Column(db.Boolean, default=False)
    share_token = db.Column(db.String(100), unique=True)
    
    def __repr__(self):
        return f'<Report {self.id}: {self.title}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'report_type': self.report_type,
            'parameters': self.parameters,
            'content': self.content,
            'file_path': self.file_path,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_shared': self.is_shared,
            'share_token': self.share_token
        }
