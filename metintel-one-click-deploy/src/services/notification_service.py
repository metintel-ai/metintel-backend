"""
Smart Notification Service for PreciousAI Application
Advanced notification system integrated from PreciousPredictor codebase
Supports multiple channels and intelligent notification management
"""

import os
import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NotificationData:
    """Structure for notification data"""
    user_id: int
    title: str
    message: str
    notification_type: str  # 'target_triggered', 'recommendation', 'market_alert', 'system'
    priority: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    channels: List[str]  # ['email', 'sms', 'push', 'in_app']
    data: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None

class SmartNotificationService:
    """
    Smart Notification Service providing:
    - Multi-channel notifications (email, SMS, push, in-app)
    - Priority-based notification management
    - User preference-based filtering
    - Notification history and tracking
    - Rate limiting and spam prevention
    """
    
    def __init__(self):
        """Initialize the notification service"""
        # Email configuration
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.from_email = os.getenv('FROM_EMAIL', 'noreply@preciousai.com')
        
        # SMS configuration (placeholder for future implementation)
        self.sms_api_key = os.getenv('SMS_API_KEY')
        self.sms_from_number = os.getenv('SMS_FROM_NUMBER')
        
        # Push notification configuration (placeholder for future implementation)
        self.push_api_key = os.getenv('PUSH_API_KEY')
        
        # Database configuration
        self.db_path = "/home/ubuntu/precious-metals-api/notifications.db"
        
        # Rate limiting configuration
        self.rate_limits = {
            'email': {'count': 10, 'period': 3600},  # 10 emails per hour
            'sms': {'count': 5, 'period': 3600},     # 5 SMS per hour
            'push': {'count': 20, 'period': 3600},   # 20 push notifications per hour
        }
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the notifications database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Notifications table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT NOT NULL,
                        notification_type TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        channels TEXT NOT NULL,
                        data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        sent_at TIMESTAMP,
                        status TEXT DEFAULT 'pending',
                        error_message TEXT
                    )
                """)
                
                # Notification delivery tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS notification_deliveries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        notification_id INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        status TEXT NOT NULL,
                        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        error_message TEXT,
                        FOREIGN KEY (notification_id) REFERENCES notifications (id)
                    )
                """)
                
                # Rate limiting tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS notification_rate_limits (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("Notification database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing notification database: {str(e)}")
            raise
    
    def send_notification(self, notification: NotificationData) -> bool:
        """Send a notification through specified channels"""
        try:
            # Save notification to database
            notification_id = self._save_notification(notification)
            
            # Check if notification should be sent based on user preferences
            if not self._should_send_notification(notification):
                logger.info(f"Notification {notification_id} filtered by user preferences")
                self._update_notification_status(notification_id, 'filtered')
                return True
            
            # Check rate limits
            allowed_channels = self._check_rate_limits(notification.user_id, notification.channels)
            if not allowed_channels:
                logger.warning(f"All channels rate limited for user {notification.user_id}")
                self._update_notification_status(notification_id, 'rate_limited')
                return False
            
            # Send through each allowed channel
            success = True
            for channel in allowed_channels:
                try:
                    if channel == 'email':
                        self._send_email(notification)
                    elif channel == 'sms':
                        self._send_sms(notification)
                    elif channel == 'push':
                        self._send_push(notification)
                    elif channel == 'in_app':
                        self._send_in_app(notification)
                    
                    # Record successful delivery
                    self._record_delivery(notification_id, channel, 'sent')
                    self._record_rate_limit(notification.user_id, channel)
                    
                except Exception as e:
                    logger.error(f"Error sending {channel} notification: {str(e)}")
                    self._record_delivery(notification_id, channel, 'failed', str(e))
                    success = False
            
            # Update notification status
            status = 'sent' if success else 'partial'
            self._update_notification_status(notification_id, status)
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            return False
    
    def _save_notification(self, notification: NotificationData) -> int:
        """Save notification to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO notifications 
                    (user_id, title, message, notification_type, priority, channels, data, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    notification.user_id,
                    notification.title,
                    notification.message,
                    notification.notification_type,
                    notification.priority,
                    json.dumps(notification.channels),
                    json.dumps(notification.data) if notification.data else None,
                    notification.expires_at
                ))
                
                notification_id = cursor.lastrowid
                conn.commit()
                
                return notification_id
                
        except Exception as e:
            logger.error(f"Error saving notification: {str(e)}")
            raise
    
    def _should_send_notification(self, notification: NotificationData) -> bool:
        """Check if notification should be sent based on user preferences"""
        try:
            # Get user preferences from AI agent service
            from src.services.enhanced_ai_agent_service import enhanced_ai_agent
            preferences = enhanced_ai_agent.get_user_preferences(notification.user_id)
            
            if not preferences.get('agent_enabled', True):
                return False
            
            # Check notification frequency preference
            frequency = preferences.get('notification_frequency', 'IMPORTANT')
            
            if frequency == 'CRITICAL' and notification.priority not in ['CRITICAL']:
                return False
            elif frequency == 'IMPORTANT' and notification.priority not in ['CRITICAL', 'HIGH']:
                return False
            # 'ALL' allows all notifications
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking notification preferences: {str(e)}")
            return True  # Default to sending if we can't check preferences
    
    def _check_rate_limits(self, user_id: int, channels: List[str]) -> List[str]:
        """Check rate limits for channels and return allowed channels"""
        try:
            allowed_channels = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for channel in channels:
                    if channel not in self.rate_limits:
                        allowed_channels.append(channel)
                        continue
                    
                    limit_config = self.rate_limits[channel]
                    cutoff_time = datetime.now() - timedelta(seconds=limit_config['period'])
                    
                    cursor.execute("""
                        SELECT COUNT(*) FROM notification_rate_limits 
                        WHERE user_id = ? AND channel = ? AND sent_at > ?
                    """, (user_id, channel, cutoff_time))
                    
                    count = cursor.fetchone()[0]
                    
                    if count < limit_config['count']:
                        allowed_channels.append(channel)
                    else:
                        logger.warning(f"Rate limit exceeded for user {user_id} channel {channel}")
            
            return allowed_channels
            
        except Exception as e:
            logger.error(f"Error checking rate limits: {str(e)}")
            return channels  # Default to allowing all channels if we can't check
    
    def _send_email(self, notification: NotificationData):
        """Send email notification"""
        if not self.smtp_username or not self.smtp_password:
            logger.warning("Email credentials not configured")
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = f"user{notification.user_id}@example.com"  # In real app, get from user profile
            msg['Subject'] = f"[PreciousAI] {notification.title}"
            
            # Create HTML body
            html_body = self._create_email_template(notification)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to user {notification.user_id}")
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            raise
    
    def _create_email_template(self, notification: NotificationData) -> str:
        """Create HTML email template"""
        priority_colors = {
            'CRITICAL': '#dc2626',
            'HIGH': '#ea580c',
            'MEDIUM': '#d97706',
            'LOW': '#65a30d'
        }
        
        priority_color = priority_colors.get(notification.priority, '#6b7280')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{notification.title}</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px 10px 0 0; text-align: center;">
                <h1 style="color: white; margin: 0; font-size: 24px;">PreciousAI</h1>
                <p style="color: #e2e8f0; margin: 5px 0 0 0;">AI-Powered Precious Metals Intelligence</p>
            </div>
            
            <div style="background: white; padding: 30px; border: 1px solid #e5e7eb; border-top: none;">
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <div style="width: 4px; height: 40px; background-color: {priority_color}; margin-right: 15px; border-radius: 2px;"></div>
                    <div>
                        <h2 style="margin: 0; color: #1f2937; font-size: 20px;">{notification.title}</h2>
                        <span style="background-color: {priority_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">
                            {notification.priority} PRIORITY
                        </span>
                    </div>
                </div>
                
                <div style="background-color: #f9fafb; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                    <p style="margin: 0; white-space: pre-line;">{notification.message}</p>
                </div>
                
                {self._create_notification_data_section(notification)}
                
                <div style="text-align: center; margin-top: 30px;">
                    <a href="https://preciousai.com" style="background-color: #3b82f6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold;">
                        View in PreciousAI
                    </a>
                </div>
            </div>
            
            <div style="background-color: #f3f4f6; padding: 20px; border-radius: 0 0 10px 10px; text-align: center; font-size: 12px; color: #6b7280;">
                <p style="margin: 0;">This notification was sent by PreciousAI AI Agent</p>
                <p style="margin: 5px 0 0 0;">Sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_notification_data_section(self, notification: NotificationData) -> str:
        """Create additional data section for email"""
        if not notification.data:
            return ""
        
        data = notification.data
        sections = []
        
        # Target information
        if 'target_price' in data:
            sections.append(f"""
                <div style="margin-bottom: 15px;">
                    <strong>Target Price:</strong> ${data['target_price']:.2f}
                </div>
            """)
        
        if 'current_price' in data:
            sections.append(f"""
                <div style="margin-bottom: 15px;">
                    <strong>Current Price:</strong> ${data['current_price']:.2f}
                </div>
            """)
        
        if 'confidence_score' in data:
            sections.append(f"""
                <div style="margin-bottom: 15px;">
                    <strong>AI Confidence:</strong> {data['confidence_score']:.1f}%
                </div>
            """)
        
        if 'metal_type' in data:
            sections.append(f"""
                <div style="margin-bottom: 15px;">
                    <strong>Metal:</strong> {data['metal_type']}
                </div>
            """)
        
        if sections:
            return f"""
                <div style="border-top: 1px solid #e5e7eb; padding-top: 20px; margin-top: 20px;">
                    <h3 style="margin: 0 0 15px 0; color: #374151; font-size: 16px;">Details</h3>
                    {''.join(sections)}
                </div>
            """
        
        return ""
    
    def _send_sms(self, notification: NotificationData):
        """Send SMS notification (placeholder for future implementation)"""
        logger.info(f"SMS notification would be sent to user {notification.user_id}: {notification.title}")
        # TODO: Implement SMS sending using Twilio or similar service
    
    def _send_push(self, notification: NotificationData):
        """Send push notification (placeholder for future implementation)"""
        logger.info(f"Push notification would be sent to user {notification.user_id}: {notification.title}")
        # TODO: Implement push notifications using Firebase or similar service
    
    def _send_in_app(self, notification: NotificationData):
        """Send in-app notification (save to database for frontend to retrieve)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO in_app_notifications 
                    (user_id, title, message, notification_type, priority, data, created_at, read_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    notification.user_id,
                    notification.title,
                    notification.message,
                    notification.notification_type,
                    notification.priority,
                    json.dumps(notification.data) if notification.data else None,
                    datetime.now(),
                    None
                ))
                
                conn.commit()
                logger.info(f"In-app notification saved for user {notification.user_id}")
                
        except Exception as e:
            # Create in-app notifications table if it doesn't exist
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS in_app_notifications (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            title TEXT NOT NULL,
                            message TEXT NOT NULL,
                            notification_type TEXT NOT NULL,
                            priority TEXT NOT NULL,
                            data TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            read_at TIMESTAMP
                        )
                    """)
                    conn.commit()
                    
                    # Retry the insert
                    cursor.execute("""
                        INSERT INTO in_app_notifications 
                        (user_id, title, message, notification_type, priority, data, created_at, read_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        notification.user_id,
                        notification.title,
                        notification.message,
                        notification.notification_type,
                        notification.priority,
                        json.dumps(notification.data) if notification.data else None,
                        datetime.now(),
                        None
                    ))
                    conn.commit()
                    
            except Exception as e2:
                logger.error(f"Error creating in-app notification: {str(e2)}")
                raise
    
    def _record_delivery(self, notification_id: int, channel: str, status: str, error_message: str = None):
        """Record notification delivery status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO notification_deliveries 
                    (notification_id, channel, status, error_message)
                    VALUES (?, ?, ?, ?)
                """, (notification_id, channel, status, error_message))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording delivery: {str(e)}")
    
    def _record_rate_limit(self, user_id: int, channel: str):
        """Record rate limit usage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO notification_rate_limits (user_id, channel)
                    VALUES (?, ?)
                """, (user_id, channel))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording rate limit: {str(e)}")
    
    def _update_notification_status(self, notification_id: int, status: str):
        """Update notification status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE notifications 
                    SET status = ?, sent_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, notification_id))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating notification status: {str(e)}")
    
    def get_user_notifications(self, user_id: int, limit: int = 20, unread_only: bool = False) -> List[Dict]:
        """Get in-app notifications for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT id, title, message, notification_type, priority, data, created_at, read_at
                    FROM in_app_notifications 
                    WHERE user_id = ?
                """
                
                if unread_only:
                    query += " AND read_at IS NULL"
                
                query += " ORDER BY created_at DESC LIMIT ?"
                
                cursor.execute(query, (user_id, limit))
                
                columns = [desc[0] for desc in cursor.description]
                notifications = []
                
                for row in cursor.fetchall():
                    notification = dict(zip(columns, row))
                    if notification['data']:
                        notification['data'] = json.loads(notification['data'])
                    notifications.append(notification)
                
                return notifications
                
        except Exception as e:
            logger.error(f"Error getting user notifications: {str(e)}")
            return []
    
    def mark_notification_read(self, notification_id: int, user_id: int):
        """Mark an in-app notification as read"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE in_app_notifications 
                    SET read_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND user_id = ?
                """, (notification_id, user_id))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error marking notification as read: {str(e)}")
    
    def send_target_triggered_notification(self, user_id: int, target_data: Dict[str, Any]):
        """Send notification when investment target is triggered"""
        notification = NotificationData(
            user_id=user_id,
            title=f"🎯 Investment Target Triggered: {target_data['metal_type']}",
            message=f"Your {target_data['target_type'].replace('_', ' ')} target for {target_data['metal_type']} at ${target_data['target_price']:.2f} has been triggered!\n\nCurrent Price: ${target_data['current_price']:.2f}",
            notification_type='target_triggered',
            priority='HIGH',
            channels=['email', 'in_app'],
            data=target_data
        )
        
        return self.send_notification(notification)
    
    def send_recommendation_notification(self, user_id: int, recommendation_data: Dict[str, Any]):
        """Send notification for AI recommendation"""
        confidence = recommendation_data.get('confidence_score', 0)
        priority = 'HIGH' if confidence >= 80 else 'MEDIUM'
        
        notification = NotificationData(
            user_id=user_id,
            title=f"🤖 AI Recommendation: {recommendation_data['action']} {recommendation_data['metal_type']}",
            message=f"AI recommends {recommendation_data['action']} {recommendation_data['metal_type']} with {confidence:.1f}% confidence.\n\n{recommendation_data.get('reasoning', '')[:200]}...",
            notification_type='recommendation',
            priority=priority,
            channels=['email', 'in_app'],
            data=recommendation_data
        )
        
        return self.send_notification(notification)
    
    def send_market_alert_notification(self, user_id: int, alert_data: Dict[str, Any]):
        """Send market alert notification"""
        notification = NotificationData(
            user_id=user_id,
            title=f"📈 Market Alert: {alert_data['metal_type']}",
            message=f"Significant market movement detected for {alert_data['metal_type']}.\n\n{alert_data.get('analysis', '')}",
            notification_type='market_alert',
            priority='MEDIUM',
            channels=['email', 'in_app'],
            data=alert_data
        )
        
        return self.send_notification(notification)

# Global instance
notification_service = SmartNotificationService()

