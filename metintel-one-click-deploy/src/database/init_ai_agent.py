"""
Database initialization script for AI Agent functionality
Integrates advanced features from PreciousPredictor codebase
"""

import os
import sys
import logging
from datetime import datetime

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ai_agent import AIAgentDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_ai_agent_database():
    """Initialize AI agent database tables"""
    try:
        # For now, we'll create a simple SQLite database for testing
        # In production, this would connect to PostgreSQL
        import sqlite3
        
        # Create database connection
        db_path = "/home/ubuntu/precious-metals-api/ai_agent.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        logger.info("Creating AI agent database tables...")
        
        # Get the SQL statements from our model
        create_tables_sql = AIAgentDatabase.get_create_tables_sql()
        
        # Execute each statement separately (SQLite doesn't support multiple statements in one execute)
        statements = create_tables_sql.split(';')
        for statement in statements:
            statement = statement.strip()
            if statement:
                try:
                    cursor.execute(statement)
                    logger.info(f"Executed: {statement[:50]}...")
                except Exception as e:
                    logger.error(f"Error executing statement: {e}")
                    logger.error(f"Statement: {statement}")
        
        # Insert sample data
        logger.info("Inserting sample data...")
        sample_data_sql = AIAgentDatabase.get_sample_data_sql()
        
        statements = sample_data_sql.split(';')
        for statement in statements:
            statement = statement.strip()
            if statement and not statement.startswith('--'):
                try:
                    cursor.execute(statement)
                    logger.info(f"Executed sample data: {statement[:50]}...")
                except Exception as e:
                    logger.error(f"Error executing sample data: {e}")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info("AI agent database initialized successfully!")
        logger.info(f"Database created at: {db_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing AI agent database: {str(e)}")
        return False

def verify_database():
    """Verify that all tables were created successfully"""
    try:
        import sqlite3
        
        db_path = "/home/ubuntu/precious-metals-api/ai_agent.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        expected_tables = [
            'investment_targets',
            'ai_recommendations', 
            'agent_performance',
            'user_ai_preferences',
            'user_portfolios',
            'portfolio_transactions'
        ]
        
        logger.info("Verifying database tables...")
        for table_name in expected_tables:
            if any(table_name in table[0] for table in tables):
                logger.info(f"✅ Table '{table_name}' created successfully")
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                logger.info(f"   - Contains {count} rows")
            else:
                logger.error(f"❌ Table '{table_name}' not found")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error verifying database: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting AI Agent Database Initialization...")
    
    # Initialize database
    if init_ai_agent_database():
        logger.info("✅ Database initialization completed successfully!")
        
        # Verify database
        if verify_database():
            logger.info("✅ Database verification completed successfully!")
            logger.info("🎉 AI Agent database is ready for use!")
        else:
            logger.error("❌ Database verification failed!")
    else:
        logger.error("❌ Database initialization failed!")

