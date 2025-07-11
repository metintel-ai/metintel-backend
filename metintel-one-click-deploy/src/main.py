import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from src.models.user import db
from src.models.metals import Metal, PriceData, Prediction, PredictionFactor
from src.routes.user import user_bp
from src.routes.prices import prices_bp
from src.routes.predictions import predictions_bp
from src.routes.news import news_bp
from src.routes.met_ask import met_ask_bp
from src.routes.ai_agent import ai_agent_bp
from src.routes.enhanced_ai_agent import enhanced_ai_agent_bp
import logging

# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Enable CORS for all routes
CORS(app, origins="*")

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'asdf#FGSgvasgf$5$WGT')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Register blueprints
app.register_blueprint(user_bp, url_prefix='/api/users')
app.register_blueprint(prices_bp, url_prefix='/api/prices')
app.register_blueprint(predictions_bp, url_prefix='/api/predictions')
app.register_blueprint(news_bp, url_prefix='/api/news')
app.register_blueprint(met_ask_bp, url_prefix='/api/met-ask')
app.register_blueprint(ai_agent_bp, url_prefix='/api/ai-agent')
app.register_blueprint(enhanced_ai_agent_bp, url_prefix='/api/enhanced-ai-agent')

# Initialize database
db.init_app(app)

with app.app_context():
    db.create_all()
    
    # Initialize default metals if they don't exist
    metals_data = [
        {'symbol': 'USDXAU', 'name': 'Gold'},
        {'symbol': 'USDXAG', 'name': 'Silver'},
        {'symbol': 'USDXPT', 'name': 'Platinum'},
        {'symbol': 'USDXPD', 'name': 'Palladium'}
    ]
    
    for metal_data in metals_data:
        existing_metal = Metal.query.filter_by(symbol=metal_data['symbol']).first()
        if not existing_metal:
            metal = Metal(
                symbol=metal_data['symbol'],
                name=metal_data['name'],
                unit='troy_ounce',
                is_active=True
            )
            db.session.add(metal)
    
    try:
        db.session.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        db.session.rollback()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'precious-metals-api',
        'version': '1.0.0',
        'timestamp': os.popen('date -u +"%Y-%m-%dT%H:%M:%SZ"').read().strip()
    }

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "Frontend not deployed. API is running at /api/", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)

# Production configuration
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=False)
