"""
Word Manifold API initialization.
"""

from flask import Flask
from flask_cors import CORS
from .routes.embeddings import embeddings_bp
from .routes.visualizations import visualizations_bp

def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Apply configuration
    if config:
        app.config.update(config)
    
    # Register blueprints
    app.register_blueprint(embeddings_bp, url_prefix='/api/v1/embeddings')
    app.register_blueprint(visualizations_bp, url_prefix='/api/v1/visualizations')
    
    return app 