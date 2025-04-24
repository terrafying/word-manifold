"""Visualization server for handling visualization requests."""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from pathlib import Path
import tempfile
import base64
import io
import logging
import traceback
from typing import Dict, Any, List
import os

from .manifold_vis import ManifoldVisualizer
from .interactive import InteractiveManifoldVisualizer
from ..embeddings.word_embeddings import WordEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Security headers
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

_embeddings = None
_visualizers: Dict[str, Any] = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

def get_embeddings(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> WordEmbeddings:
    """Get or create embeddings instance."""
    global _embeddings
    if _embeddings is None:
        try:
            logger.info(f"Initializing embeddings with model {model_name}")
            _embeddings = WordEmbeddings(model_name=model_name)
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}\n{traceback.format_exc()}")
            raise
    return _embeddings

@app.route('/api/visualize/static', methods=['POST'])
def create_static_visualization():
    """Create static manifold visualization."""
    try:
        data = request.json
        terms = data.get('terms', [])
        model = data.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        dimensions = data.get('dimensions', 2)
        
        logger.info(f"Creating static visualization for terms: {terms}")
        
        # Get embeddings
        embeddings = get_embeddings(model)
        
        # Process terms
        term_embeddings = []
        term_lists = []
        
        for term in terms:
            logger.info(f"Getting embedding for term: {term}")
            embedding = embeddings.get_embedding(term)
            if embedding is not None:
                term_embeddings.append(embedding)
                term_lists.append([term])
            else:
                logger.warning(f"No embedding found for term: {term}")
                
        if not term_embeddings:
            logger.error("No valid embeddings found for any terms")
            return jsonify({'error': 'No valid embeddings found for terms'}), 400
            
        # Create visualization
        logger.info("Creating ManifoldVisualizer")
        visualizer = ManifoldVisualizer(
            embeddings=np.array(term_embeddings),
            terms=term_lists,
            n_components=dimensions
        )
        
        # Generate visualization
        logger.info("Preparing visualization data")
        data = visualizer.prepare_data()
        logger.info("Plotting visualization")
        fig = visualizer.plot(data)
        
        # Save to buffer
        logger.info("Saving visualization to buffer")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        
        logger.info("Visualization created successfully")
        return jsonify({
            'image': img_base64,
            'type': 'png',
            'dimensions': dimensions
        })
        
    except Exception as e:
        error_msg = f"Error creating static visualization: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/visualize/interactive/create', methods=['POST'])
def create_interactive_visualization():
    """Create interactive visualization session."""
    try:
        data = request.json
        terms = data.get('terms', [])
        model = data.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        dimensions = data.get('dimensions', 2)
        
        logger.info(f"Creating interactive visualization for terms: {terms}")
        
        # Get embeddings
        embeddings = get_embeddings(model)
        
        # Process terms
        term_embeddings = []
        term_lists = []
        
        for term in terms:
            logger.info(f"Getting embedding for term: {term}")
            embedding = embeddings.get_embedding(term)
            if embedding is not None:
                term_embeddings.append(embedding)
                term_lists.append([term])
            else:
                logger.warning(f"No embedding found for term: {term}")
                
        if not term_embeddings:
            logger.error("No valid embeddings found for any terms")
            return jsonify({'error': 'No valid embeddings found for terms'}), 400
            
        # Create visualization
        logger.info("Creating interactive visualization")
        session_id = base64.urlsafe_b64encode(os.urandom(16)).decode()
        visualizer = InteractiveManifoldVisualizer(
            embeddings=np.array(term_embeddings),
            terms=term_lists
        )
        
        # Store visualizer
        _visualizers[session_id] = visualizer
        
        # Generate initial visualization
        logger.info("Preparing visualization data")
        data = visualizer.prepare_data()
        logger.info("Plotting visualization")
        fig = visualizer.plot(data)
        
        # Save initial state to buffer
        logger.info("Saving visualization to buffer")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        
        logger.info("Interactive visualization created successfully")
        return jsonify({
            'session_id': session_id,
            'image': img_base64,
            'type': 'png',
            'dimensions': dimensions,
            'state': visualizer.get_state()
        })
        
    except Exception as e:
        error_msg = f"Error creating interactive visualization: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/visualize/interactive/update', methods=['POST'])
def update_interactive_visualization():
    """Update interactive visualization state."""
    try:
        data = request.json
        session_id = data.get('session_id')
        state = data.get('state', {})
        
        logger.info(f"Updating visualization for session: {session_id}")
        
        if session_id not in _visualizers:
            logger.error(f"Invalid session ID: {session_id}")
            return jsonify({'error': 'Invalid session ID'}), 404
            
        visualizer = _visualizers[session_id]
        
        # Update visualization state
        logger.info(f"Updating state: {state}")
        for key, value in state.items():
            if key in visualizer.controls:
                visualizer.controls[key].set_val(value)
        
        # Update visualization
        logger.info("Updating visualization")
        visualizer.update()
        
        # Get updated image
        logger.info("Saving updated visualization")
        buf = io.BytesIO()
        visualizer._figure.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        
        logger.info("Visualization updated successfully")
        return jsonify({
            'image': img_base64,
            'type': 'png',
            'state': visualizer.get_state()
        })
        
    except Exception as e:
        error_msg = f"Error updating visualization: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/visualize/interactive/close', methods=['POST'])
def close_interactive_visualization():
    """Close interactive visualization session."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        logger.info(f"Closing visualization session: {session_id}")
        
        if session_id in _visualizers:
            visualizer = _visualizers[session_id]
            visualizer.close()
            del _visualizers[session_id]
            
        logger.info("Session closed successfully")
        return jsonify({'status': 'success'})
        
    except Exception as e:
        error_msg = f"Error closing visualization: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

def run_server(host: str = 'localhost', port: int = 5000, debug: bool = False):
    """Run the visualization server.
    
    Args:
        host: Host address to bind to
        port: Port to listen on
        debug: Whether to run in debug mode
    """
    logger.info(f"Starting visualization server on {host}:{port} (debug={debug})")
    
    if debug:
        # Enable more detailed Flask logging
        logging.getLogger('flask').setLevel(logging.DEBUG)
        # Enable Werkzeug debugger
        os.environ['FLASK_ENV'] = 'development'
        os.environ['FLASK_DEBUG'] = '1'
        
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_server() 