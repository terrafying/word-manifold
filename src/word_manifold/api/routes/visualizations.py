"""
API routes for visualization functionality.
"""

from flask import Blueprint, request, jsonify, send_file
from typing import Dict, Any, Optional
import tempfile
from pathlib import Path
import json
import logging
from ...visualization.engines.timeseries import TimeSeriesEngine
from ...visualization.renderers.timeseries import TimeSeriesRenderer
from ...embeddings.word_embeddings import WordEmbeddings
from ..routes.embeddings import get_embeddings

# Create blueprint
visualizations_bp = Blueprint('visualizations', __name__)
logger = logging.getLogger(__name__)

@visualizations_bp.route('/timeseries', methods=['POST'])
def create_timeseries():
    """
    Create time series visualization.
    
    Request body:
    {
        "terms": ["term1", "term2", ...],
        "timeframe": "1d",  # optional
        "interval": "1h",   # optional
        "pattern_type": "cyclic",  # optional
        "interactive": true,  # optional
        "show_values": true,  # optional
        "show_connections": true,  # optional
        "hexagram_data": {...}  # optional
    }
    """
    try:
        data = request.get_json()
        terms = data.get('terms', [])
        
        if not terms:
            return jsonify({
                'error': 'No terms provided'
            }), 400
            
        # Get embeddings
        embeddings = get_embeddings()
        
        # Initialize engine and renderer
        engine = TimeSeriesEngine(
            word_embeddings=embeddings,
            pattern_type=data.get('pattern_type', 'cyclic'),
            timeframe=data.get('timeframe', '1d'),
            interval=data.get('interval', '1h')
        )
        
        renderer = TimeSeriesRenderer()
        
        # Process data
        processed_data = engine.process_data({
            'terms': terms,
            'timeframe': data.get('timeframe'),
            'interval': data.get('interval'),
            'pattern_type': data.get('pattern_type'),
            'hexagram_data': data.get('hexagram_data')
        })
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Render visualization
            output_file = renderer.render_local(
                processed_data,
                output_path,
                interactive=data.get('interactive', False),
                show_values=data.get('show_values', True),
                show_connections=data.get('show_connections', True)
            )
            
            # Get insights
            insights = renderer.get_insights()
            
            # Return visualization and insights
            return jsonify({
                'visualization': output_file.read_bytes().decode('utf-8') if output_file.suffix == '.html' else output_file.read_bytes(),
                'format': output_file.suffix[1:],
                'insights': insights,
                'metadata': processed_data['metadata']
            })
            
    except Exception as e:
        logger.error(f"Error creating time series visualization: {e}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500

@visualizations_bp.route('/magic', methods=['POST'])
def create_magic_structure():
    """
    Create magic structure visualization.
    
    Request body:
    {
        "dimension": 2,
        "size": 3,
        "terms": ["term1", "term2", ...],  # optional
        "interactive": true,  # optional
        "show_values": true,  # optional
        "show_connections": true,  # optional
        "color_scheme": "viridis"  # optional
    }
    """
    try:
        from ...visualization.magic_visualizer import MagicVisualizer
        
        data = request.get_json()
        dimension = data.get('dimension', 2)
        size = data.get('size', 3)
        
        # Get embeddings if terms provided
        embeddings = None
        if 'terms' in data:
            embeddings = get_embeddings()
        
        # Initialize visualizer
        visualizer = MagicVisualizer(
            word_embeddings=embeddings,
            enable_semantic_weighting='terms' in data,
            color_scheme=data.get('color_scheme', 'viridis')
        )
        
        # Generate structure
        structure = visualizer.generate_magic_structure(
            dimension=dimension,
            size=size,
            terms=data.get('terms')
        )
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Create visualization
            if data.get('interactive', False):
                output_file = output_path / 'magic_structure.html'
            else:
                output_file = output_path / 'magic_structure.png'
                
            visualizer.visualize(
                structure=structure,
                title=f"{size}D Magic Structure",
                show_values=data.get('show_values', True),
                show_connections=data.get('show_connections', True),
                interactive=data.get('interactive', False),
                save_path=str(output_file)
            )
            
            # Return visualization
            return jsonify({
                'visualization': output_file.read_bytes().decode('utf-8') if output_file.suffix == '.html' else output_file.read_bytes(),
                'format': output_file.suffix[1:],
                'magic_constant': float(structure.magic_constant),
                'is_magic': structure.is_magic()
            })
            
    except Exception as e:
        logger.error(f"Error creating magic structure visualization: {e}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500 