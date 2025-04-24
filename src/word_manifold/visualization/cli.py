"""
Command Line Interface for Word Manifold Visualizations.

This module provides a unified CLI for invoking different types of visualizations,
including ritual evolution animations, semantic shape analysis, automata, and symbolic visualizations.
"""

import click
import logging
from pathlib import Path
import requests
import threading
import webbrowser
import time
from typing import Optional, List, Dict, Any
import numpy as np
import tempfile
import sys

from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.manifold.vector_manifold import VectorManifold
from word_manifold.visualization.hypertools_visualizer import HyperToolsVisualizer
from word_manifold.visualization.shape_visualizer import ShapeVisualizer
from word_manifold.visualization.symbolic_visualizer import SymbolicVisualizer
from word_manifold.visualization.semantic_tree_visualizer import SemanticTreeVisualizer
from word_manifold.visualization.manifold_vis import ManifoldVisualizer
from word_manifold.visualization.interactive import InteractiveManifoldVisualizer
from word_manifold.automata.cellular_rules import create_predefined_rules
from word_manifold.visualization.server import run_server

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Server configuration
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 5000
SERVER_URL = f'http://{DEFAULT_HOST}:{DEFAULT_PORT}'

def ensure_server_running():
    """Start visualization server if not running."""
    try:
        response = requests.get(f'{SERVER_URL}/health', timeout=1)
        if response.status_code == 200:
            return
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        logger.info("Starting visualization server...")
        
        # Start server in background thread
        server_thread = threading.Thread(
            target=run_server,
            args=(DEFAULT_HOST, DEFAULT_PORT),
            daemon=True
        )
        server_thread.start()
        
        # Wait for server to start
        max_retries = 10
        for i in range(max_retries):
            try:
                time.sleep(1)
                response = requests.get(f'{SERVER_URL}/health', timeout=1)
                if response.status_code == 200:
                    logger.info("Visualization server is ready")
                    return
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if i == max_retries - 1:
                    logger.error("Failed to start visualization server")
                    raise click.ClickException("Could not start visualization server")
                logger.info(f"Waiting for server to start (attempt {i+1}/{max_retries})...")

def handle_server_request(method: str, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Handle server request with proper error handling."""
    try:
        if method.upper() == 'GET':
            response = requests.get(f'{SERVER_URL}{endpoint}', timeout=5)
        else:
            response = requests.post(f'{SERVER_URL}{endpoint}', json=data, timeout=5)
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise click.ClickException("Server request timed out")
    except requests.exceptions.ConnectionError:
        raise click.ClickException("Could not connect to visualization server")
    except requests.exceptions.HTTPError as e:
        error_msg = "Unknown error"
        try:
            error_msg = response.json().get('error', error_msg)
        except:
            pass
        raise click.ClickException(f"Server error: {error_msg}")
    except Exception as e:
        raise click.ClickException(f"Unexpected error: {str(e)}")

@click.group()
def cli():
    """Word Manifold visualization and analysis tools."""
    pass

@cli.command()
@click.option('--interactive/--no-interactive', default=True, help='Enable/disable interactive mode')
@click.option('--output-dir', default='visualizations', help='Output directory for visualizations')
@click.option('--dimensions', default=3, help='Number of dimensions for visualization')
@click.option('--model', default='glove-wiki-gigaword-300', help='model to use for embeddings')
def visualize(interactive, output_dir, dimensions, model):
    """Create visualizations of the word manifold."""
    try:
        # Initialize embeddings and manifold
        embeddings = WordEmbeddings(model_name=model)
        
        # Define example terms
        terms = [
            "light", "darkness", "wisdom", "understanding",
            "beauty", "strength", "mercy", "severity"
        ]
        
        # Load terms into embeddings
        embeddings.load_terms(terms)
        manifold = VectorManifold(embeddings)
        
        # Create output directories
        hypertools_dir = Path(output_dir) / "hypertools"
        shapes_dir = Path(output_dir) / "shapes"
        hypertools_dir.mkdir(parents=True, exist_ok=True)
        shapes_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizers
        hypertools_vis = HyperToolsVisualizer(
            word_embeddings=embeddings,
            output_dir=str(hypertools_dir),
            interactive=interactive,
            n_dimensions=dimensions
        )
        
        shape_vis = ShapeVisualizer()
        
        # Create term trajectories through ritual phases
        term_trajectories = {}
        rules = create_predefined_rules()
        
        for term in terms:
            trajectories = []
            vector = manifold.embeddings.get_embedding(term)
            
            # Initial state
            trajectories.append(vector.copy())
            
            # Apply transformations
            current = vector.copy()
            current_manifold = VectorManifold(embeddings)
            current_manifold.add_cell(term, current)
            
            # Phase 1: Equilibrium
            try:
                transformed_manifold = rules["equilibrium"].apply(current_manifold)
                transformed = transformed_manifold.get_cell(term).centroid
                trajectories.append(transformed.copy())
                current_manifold = transformed_manifold
            except Exception as e:
                logger.warning(f"Error applying equilibrium rule: {e}")
                trajectories.append(current.copy())
            
            # Phase 2: The Great Work
            try:
                transformed_manifold = rules["great_work"].apply(current_manifold)
                transformed = transformed_manifold.get_cell(term).centroid
                trajectories.append(transformed.copy())
                current_manifold = transformed_manifold
            except Exception as e:
                logger.warning("Error applying great work rule", exc_info=e, stack_info=True, stacklevel=2)
                trajectories.append(current.copy())
            
            # Phase 3: Star
            try:
                transformed_manifold = rules["star"].apply(current_manifold)
                transformed = transformed_manifold.get_cell(term).centroid
                trajectories.append(transformed.copy())
            except Exception as e:
                logger.warning("Error applying star rule", exc_info=e, stack_info=True, stacklevel=2)
                trajectories.append(current.copy())
            
            term_trajectories[term] = trajectories
        
        # Create static visualization
        try:
            hypertools_vis.visualize_term_evolution(
                term_trajectories=term_trajectories,
                phase_names=["Initial", "Equilibrium", "Great Work", "Illumination"],
                title="Ritual Term Evolution"
            )
        except Exception as e:
            logger.error("Error creating static visualization", exc_info=e, stack_info=True, stacklevel=2)
        
        # Create animation
        try:
            hypertools_vis.create_animated_ritual(
                term_trajectories=term_trajectories,
                phase_names=["Initial", "Equilibrium", "Great Work", "Illumination"],
                title="Ritual Evolution Animation",
                duration=10.0,
                fps=30,
                add_trails=True
            )
        except Exception as e:
            logger.error(f"Error creating animation: {e}")
            
        # Create shape visualization
        ritual_text = """
        I am the flame that burns in every heart of man,
        and in the core of every star.
        I am Life, and the giver of Life,
        yet therefore is the knowledge of me the knowledge of death.
        I am alone: there is no God where I am.
        """
        
        try:
            # Save shape field visualization
            shape_vis.create_shape_field(
                ritual_text,
                chunk_size=1  # Process one sentence at a time
            )
            # Move the generated file to the shapes directory
            for file in Path('.').glob('shape_field*.png'):
                file.rename(shapes_dir / file.name)
        except Exception as e:
            logger.error(f"Error creating shape field: {e}")
        
        # Create comparative visualization
        texts = [
            "The mysteries of the universe unfold in sacred geometry.",
            "Divine passion ignites the soul with transformative fire.",
            "Wisdom flows like water through the channels of understanding."
        ]
        
        try:
            # Save comparative visualization
            shape_vis.create_comparative_visualization(
                texts,
                labels=["Mystery", "Passion", "Wisdom"]
            )
            # Move the generated file to the shapes directory
            for file in Path('.').glob('comparative*.png'):
                file.rename(shapes_dir / file.name)
        except Exception as e:
            logger.error("Error creating comparative visualization", exc_info=e, stack_info=True, stacklevel=2)
            
        logger.info("Visualizations have been saved to %s", output_dir)
            
    except Exception as e:
        logger.error("Error in visualization", exc_info=e, stack_info=True, stacklevel=2)
        raise click.ClickException(str(e))

@cli.command()
@click.argument('text')
@click.option('--output-dir', default='visualizations/symbolic', help='Output directory for symbolic visualizations')
@click.option('--width', default=80, help='Width of the symbolic visualization')
@click.option('--height', default=40, help='Height of the symbolic visualization')
def symbolic(text: str, output_dir: str, width: int, height: int):
    """Create symbolic visualizations of text."""
    try:
        embeddings = WordEmbeddings()
        visualizer = SymbolicVisualizer(embeddings, width=width, height=height)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create static visualization
        static_viz = visualizer.visualize_term(text)
        static_path = output_path / f"{text.replace(' ', '_')}_symbolic.txt"
        static_path.write_text(static_viz)
        
        # Create semantic field if text contains multiple words
        terms = text.split()
        if len(terms) > 1:
            field_viz = visualizer.create_semantic_field(terms)
            field_path = output_path / f"{text.replace(' ', '_')}_field.txt"
            field_path.write_text(field_viz)
            
        logger.info(f"Symbolic visualizations saved to {output_dir}")
        
    except Exception as e:
        logger.error("Error in symbolic visualization", exc_info=e)
        raise click.ClickException(str(e))

@cli.command()
@click.argument('text')
@click.option('--output-dir', default='visualizations/semantic_tree', help='Output directory for semantic tree visualizations')
@click.option('--max-depth', default=3, help='Maximum depth of the semantic tree')
@click.option('--branching-factor', default=4, help='Maximum children per node')
def semantic_tree(text: str, output_dir: str, max_depth: int, branching_factor: int):
    """Generate semantic tree visualization for text."""
    try:
        embeddings = WordEmbeddings()
        visualizer = SemanticTreeVisualizer(embeddings)
        
        # Find related terms
        related_terms = []
        for term in text.split():
            similar = embeddings.find_similar_terms(term, k=5)
            related_terms.extend([t for t, _ in similar])
        
        # Build and visualize tree
        tree = visualizer.build_semantic_tree(
            root_text=text,
            related_terms=related_terms,
            max_depth=max_depth,
            branching_factor=branching_factor
        )
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save visualization
        viz_path = output_path / f"{text.replace(' ', '_')}_tree.png"
        visualizer.visualize_tree(tree, save_path=str(viz_path))
        
        logger.info(f"Semantic tree visualization saved to {viz_path}")
        
    except Exception as e:
        logger.error("Error in semantic tree visualization", exc_info=e)
        raise click.ClickException(str(e))

@cli.command()
@click.argument('text')
@click.option('--output-dir', default='visualizations/shapes', help='Output directory for shape visualizations')
@click.option('--chunk-size', default=1, help='Number of sentences per chunk')
def shapes(text: str, output_dir: str, chunk_size: int):
    """Generate shape field visualizations for text."""
    try:
        visualizer = ShapeVisualizer()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create shape field visualization
        visualizer.create_shape_field(text, chunk_size=chunk_size)
        
        # Move generated files to output directory
        for file in Path('.').glob('shape_field*.png'):
            file.rename(output_path / file.name)
            
        logger.info(f"Shape visualizations saved to {output_dir}")
        
    except Exception as e:
        logger.error("Error in shape visualization", exc_info=e)
        raise click.ClickException(str(e))

@cli.command()
@click.argument('text')
@click.option('--rule', default='equilibrium', type=click.Choice(['equilibrium', 'great_work', 'star']), help='Automata rule to apply')
@click.option('--output-dir', default='visualizations/automata', help='Output directory for automata visualizations')
@click.option('--iterations', default=10, help='Number of iterations to run the automata')
def automata(text: str, rule: str, output_dir: str, iterations: int):
    """Apply cellular automata rules to transform text."""
    try:
        embeddings = WordEmbeddings()
        manifold = VectorManifold(embeddings)
        rules = create_predefined_rules()
        
        # Initialize manifold with text
        for term in text.split():
            vector = embeddings.get_embedding(term)
            if vector is not None:
                manifold.add_cell(term, vector)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Apply rule iteratively and visualize
        current_manifold = manifold
        visualizer = HyperToolsVisualizer(embeddings, output_dir=str(output_path))
        
        for i in range(iterations):
            # Apply rule
            transformed_manifold = rules[rule].apply(current_manifold)
            
            # Visualize current state
            terms = list(transformed_manifold.cells.keys())
            vectors = [cell.centroid for cell in transformed_manifold.cells.values()]
            
            visualizer.visualize_terms(
                terms,
                title=f"{rule.title()} Transformation - Iteration {i+1}",
                save_path=str(output_path / f"iteration_{i+1:03d}.png")
            )
            
            current_manifold = transformed_manifold
            
        logger.info(f"Automata visualizations saved to {output_dir}")
        
    except Exception as e:
        logger.error("Error in automata visualization", exc_info=e)
        raise click.ClickException(str(e))

@cli.command()
@click.argument('terms', nargs=-1, required=True)
@click.option('--model', '-m', default='all-MiniLM-L6-v2', help='Name of the sentence transformer model to use')
@click.option('--dimensions', '-d', default=3, help='Number of dimensions for visualization')
@click.option('--output', '-o', default='visualizations', help='Output directory for visualizations')
@click.option('--save-format', '-f', default='png', help='Format to save visualizations in (png, jpg, svg)')
def manifold(terms: List[str], model: str, dimensions: int, output: str, save_format: str):
    """Generate static manifold visualization for given terms."""
    ensure_server_running()
    
    try:
        # Validate input
        if not terms:
            raise click.ClickException("Please provide at least one term to visualize")
            
        if dimensions < 2 or dimensions > 3:
            raise click.ClickException("Dimensions must be 2 or 3")
            
        # Create output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create request
        response = handle_server_request('POST', '/api/visualize/static', {
            'terms': terms,
            'model': model,
            'dimensions': dimensions
        })
        
        if response.get('error'):
            raise click.ClickException(response['error'])
            
        # Save visualization
        import base64
        img_data = base64.b64decode(response['image'])
        output_file = output_path / f"manifold_visualization.{save_format}"
        output_file.write_bytes(img_data)
        
        click.echo(f"Visualization saved to {output_file}")
        
    except Exception as e:
        raise click.ClickException(str(e))

@cli.command()
@click.argument('terms', nargs=-1, required=True)
@click.option('--model', '-m', default='all-MiniLM-L6-v2', help='Name of the sentence transformer model to use')
@click.option('--dimensions', '-d', default=3, help='Number of dimensions for visualization')
@click.option('--port', '-p', default=DEFAULT_PORT, help='Port for visualization server')
def interactive(terms: List[str], model: str, dimensions: int, port: int):
    """Launch interactive visualization for given terms."""
    ensure_server_running()
    
    try:
        # Validate input
        if not terms:
            raise click.ClickException("Please provide at least one term to visualize")
            
        if dimensions < 2 or dimensions > 3:
            raise click.ClickException("Dimensions must be 2 or 3")
        
        # Create interactive session
        response = handle_server_request('POST', '/api/visualize/interactive/create', {
            'terms': terms,
            'model': model,
            'dimensions': dimensions
        })
        
        if response.get('error'):
            raise click.ClickException(response['error'])
            
        session_data = response
        session_id = session_data['session_id']
        
        # Open browser
        url = f'{SERVER_URL}/visualize/{session_id}'
        click.echo(f"Opening visualization in browser: {url}")
        webbrowser.open(url)
        
        # Keep server running until user interrupts
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            # Clean up session
            handle_server_request('POST', f'/api/visualize/interactive/{session_id}/close')
            click.echo("\nVisualization session closed")
            
    except Exception as e:
        raise click.ClickException(str(e))

@cli.command()
@click.argument('session_id')
def close_session(session_id: str):
    """Close an interactive visualization session."""
    ensure_server_running()
    
    try:
        response = handle_server_request('POST', f'/api/visualize/interactive/{session_id}/close')
        if response.get('error'):
            raise click.ClickException(response['error'])
        click.echo("Session closed successfully")
    except Exception as e:
        raise click.ClickException(str(e))

cli.add_command(manifold)
cli.add_command(interactive)
cli.add_command(close_session)

if __name__ == '__main__':
    cli() 