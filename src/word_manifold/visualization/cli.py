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
try:
    import psutil
except ImportError:
    psutil = None
import json
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np
import tempfile
import sys
from functools import wraps
import yaml
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
import asyncio

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib imports

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def lazy_import(module_name: str) -> Callable:
    """Lazily import a module only when needed."""
    def get_module():
        import importlib
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            logger.error(f"Failed to import {module_name}: {e}")
            raise click.ClickException(f"Required module {module_name} not found. Please install required dependencies.")
    return get_module

# Lazy imports for heavy dependencies
get_word_embeddings = lazy_import('word_manifold.embeddings.word_embeddings')
get_vector_manifold = lazy_import('word_manifold.manifold.vector_manifold')
get_hypertools = lazy_import('word_manifold.visualization.hypertools_visualizer')
get_shape_vis = lazy_import('word_manifold.visualization.shape_visualizer')
get_symbolic_vis = lazy_import('word_manifold.visualization.symbolic_visualizer')
get_semantic_tree = lazy_import('word_manifold.visualization.semantic_tree_visualizer')
get_manifold_vis = lazy_import('word_manifold.visualization.manifold_vis')
get_interactive_vis = lazy_import('word_manifold.visualization.interactive')
get_cellular_rules = lazy_import('word_manifold.automata.cellular_rules')
get_server = lazy_import('word_manifold.visualization.server')
get_remote_server = lazy_import('word_manifold.visualization.remote_server')
get_ascii_engine = lazy_import('word_manifold.visualization.engines.ascii')
get_ascii_renderer = lazy_import('word_manifold.visualization.renderers.ascii')
get_audiovis = lazy_import('word_manifold.visualization.audiovis_viewer')

# Progress bar support (lightweight)
try:
    from tqdm import tqdm
except ImportError:
    # Fallback progress bar if tqdm not available
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, *args):
            pass

# Server configuration
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 5000
DEFAULT_TIMEOUTS = {
    'server': 15,      # Default server request timeout
    'health': 5,       # Health check timeout
    'shutdown': 10,    # Graceful shutdown timeout
    'startup': 30,     # Server startup timeout
    'process': 5,      # Process operation timeout
    'websocket': 60,   # WebSocket connection timeout
}
SERVER_URL = f'http://{DEFAULT_HOST}:{DEFAULT_PORT}'

# Default configuration
DEFAULT_CONFIG = {
    'server': {
        'host': DEFAULT_HOST,
        'port': DEFAULT_PORT,
        'timeout': DEFAULT_TIMEOUTS['server'],
        'max_workers': 4,
        'request_queue_size': 100,
        'max_connections': 1000,
        'shutdown_timeout': DEFAULT_TIMEOUTS['shutdown'],
    },
    'security': {
        'enable_cors': True,  # Enable CORS by default
        'allowed_origins': ['*'],  # Allow all origins by default
        'require_auth': False,  # Disable authentication requirement
    },
    'logging': {
        'level': 'INFO',
        'file': 'server.log',
        'max_size': '10MB',
        'backup_count': 5,
    }
}

# Retry configuration
try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_log,
        after_log,
    )
    RETRY_ENABLED = True
except ImportError:
    logger.warning("tenacity not found, retry functionality disabled")
    RETRY_ENABLED = False
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

def validate_server_config(config: Dict) -> Tuple[bool, List[str]]:
    """Validate server configuration."""
    errors = []
    
    # Validate server section
    server = config.get('server', {})
    if not isinstance(server.get('port', 0), int) or not 0 < server.get('port', 0) < 65536:
        errors.append("Port must be an integer between 1 and 65535")
    if not isinstance(server.get('timeout', 0), (int, float)) or server.get('timeout', 0) <= 0:
        errors.append("Timeout must be a positive number")
    
    # Validate security section
    security = config.get('security', {})
    if not isinstance(security.get('enable_cors', False), bool):
        errors.append("enable_cors must be a boolean")
    if not isinstance(security.get('allowed_origins', []), list):
        errors.append("allowed_origins must be a list")
        
    return len(errors) == 0, errors

def get_server_process(port: int) -> Optional['psutil.Process']:
    """Find server process by port."""
    if psutil is None:
        logger.warning("psutil not available, cannot detect server process")
        return None
        
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.net_connections(kind='inet'):
                    if conn.laddr.port == port and conn.status == 'LISTEN':
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
                continue
    except Exception as e:
        logger.warning(f"Error checking server process: {e}")
        return None
    return None

def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from file or return defaults."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
            # Merge with defaults
            return {**DEFAULT_CONFIG, **user_config}
    return DEFAULT_CONFIG

# Server configuration
config = load_config(Path('word_manifold.yaml'))
DEFAULT_HOST = config['server']['host']
DEFAULT_PORT = config['server']['port']
SERVER_URL = f'http://{DEFAULT_HOST}:{DEFAULT_PORT}'

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError))
)
def handle_server_request(method: str, endpoint: str, data: Dict[str, Any] = None, timeout: int = DEFAULT_TIMEOUTS['server']) -> Dict[str, Any]:
    """Handle server request with proper error handling and retries."""
    try:
        if method.upper() == 'GET':
            response = requests.get(f'{SERVER_URL}{endpoint}', timeout=timeout)
        else:
            response = requests.post(f'{SERVER_URL}{endpoint}', json=data, timeout=timeout)
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout as exc:
        raise click.ClickException("Request timed out") from exc
    except requests.exceptions.ConnectionError as exc:
        raise click.ClickException("Connection error") from exc
    except requests.exceptions.HTTPError as exc:
        error_msg = f"Server returned {exc.response.status_code}"
        try:
            error_msg = response.json().get('error', error_msg)
        except:
            pass
        raise click.ClickException(f"Server error: {error_msg}") from exc
    except Exception as e:
        raise click.ClickException(f"Unexpected error: {str(e)}") from e

def ensure_server_running():
    """Start visualization server if not running."""
    try:
        response = requests.get(f'{SERVER_URL}/health', timeout=DEFAULT_TIMEOUTS['health'])
        if response.status_code == 200:
            return
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        logger.info("Starting visualization server...")
        
        # Import server module only when needed
        try:
            from word_manifold.visualization.server import run_server
        except ImportError:
            raise click.ClickException("Server module not found. Please install with 'pip install word-manifold[server]'")
        
        # Start server in background thread
        server_thread = threading.Thread(
            target=run_server,
            args=(DEFAULT_HOST, DEFAULT_PORT),
            daemon=True
        )
        server_thread.start()
        
        # Wait for server to start
        max_retries = DEFAULT_TIMEOUTS['startup'] // DEFAULT_TIMEOUTS['health']
        for i in range(max_retries):
            try:
                time.sleep(1)
                response = requests.get(f'{SERVER_URL}/health', timeout=DEFAULT_TIMEOUTS['health'])
                if response.status_code == 200:
                    logger.info("Visualization server is ready")
                    return
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if i == max_retries - 1:
                    logger.error("Failed to start visualization server")
                    raise click.ClickException("Could not start visualization server")
                logger.info(f"Waiting for server to start (attempt {i+1}/{max_retries})...")

def with_progress(desc: str = None):
    """Progress bar decorator for long-running operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tqdm(desc=desc or func.__name__, leave=True) as pbar:
                result = func(*args, **kwargs)
                pbar.update(1)
                return result
        return wrapper
    return decorator

def get_retry_decorator():
    """Get the appropriate retry decorator based on availability."""
    if not RETRY_ENABLED:
        return lambda x: x  # No-op decorator
        
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError)),
        before=before_log(logger, logging.DEBUG),
        after=before_log(logger, logging.DEBUG),
    )

@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
@click.option('--config', type=click.Path(exists=True, path_type=Path), help='Path to configuration file')
def cli(debug, config):
    """Word Manifold visualization and analysis tools."""
    # Print package location to verify we're using local source
    import word_manifold
    logger.info(f"Using word-manifold package from: {Path(word_manifold.__file__).parent}")
    
    # Set logging level based on debug flag
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    # Propagate to other loggers
    logging.getLogger('word_manifold').setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Load configuration if provided
    if config:
        global SERVER_URL
        new_config = load_config(config)
        SERVER_URL = f"http://{new_config['server']['host']}:{new_config['server']['port']}"

@cli.group()
def server():
    """Server management commands."""
    pass

@server.command()
@click.option('--host', default=DEFAULT_HOST, help='Server host address')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
@click.option('--remote/--local', default=False, help='Run in remote or local mode')
@click.option('--workers', default=1, type=int, help='Number of worker processes')
@click.option('--daemon/--no-daemon', default=False, help='Run server as daemon process')
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
def start(host: str, port: int, remote: bool, workers: int, daemon: bool, debug: bool):
    """Start the visualization server."""
    try:
        # Set logging level based on debug flag
        if debug:
            logger.setLevel(logging.DEBUG)
            logging.getLogger('word_manifold').setLevel(logging.DEBUG)
            logging.getLogger('flask').setLevel(logging.DEBUG)
        
        # Check if server is already running
        if psutil is not None:
            existing_proc = get_server_process(port)
            if existing_proc:
                logger.info(f"Server already running on port {port} (PID: {existing_proc.pid})")
                return
        
        # Import server modules only when needed
        try:
            if remote:
                from word_manifold.visualization.remote_server import run_remote_server
                logger.info(f"Starting remote server on {host}:{port} with {workers} workers")
                run_remote_server(host=host, port=port, workers=workers, daemon=daemon)
            else:
                from word_manifold.visualization.server import run_server
                logger.info(f"Starting local server on {host}:{port}")
                # Update global server URL for local usage
                global SERVER_URL
                SERVER_URL = f"http://{host}:{port}"
                run_server(host=host, port=port, debug=debug)
        except ImportError as e:
            raise click.ClickException(f"Server modules not found. Please install with 'pip install word-manifold[server]': {e}")
            
    except Exception as e:
        logger.error("Failed to start server", exc_info=e)
        raise click.ClickException(f"Failed to start server: {str(e)}")

@server.command()
@click.option('--host', default=DEFAULT_HOST, help='Server host address')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
@click.option('--config', type=click.Path(exists=True, path_type=Path), help='Path to server configuration file')
def status(host: str, port: int, config: Optional[Path]):
    """Check detailed server status."""
    url = f"http://{host}:{port}"
    
    try:
        # Check basic connectivity
        health_response = requests.get(f"{url}/health", timeout=DEFAULT_TIMEOUTS['health'])
        if health_response.status_code != 200:
            click.echo(click.style("✗ Server is not healthy", fg="red"))
            return
            
        # Get process information
        proc = get_server_process(port)
        if proc:
            try:
                # Server stats
                memory = proc.memory_info()
                cpu_percent = proc.cpu_percent(interval=1.0)
                
                click.echo(click.style("✓ Server is running", fg="green"))
                click.echo("\nServer Information:")
                click.echo(f"  PID: {proc.pid}")
                click.echo(f"  Status: {proc.status()}")
                click.echo(f"  Uptime: {time.time() - proc.create_time():.1f} seconds")
                click.echo(f"  CPU Usage: {cpu_percent:.1f}%")
                click.echo(f"  Memory Usage: {memory.rss / 1024 / 1024:.1f} MB")
                
                # Get server metrics
                try:
                    metrics = requests.get(f"{url}/metrics", timeout=DEFAULT_TIMEOUTS['server'])
                    click.echo("\nServer Metrics:")
                    click.echo(f"  Active Connections: {metrics.get('active_connections', 'N/A')}")
                    click.echo(f"  Requests/sec: {metrics.get('requests_per_second', 'N/A')}")
                    click.echo(f"  Average Response Time: {metrics.get('avg_response_time', 'N/A')} ms")
                    click.echo(f"  Error Rate: {metrics.get('error_rate', 'N/A')}%")
                except Exception as e:
                    logger.debug(f"Could not fetch metrics: {e}")
            except psutil.Error as e:
                logger.warning(f"Could not get process stats: {e}")
                click.echo(click.style("✓ Server is running but stats unavailable", fg="yellow"))
                
            # Configuration
            if config:
                with open(config) as f:
                    cfg = yaml.safe_load(f)
                is_valid, errors = validate_server_config(cfg)
                click.echo("\nConfiguration Status:")
                if is_valid:
                    click.echo(click.style("  ✓ Configuration is valid", fg="green"))
                else:
                    click.echo(click.style("  ✗ Configuration has errors:", fg="red"))
                    for error in errors:
                        click.echo(f"    - {error}")
        else:
            click.echo(click.style("✗ Server process not found", fg="red"))
            
    except requests.exceptions.RequestException:
        click.echo(click.style("✗ Server is not responding", fg="red"))
        # Check if process exists but is not responding
        proc = get_server_process(port)
        if proc:
            try:
                click.echo("\nProcess Information (Not Responding):")
                click.echo(f"  PID: {proc.pid}")
                click.echo(f"  Status: {proc.status()}")
                click.echo("Consider using 'word-manifold server restart' to restart the server")
            except psutil.Error:
                click.echo("Process information unavailable")
    except Exception as e:
        raise click.ClickException(f"Error checking server status: {str(e)}")

@server.command()
@click.option('--host', default=DEFAULT_HOST, help='Server host address')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
@click.option('--force', is_flag=True, help='Force stop without graceful shutdown')
def stop(host: str, port: int, force: bool):
    """Stop the server with graceful shutdown."""
    url = f"http://{host}:{port}/shutdown"
    proc = get_server_process(port)
    
    try:
        if not force:
            # Try graceful shutdown first
            response = requests.post(url, timeout=DEFAULT_TIMEOUTS['shutdown'])
            if response.status_code == 200:
                click.echo(click.style("✓ Server stopped gracefully", fg="green"))
                return
                
        # Force stop if requested or graceful shutdown failed
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=DEFAULT_TIMEOUTS['process'])  # Wait for process to terminate
                click.echo(click.style("✓ Server stopped", fg="yellow"))
            except psutil.TimeoutExpired:
                proc.kill()  # Force kill if terminate didn't work
                click.echo(click.style("✓ Server forcefully killed", fg="red"))
        else:
            click.echo(click.style("✗ Server is not running", fg="yellow"))
            
    except requests.exceptions.RequestException:
        if proc:
            proc.kill()
            click.echo(click.style("✓ Server forcefully killed", fg="red"))
        else:
            click.echo(click.style("✗ Server is not running", fg="yellow"))
    except Exception as e:
        raise click.ClickException(f"Error stopping server: {str(e)}")

@server.command()
@click.option('--host', default=DEFAULT_HOST, help='Server host address')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
def restart(host: str, port: int):
    """Restart the server."""
    try:
        # Stop server
        click.echo("Stopping server...")
        stop.callback(host=host, port=port, force=False)
        
        # Wait for port to be available
        max_wait = 10
        for i in range(max_wait):
            if not get_server_process(port):
                break
            time.sleep(1)
            
        # Start server
        click.echo("Starting server...")
        start.callback(host=host, port=port, remote=False, workers=1, daemon=False)
        
    except Exception as e:
        raise click.ClickException(f"Error restarting server: {str(e)}")

@cli.command()
@click.option('--interactive/--no-interactive', default=True, help='Enable/disable interactive mode')
@click.option('--output-dir', default='visualizations', help='Output directory for visualizations')
@click.option('--dimensions', default=3, help='Number of dimensions for visualization')
@click.option('--model', default='glove-wiki-gigaword-300', help='model to use for embeddings')
def visualize(interactive, output_dir, dimensions, model):
    """Create visualizations of the word manifold."""
    try:
        # Initialize embeddings and manifold
        embeddings = get_word_embeddings().WordEmbeddings(model_name=model)
        
        # Define example terms
        terms = [
            "light", "darkness", "wisdom", "understanding",
            "beauty", "strength", "mercy", "severity"
        ]
        
        # Load terms into embeddings
        embeddings.load_terms(terms)
        manifold = get_vector_manifold().VectorManifold(embeddings)
        
        # Create output directories
        hypertools_dir = Path(output_dir) / "hypertools"
        shapes_dir = Path(output_dir) / "shapes"
        hypertools_dir.mkdir(parents=True, exist_ok=True)
        shapes_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizers
        hypertools_vis = get_hypertools().HyperToolsVisualizer(
            word_embeddings=embeddings,
            output_dir=str(hypertools_dir),
            interactive=interactive,
            n_dimensions=dimensions
        )
        
        shape_vis = get_shape_vis().ShapeVisualizer()
        
        # Create term trajectories through ritual phases
        term_trajectories = {}
        rules = get_cellular_rules().create_predefined_rules()
        
        for term in terms:
            trajectories = []
            vector = manifold.embeddings.get_embedding(term)
            
            # Initial state
            trajectories.append(vector.copy())
            
            # Apply transformations
            current = vector.copy()
            current_manifold = get_vector_manifold().VectorManifold(embeddings)
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
        # Lazy load dependencies
        WordEmbeddings = get_word_embeddings().WordEmbeddings
        SymbolicVisualizer = get_symbolic_vis().SymbolicVisualizer
        
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
        embeddings = get_word_embeddings().WordEmbeddings()
        visualizer = get_semantic_tree().SemanticTreeVisualizer(embeddings)
        
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
        visualizer = get_shape_vis().ShapeVisualizer()
        
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
        embeddings = get_word_embeddings().WordEmbeddings()
        manifold = get_vector_manifold().VectorManifold(embeddings)
        rules = get_cellular_rules().create_predefined_rules()
        
        # Initialize manifold with text
        for term in text.split():
            vector = embeddings.get_embedding(term)
            if vector is not None:
                manifold.add_cell(term, vector)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Apply rule iteratively and visualize
        current_manifold = manifold
        visualizer = get_hypertools().HyperToolsVisualizer(embeddings, output_dir=str(output_path))
        
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
@click.argument('terms', nargs=-1, required=False)
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
@click.argument('terms', nargs=-1, required=False)
@click.option('--timeframe', '-t', default='1d', help='Timeframe to analyze (e.g. 1h, 1d, 1w, 1m)')
@click.option('--interval', '-i', default='1h', help='Sampling interval (e.g. 1m, 5m, 1h)')
@click.option('--output-dir', default='visualizations/timeseries', help='Output directory for visualizations')
@click.option('--casting-method', type=click.Choice(['yarrow', 'coins', 'bones']), default='bones', help='Method for casting hexagrams')
@click.option('--local/--server', default=False, help='Use local visualization instead of server')
@click.option('--server-url', default=None, help='Remote visualization server URL')
@click.option('--pattern-type', type=click.Choice(['cyclic', 'linear']), default='cyclic', help='Type of temporal pattern to generate')
@click.option('--cloud-provider', type=click.Choice(['s3', 'gcs', 'azure', 'huggingface']), help='Cloud storage provider')
@click.option('--cloud-credentials', type=click.Path(exists=True, path_type=Path), help='Path to cloud credentials file')
@with_progress("Generating time-series visualization")
def timeseries(
    terms: Optional[List[str]],
    timeframe: str,
    interval: str,
    output_dir: str,
    casting_method: str,
    local: bool,
    server_url: Optional[str],
    pattern_type: str,
    cloud_provider: Optional[str],
    cloud_credentials: Optional[Path]
):
    """Generate time-series visualization showing term evolution over time.
    
    If no terms provided, uses I Ching-based temporal analysis.
    Supports cloud storage through various providers for sharing and collaboration.
    Can use either local rendering or a remote visualization server.
    """
    try:
        # Import required modules
        from word_manifold.automata.hexagram_rules import (
            create_hexagram_rules, cast_hexagram, get_default_text,
            CastingMethod, DEFAULT_TEXTS
        )
        from word_manifold.embeddings.word_embeddings import WordEmbeddings
        from word_manifold.visualization.engines.timeseries import TimeSeriesEngine
        from word_manifold.visualization.renderers.timeseries import TimeSeriesRenderer
        
        # Determine server URL
        vis_server_url = server_url or SERVER_URL
        if not local and not vis_server_url:
            raise click.ClickException("Server URL must be provided when using remote visualization")
            
        # Ensure server is running if using local server
        if not local and not server_url:
            ensure_server_running()
        
        # Load cloud credentials if provided
        cloud_creds = None
        if cloud_credentials:
            try:
                with open(cloud_credentials) as f:
                    cloud_creds = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cloud credentials: {e}")
                cloud_provider = None
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings with temporal and I Ching concepts
        embeddings = WordEmbeddings()
        default_terms = [
            # Temporal concepts
            "past", "present", "future", "time", "cycle",
            "moment", "duration", "change", "flow", "rhythm",
            
            # I Ching temporal aspects
            "transformation", "movement", "stillness",
            "becoming", "returning", "advancing", "retreating",
            
            # Natural cycles
            "dawn", "noon", "dusk", "midnight",
            "spring", "summer", "autumn", "winter",
            
            # Metaphysical time concepts
            "eternal", "temporal", "cyclical", "linear",
            "beginning", "end", "renewal", "decay"
        ]
        embeddings.load_terms(default_terms)
        
        # Initialize visualization components
        engine = TimeSeriesEngine(embeddings)
        renderer = TimeSeriesRenderer(
            cloud_provider=cloud_provider,
            cloud_credentials=cloud_creds
        )
        
        hexagram_data = None
        # If no terms provided, use I Ching-based analysis
        if not terms:
            # Cast hexagram to determine temporal focus
            method = CastingMethod(casting_method)
            hexagram, changing_lines = cast_hexagram(method)
            
            # Select terms based on hexagram properties
            lower_trigram, upper_trigram = hexagram.get_trigrams()
            
            # Map trigrams to temporal aspects
            trigram_time_mapping = {
                'heaven': ["future", "potential", "aspiration"],
                'earth': ["past", "foundation", "memory"],
                'thunder': ["sudden", "initiative", "awakening"],
                'water': ["flow", "transition", "adaptation"],
                'mountain': ["stillness", "presence", "stability"],
                'lake': ["reflection", "clarity", "insight"],
                'fire': ["transformation", "awareness", "illumination"],
                'wind': ["gradual", "penetration", "subtle"]
            }
            
            # Get temporal terms from trigrams
            terms = []
            for trigram in [lower_trigram, upper_trigram]:
                if str(trigram) in trigram_time_mapping:
                    terms.extend(trigram_time_mapping[str(trigram)])
            
            # Add changing line influences
            if changing_lines:
                terms.extend(["transformation", "change", "becoming"])
            
            logger.info(f"Using I Ching-derived temporal terms: {', '.join(terms)}")
            
            # Store hexagram data for insights
            hexagram_data = {
                'hexagram': hexagram,
                'changing_lines': changing_lines,
                'trigram_mapping': trigram_time_mapping
            }
        
        # Load terms into embeddings if not already loaded
        embeddings.load_terms(terms)
        
        # Process data using engine
        data = {
            'terms': terms if terms else default_terms,
            'timeframe': timeframe,
            'interval': interval,
            'pattern_type': pattern_type,
            'embeddings': [embeddings.get_embedding(t).tolist() for t in (terms if terms else default_terms)],
            'hexagram_data': hexagram_data
        }
        
        try:
            if local:
                # Use local renderer with cloud upload if configured
                output_file = renderer.render_local(
                    data,
                    output_path,
                    upload_to_cloud=bool(cloud_provider)
                )
            else:
                # Try server-based visualization with cloud storage
                response = renderer.render_server(
                    data,
                    vis_server_url,
                    endpoint='/api/v1/visualize/timeseries',
                    store_in_cloud=bool(cloud_provider)
                )
                
                if response.get('error'):
                    raise click.ClickException(response['error'])
                    
                # Save visualization
                import base64
                img_data = base64.b64decode(response['image'])
                output_file = output_path / f"timeseries_{timeframe}.{response['format']}"
                output_file.write_bytes(img_data)
                
                # Store any insights returned from server
                if response.get('insights'):
                    insights_file = output_path / 'insights.json'
                    with open(insights_file, 'w') as f:
                        json.dump(response['insights'], f, indent=2)
                
        except Exception as e:
            if not local and not server_url:
                logger.warning(f"Server visualization failed: {e}. Falling back to local visualization...")
                output_file = renderer.render_local(data, output_path)
            else:
                raise
        
        # Generate insights with cloud storage if configured
        if local or not response.get('insights'):
            renderer.render_insights(
                data,
                hexagram_data,
                output_path,
                upload_to_cloud=bool(cloud_provider)
            )
        
        # Display results
        click.echo(f"Time-series visualization saved to {output_file}")
        
        # Display cloud storage information if available
        insights = renderer.get_insights()
        cloud_info = [i for i in insights if "Cloud Storage Information" in i]
        if cloud_info:
            click.echo("\nCloud Storage Information:")
            for info in cloud_info:
                click.echo(info)
        
    except Exception as e:
        logger.error("Error in time-series visualization", exc_info=e)
        raise click.ClickException(str(e)) from e
    finally:
        # Ensure proper cleanup of embeddings
        try:
            embeddings.term_manager.shutdown()
        except:
            pass

@cli.command()
@click.argument('text', required=False)
@click.option('--output-dir', default='visualizations/hexagrams', help='Output directory for hexagram visualizations')
@click.option('--casting-method', type=click.Choice(['yarrow', 'coins', 'bones']), default='bones', help='Method for casting hexagrams')
def hexagrams(text: Optional[str], output_dir: str, casting_method: str):
    """Generate hexagram-based visualizations for text transformations. If no text is provided, casts the oracle bones."""
    try:
        # Import required modules
        from word_manifold.automata.hexagram_rules import (
            create_hexagram_rules, cast_hexagram, get_default_text,
            CastingMethod
        )
        from word_manifold.visualization.hexagram_visualizer import HexagramVisualizer
        from word_manifold.embeddings.word_embeddings import WordEmbeddings
        from word_manifold.manifold.vector_manifold import VectorManifold
        from word_manifold.visualization.shape_visualizer import ShapeVisualizer
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings with default sacred terms and Plato's Cave concepts
        embeddings = WordEmbeddings()
        default_terms = [
            # Traditional sacred terms
            "light", "dark", "above", "below", "spirit", "matter",
            "wisdom", "understanding", "mercy", "severity",
            "beauty", "foundation", "kingdom", "crown",
            
            # Plato's Cave concepts
            "shadow", "reality", "illusion", "truth",
            "perception", "knowledge", "enlightenment", "ignorance",
            "chains", "freedom", "cave", "sun",
            
            # Parallel concepts from I Ching
            "change", "transformation", "movement", "stillness",
            "heaven", "earth", "mountain", "lake",
            
            # Hermetic parallels
            "as_above", "so_below", "within", "without",
            "mind", "manifestation", "vibration", "polarity"
        ]
        embeddings.load_terms(default_terms)
        
        # Initialize components
        manifold = VectorManifold(embeddings)
        rules = create_hexagram_rules()
        visualizer = HexagramVisualizer()
        shape_vis = ShapeVisualizer()
        
        # If no text provided, cast hexagram and use default text
        if not text:
            # Cast hexagram using specified method
            method = CastingMethod(casting_method)
            hexagram, changing_lines = cast_hexagram(method)
            
            # Get default text
            text = get_default_text()
            
            # Create special visualization for oracle casting
            save_path = output_path / f"oracle_casting_{hexagram.number}.png"
            visualizer.draw_transformation(
                hexagram,
                hexagram.get_nuclear_hexagram(),
                changing_lines,
                str(save_path)
            )
            logger.info(f"Cast hexagram {hexagram.number} ({hexagram.name}) with text: {text}")
            
            # Create Plato's Cave parallel visualization
            cave_text = """
            As shadows dance upon the cave wall,
            So do the changing lines reveal truth.
            What seems fixed is but illusion,
            As above in heaven, so below in earth.
            Through transformation comes enlightenment,
            As the prisoner turns to face the sun.
            """
            
            # Generate shape field visualization showing the relationship
            # between shadows (hexagrams) and reality (transformations)
            shape_vis.create_shape_field(
                cave_text,
                chunk_size=1  # Process one sentence at a time
            )
            
            # Create comparative visualization of different levels of reality
            reality_levels = [
                "The shadows on the wall are like our limited perceptions",
                "The objects casting shadows are like the true forms",
                "The sun outside is like the eternal principles that generate all change"
            ]
            shape_vis.create_comparative_visualization(
                texts=reality_levels,
                labels=["Shadows", "Objects", "Source"]
            )
        
        # Add text to manifold
        for term in text.split():
            vector = embeddings.get_embedding(term)
            if vector is not None:
                manifold.add_cell(term, vector)
        
        # Apply and visualize each hexagram rule
        for rule_name, rule in rules.items():
            # Skip rules that don't match the text's semantic properties
            if not rule.parameters.numerological_weights:
                continue
                
            # Visualize rule application
            save_path = output_path / f"{rule_name}.png"
            visualizer.visualize_rule_application(
                rule,
                manifold,
                str(save_path)
            )
            
        logger.info(f"Hexagram visualizations saved to {output_dir}")
        
    except Exception as e:
        logger.error("Error in hexagram visualization", exc_info=e)
        raise click.ClickException(str(e))

@cli.command()
@click.argument('pattern_type', type=click.Choice(['mandala', 'wave', 'field', 'blend']))
@click.option('--width', default=80, help='Width of the pattern')
@click.option('--height', default=40, help='Height of the pattern')
@click.option('--radius', default=15, help='Radius for mandala patterns')
@click.option('--complexity', default=2.0, help='Pattern complexity (higher = more intricate)')
@click.option('--density', default=0.6, help='Pattern density (0.0 to 1.0)')
@click.option('--style', type=click.Choice([
    'mystical', 'geometric', 'natural', 'runic', 'mathematical',
    'cyberpunk', 'ethereal', 'ancient', 'digital', 'cosmic', 'braille',
    'rainbow_pulse', 'acid', 'plasma', 'fractal', 'quantum', 'neural', 'dream',
    'spectrum', 'intensity', 'frequency'
]), default='mystical', help='Visual style to use')
@click.option('--field-type', type=click.Choice([
    'organic', 'crystalline', 'flowing', 'chaotic'
]), default='organic', help='Type of field pattern')
@click.option('--wave-type', type=click.Choice([
    'sine', 'square', 'triangle', 'sawtooth'
]), default='sine', help='Type of wave pattern')
@click.option('--layers', default=3, help='Number of mandala layers')
@click.option('--symmetry', default=8, help='Number of mandala symmetry axes')
@click.option('--interference/--no-interference', default=False, help='Add wave interference patterns')
@click.option('--animate/--no-animate', default=False, help='Create animation')
@click.option('--frames', default=60, help='Number of animation frames')
@click.option('--frame-delay', default=0.05, help='Delay between animation frames')
@click.option('--theme', type=click.Choice([
    'none', 'fire', 'water', 'earth', 'air', 'cosmic', 'mystic',
    'rainbow', 'sunset', 'forest', 'ocean', 'ethereal', 'void'
]), default='none', help='Color theme to use')
@click.option('--output-dir', default='visualizations/ascii', help='Output directory')
@click.option('--blend-with', type=click.Choice(['none', 'mandala', 'wave', 'field']), default='none', help='Pattern to blend with')
@click.option('--blend-mode', type=click.Choice([
    'overlay', 'add', 'multiply', 'screen', 'difference'
]), default='overlay', help='Blend mode to use')
@click.option('--blend-alpha', default=0.7, help='Blend opacity (0.0 to 1.0)')
@click.option('--blend-width', default=None, type=int, help='Width for blended pattern')
@click.option('--blend-height', default=None, type=int, help='Height for blended pattern')
@click.option('--output-format', type=click.Choice(['text', 'png', 'gif']), default='text', help='Output format')
@click.option('--image-scale', default=1.0, help='Scale factor for image output')
@click.option('--image-glow/--no-image-glow', default=True, help='Add glow effect to image')
@click.option('--image-blur/--no-image-blur', default=True, help='Add blur effect to image')
@click.option('--image-enhance/--no-image-enhance', default=True, help='Enhance image contrast/brightness')
@click.option('--background-color', default=None, help='Background color in hex format (e.g. #000000)')
@click.option('--effect', type=click.Choice([
    'fractals', 'neural', 'quantum', 'flow_field', 'reaction_diffusion'
]), multiple=True, help='Add visual effects (can specify multiple)')
@click.option('--audio', type=click.Path(exists=True), help='Audio file for reactive effects')
@click.option('--audio-intensity', default=1.0, help='Intensity of audio reactive effects (0.0 to 2.0)')
@click.option('--message', help='Subliminal message to embed')
@click.option('--symbol', type=click.Choice([
    'protection', 'wisdom', 'power', 'harmony', 'transformation',
    'unity', 'transcendence', 'infinity', 'consciousness', 'enlightenment'
]), multiple=True, help='Symbols to embed (can specify multiple)')
@click.option('--sigil', type=click.Choice([
    'focus', 'clarity', 'energy', 'peace', 'growth'
]), multiple=True, help='Sigils to embed (can specify multiple)')
def ascii(
    pattern_type: str,
    width: int,
    height: int,
    radius: int,
    complexity: float,
    density: float,
    style: str,
    field_type: str,
    wave_type: str,
    layers: int,
    symmetry: int,
    interference: bool,
    animate: bool,
    frames: int,
    frame_delay: float,
    theme: str,
    output_dir: str,
    blend_with: str,
    blend_mode: str,
    blend_alpha: float,
    blend_width: Optional[int],
    blend_height: Optional[int],
    output_format: str,
    image_scale: float,
    image_glow: bool,
    image_blur: bool,
    image_enhance: bool,
    background_color: Optional[str],
    effect: Tuple[str, ...],
    audio: Optional[str],
    audio_intensity: float,
    message: Optional[str],
    symbol: Tuple[str, ...],
    sigil: Tuple[str, ...]
):
    """Generate rich ASCII art patterns and animations with audio reactivity and subliminal effects.
    
    Examples:
    \b
    # Create an audio-reactive mandala that pulses with the beat
    word-manifold ascii mandala --style spectrum --output-format gif --audio music.mp3
    
    # Generate a mystical pattern with embedded sigils and symbols
    word-manifold ascii field --style mystical --sigil focus --sigil energy --symbol wisdom
    
    # Create a quantum pattern with subliminal message
    word-manifold ascii wave --style quantum --message "expand consciousness" --effect quantum
    
    # Make an intense audio-reactive animation with multiple effects
    word-manifold ascii mandala --style acid --audio trance.mp3 --audio-intensity 2.0 \\
        --effect fractals --effect flow_field --symbol enlightenment --symbol transcendence
    """
    try:
        # Initialize components
        engine = get_ascii_engine().ASCIIEngine()
        ascii_renderer = get_ascii_renderer().ASCIIRenderer()
        
        # Initialize image renderer if needed
        image_renderer = None
        if output_format in ['png', 'gif']:
            from word_manifold.visualization.renderers.image_renderer import ImageRenderer
            image_renderer = ImageRenderer(font_size=20)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate primary pattern
        if pattern_type == 'mandala':
            pattern = engine.generate_mandala(
                radius=radius,
                complexity=complexity,
                style=style,
                layers=layers,
                symmetry=symmetry
            )
        elif pattern_type == 'wave':
            pattern = engine.generate_field(
                width=width,
                height=height,
                density=density,
                style=style,
                pattern_type='flowing'
            )
            engine.add_wave_pattern(
                pattern,
                frequency=0.1,
                phase=0,
                wave_type=wave_type,
                interference=interference
            )
            if interference:
                engine.add_wave_pattern(
                    pattern,
                    frequency=0.15,
                    phase=math.pi/3,
                    wave_type=wave_type,
                    interference=True
                )
        elif pattern_type == 'field':
            pattern = engine.generate_field(
                width=width,
                height=height,
                density=density,
                style=style,
                pattern_type=field_type
            )
        
        # Handle blending if requested
        if blend_with != 'none':
            # Create blend pattern
            if blend_with == 'mandala':
                blend_pattern = engine.generate_mandala(
                    radius=min(width, height) // 4,
                    complexity=complexity,
                    style=style,
                    layers=layers,
                    symmetry=symmetry
                )
            elif blend_with == 'wave':
                blend_pattern = engine.generate_field(
                    width=width,
                    height=height,
                    density=density,
                    style=style,
                    pattern_type='flowing'
                )
                engine.add_wave_pattern(
                    blend_pattern,
                    frequency=0.15,
                    phase=math.pi/3,
                    wave_type=wave_type,
                    interference=interference
                )
            else:  # field
                blend_pattern = engine.generate_field(
                    width=width,
                    height=height,
                    density=density,
                    style=style,
                    pattern_type=field_type
                )
            
            # Determine blend dimensions
            blend_width = blend_width or max(pattern.width, blend_pattern.width)
            blend_height = blend_height or max(pattern.height, blend_pattern.height)
            
            # Resize patterns if needed
            if pattern.width != blend_width or pattern.height != blend_height:
                pattern = engine.resize_pattern(pattern, blend_width, blend_height)
            if blend_pattern.width != blend_width or blend_pattern.height != blend_height:
                blend_pattern = engine.resize_pattern(blend_pattern, blend_width, blend_height)
            
            # Blend patterns
            pattern = engine.blend_patterns(
                pattern,
                blend_pattern,
                alpha=blend_alpha,
                blend_mode=blend_mode
            )
        
        # Handle output format
        if output_format == 'text':
            # Save ASCII text
            static_path = output_path / f"{pattern_type}_{style}.txt"
            ascii_renderer.save_pattern(
                pattern,
                static_path,
                include_metadata=True
            )
            logger.info(f"Saved ASCII pattern to {static_path}")
            
            if animate:
                frames_path = output_path / f"{pattern_type}_{style}_animation.txt"
                animation_frames = engine.create_animation_frames(pattern, n_frames=frames)
                ascii_renderer.save_animation(animation_frames, frames_path)
                logger.info(f"Saved ASCII animation to {frames_path}")
                
                # Display animation in terminal if supported
                if ascii_renderer.supports_color and theme != 'none':
                    logger.info("\nDisplaying animation (Ctrl+C to stop)...")
                    ascii_renderer.render_animation(
                        animation_frames,
                        frame_delay=frame_delay,
                        theme=theme,
                        loop=True
                    )
            
            # Display static pattern in terminal if theme requested
            elif theme != 'none' and ascii_renderer.supports_color:
                logger.info("\nDisplaying pattern:")
                ascii_renderer.render_pattern(pattern, theme=theme)
                
        else:  # png or gif
            # Convert background color if provided
            bg_color = None
            if background_color:
                bg_color = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            if animate and output_format == 'gif':
                # Create and save animated GIF
                animation_frames = engine.create_animation_frames(pattern, n_frames=frames)
                gif_path = output_path / f"{pattern_type}_{style}.gif"
                
                # Prepare render parameters
                render_params = {
                    'style': style,
                    'scale': image_scale,
                    'duration': int(frame_delay * 1000),
                    'glow': image_glow,
                    'blur': image_blur,
                    'enhance': image_enhance,
                }
                
                # Add effects if specified
                if effect:
                    render_params['psychedelic_effects'] = list(effect)
                
                # Add audio reactivity if specified
                if audio:
                    render_params.update({
                        'audio_file': audio,
                        'audio_intensity': audio_intensity
                    })
                
                # Add subliminal effects if specified
                if message:
                    render_params['subliminal_message'] = message
                if symbol:
                    render_params['subliminal_symbols'] = list(symbol)
                if sigil:
                    render_params['subliminal_sigils'] = list(sigil)
                
                image_renderer.save_animation(
                    animation_frames,
                    str(gif_path),
                    **render_params
                )
                logger.info(f"Saved animated GIF to {gif_path}")
            else:
                # Create and save static image
                img_path = output_path / f"{pattern_type}_{style}.png"
                
                # Prepare render parameters
                render_params = {
                    'style': style,
                    'scale': image_scale,
                    'glow': image_glow,
                    'blur': image_blur,
                    'enhance': image_enhance,
                    'background_color': bg_color,
                }
                
                # Add effects if specified
                if effect:
                    render_params['psychedelic_effects'] = list(effect)
                
                # Add audio reactivity if specified
                if audio:
                    render_params.update({
                        'audio_file': audio,
                        'audio_intensity': audio_intensity
                    })
                
                # Add subliminal effects if specified
                if message:
                    render_params['subliminal_message'] = message
                if symbol:
                    render_params['subliminal_symbols'] = list(symbol)
                if sigil:
                    render_params['subliminal_sigils'] = list(sigil)
                
                image_renderer.save_pattern(
                    pattern,
                    str(img_path),
                    **render_params
                )
                logger.info(f"Saved image to {img_path}")
            
    except Exception as e:
        logger.error("Error generating visualization", exc_info=e)
        raise click.ClickException(str(e))

@cli.command()
@click.option('--host', default=DEFAULT_HOST, help='Server host')
@click.option('--port', default=8765, help='Server port')
@click.option('--width', default=80, help='Visualization width')
@click.option('--height', default=40, help='Visualization height')
@click.option('--fps', default=30, help='Target frames per second')
@click.option('--pattern', type=click.Choice(['wave', 'mandala', 'field']), default='wave', help='Initial pattern style')
@click.option('--color-mode', type=click.Choice(['spectrum', 'intensity', 'frequency']), default='spectrum', help='Initial color mode')
@click.option('--sample-rate', default=44100, help='Audio sample rate')
@click.option('--block-size', default=2048, help='Audio block size')
@click.option('--device', type=int, help='Audio input device ID')
@click.option('--browser/--no-browser', default=True, help='Open browser automatically')
@click.option('--template', type=click.Path(exists=True), help='Custom visualization template')
def audiovis(
    host: str,
    port: int,
    width: int,
    height: int,
    fps: int,
    pattern: str,
    color_mode: str,
    sample_rate: int,
    block_size: int,
    device: int,
    browser: bool,
    template: Optional[Path]
):
    """Launch audio-reactive ASCII visualization.
    
    This command starts a WebSocket server that:
    1. Captures and analyzes audio input in real-time
    2. Generates ASCII art patterns based on audio features
    3. Serves a web interface for visualization
    
    The visualization can be viewed in any modern web browser and includes:
    - Multiple pattern styles (wave, mandala, field)
    - Different color modes (spectrum, intensity, frequency)
    - Real-time audio spectrum display
    - Performance metrics
    
    Example usage:
    \b
    # Start visualization with default settings
    word-manifold audiovis
    
    \b
    # Use specific audio device and custom settings
    word-manifold audiovis --device 1 --pattern mandala --fps 60
    
    \b
    # Use custom visualization template
    word-manifold audiovis --template my_template.html
    """
    try:
        # Check for sounddevice dependency
        try:
            import sounddevice as sd
        except ImportError:
            raise click.ClickException(
                "sounddevice not found. Please install with: pip install sounddevice"
            )
        
        # Initialize audio and visualizer configs
        AudioConfig = get_audiovis().AudioConfig
        VisualizerConfig = get_audiovis().VisualizerConfig
        
        audio_config = AudioConfig(
            sample_rate=sample_rate,
            block_size=block_size,
            device=device
        )
        
        vis_config = VisualizerConfig(
            width=width,
            height=height,
            fps=fps,
            pattern_style=pattern,
            color_mode=color_mode
        )
        
        # Create visualizer
        AudioVisualizer = get_audiovis().AudioVisualizer
        visualizer = AudioVisualizer(
            audio_config=audio_config,
            vis_config=vis_config
        )
        
        # Start visualization in background thread
        vis_thread = threading.Thread(target=visualizer.start)
        vis_thread.daemon = True
        vis_thread.start()
        
        # Get template path
        if template:
            template_path = Path(template)
        else:
            template_dir = Path(__file__).parent / 'templates'
            template_path = template_dir / 'audiovis.html'
        
        if not template_path.exists():
            logger.error(f"Template not found: {template_path}")
            return
            
        # Create HTTP server for serving template
        from aiohttp import web
        import aiohttp_cors
        
        async def serve_template(request):
            return web.FileResponse(template_path)
            
        async def run_server():
            app = web.Application()
            app.router.add_get('/', serve_template)
            
            # Configure CORS
            cors = aiohttp_cors.setup(app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*"
                )
            })
            
            for route in list(app.router.routes()):
                cors.add(route)
            
            # Start server
            server_url = f'http://{host}:{port}'
            ws_url = f'ws://{host}:{port}'
            
            logger.info(f"Starting visualization server at {server_url}")
            logger.info(f"WebSocket endpoint: {ws_url}")
            
            if browser:
                # Open browser after short delay
                def open_browser():
                    time.sleep(1.5)  # Wait for server to start
                    webbrowser.open(server_url)
                
                browser_thread = threading.Thread(target=open_browser)
                browser_thread.daemon = True
                browser_thread.start()
            
            # Run server
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            
            # Start WebSocket server
            await visualizer.start_server(host=host, port=port)
            
            try:
                # Run forever
                await asyncio.Future()
            except KeyboardInterrupt:
                logger.info("Stopping visualization server...")
            finally:
                visualizer.stop()
                await runner.cleanup()
        
        # Run the async server
        asyncio.run(run_server())
            
    except Exception as e:
        logger.error(f"Error starting visualization: {e}", exc_info=True)
        raise click.ClickException(str(e))

cli.add_command(manifold)
cli.add_command(timeseries)
cli.add_command(server)
cli.add_command(audiovis)

if __name__ == '__main__':
    cli.main(standalone_mode=False) 