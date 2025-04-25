"""CLI command for audio-reactive visualization."""

import click
import logging
import webbrowser
import asyncio
import threading
import time
from pathlib import Path
from ...audiovis_viewer import AudioVisualizer, AudioConfig, VisualizerConfig

logger = logging.getLogger(__name__)

@click.command()
@click.option('--host', default='localhost', help='Server host')
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
    browser: bool
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
    """
    try:
        # Initialize audio and visualizer configs
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
        visualizer = AudioVisualizer(
            audio_config=audio_config,
            vis_config=vis_config
        )
        
        # Start visualization in background thread
        vis_thread = threading.Thread(target=visualizer.start)
        vis_thread.daemon = True
        vis_thread.start()
        
        # Ensure template is available
        template_dir = Path(__file__).parent.parent.parent / 'templates'
        template_path = template_dir / 'audiovis.html'
        
        if not template_path.exists():
            logger.error(f"Template not found: {template_path}")
            return
            
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
        try:
            asyncio.run(visualizer.start_server(host=host, port=port))
        except KeyboardInterrupt:
            logger.info("Stopping visualization server...")
        finally:
            visualizer.stop()
            
    except Exception as e:
        logger.error(f"Error starting visualization: {e}", exc_info=True)
        raise click.ClickException(str(e)) 