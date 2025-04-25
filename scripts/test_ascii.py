"""
ASCII Visualization Test

Tests enhanced ASCII visualization with rich colors and UTF-8 characters.
Uses parallel processing and debugging utilities.
"""

import logging
import os
from pathlib import Path
import sys
from word_manifold.utils.debug import (
    AsyncTaskManager, DebugContext, log_errors,
    time_it, memory_usage, profile_function
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_ray():
    """Initialize Ray if needed."""
    try:
        import ray
        if not ray.is_initialized():
            # Try to connect to existing Ray instance first
            try:
                ray.init(address='auto', ignore_reinit_error=True)
                logger.info("Connected to existing Ray instance")
            except:
                # Start Ray locally with minimal settings
                ray.init(
                    num_cpus=1,  # Use single CPU for testing
                    include_dashboard=False,  # Disable dashboard
                    ignore_reinit_error=True,
                    logging_level=logging.WARNING,  # Reduce Ray logging
                    _temp_dir=os.path.expanduser("~/ray_tmp")  # Use home directory for temp files
                )
                logger.info("Started new Ray instance")
    except ImportError:
        logger.warning("Ray not available, running in local mode")
    except Exception as e:
        logger.warning(f"Could not initialize Ray: {e}, running in local mode")

def import_visualization():
    """Import visualization modules with proper error handling."""
    with DebugContext("Importing modules"):
        try:
            from word_manifold.visualization.engines.ascii import ASCIIEngine
            from word_manifold.visualization.renderers.ascii import ASCIIRenderer
            from word_manifold.types.patterns import Mandala, Field
            return ASCIIEngine, ASCIIRenderer, Mandala, Field
        except ImportError as e:
            logger.error(f"Failed to import visualization modules: {e}")
            sys.exit(1)

@log_errors
@time_it
def test_mandalas(task_mgr: AsyncTaskManager):
    """Test mandala patterns with different styles."""
    logger.info("Testing mandala patterns...")
    
    ASCIIEngine, ASCIIRenderer, _, _ = import_visualization()
    engine = ASCIIEngine()
    renderer = ASCIIRenderer()
    
    try:
        # Test different styles in parallel
        styles = ['mystical', 'geometric', 'natural']
        themes = ['mystic', 'fire', 'water']
        futures = []
        
        for style, theme in zip(styles, themes):
            # Submit mandala generation as async task
            future = task_mgr.submit_task(
                f"mandala_{style}",
                engine.generate_mandala,
                radius=15,
                complexity=2.0,
                style=style
            )
            futures.append((future, style, theme))
        
        # Render results as they complete
        for future, style, theme in futures:
            mandala = task_mgr.get_result(f"mandala_{style}")
            print(f"\n{style.title()} Mandala with {theme} theme:")
            print(renderer.render_pattern(
                mandala,
                theme=theme,
                add_border=True
            ))
            
    except Exception as e:
        logger.error(f"Error in mandala test: {e}")
        raise

@log_errors
@time_it
def test_wave_patterns(task_mgr: AsyncTaskManager):
    """Test wave patterns with different styles."""
    logger.info("Testing wave patterns...")
    
    ASCIIEngine, ASCIIRenderer, _, _ = import_visualization()
    engine = ASCIIEngine()
    renderer = ASCIIRenderer()
    
    try:
        # Test different wave types in parallel
        wave_types = ['sine', 'square', 'triangle']
        themes = ['water', 'earth', 'air']
        futures = []
        
        for wave_type, theme in zip(wave_types, themes):
            # Create field
            field = engine.generate_field(width=50, height=20, style='natural')
            
            # Submit wave pattern generation as async task
            future = task_mgr.submit_task(
                f"wave_{wave_type}",
                engine.add_wave_pattern,
                field,
                frequency=0.1,
                wave_type=wave_type,
                amplitude=1.0
            )
            futures.append((future, field, wave_type, theme))
        
        # Render results as they complete
        for future, field, wave_type, theme in futures:
            task_mgr.get_result(f"wave_{wave_type}")  # Wait for pattern generation
            print(f"\n{wave_type.title()} Wave with {theme} theme:")
            print(renderer.render_pattern(
                field,
                theme=theme,
                add_border=True
            ))
            
    except Exception as e:
        logger.error(f"Error in wave pattern test: {e}")
        raise

@log_errors
@time_it
def test_pattern_blending(task_mgr: AsyncTaskManager):
    """Test pattern blending with different modes."""
    logger.info("Testing pattern blending...")
    
    ASCIIEngine, ASCIIRenderer, _, _ = import_visualization()
    engine = ASCIIEngine()
    renderer = ASCIIRenderer()
    
    try:
        # Create base patterns with matching dimensions
        size = 31  # Odd number for mandala centering
        radius = (size - 1) // 2
        
        # Create field pattern
        field = engine.generate_field(width=size, height=size, style='natural')
        engine.add_wave_pattern(field, frequency=0.2, wave_type='sine')
        
        # Create mandala pattern
        mandala = engine.generate_mandala(radius=radius, style='mystical')
        
        # Test different blend modes in parallel
        blend_modes = ['overlay', 'add', 'multiply']
        themes = ['cosmic', 'fire', 'mystic']
        futures = []
        
        for mode, theme in zip(blend_modes, themes):
            future = task_mgr.submit_task(
                f"blend_{mode}",
                engine.blend_patterns,
                field,
                mandala,
                alpha=0.7,
                blend_mode=mode
            )
            futures.append((future, mode, theme))
        
        # Render results as they complete
        for future, mode, theme in futures:
            blended = task_mgr.get_result(f"blend_{mode}")
            print(f"\nBlend Mode '{mode}' with {theme} theme:")
            print(renderer.render_pattern(
                blended,
                theme=theme,
                add_border=True
            ))
            
    except Exception as e:
        logger.error(f"Error in blending test: {e}")
        raise

@log_errors
@time_it
def test_animations():
    """Test animations with different effects."""
    logger.info("Testing animations...")
    
    ASCIIEngine, ASCIIRenderer, _, _ = import_visualization()
    engine = ASCIIEngine()
    renderer = ASCIIRenderer()
    
    try:
        # Create different animation types
        print("\nAnimation Types Available:")
        print("1. Rotating Mandala (cosmic theme)")
        print("2. Pulsing Mandala (fire theme)")
        print("3. Wave Field (water theme)")
        
        choice = input("\nSelect animation type (1-3) or press Enter for all: ").strip()
        
        if not choice or choice == '1':
            with DebugContext("Rotating Mandala"):
                mandala = engine.generate_mandala(radius=10, style='mystical')
                frames = engine.create_animation_frames(
                    mandala,
                    n_frames=30,
                    animation_type='rotate'
                )
                
                print("\nRotating Mandala Animation (Press Ctrl+C to stop)...")
                renderer.render_animation(
                    frames,
                    frame_delay=0.1,
                    theme='cosmic',
                    add_border=True
                )
            
        if not choice or choice == '2':
            with DebugContext("Pulsing Mandala"):
                mandala = engine.generate_mandala(radius=10, style='geometric')
                frames = engine.create_animation_frames(
                    mandala,
                    n_frames=30,
                    animation_type='pulse'
                )
                
                print("\nPulsing Mandala Animation (Press Ctrl+C to stop)...")
                renderer.render_animation(
                    frames,
                    frame_delay=0.1,
                    theme='fire',
                    add_border=True
                )
            
        if not choice or choice == '3':
            with DebugContext("Wave Field"):
                field = engine.generate_field(width=50, height=20, style='natural')
                frames = engine.create_animation_frames(
                    field,
                    n_frames=30,
                    animation_type='wave'
                )
                
                print("\nWave Field Animation (Press Ctrl+C to stop)...")
                renderer.render_animation(
                    frames,
                    frame_delay=0.1,
                    theme='water',
                    add_border=True
                )
            
    except KeyboardInterrupt:
        print("\nAnimation stopped by user")
    except Exception as e:
        logger.error(f"Error in animation test: {e}")
        raise

@profile_function
def main():
    """Run all tests."""
    try:
        # Create output directory
        output_dir = Path("test_outputs/ascii")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Ray if needed (but don't require it)
        init_ray()
        
        # Create task manager for parallel processing
        task_mgr = AsyncTaskManager(max_workers=4)
        
        try:
            # Show initial memory usage
            mem = memory_usage()
            if mem:
                logger.info(f"Initial memory usage: {mem['rss']:.1f}MB RSS")
            
            # Run tests
            with DebugContext("Running tests", log_level=logging.INFO):
                test_mandalas(task_mgr)
                test_wave_patterns(task_mgr)
                test_pattern_blending(task_mgr)
                test_animations()
            
            # Show final memory usage
            mem = memory_usage()
            if mem:
                logger.info(f"Final memory usage: {mem['rss']:.1f}MB RSS")
                
        finally:
            task_mgr.shutdown()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Cleanup Ray if it was initialized
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown complete")
        except:
            pass

if __name__ == "__main__":
    main() 