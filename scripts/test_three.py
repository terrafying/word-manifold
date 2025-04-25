"""
Three.js Visualization Test

Tests the Three.js visualization system with hyperdimensional projections.
"""

import logging
import os
from pathlib import Path
import sys
import numpy as np
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

def import_visualization():
    """Import visualization modules with proper error handling."""
    with DebugContext("Importing modules"):
        try:
            from word_manifold.visualization.renderers.three_renderer import ThreeRenderer
            from word_manifold.visualization.engines.projection import ProjectionEngine
            return ThreeRenderer, ProjectionEngine
        except ImportError as e:
            logger.error(f"Failed to import visualization modules: {e}")
            sys.exit(1)

@log_errors
@time_it
def test_static_visualization(task_mgr: AsyncTaskManager):
    """Test static 3D visualization."""
    logger.info("Testing static visualization...")
    
    ThreeRenderer, ProjectionEngine = import_visualization()
    
    try:
        # Create test data
        n_points = 100
        n_dims = 5
        points = np.random.randn(n_points, n_dims)
        
        # Create edges between nearby points
        edges = []
        for i in range(n_points):
            distances = np.linalg.norm(points - points[i], axis=1)
            nearest = np.argsort(distances)[1:4]  # 3 nearest neighbors
            edges.extend([(i, j) for j in nearest])
            
        # Add colors based on first three dimensions
        colors = (points[:, :3] + 2) / 4  # Normalize to [0, 1]
        
        # Add sizes based on distance from origin
        sizes = np.linalg.norm(points, axis=1)
        sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 0.5 + 0.1
        
        # Initialize components
        projection = ProjectionEngine(projection_type="stereographic")
        renderer = ThreeRenderer(
            background_color="#000000",
            ambient_intensity=0.4,
            directional_intensity=0.6
        )
        
        # Save static visualization
        output_path = Path("visualizations/three/static.html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        renderer.save_static(
            points=points,
            output_path=output_path,
            edges=edges,
            colors=colors,
            sizes=sizes
        )
        
        logger.info(f"Static visualization saved to {output_path}")
            
    except Exception as e:
        logger.error(f"Error in static visualization test: {e}")
        raise

@log_errors
@time_it
def test_animation(task_mgr: AsyncTaskManager):
    """Test animated dimensional transitions."""
    logger.info("Testing animation...")
    
    ThreeRenderer, ProjectionEngine = import_visualization()
    
    try:
        # Create test data
        n_points = 50
        n_dims = 4
        points = np.random.randn(n_points, n_dims)
        
        # Create edges between nearby points
        edges = []
        for i in range(n_points):
            distances = np.linalg.norm(points - points[i], axis=1)
            nearest = np.argsort(distances)[1:4]
            edges.extend([(i, j) for j in nearest])
            
        # Add colors and sizes
        colors = (points[:, :3] + 2) / 4
        sizes = np.linalg.norm(points, axis=1)
        sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 0.5 + 0.1
        
        # Initialize components
        projection = ProjectionEngine(projection_type="stereographic")
        renderer = ThreeRenderer(
            background_color="#000000",
            ambient_intensity=0.4,
            directional_intensity=0.6
        )
        
        # Create animation
        renderer.create_animation(
            points=points,
            n_steps=30,
            edges=edges,
            colors=colors,
            sizes=sizes
        )
        
        logger.info("Animation server started")
            
    except Exception as e:
        logger.error(f"Error in animation test: {e}")
        raise

@profile_function
def main():
    """Run all tests."""
    mem = memory_usage()
    if isinstance(mem, dict):
        mem = mem.get('rss', 0)
    logger.info(f"Initial memory usage: {mem:.1f}MB RSS")
    
    with DebugContext("Running tests"):
        task_mgr = AsyncTaskManager()
        
        try:
            # Run tests
            test_static_visualization(task_mgr)
            test_animation(task_mgr)
            
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            raise
        finally:
            task_mgr.shutdown()
            
    mem = memory_usage()
    if isinstance(mem, dict):
        mem = mem.get('rss', 0)
    logger.info(f"Final memory usage: {mem:.1f}MB RSS")

if __name__ == '__main__':
    main() 