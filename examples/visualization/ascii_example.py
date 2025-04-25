"""
ASCII Visualization Example

Demonstrates core ASCII visualization capabilities with minimal dependencies.
"""

import logging
import math
from pathlib import Path

from word_manifold.visualization.engines.ascii import ASCIIEngine
from word_manifold.visualization.renderers.ascii import ASCIIRenderer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run ASCII visualization examples."""
    try:
        # Initialize components
        engine = ASCIIEngine()
        renderer = ASCIIRenderer()
        
        # Create output directory
        output_dir = Path("visualizations/ascii")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Example 1: Mandala Pattern
        logger.info("Creating mandala pattern...")
        mandala = engine.generate_mandala(
            radius=15,
            complexity=2.0  # Higher complexity = more intricate pattern
        )
        
        # Save static mandala with metadata
        renderer.save_pattern(
            mandala,
            output_dir / "mandala.txt",
            include_metadata=True
        )
        
        # Create rotating animation
        frames = engine.create_animation_frames(mandala, n_frames=60)
        renderer.save_animation(
            frames,
            output_dir / "mandala_animation.txt"
        )
        
        # Example 2: Wave Field
        logger.info("Creating wave field pattern...")
        field = engine.generate_field(
            width=80,
            height=30,
            density=0.6  # Lower density = sparser pattern
        )
        
        # Add multiple wave patterns
        engine.add_wave_pattern(field, frequency=0.1, phase=0)
        engine.add_wave_pattern(field, frequency=0.2, phase=math.pi/4)
        
        # Save field pattern
        renderer.save_pattern(
            field,
            output_dir / "wave_field.txt"
        )
        
        # Example 3: Pattern Blending
        logger.info("Creating blended pattern...")
        # Create a smaller mandala to blend with the field
        small_mandala = engine.generate_mandala(radius=10)
        
        # Blend patterns with 70% opacity for the mandala
        blended = engine.blend_patterns(
            field,
            small_mandala,
            alpha=0.7
        )
        
        # Save blended pattern
        renderer.save_pattern(
            blended,
            output_dir / "blended.txt"
        )
        
        # Display results
        logger.info("\nCreated visualizations:")
        logger.info("- Mandala: %s", output_dir / "mandala.txt")
        logger.info("- Mandala Animation: %s", output_dir / "mandala_animation.txt")
        logger.info("- Wave Field: %s", output_dir / "wave_field.txt")
        logger.info("- Blended Pattern: %s", output_dir / "blended.txt")
        
        # Display animation in terminal if supported
        if renderer.supports_color:
            logger.info("\nDisplaying mandala animation (Ctrl+C to stop)...")
            renderer.render_animation(
                frames,
                frame_delay=0.05,
                color='cyan',
                loop=True
            )
        
    except Exception as e:
        logger.error("Error in ASCII visualization example", exc_info=e)

if __name__ == "__main__":
    main() 