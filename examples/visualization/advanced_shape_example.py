"""
Advanced Shape Visualizations

This module demonstrates advanced shape visualization capabilities,
including animated transformations, interactive shape fields,
and semantic geometry analysis.
"""

import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from tqdm import tqdm

from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.visualization.shape_visualizer import ShapeVisualizer, ExportConfig
from word_manifold.visualization.engines.shape_engine import ShapeEngine
from word_manifold.visualization.renderers.shape_renderer import ShapeRenderer
from word_manifold.visualization.renderers.interactive import InteractiveRenderer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedShapeVisualization:
    """Advanced shape visualization capabilities."""
    
    def __init__(self, output_dir: str = "visualizations/advanced_shapes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embeddings = WordEmbeddings()
        self.shape_engine = ShapeEngine()
        self.shape_renderer = ShapeRenderer()
        self.interactive_renderer = InteractiveRenderer()
        
        # Configure shape visualizer
        self.visualizer = ShapeVisualizer(
            export_config=ExportConfig(
                output_dir=str(self.output_dir),
                format="mp4",
                save_frames=True,
                frame_format="png",
                dpi=300
            )
        )
    
    def create_shape_field_evolution(
        self,
        text: str,
        n_frames: int = 60,
        duration: float = 10.0,
        interpolation_steps: int = 30,
        add_trails: bool = True
    ) -> Tuple[str, str]:
        """Create an evolving shape field visualization."""
        
        # Process text into semantic shapes
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        shapes = []
        
        for sentence in tqdm(sentences, desc="Processing sentences"):
            # Generate shape field for each sentence
            field = self.shape_engine.generate_shape_field(
                sentence,
                resolution=50,
                complexity=0.8
            )
            shapes.append(field)
        
        # Create animation
        animation, frames = self.visualizer.visualize_shape_evolution(
            shapes,
            duration=duration,
            interpolation_steps=interpolation_steps,
            add_trails=add_trails
        )
        
        # Export animation and frames
        animation_path = self.visualizer.export_animation(
            animation,
            "shape_field_evolution",
            frames
        )
        
        # Create interactive visualization
        fig = self.interactive_renderer.create_interactive_shapes(
            shapes,
            title="Interactive Shape Field Evolution"
        )
        
        interactive_path = self.output_dir / "interactive_shape_field.html"
        fig.write_html(str(interactive_path))
        
        return str(animation_path), str(interactive_path)
    
    def create_geometric_transformation(
        self,
        source_text: str,
        target_text: str,
        n_steps: int = 20,
        add_intermediates: bool = True
    ) -> Tuple[str, Dict]:
        """Create visualization of geometric transformation between texts."""
        
        # Generate source and target shapes
        source_shape = self.shape_engine.generate_shape_field(source_text)
        target_shape = self.shape_engine.generate_shape_field(target_text)
        
        # Create transformation sequence
        transformations = self.shape_engine.create_transformation_sequence(
            source_shape,
            target_shape,
            n_steps=n_steps,
            add_intermediates=add_intermediates
        )
        
        # Create animation
        animation = self.visualizer.create_transformation_animation(
            transformations,
            duration=10.0,
            fps=30
        )
        
        # Export animation
        output_path = self.output_dir / "geometric_transformation.mp4"
        animation.save(str(output_path))
        
        # Generate transformation metrics
        metrics = self.shape_engine.analyze_transformation(
            source_shape,
            target_shape,
            transformations
        )
        
        return str(output_path), metrics
    
    def create_semantic_geometry(
        self,
        texts: List[str],
        n_dimensions: int = 3,
        show_connections: bool = True
    ) -> str:
        """Create visualization of semantic geometry relationships."""
        
        # Generate shapes for each text
        shapes = []
        for text in texts:
            shape = self.shape_engine.generate_shape_field(
                text,
                n_dimensions=n_dimensions
            )
            shapes.append(shape)
        
        # Create geometric relationship visualization
        fig = self.shape_renderer.create_geometric_relationships(
            shapes,
            labels=texts,
            show_connections=show_connections
        )
        
        # Save visualization
        output_path = self.output_dir / "semantic_geometry.html"
        fig.write_html(str(output_path))
        
        return str(output_path)

def main():
    """Run advanced shape visualization examples."""
    try:
        # Initialize visualization
        viz = AdvancedShapeVisualization()
        
        # Example 1: Shape Field Evolution
        ritual_text = """
        In the depths of darkness, a spark ignites.
        Through sacred geometry, patterns emerge.
        The flame dances with divine symmetry.
        Shadows and light interweave their dance.
        In perfect balance, transformation occurs.
        The eternal forms reveal their truth.
        """
        
        animation_path, interactive_path = viz.create_shape_field_evolution(
            text=ritual_text,
            n_frames=90,
            duration=15.0,
            interpolation_steps=45,
            add_trails=True
        )
        
        logger.info(f"Created shape field evolution: {animation_path}")
        logger.info(f"Created interactive visualization: {interactive_path}")
        
        # Example 2: Geometric Transformation
        source = "The shadow cast upon the cave wall"
        target = "The eternal form revealed in light"
        
        transform_path, metrics = viz.create_geometric_transformation(
            source_text=source,
            target_text=target,
            n_steps=30,
            add_intermediates=True
        )
        
        logger.info(f"Created geometric transformation: {transform_path}")
        logger.info("Transformation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"- {metric}: {value}")
        
        # Example 3: Semantic Geometry
        texts = [
            "The perfect circle of eternity",
            "The square foundation of matter",
            "The triangle of divine harmony",
            "The spiral of evolution"
        ]
        
        geometry_path = viz.create_semantic_geometry(
            texts=texts,
            n_dimensions=3,
            show_connections=True
        )
        
        logger.info(f"Created semantic geometry visualization: {geometry_path}")
        
    except Exception as e:
        logger.error("Error in advanced shape visualization example", exc_info=e)

if __name__ == "__main__":
    main() 