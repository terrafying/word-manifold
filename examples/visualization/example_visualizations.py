"""
Example Visualizations

This module demonstrates the core visualization capabilities:
1. Ritual evolution visualization using HyperTools
2. Semantic shape visualization for textual analysis
3. Symbolic ASCII art visualization
"""

import logging
from pathlib import Path
import os
from typing import Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt

from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.manifold.vector_manifold import VectorManifold
from word_manifold.visualization.hypertools_visualizer import HyperToolsVisualizer
from word_manifold.visualization.shape_visualizer import ShapeVisualizer, ExportConfig
from word_manifold.visualization.symbolic_visualizer import SymbolicVisualizer
from word_manifold.automata.cellular_rules import create_predefined_rules

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationResult:
    """Track the success and outputs of visualization operations."""
    def __init__(self):
        self.ritual_evolution_success = False
        self.shape_field_success = False
        self.comparative_success = False
        self.symbolic_success = False
        self.saved_files: List[str] = []
    
    def add_saved_file(self, filepath: str):
        """Record a successfully saved visualization file."""
        self.saved_files.append(filepath)
    
    @property
    def any_success(self) -> bool:
        """Check if any visualization was successful."""
        return any([
            self.ritual_evolution_success,
            self.shape_field_success,
            self.comparative_success,
            self.symbolic_success
        ])
    
    def get_summary(self) -> str:
        """Get a summary of the visualization results."""
        if not self.any_success:
            return "\nNo visualizations were successfully created."
        
        summary = "\nSuccessfully created visualizations:"
        if self.ritual_evolution_success:
            summary += "\n- Ritual evolution visualization"
        if self.shape_field_success:
            summary += "\n- Shape field visualization"
        if self.comparative_success:
            summary += "\n- Comparative visualization"
        if self.symbolic_success:
            summary += "\n- Symbolic ASCII visualization"
        
        if self.saved_files:
            summary += "\n\nFiles saved:"
            for filepath in self.saved_files:
                summary += f"\n- {filepath}"
        
        return summary

def ritual_evolution_example(result: VisualizationResult):
    """Demonstrate ritual evolution visualization."""
    try:
        # Initialize embeddings and manifold
        embeddings = WordEmbeddings()
        
        # Define ritual terms
        terms = [
            "light", "darkness", "wisdom", "understanding",
            "beauty", "strength", "mercy", "severity"
        ]
        
        # Load terms into embeddings
        embeddings.load_terms(terms)
        manifold = VectorManifold(embeddings)
        
        # Create visualizer
        visualizer = HyperToolsVisualizer(
            word_embeddings=embeddings,
            output_dir="visualizations/hypertools",
            n_dimensions=3
        )
        
        # Create term trajectories through ritual phases
        rules = create_predefined_rules()
        term_trajectories = {}
        
        # Apply transformations and track positions
        for term in terms:
            positions = []
            current_pos = embeddings.get_embedding(term)
            positions.append(current_pos)
            
            # Apply each rule and track position
            for rule in rules.values():
                transformed = manifold.transform(
                    current_pos.reshape(1, -1),
                    {'rule': rule.name}
                )
                positions.append(transformed[0])
                current_pos = transformed[0]
            
            term_trajectories[term] = np.array(positions)
        
        # Create visualization
        viz_path = visualizer.visualize_term_evolution(
            term_trajectories,
            title="Ritual Term Evolution",
            save_path="visualizations/ritual_evolution.png"
        )
        
        if viz_path:
            result.add_saved_file(viz_path)
            result.ritual_evolution_success = True
            
    except Exception as e:
        logger.error("Error in ritual evolution example", exc_info=e)

def semantic_shape_example(result: VisualizationResult):
    """Demonstrate semantic shape visualization."""
    try:
        # Initialize visualizer with explicit output directory
        visualizer = ShapeVisualizer(
            export_config=ExportConfig(
                output_dir="visualizations/shapes",
                format="mp4",
                save_frames=True
            )
        )
        
        # Example 1: Ritual text visualization
        ritual_text = """
        I am the flame that burns in every heart of man,
        and in the core of every star.
        I am Life, and the giver of Life,
        yet therefore is the knowledge of me the knowledge of death.
        I am alone: there is no God where I am.
        """
        
        try:
            # Create and save shape field animation
            animation, frames = visualizer.visualize_shape_evolution(
                ritual_text.split('\n'),  # Process each line
                duration=10.0,  # 10 second animation
                interpolation_steps=45  # Smoother transitions
            )
            
            # Export the animation
            output_path = visualizer.export_animation(
                animation,
                "ritual_shape_field",
                frames
            )
            
            if output_path:
                result.add_saved_file(str(output_path))
                result.shape_field_success = True
                
        except Exception as e:
            logger.error("Error creating shape field", exc_info=e)
        
        # Example 2: Comparative visualization
        texts = [
            "The mysteries of the universe unfold in sacred geometry.",
            "Divine passion ignites the soul with transformative fire.",
            "Wisdom flows like water through the channels of understanding."
        ]
        
        try:
            # Create and save comparative visualization
            animation, frames = visualizer.visualize_shape_evolution(
                texts,
                duration=15.0,  # 15 second animation
                interpolation_steps=45  # Smoother transitions
            )
            
            # Export the animation
            output_path = visualizer.export_animation(
                animation,
                "comparative_shapes",
                frames
            )
            
            if output_path:
                result.add_saved_file(str(output_path))
                result.comparative_success = True
                
        except Exception as e:
            logger.error("Error creating comparative visualization", exc_info=e)
            
    except Exception as e:
        logger.error("Error in semantic shape example", exc_info=e)

def symbolic_visualization_example(result: VisualizationResult):
    """Demonstrate symbolic ASCII visualization."""
    try:
        # Initialize embeddings and visualizer
        embeddings = WordEmbeddings()
        visualizer = SymbolicVisualizer(
            word_embeddings=embeddings,
            width=100,  # Wider field for better patterns
            height=50   # Taller field for better patterns
        )
        
        # Example 1: Single term visualization
        term = "enlightenment"
        ascii_art = visualizer.visualize_term(term)
        
        # Save to file
        output_path = "visualizations/symbolic/enlightenment_mandala.txt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(ascii_art)
        result.add_saved_file(output_path)
        
        # Example 2: Transformation sequence
        frames = visualizer.visualize_transformation(
            "chaos",
            "order",
            steps=10
        )
        
        # Save transformation frames
        output_path = "visualizations/symbolic/chaos_to_order.txt"
        with open(output_path, 'w') as f:
            for i, frame in enumerate(frames):
                f.write(f"Frame {i+1}:\n")
                f.write(frame)
                f.write("\n\n" + "="*80 + "\n\n")
        result.add_saved_file(output_path)
        
        # Example 3: Semantic field
        terms = [
            "wisdom", "understanding", "knowledge",
            "beauty", "severity", "mercy"
        ]
        field = visualizer.create_semantic_field(terms)
        
        # Save field visualization
        output_path = "visualizations/symbolic/semantic_field.txt"
        with open(output_path, 'w') as f:
            f.write(field)
        result.add_saved_file(output_path)
        
        result.symbolic_success = True
        
    except Exception as e:
        logger.error("Error in symbolic visualization example", exc_info=e)

def main():
    """Run all visualization examples."""
    result = VisualizationResult()
    
    print("Creating ritual evolution visualization...")
    ritual_evolution_example(result)
    
    print("\nCreating semantic shape visualization...")
    semantic_shape_example(result)
    
    print("\nCreating symbolic ASCII visualization...")
    symbolic_visualization_example(result)
    
    print(result.get_summary())

if __name__ == "__main__":
    main() 