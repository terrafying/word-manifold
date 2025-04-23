"""
Example Visualizations

This module demonstrates the core visualization capabilities:
1. Ritual evolution visualization using HyperTools
2. Semantic shape visualization for textual analysis
"""

import logging
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional

from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.manifold.vector_manifold import VectorManifold
from word_manifold.visualization.hypertools_visualizer import HyperToolsVisualizer
from word_manifold.visualization.shape_visualizer import ShapeVisualizer
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
            self.comparative_success
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
        
        if self.saved_files:
            summary += "\n\nFiles saved:"
            for filepath in self.saved_files:
                summary += f"\n- {filepath}"
        
        return summary

def ritual_evolution_example(result: VisualizationResult):
    """Demonstrate ritual evolution visualization."""
    try:
        # Initialize embeddings and manifold
        embeddings = WordEmbeddings(model_name='en_core_web_sm')
        
        # Define ritual sequence with key terms
        terms = [
            "light", "darkness", "wisdom", "understanding",
            "beauty", "strength", "mercy", "severity"
        ]
        
        # Load terms into embeddings
        embeddings.load_terms(terms)
        
        # Create term trajectories through ritual phases
        term_trajectories = {term: [] for term in terms}
        rules = create_predefined_rules()
        
        # For each phase, create a new manifold and apply the rule
        for term in terms:
            # Get initial vector
            initial_vector = embeddings.get_embedding(term)
            term_trajectories[term].append(initial_vector)
            
            current_vector = initial_vector.copy()
            
            # Phase 1: Equilibrium
            try:
                manifold = VectorManifold(embeddings)
                rules["equilibrium"].apply(manifold, generation=0)
                transformed = manifold.get_term_cell(term).centroid
                term_trajectories[term].append(transformed)
                current_vector = transformed
            except Exception as e:
                logger.warning(f"Error applying equilibrium rule for term {term}", exc_info=e)
                term_trajectories[term].append(current_vector)
            
            # Phase 2: The Great Work
            try:
                manifold = VectorManifold(embeddings)
                rules["great_work"].apply(manifold, generation=1)
                transformed = manifold.get_term_cell(term).centroid
                term_trajectories[term].append(transformed)
                current_vector = transformed
            except Exception as e:
                logger.warning(f"Error applying great work rule for term {term}", exc_info=e)
                term_trajectories[term].append(current_vector)
            
            # Phase 3: Star
            try:
                manifold = VectorManifold(embeddings)
                rules["star"].apply(manifold, generation=2)
                transformed = manifold.get_term_cell(term).centroid
                term_trajectories[term].append(transformed)
            except Exception as e:
                logger.warning(f"Error applying star rule for term {term}", exc_info=e)
                term_trajectories[term].append(current_vector)
        
        # Create visualizer
        visualizer = HyperToolsVisualizer(
            output_dir="visualizations/rituals",
            interactive=True,
            n_dimensions=3
        )
        
        # Create static visualization
        try:
            filepath = visualizer.visualize_term_evolution(
                term_trajectories=term_trajectories,
                phase_names=["Initial", "Equilibrium", "Great Work", "Illumination"],
                title="Ritual Term Evolution"
            )
            if filepath:
                result.add_saved_file(filepath)
                result.ritual_evolution_success = True
        except Exception as e:
            logger.error("Error creating static visualization", exc_info=e)
        
        # Create animation
        try:
            filepath = visualizer.create_animated_ritual(
                term_trajectories=term_trajectories,
                phase_names=["Initial", "Equilibrium", "Great Work", "Illumination"],
                title="Ritual Evolution Animation",
                duration=10.0,
                fps=30,
                add_trails=True
            )
            if filepath:
                result.add_saved_file(filepath)
                result.ritual_evolution_success = True
        except Exception as e:
            logger.error("Error creating animation", exc_info=e)
            
    except Exception as e:
        logger.error("Error in ritual evolution example", exc_info=e)

def semantic_shape_example(result: VisualizationResult):
    """Demonstrate semantic shape visualization."""
    try:
        # Initialize visualizer
        visualizer = ShapeVisualizer()
        
        # Example 1: Ritual text visualization
        ritual_text = """
        I am the flame that burns in every heart of man,
        and in the core of every star.
        I am Life, and the giver of Life,
        yet therefore is the knowledge of me the knowledge of death.
        I am alone: there is no God where I am.
        """
        
        try:
            filepath = visualizer.create_shape_field(
                ritual_text,
                chunk_size=1  # Process one sentence at a time
            )
            if filepath:
                result.add_saved_file(filepath)
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
            filepath = visualizer.create_comparative_visualization(
                texts,
                labels=["Mystery", "Passion", "Wisdom"]
            )
            if filepath:
                result.add_saved_file(filepath)
                result.comparative_success = True
        except Exception as e:
            logger.error("Error creating comparative visualization", exc_info=e)
            
    except Exception as e:
        logger.error("Error in semantic shape example", exc_info=e)

def main():
    """Run visualization examples."""
    # Set environment variable for tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Initialize result tracking
    result = VisualizationResult()
    
    try:
        ritual_evolution_example(result)
        semantic_shape_example(result)
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
    
    # Print appropriate summary
    print(result.get_summary())

if __name__ == "__main__":
    main() 