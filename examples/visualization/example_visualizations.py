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

from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.manifold.vector_manifold import VectorManifold
from word_manifold.visualization.hypertools_visualizer import HyperToolsVisualizer
from word_manifold.visualization.shape_visualizer import ShapeVisualizer
from word_manifold.automata.cellular_rules import create_predefined_rules

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ritual_evolution_example():
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
        manifold = VectorManifold(embeddings)
        fig, ax = plt.subplots(figsize=(12, 8))
        manifold.visualize_manifold(
            ax=ax,
            show_flow=True,
            show_boundaries=True,
            show_labels=True
        )
        plt.show()
        # Create visualizer
        visualizer = HyperToolsVisualizer(
            output_dir="visualizations/rituals",
            interactive=True,
            n_dimensions=3  # Use 3D for better stability
        )
        
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
            
            # Phase 1: Equilibrium
            try:
                transformed = rules["equilibrium"].apply(current.reshape(1, -1))
                trajectories.append(transformed[0])
            except Exception as e:
                logger.warning(f"Error applying equilibrium rule: {e}")
                trajectories.append(current.copy())
            
            # Phase 2: The Great Work
            try:
                transformed = rules["great_work"].apply(transformed)
                trajectories.append(transformed[0])
            except Exception as e:
                logger.warning(f"Error applying great work rule: {e}")
                trajectories.append(current.copy())
            
            # Phase 3: Star
            try:
                transformed = rules["star"].apply(transformed)
                trajectories.append(transformed[0])
            except Exception as e:
                logger.warning("Error applying star rule", exc_info=e, stack_info=True, stacklevel=2)
                trajectories.append(current.copy())
            
            term_trajectories[term] = trajectories
        
        # Create static visualization
        try:
            visualizer.visualize_term_evolution(
                term_trajectories=term_trajectories,
                phase_names=["Initial", "Equilibrium", "Great Work", "Illumination"],
                title="Ritual Term Evolution"
            )
        except Exception as e:
            logger.error("Error creating static visualization", exc_info=e, stack_info=True, stacklevel=2)
        
        # Create animation
        try:
            visualizer.create_animated_ritual(
                term_trajectories=term_trajectories,
                phase_names=["Initial", "Equilibrium", "Great Work", "Illumination"],
                title="Ritual Evolution Animation",
                duration=10.0,
                fps=30,
                add_trails=True
            )
        except Exception as e:
            logger.error("Error creating animation", exc_info=e, stack_info=True, stacklevel=2)
            
    except Exception as e:
        logger.error("Error in ritual evolution example", exc_info=e, stack_info=True, stacklevel=2)

def semantic_shape_example():
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
            visualizer.create_shape_field(
                ritual_text,
                chunk_size=1  # Process one sentence at a time
            )
        except Exception as e:
            logger.error("Error creating shape field", exc_info=e, stack_info=True, stacklevel=2)
        
        # Example 2: Comparative visualization
        texts = [
            "The mysteries of the universe unfold in sacred geometry.",
            "Divine passion ignites the soul with transformative fire.",
            "Wisdom flows like water through the channels of understanding."
        ]
        
        try:
            visualizer.create_comparative_visualization(
                texts,
                labels=["Mystery", "Passion", "Wisdom"]
            )
        except Exception as e:
            logger.error("Error creating comparative visualization", exc_info=e, stack_info=True, stacklevel=2)
            
    except Exception as e:
        logger.error("Error in semantic shape example", exc_info=e, stack_info=True, stacklevel=2)

def main():
    """Run visualization examples."""
    # Create output directories
    Path("visualizations/rituals").mkdir(parents=True, exist_ok=True)
    Path("visualizations/shapes").mkdir(parents=True, exist_ok=True)
    
    print("Running ritual evolution visualization example...")
    ritual_evolution_example()
    
    print("\nRunning semantic shape visualization example...")
    semantic_shape_example()
    
    print("\nVisualizations have been saved to the visualizations directory.")

if __name__ == "__main__":
    main() 