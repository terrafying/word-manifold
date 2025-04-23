"""
Command Line Interface for Word Manifold Visualizations.

This module provides a unified CLI for invoking different types of visualizations,
including ritual evolution animations and semantic shape analysis.
"""

import click
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.manifold.vector_manifold import VectorManifold
from word_manifold.visualization.hypertools_visualizer import HyperToolsVisualizer
from word_manifold.visualization.shape_visualizer import ShapeVisualizer
from word_manifold.automata.cellular_rules import create_predefined_rules

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Word Manifold visualization tools."""
    pass

@cli.command()
@click.option('--interactive/--no-interactive', default=True, help='Enable/disable interactive mode')
@click.option('--output-dir', default='visualizations', help='Output directory for visualizations')
@click.option('--dimensions', default=3, help='Number of dimensions for visualization')
@click.option('--model', default='en_core_web_sm', help='Spacy model to use for embeddings')
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

if __name__ == '__main__':
    cli() 