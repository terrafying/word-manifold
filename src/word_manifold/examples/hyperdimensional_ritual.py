#!/usr/bin/env python3
"""
Hyperdimensional Ritual Visualization Example

This example demonstrates the visualization of semantic transformations
in 4+ dimensional space during a Thelemic ritual working. It shows how
concepts evolve and interact across multiple dimensions of meaning.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.manifold.vector_manifold import VectorManifold
from word_manifold.visualization.hypertools_visualizer import HyperToolsVisualizer
from word_manifold.examples.ritual_evolution import RitualWorking
from word_manifold.automata.cellular_rules import create_predefined_rules

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperdimensionalRitual:
    """
    A class demonstrating hyperdimensional visualization of Thelemic rituals.
    """
    
    def __init__(
        self,
        n_dimensions: int = 5,  # We'll use 5D for richer semantic representation
        output_dir: str = "visualizations/hyperdimensional"
    ):
        self.n_dimensions = n_dimensions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embeddings = None
        self.manifold = None
        self.visualizer = None
        self.ritual = None
        
    def prepare_components(self):
        """Initialize all necessary components."""
        logger.info(f"Preparing {self.n_dimensions}D ritual visualization...")
        
        # Initialize embeddings with extended Thelemic vocabulary
        self.embeddings = WordEmbeddings(model_name="bert-base-uncased")
        
        # Define key Thelemic concepts to track
        thelemic_concepts = {
            # Core principles
            "will", "love", "law", "liberty", "truth", "light", "life",
            
            # Thelemic terminology
            "thelema", "agape", "abrahadabra", "aiwass", "nuit", "hadit",
            "ra-hoor-khuit", "babalon", "therion", "aeon",
            
            # Magical concepts
            "magick", "ritual", "invocation", "evocation", "banishing",
            "pentagram", "hexagram", "circle", "altar", "wand",
            
            # States of consciousness
            "consciousness", "enlightenment", "initiation", "knowledge",
            "understanding", "wisdom", "folly", "silence", "speech",
            
            # Elements and forces
            "fire", "water", "air", "earth", "spirit", "force", "form",
            "energy", "matter", "light", "darkness",
            
            # Tree of Life concepts
            "kether", "chokmah", "binah", "chesed", "geburah", "tiphareth",
            "netzach", "hod", "yesod", "malkuth"
        }
        
        # Load embeddings
        self.embeddings.load_terms(thelemic_concepts)
        
        # Create manifold with higher dimensions
        self.manifold = VectorManifold(
            word_embeddings=self.embeddings,
            n_cells=22,  # Corresponding to Major Arcana
            random_state=93,  # Thelemic significance
            reduction_dims=self.n_dimensions
        )
        
        # Initialize visualizer
        self.visualizer = HyperToolsVisualizer(
            output_dir=str(self.output_dir),
            n_dimensions=self.n_dimensions,
            color_palette="plasma",  # For magical aesthetics
            interactive=True
        )
        
        # Create ritual working
        self.ritual = RitualWorking(
            ritual_name="Hyperdimensional True Will",
            ritual_intent="To manifest True Will across multiple dimensions of consciousness"
        )
        self.ritual.prepare_components()
        
        logger.info("All components prepared")
        
    def visualize_ritual_transformation(self):
        """
        Visualize the semantic transformations during the ritual.
        """
        if not all([self.embeddings, self.manifold, self.visualizer, self.ritual]):
            raise ValueError("Components not initialized. Call prepare_components() first.")
            
        logger.info("Beginning ritual visualization...")
        
        # Get initial term embeddings
        initial_embeddings = np.array([
            self.embeddings.get_embedding(term) 
            for term in self.embeddings.terms
        ])
        
        # Track term positions through ritual phases
        term_trajectories = {term: [] for term in self.embeddings.terms}
        
        # Define ritual phases
        phases = [
            ("Preparation", self.ritual.system.rules["tower"]),
            ("Purification", self.ritual.system.rules["hermit"]),
            ("Invocation", self.ritual.system.rules["magician"]),
            ("Transformation", self.ritual.system.rules["true_will"]),
            ("Integration", self.ritual.system.rules["liberty"]),
            ("Illumination", self.ritual.system.rules["star"])
        ]
        
        # Collect embeddings through each phase
        current_embeddings = initial_embeddings.copy()
        
        for phase_name, rule in phases:
            logger.info(f"Processing {phase_name} phase...")
            
            # Apply the rule
            rule.apply(self.manifold, self.ritual.system.generation)
            self.ritual.system.generation += 1
            
            # Get transformed embeddings using transform method instead of transform_vectors
            transformed = self.manifold.transform(current_embeddings)
            
            # Store positions for each term
            for i, term in enumerate(self.embeddings.terms):
                term_trajectories[term].append(transformed[i])
            
            # Update current state
            current_embeddings = transformed
            
        # Create hyperdimensional visualization
        logger.info("Creating hyperdimensional visualization...")
        
        # Prepare data for visualization
        all_points = []
        labels = []
        groups = []  # For coloring by phase
        
        for term in self.embeddings.terms:
            positions = term_trajectories[term]
            for phase_idx, pos in enumerate(positions):
                all_points.append(pos)
                labels.append(f"{term} (Phase {phase_idx+1})")
                groups.append(phase_idx)
                
        # Convert to numpy array
        points = np.array(all_points)
        
        # Create interactive visualization with trails and force field
        viz_path = self.visualizer.visualize_vector_space(
            vectors=points,
            labels=labels,
            title="Hyperdimensional Ritual Evolution",
            group_by=groups,
            legend_labels=[phase[0] for phase in phases],
            auto_rotate=True,  # Enable automatic rotation
            show_trails=True,  # Enable motion trails
            show_force_field=True  # Enable force field visualization
        )
        
        logger.info(f"Visualization saved to: {viz_path}")
        
        return viz_path

def main():
    """Run the hyperdimensional ritual visualization example."""
    # Create and prepare the ritual
    ritual = HyperdimensionalRitual(n_dimensions=5)
    ritual.prepare_components()
    
    # Create visualization
    viz_path = ritual.visualize_ritual_transformation()
    
    logger.info(f"""
    Hyperdimensional ritual visualization complete!
    
    The visualization shows the evolution of Thelemic concepts through a 5-dimensional
    semantic space. You can interact with the visualization using:
    
    - Arrow keys: Rotate in primary dimensions
    - 1-5: Select dimension for X axis
    - Shift + 1-5: Select dimension for Y axis
    - Ctrl + 1-5: Select dimension for Z axis
    - Space: Toggle auto-rotation
    - +/-: Adjust rotation speed
    - R: Reset view
    - T: Toggle motion trails
    - F: Toggle force field visualization
    
    The visualization has been saved to: {viz_path}
    
    Features:
    - Motion trails show the path of concepts through semantic space
    - Force field arrows indicate the flow of semantic energy
    - Colors indicate ritual phases
    - Interactive controls for exploring all 5 dimensions
    
    "Do what thou wilt shall be the whole of the Law."
    """)

if __name__ == "__main__":
    main() 