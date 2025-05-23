"""
Example demonstrating force field visualization in semantic space.

This example creates a dynamic visualization showing how concepts move through
a semantic force field generated by attractor and repulsor terms.
"""

import numpy as np
import logging
from word_manifold.embeddings import WordEmbeddings
from word_manifold.manifold.vector_manifold import VectorManifold
from word_manifold.visualization.hypertools_visualizer import HyperToolsVisualizer

logger = logging.getLogger(__name__)

class ForceFieldDemo:
    def __init__(self, n_dimensions=3):
        """Initialize the force field demonstration.
        
        Args:
            n_dimensions (int): Number of dimensions for the semantic space
        """
        self.n_dimensions = n_dimensions
        self.embeddings = None
        self.manifold = None
        self.visualizer = None
        
    def prepare_components(self):
        """Initialize required components."""
        # Initialize embeddings with contrasting concept pairs
        self.embeddings = WordEmbeddings(
            terms=[
                # Attractors (positive concepts)
                "love", "light", "wisdom", "truth", "beauty",
                # Repulsors (challenging concepts) 
                "fear", "darkness", "ignorance", "illusion", "chaos",
                # Test particles (neutral concepts)
                "mind", "spirit", "soul", "will", "consciousness",
                "energy", "power", "knowledge", "understanding", "transformation"
            ],
            n_dimensions=self.n_dimensions
        )
        
        # Initialize manifold
        self.manifold = SemanticManifold(
            embeddings=self.embeddings,
            learning_rate=0.1
        )
        
        # Initialize visualizer
        self.visualizer = HyperToolsVisualizer(
            force_field_resolution=20,  # Higher resolution force field
            force_field_strength=2.0,   # Stronger forces
            trail_length=10,            # Longer trails
            trail_fade=0.8              # Slower trail fade
        )
        
    def simulate_force_field(self, n_steps=100):
        """Simulate particle movement in the force field.
        
        Args:
            n_steps (int): Number of simulation steps
            
        Returns:
            str: Path to the generated visualization
        """
        if not all([self.embeddings, self.manifold, self.visualizer]):
            raise ValueError("Components not initialized. Call prepare_components() first.")
            
        logger.info("Beginning force field simulation...")
        
        # Get initial embeddings
        embeddings = np.array([
            self.embeddings.get_embedding(term)
            for term in self.embeddings.terms
        ])
        
        # Track positions over time
        positions = []
        labels = []
        groups = []
        
        # Classify terms
        term_types = {
            "attractor": self.embeddings.terms[:5],
            "repulsor": self.embeddings.terms[5:10],
            "particle": self.embeddings.terms[10:]
        }
        
        # Simulate movement
        current_pos = embeddings.copy()
        
        for step in range(n_steps):
            # Store current state
            for i, term in enumerate(self.embeddings.terms):
                positions.append(current_pos[i])
                labels.append(f"{term} (t={step})")
                
                # Color by term type
                if term in term_types["attractor"]:
                    groups.append(0)  # Attractors
                elif term in term_types["repulsor"]:
                    groups.append(1)  # Repulsors
                else:
                    groups.append(2)  # Test particles
            
            # Update positions based on forces
            forces = np.zeros_like(current_pos)
            
            # Attractive forces from positive concepts
            for i, term in enumerate(term_types["particle"]):
                particle_idx = self.embeddings.terms.index(term)
                
                # Attraction to positive concepts
                for attractor in term_types["attractor"]:
                    attractor_idx = self.embeddings.terms.index(attractor)
                    diff = current_pos[attractor_idx] - current_pos[particle_idx]
                    dist = np.linalg.norm(diff)
                    force = diff / (dist ** 2 + 1e-6)  # Avoid division by zero
                    forces[particle_idx] += force
                
                # Repulsion from negative concepts
                for repulsor in term_types["repulsor"]:
                    repulsor_idx = self.embeddings.terms.index(repulsor)
                    diff = current_pos[particle_idx] - current_pos[repulsor_idx]
                    dist = np.linalg.norm(diff)
                    force = diff / (dist ** 2 + 1e-6)
                    forces[particle_idx] += force
            
            # Update positions (only move test particles)
            for term in term_types["particle"]:
                idx = self.embeddings.terms.index(term)
                current_pos[idx] += 0.1 * forces[idx]  # Scale force effect
        
        # Convert to numpy array
        points = np.array(positions)
        
        # Create visualization
        logger.info("Creating force field visualization...")
        viz_path = self.visualizer.visualize_vector_space(
            vectors=points,
            labels=labels,
            title="Semantic Force Field Demonstration",
            group_by=groups,
            legend_labels=["Attractors", "Repulsors", "Test Particles"],
            auto_rotate=True,
            show_trails=True,
            show_force_field=True
        )
        
        logger.info(f"Visualization saved to: {viz_path}")
        return viz_path

def main():
    """Run the force field visualization demo."""
    # Create and run simulation
    demo = ForceFieldDemo(n_dimensions=3)
    demo.prepare_components()
    viz_path = demo.simulate_force_field()
    
    logger.info(f"""
    Force field visualization complete!
    
    This demonstration shows how concepts move through a semantic force field:
    - Red points are attractors (positive concepts)
    - Blue points are repulsors (negative concepts)
    - Green points are test particles that move based on the forces
    
    The visualization shows:
    - Force field arrows indicating semantic "flow"
    - Motion trails showing concept trajectories
    - Test particles being attracted to positive concepts
    - Test particles being repelled by negative concepts
    
    You can interact with the visualization using:
    - Arrow keys: Rotate view
    - Space: Toggle auto-rotation
    - T: Toggle motion trails
    - F: Toggle force field
    - R: Reset view
    
    The visualization has been saved to: {viz_path}
    """)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 