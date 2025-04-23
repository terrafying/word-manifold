"""
Semantic Crystallization Visualization Example.

This example demonstrates how meanings crystallize and accumulate during a reading sequence,
showing the progressive transformation of semantic space as concepts are integrated.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from word_manifold.embeddings import WordEmbeddings
from word_manifold.manifold import VectorManifold
from word_manifold.visualization.hypertools_visualizer import HyperToolsVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReadingStep:
    """Represents a single step in the reading sequence."""
    card: str  # The card or concept being integrated
    keywords: List[str]  # Associated keywords/meanings
    position: str  # Position or aspect in the reading (e.g., "past", "present", "future")
    influence: float = 1.0  # Relative influence of this step (0-1)

terms=[
                # Major Arcana
                "fool", "magician", "priestess", "empress", "emperor",
                "hierophant", "lovers", "chariot", "strength", "hermit",
                "wheel", "justice", "hanged", "death", "temperance",
                "devil", "tower", "star", "moon", "sun",
                "judgement", "world",
                
                # Elements and Qualities
                "fire", "water", "air", "earth",
                "active", "passive", "fixed", "mutable",
                
                # Abstract Concepts
                "beginning", "potential", "wisdom", "fertility", "authority",
                "tradition", "choice", "triumph", "courage", "solitude",
                "fate", "balance", "sacrifice", "transformation", "harmony",
                "bondage", "awakening", "hope", "intuition", "vitality",
                "rebirth", "completion",
                
                # Temporal Aspects
                "past", "present", "future",
                "conscious", "unconscious", "higher-self",
                "internal", "external", "synthesis"
            ]

class SemanticCrystallization:
    """
    Visualizes the crystallization of meaning during a reading sequence.
    
    This class tracks how semantic space transforms as each new card or concept
    is integrated, showing the accumulation and crystallization of meaning over time.
    """
    
    def __init__(
        self,
        n_dimensions: int = 5,
        output_dir: str = "visualizations/crystallization",
        decay_rate: float = 0.9  # How quickly previous influences fade
    ):
        self.n_dimensions = n_dimensions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.decay_rate = decay_rate
        
        # Initialize components
        self.embeddings = None
        self.manifold = None
        self.visualizer = None
        
        # Track reading state
        self.reading_steps: List[ReadingStep] = []
        self.crystallized_vectors: Dict[str, np.ndarray] = {}
        self.transformation_history: List[Dict[str, np.ndarray]] = []
        
    def prepare_components(self):
        """Initialize required components."""
        logger.info("Preparing semantic crystallization components...")
        
        # Initialize embeddings with tarot and interpretive vocabulary
        self.embeddings = WordEmbeddings()
        self.embeddings.load_terms(set(terms))
        # Initialize manifold
        self.manifold = VectorManifold(
            word_embeddings=self.embeddings,
            n_cells=22,  # Major Arcana
            reduction_dims=self.n_dimensions
        )
        
        # Initialize visualizer with special settings for crystallization
        self.visualizer = HyperToolsVisualizer(
            output_dir=str(self.output_dir),
            n_dimensions=self.n_dimensions,
            color_palette="plasma",)
        # Args yet to be added
        #     force_field_resolution=30,  # Higher resolution for detailed force fields
        #     force_field_strength=1.5,   # Stronger forces to show crystallization
        #     trail_length=30,            # Longer trails to show evolution
        #     trail_fade=0.95             # Slower fade for persistence
        # )
        
        logger.info("Components prepared")
        
    def add_reading_step(
        self,
        card: str,
        keywords: List[str],
        position: str,
        influence: float = 1.0
    ) -> None:
        """
        Add a new step to the reading sequence.
        
        Args:
            card: The card or concept being integrated
            keywords: Associated keywords/meanings
            position: Position or aspect in the reading
            influence: Relative influence of this step (0-1)
        """
        step = ReadingStep(card=card, keywords=keywords, 
                         position=position, influence=influence)
        self.reading_steps.append(step)
        
        # Update semantic crystallization
        self._integrate_step(step)
        
    def _integrate_step(self, step: ReadingStep) -> None:
        """Integrate a new reading step into the crystallized semantic space."""
        # Get embeddings for card and keywords
        vectors_to_integrate = []
        weights = []
        
        # Add card embedding
        if step.card in self.embeddings.terms:
            card_vector = self.embeddings.get_embedding(step.card)
            vectors_to_integrate.append(card_vector)
            weights.append(1.0)  # Primary weight for the card
            
        # Add keyword embeddings
        for keyword in step.keywords:
            if keyword in self.embeddings.terms:
                keyword_vector = self.embeddings.get_embedding(keyword)
                vectors_to_integrate.append(keyword_vector)
                weights.append(0.5)  # Lower weight for associated keywords
                
        if not vectors_to_integrate:
            logger.warning(f"No valid embeddings found for step: {step.card}")
            return
            
        # Convert to numpy arrays
        vectors = np.array(vectors_to_integrate)
        weights = np.array(weights)
        weights /= weights.sum()  # Normalize weights
        
        # Compute weighted average for this step
        step_vector = np.average(vectors, weights=weights, axis=0)
        
        # Apply manifold transformation
        transformed = self.manifold.transform(step_vector.reshape(1, -1))[0]
        
        # Decay previous crystallizations
        for term in self.crystallized_vectors:
            self.crystallized_vectors[term] *= self.decay_rate
            
        # Add new crystallization
        self.crystallized_vectors[step.card] = transformed
        
        # Store current state in history
        self.transformation_history.append(self.crystallized_vectors.copy())
        
    def visualize_crystallization(self) -> str:
        """
        Create visualization of the semantic crystallization process.
        
        Returns:
            Path to the generated visualization
        """
        if not self.transformation_history:
            raise ValueError("No reading steps to visualize")
            
        logger.info("Creating semantic crystallization visualization...")
        
        # Prepare visualization data
        all_vectors = []
        labels = []
        groups = []  # For coloring by reading position
        
        # Position mapping for coloring
        position_to_group = {}
        
        # Process each state in the transformation history
        for step_idx, state in enumerate(self.transformation_history):
            step = self.reading_steps[step_idx]
            
            # Assign group for position if not seen before
            if step.position not in position_to_group:
                position_to_group[step.position] = len(position_to_group)
                
            # Add vectors for this state
            for card, vector in state.items():
                all_vectors.append(vector)
                labels.append(f"{card} ({step.position})")
                groups.append(position_to_group[step.position])
                
        # Convert to numpy array
        points = np.array(all_vectors)
        
        # Create visualization
        viz_path = self.visualizer.visualize_vector_space(
            vectors=points,
            labels=labels,
            title="Semantic Crystallization in Reading",
            group_by=groups,
            legend_labels=list(position_to_group.keys()),
            auto_rotate=True,
            show_trails=True,
            show_force_field=True
        )
        
        logger.info(f"Visualization saved to: {viz_path}")
        return viz_path

def main():
    """Run the semantic crystallization example."""
    # Create and prepare crystallization viewer
    crystal = SemanticCrystallization(n_dimensions=5)
    crystal.prepare_components()
    
    # Example Celtic Cross reading sequence
    reading_sequence = [
        # Central cross
        ("present", "tower", ["disruption", "awakening", "revelation"]),
        ("challenge", "death", ["transformation", "ending", "rebirth"]),
        ("conscious", "hermit", ["introspection", "guidance", "solitude"]),
        ("unconscious", "moon", ["intuition", "fear", "illusion"]),
        ("past", "wheel", ["cycles", "fate", "turning-point"]),
        ("future", "star", ["hope", "inspiration", "healing"]),
        
        # Staff
        ("self", "magician", ["power", "skill", "manifestation"]),
        ("environment", "world", ["completion", "integration", "achievement"]),
        ("hopes", "sun", ["joy", "success", "vitality"]),
        ("outcome", "judgement", ["awakening", "rebirth", "transformation"])
    ]
    
    # Add each step
    for position, card, keywords in reading_sequence:
        crystal.add_reading_step(
            card=card,
            keywords=keywords,
            position=position,
            influence=1.0
        )
    
    # Create visualization
    viz_path = crystal.visualize_crystallization()
    
    logger.info(f"""
    Semantic Crystallization visualization complete!
    
    This visualization shows how meanings accumulate and crystallize through the reading:
    - Points represent card meanings in semantic space
    - Colors indicate different positions/aspects of the reading
    - Trails show how meanings evolve and influence each other
    - Force fields reveal the semantic "gravity" of accumulated meanings
    - The crystallization process shows how each new card builds upon previous meanings
    
    Interactive Controls:
    - Arrow keys: Rotate view
    - Space: Toggle auto-rotation
    - T: Toggle motion trails
    - F: Toggle force field
    - R: Reset view
    - +/-: Adjust rotation speed
    
    The visualization has been saved to: {viz_path}
    """)

if __name__ == "__main__":
    main() 