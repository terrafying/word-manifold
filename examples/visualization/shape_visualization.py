#!/usr/bin/env python3
"""
Example script demonstrating semantic shape visualization.

This script shows how to create and visualize semantic shapes
from different types of text, showing how meaning evolves and flows.
"""

import os

from word_manifold.visualization.shape_visualizer import ShapeVisualizer
from word_manifold.embeddings.phrase_embeddings import PhraseEmbedder

# Set environment variables to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """Run semantic shape visualization examples."""
    # Create visualizer
    visualizer = ShapeVisualizer()
    
    # Example 1: Visualize evolution of a poetic text
    print("Creating visualization of poetic evolution...")
    poetic_text = """
    The stars move still, time runs, the clock will strike,
    The devil will come, and Faustus must be damned.
    O, I'll leap up to my God! Who pulls me down?
    See, see, where Christ's blood streams in the firmament!
    One drop would save my soul, half a drop: ah, my Christ!—
    Ah, rend not my heart for naming of my Christ!
    Yet will I call on him: O, spare me, Lucifer!
    """
    
    visualizer.create_shape_field(
        poetic_text,
        chunk_size=2  # Process two lines at a time
    )
    
    # Example 2: Compare different emotional states
    print("\nCreating comparative visualization of emotional states...")
    emotional_texts = [
        # Joy and wonder
        """
        O wonder!
        How many goodly creatures are there here!
        How beauteous mankind is! O brave new world,
        That has such people in't!
        """,
        
        # Contemplation and mystery
        """
        There are more things in heaven and earth, Horatio,
        Than are dreamt of in your philosophy.
        """,
        
        # Intensity and passion
        """
        But soft! what light through yonder window breaks?
        It is the east, and Juliet is the sun.
        """
    ]
    
    visualizer.create_comparative_visualization(
        emotional_texts,
        labels=["Wonder", "Mystery", "Passion"]
    )
    
    # Example 3: Show transformation of ritual language
    print("\nCreating visualization of ritual transformation...")
    ritual_text = """
    I am the flame that burns in every heart of man,
    and in the core of every star.
    I am Life, and the giver of Life,
    yet therefore is the knowledge of me the knowledge of death.
    I am alone: there is no God where I am.
    """
    
    visualizer.create_shape_field(
        ritual_text,
        chunk_size=1  # Process one sentence at a time
    )
    
    print("\nVisualizations have been saved to the visualizations/shapes directory.")

if __name__ == "__main__":
    main() 