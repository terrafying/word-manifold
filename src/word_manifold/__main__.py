"""Main entry point for word-manifold package."""

import os
# Set tokenizer parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
from .visualization.visualizer import ManifoldVisualizer, SOUND_AVAILABLE
from .manifold.vector_manifold import VectorManifold
from .embeddings.word_embeddings import WordEmbeddings

def main():
    """Run the main visualization demo."""
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create embeddings
    embeddings = WordEmbeddings()
    
    # Define coherent term groups
    elemental_terms = {
        "earth", "air", "fire", "water",  # Classical elements
        "solid", "gas", "plasma", "liquid",  # Physical states
        "mountain", "wind", "flame", "ocean"  # Natural manifestations
    }
    
    planetary_terms = {
        "sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn",
        "light", "darkness", "war", "communication", "expansion", "love", "time"
    }
    
    polar_pairs = {
        # Fundamental opposites
        "hot", "cold",
        "wet", "dry",
        "light", "dark",
        "above", "below",
        "spirit", "matter",
        "active", "passive",
        "creation", "destruction",
        "self", "other"
    }
    
    # Combine all terms
    terms = elemental_terms.union(planetary_terms).union(polar_pairs)
    
    # Load the terms
    embeddings.load_terms(terms)
    
    # Create manifold
    manifold = VectorManifold(embeddings)
    
    # Create visualizer
    visualizer = ManifoldVisualizer(
        manifold=manifold,
        output_dir=str(output_dir),
        color_scheme="semantic"
    )
    
    # Create interactive view highlighting relationships between opposites
    visualizer.create_interactive_view(
        highlight_terms=["light", "dark", "spirit", "matter", "hot", "cold"],
        show_voronoi=True,
        show_paths=True
    )
    
    # Save the visualization
    html_path = output_dir / "manifold_visualization.html"
    visualizer.fig.write_html(str(html_path))
    print(f"\nVisualization saved to: {html_path}")
    
    # Create animation showing elemental transformations
    animation_path = output_dir / "term_animation.gif"
    visualizer.create_term_animation(
        terms=["earth", "water", "air", "fire", "earth"],  # Cycle through elements
        duration=5.0,
        fps=30
    )
    print(f"Animation saved to: {animation_path}")
    
    # Generate and play sounds for elemental terms
    if SOUND_AVAILABLE:
        for term in ["earth", "water", "air", "fire"]:
            print(f"\nPlaying sound for element: {term}")
            visualizer.play_term_sound(term)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 