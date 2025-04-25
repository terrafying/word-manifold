"""
Advanced HyperTools Visualizations

This module demonstrates advanced visualization capabilities using HyperTools,
including interactive plots, animations, and complex transformations.
"""

import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from tqdm import tqdm

from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.manifold.vector_manifold import VectorManifold
from word_manifold.visualization.hypertools_visualizer import HyperToolsVisualizer
from word_manifold.automata.cellular_rules import create_predefined_rules
from word_manifold.visualization.engines.plotly_engine import PlotlyEngine
from word_manifold.visualization.renderers.interactive import InteractiveRenderer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedVisualization:
    """Advanced visualization capabilities with interactive features."""
    
    def __init__(self, output_dir: str = "visualizations/advanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embeddings = WordEmbeddings()
        self.manifold = VectorManifold(self.embeddings)
        self.visualizer = HyperToolsVisualizer(
            word_embeddings=self.embeddings,
            output_dir=str(self.output_dir),
            interactive=True,
            n_dimensions=3
        )
        self.plotly_engine = PlotlyEngine()
        self.interactive_renderer = InteractiveRenderer()
        
    def create_ritual_evolution_animation(
        self,
        terms: List[str],
        phases: List[str],
        duration: float = 10.0,
        fps: int = 30,
        add_trails: bool = True
    ) -> str:
        """Create an animated visualization of ritual term evolution."""
        
        # Load terms and create trajectories
        self.embeddings.load_terms(terms)
        rules = create_predefined_rules()
        
        # Track positions through phases
        term_trajectories: Dict[str, List[np.ndarray]] = {}
        for term in tqdm(terms, desc="Processing terms"):
            positions = []
            current_pos = self.embeddings.get_embedding(term)
            positions.append(current_pos)
            
            # Apply transformations for each phase
            current_manifold = self.manifold.copy()
            current_manifold.add_cell(term, current_pos)
            
            for rule_name in rules:
                transformed_manifold = rules[rule_name].apply(current_manifold)
                transformed = transformed_manifold.get_cell(term).centroid
                positions.append(transformed)
                current_manifold = transformed_manifold
            
            term_trajectories[term] = positions
            
        # Create animation with trails
        animation_path = self.visualizer.create_animated_ritual(
            term_trajectories=term_trajectories,
            phase_names=phases,
            title="Advanced Ritual Evolution",
            duration=duration,
            fps=fps,
            add_trails=add_trails,
            colormap='viridis'
        )
        
        # Create interactive version
        fig = self.plotly_engine.create_interactive_evolution(
            term_trajectories=term_trajectories,
            phase_names=phases,
            title="Interactive Ritual Evolution"
        )
        
        interactive_path = self.output_dir / "interactive_ritual_evolution.html"
        fig.write_html(str(interactive_path))
        
        return str(animation_path), str(interactive_path)
    
    def create_manifold_landscape(
        self,
        base_terms: List[str],
        n_neighbors: int = 10,
        resolution: int = 50
    ) -> str:
        """Create a 3D landscape visualization of the word manifold."""
        
        # Load terms and find neighbors
        self.embeddings.load_terms(base_terms)
        all_terms = set(base_terms)
        
        # Find semantic neighbors
        for term in base_terms:
            neighbors = self.embeddings.find_similar_terms(term, k=n_neighbors)
            all_terms.update([t for t, _ in neighbors])
        
        # Get embeddings for all terms
        term_list = list(all_terms)
        embeddings_matrix = np.stack([
            self.embeddings.get_embedding(t) for t in term_list
        ])
        
        # Create landscape visualization
        fig = self.plotly_engine.create_manifold_landscape(
            embeddings=embeddings_matrix,
            terms=term_list,
            resolution=resolution,
            colorscale='Viridis'
        )
        
        # Save interactive visualization
        output_path = self.output_dir / "manifold_landscape.html"
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    def create_semantic_clusters(
        self,
        seed_terms: List[str],
        n_clusters: int = 5,
        min_terms_per_cluster: int = 10
    ) -> Tuple[str, Dict]:
        """Create clustered visualization of semantic relationships."""
        
        # Load seed terms and expand vocabulary
        self.embeddings.load_terms(seed_terms)
        expanded_terms = set(seed_terms)
        
        # Expand terms to ensure minimum cluster size
        while len(expanded_terms) < n_clusters * min_terms_per_cluster:
            for term in list(expanded_terms):
                neighbors = self.embeddings.find_similar_terms(term, k=5)
                expanded_terms.update([t for t, _ in neighbors])
                if len(expanded_terms) >= n_clusters * min_terms_per_cluster:
                    break
        
        # Get embeddings for clustering
        term_list = list(expanded_terms)
        embeddings_matrix = np.stack([
            self.embeddings.get_embedding(t) for t in term_list
        ])
        
        # Create clustered visualization
        fig, cluster_info = self.plotly_engine.create_semantic_clusters(
            embeddings=embeddings_matrix,
            terms=term_list,
            n_clusters=n_clusters
        )
        
        # Save interactive visualization
        output_path = self.output_dir / "semantic_clusters.html"
        fig.write_html(str(output_path))
        
        return str(output_path), cluster_info

def main():
    """Run advanced visualization examples."""
    try:
        # Initialize visualization
        viz = AdvancedVisualization()
        
        # Example 1: Ritual Evolution Animation
        terms = [
            "light", "darkness", "wisdom", "understanding",
            "beauty", "strength", "mercy", "severity",
            "foundation", "kingdom", "crown", "spirit"
        ]
        
        phases = [
            "Initial State",
            "Purification",
            "Transformation",
            "Illumination",
            "Integration"
        ]
        
        animation_path, interactive_path = viz.create_ritual_evolution_animation(
            terms=terms,
            phases=phases,
            duration=15.0,
            fps=60,
            add_trails=True
        )
        
        logger.info(f"Created ritual evolution animation: {animation_path}")
        logger.info(f"Created interactive visualization: {interactive_path}")
        
        # Example 2: Manifold Landscape
        base_terms = [
            "wisdom", "knowledge", "understanding",
            "truth", "reality", "illusion"
        ]
        
        landscape_path = viz.create_manifold_landscape(
            base_terms=base_terms,
            n_neighbors=15,
            resolution=75
        )
        
        logger.info(f"Created manifold landscape: {landscape_path}")
        
        # Example 3: Semantic Clusters
        seed_terms = [
            "light", "dark", "wisdom", "ignorance",
            "truth", "illusion", "reality", "shadow"
        ]
        
        clusters_path, cluster_info = viz.create_semantic_clusters(
            seed_terms=seed_terms,
            n_clusters=5,
            min_terms_per_cluster=12
        )
        
        logger.info(f"Created semantic clusters: {clusters_path}")
        logger.info("Cluster Information:")
        for cluster_id, terms in cluster_info.items():
            logger.info(f"Cluster {cluster_id}: {', '.join(terms[:5])}...")
        
    except Exception as e:
        logger.error("Error in advanced visualization example", exc_info=e)

if __name__ == "__main__":
    main() 