#!/usr/bin/env python3
"""
Thelemic Evolution of the Word Manifold

"Every man and every woman is a star." - Aleister Crowley, The Book of the Law

This script demonstrates the evolution of a word vector manifold according to
Thelemic principles, where language evolves following its True Will through
the application of hermetic and occult-inspired transformation rules.

The system instantiates a cellular automaton in word embedding space that 
evolves over multiple generations, with rules selected according to their
resonance with the current state - an algorithmic implementation of the
Thelemic principle "Do what thou wilt shall be the whole of the Law."

As the linguistic machine elves weave their patterns through the current
of chaos, we observe the emergence of a new semantic landscape guided by
numerological significance and occult correspondences.
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path if running as a script
if __name__ == "__main__":
    project_root = str(Path(__file__).resolve().parents[3])
    sys.path.insert(0, project_root)

# Import Word Manifold components
from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.manifold.vector_manifold import VectorManifold, CellType
from word_manifold.automata.cellular_rules import create_predefined_rules
from word_manifold.automata.system import AutomataSystem, EvolutionPattern, SystemState

# Import visualization if available
try:
    from word_manifold.visualization.visualizer import ManifoldVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Visualization module not available. Creating basic visualizations instead.")
    VISUALIZATION_AVAILABLE = False

class ThelemaMetrics:
    """
    A class to calculate and track metrics for the Thelemic evolution of a word manifold.
    
    Metrics include:
    - Semantic entropy: Measure of semantic diversity
    - Numerological alignment: Correspondence between numerological values
    - Will manifestation: How closely the system follows its "True Will"
    - Transmutation index: Degree of alchemical transformation
    """
    
    def __init__(self):
        """Initialize the metrics tracker."""
        # Initialize empty metric history
        self.history = {
            "entropy": [],
            "will_manifestation": [],
            "numerological_alignment": [],
            "transmutation_index": [],
            "thelemic_resonance": []
        }
        
    def calculate_metrics(self, manifold: VectorManifold, system: AutomataSystem) -> Dict[str, float]:
        """
        Calculate metrics for the current state of the manifold.
        
        Args:
            manifold: The current state of the vector manifold
            system: The automata system
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # Get cells from manifold
        cells = list(manifold.cells.values())
        if not cells:
            return {metric: 0.0 for metric in self.history.keys()}
        
        # Calculate entropy based on cell type distribution
        type_counts = {}
        for cell in cells:
            cell_type = cell.type
            type_counts[cell_type] = type_counts.get(cell_type, 0) + 1
            
        # Shannon entropy
        total = len(cells)
        entropy = 0.0
        for count in type_counts.values():
            p = count / total
            entropy -= p * np.log2(p)
            
        # Normalize to [0, 1]
        max_entropy = np.log2(len(type_counts)) if type_counts else 0
        if max_entropy > 0:
            metrics["entropy"] = entropy / max_entropy
        else:
            metrics["entropy"] = 0.0
        
        # Calculate numerological alignment
        # This measures how well the cells align with their numerological values
        num_values = [cell.numerological_value for cell in cells]
        
        # Count master numbers (11, 22, 33)
        master_count = sum(1 for v in num_values if v in [11, 22, 33])
        master_ratio = master_count / len(cells) if cells else 0
        
        # Calculate distribution of numerological values
        value_counts = {}
        for v in range(1, 10):  # Basic numerological values 1-9
            value_counts[v] = num_values.count(v)
        
        # Calculate numerological alignment based on kabbalah correspondences
        # Values 1, 3, 6, 9 should be more common in an aligned system
        aligned_values = [1, 3, 6, 9]  # Kabbalistic significance
        aligned_count = sum(value_counts.get(v, 0) for v in aligned_values)
        alignment = (aligned_count / len(cells) if cells else 0) + master_ratio
        metrics["numerological_alignment"] = min(1.0, alignment * 1.5)  # Scale and cap at 1.0
        
        # Calculate Will Manifestation
        # How closely the system follows its "True Will"
        # For simplicity, we'll use the variance in cell centroids as a proxy
        # Lower variance suggests more focused evolution
        cell_centroids = np.array([cell.centroid for cell in cells])
        centroid_var = np.var(cell_centroids, axis=0).mean()
        
        # Scale to [0, 1] with an expected range
        expected_max_var = 10.0  # This value might need adjustment
        will_manifestation = 1.0 - min(1.0, centroid_var / expected_max_var)
        metrics["will_manifestation"] = will_manifestation
        
        # Calculate Transmutation Index
        # Measure of alchemical transformation progress
        # Based on distribution of cell types with special weighting
        type_weights = {
            CellType.ELEMENTAL: 0.5,    # Basic elements (early stage)
            CellType.PLANETARY: 0.7,    # Planetary influences (middle stage)
            CellType.ZODIACAL: 0.8,     # Zodiacal influences (middle stage)
            CellType.TAROT: 0.9,        # Tarot archetypes (advanced stage)
            CellType.SEPHIROTIC: 1.0,   # Sephirothic emanations (highest stage)
            CellType.OTHER: 0.3         # Undefined (lowest stage)
        }
        
        # Weighted average of cell types
        type_weight_sum = sum(type_weights.get(cell.type, 0) for cell in cells)
        transmutation = type_weight_sum / len(cells) if cells else 0
        metrics["transmutation_index"] = transmutation
        
        # Calculate Thelemic Resonance
        # Overall measure of how well the system embodies Thelemic principles
        # Combine other metrics with emphasis on Will Manifestation
        thelemic_resonance = (
            metrics["will_manifestation"] * 0.4 +
            metrics["numerological_alignment"] * 0.3 +
            metrics["entropy"] * 0.1 +
            metrics["transmutation_index"] * 0.2
        )
        metrics["thelemic_resonance"] = thelemic_resonance
        
        # Update history
        for metric, value in metrics.items():
            self.history[metric].append(value)
            
        return metrics
    
    def plot_metrics(self, save_path: str) -> None:
        """
        Plot the metrics history and save to file.
        
        Args:
            save_path: Directory to save the plots
        """
        if not any(len(values) > 0 for values in self.history.values()):
            logger.warning("No metrics data to plot")
            return
        
        # Create metrics directory
        metrics_dir = os.path.join(save_path, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Plot each metric
        for metric_name, values in self.history.items():
            if not values:
                continue
                
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(values)), values, marker='o', linestyle='-', linewidth=2)
            plt.title(f"{metric_name.replace('_', ' ').title()} Evolution")
            plt.xlabel("Generation")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            # Add horizontal lines for significant thresholds
            plt.axhline(y=0.33, color='r', linestyle='--', alpha=0.5, label="Nigredo")
            plt.axhline(y=0.66, color='g', linestyle='--', alpha=0.5, label="Albedo")
            plt.axhline(y=0.9, color='y', linestyle='--', alpha=0.5, label="Citrinitas")
            
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(metrics_dir, f"{metric_name}_evolution.png"), dpi=300)
            plt.close()
        
        # Create combined plot
        plt.figure(figsize=(12, 8))
        for metric_name, values in self.history.items():
            if not values:
                continue
            plt.plot(range(len(values)), values, marker='.', linestyle='-', linewidth=2, label=metric_name.replace('_', ' ').title())
        
        plt.title("Thelemic Evolution Metrics")
        plt.xlabel("Generation")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        
        # Save combined plot
        plt.savefig(os.path.join(metrics_dir, "combined_metrics.png"), dpi=300)
        plt.close()
        
        # Save metrics data as JSON
        with open(os.path.join(metrics_dir, "metrics_data.json"), 'w') as f:
            json.dump(self.history, f, indent=2)
            
        logger.info(f"Metrics plots saved to {metrics_dir}")


class BasicVisualizer:
    """
    A basic visualization class for when the full visualizer is not available.
    
    This creates simple 2D plots of the manifold's reduced representation.
    """
    
    def __init__(self, manifold: VectorManifold, save_path: str):
        """
        Initialize the basic visualizer.
        
        Args:
            manifold: The vector manifold to visualize
            save_path: Directory to save visualizations
        """
        self.manifold = manifold
        self.save_path = save_path
        self.viz_dir = os.path.join(save_path, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
    def plot_manifold(self, title: str = "Manifold Visualization", filename: str = None) -> None:
        """
        Create a basic 2D plot of the manifold.
        
        Args:
            title: Title for the plot
            filename: Filename to save the plot (if None, auto-generated)
        """
        if self.manifold.reduced is None:
            logger.warning("Reduced representation not available for visualization")
            return
        
        points = self.manifold.reduced.points
        labels = self.manifold.reduced.labels
        
        if points.shape[1] < 2:
            logger.warning("Not enough dimensions for visualization")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot
        scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', 
                            alpha=0.7, s=100, edgecolors='w', linewidths=0.5)
        
        # Plot cell centroids
        centroids = self.manifold.reduced.cell_centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', 
                   s=150, linewidths=2, label='Cell Centroids')
        
        # Add a colorbar
        plt.colorbar(scatter, label='Cell ID')
        
        # Add cell IDs as text labels to centroids
        for i, centroid in enumerate(centroids):
            plt.text(centroid[0], centroid[1], str(i), 
                    fontsize=10, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'))
        
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        if filename is None:
            filename = os.path.join(self.viz_dir, f"manifold_{int(time.time())}.png")
        else:
            filename = os.path.join(self.viz_dir, filename)
            
        plt.savefig(filename, dpi=300)
        plt.close()
        
        logger.info(f"Basic visualization saved to {filename}")
        
    def create_animation(self, output_file: str, fps: int = 2) -> None:
        """
        Create a simple animation from saved visualization frames.
        
        Args:
            output_file: Path to save the animation
            fps: Frames per second
        """
        logger.info("Animation creation requires the full visualization module")
        logger.info("Individual frame images are available in the visualization directory")


def run_thelemic_evolution(
    generations: int = 22,  # One for each Major Arcana
    save_path: str = None,
    model_name: str = "bert-base-uncased",
    n_cells: int = 22,
    random_state: int = 93  # Significant in Thelema/Crowley's work
):
    """
    Run the Thelemic evolution of the word manifold.
    
    Args:
        generations: Number of generations to evolve
        save_path: Directory to save results
        model_name: Transformer model for word embeddings
        n_cells: Number of cells in the manifold
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (system, metrics_tracker)
    """
    start_time = time.time()
    
    # Create directories
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Results will be saved to {save_path}")
    
    # Setup occult banner
    logger.info("=" * 80)
    logger.info("                    THELEMIC EVOLUTION OF THE WORD MANIFOLD")
    logger.info("                  \"Do what thou wilt shall be the whole of the Law\"")
    logger.info("=" * 80)
    
    # Initialize WordEmbeddings
    logger.info("Initializing WordEmbeddings with occult terms")
    embeddings = WordEmbeddings(model_name=model_name)
    embeddings.load_terms()  # Uses default OCCULT_TERMS set
    
    # Initialize VectorManifold
    logger.info(f"Creating VectorManifold with {n_cells} cells")
    manifold = VectorManifold(
        embeddings,
        n_cells=n_cells,
        random_state=random_state
    )
    
    # Create predefined rules
    logger.info("Creating predefined cellular rules")
    rules = create_predefined_rules()
    
    # Initialize AutomataSystem with Thelemic evolution pattern
    logger.info("Initializing AutomataSystem with Thelemic evolution pattern")
    system = AutomataSystem(
        manifold=manifold,
        rules_dict=rules,
        evolution_pattern=EvolutionPattern.THELEMIC,
        save_path=save_path
    )
    
    # Initialize metrics tracker
    metrics_tracker = ThelemaMetrics()
    
    # Initialize visualizer
    if VISUALIZATION_AVAILABLE and save_path:
        logger.info("Initializing ManifoldVisualizer")
        visualizer = ManifoldVisualizer(manifold, save_path=save_path)
    else:
        logger.info("Initializing BasicVisualizer")
        visualizer = BasicVisualizer(manifold, save_path=save_path)
    
    # Create initial visualization
    logger.info("Creating initial visualization")
    visualizer.plot_manifold(
        title="Initial State (Generation 0)",
        filename="generation_0000.png"
    )
    
    # Calculate initial metrics
    initial_metrics = metrics_tracker.calculate_metrics(manifold, system)
    logger.info("Initial metrics:")
    for metric, value in initial_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Run evolution for specified number of generations
    logger.info(f"Running Thelemic evolution for {generations} generations")
    
    for gen in range(1, generations + 1):
        gen_start = time.time()
        logger.info(f"Generation {gen}/{generations}")
        
        # Evolve the system for one generation
        system.evolve(1)
        
        # Create visualization (every 2 generations to save time)
        if gen % 2 == 0 or gen == generations:
            logger.info(f"Creating visualization for generation {gen}")
            visualizer.plot_manifold(
                title=f"Generation {gen}",
                filename=f"generation_{gen:04d}.png"
            )
        
        # Calculate and log metrics
        metrics = metrics_tracker.calculate_metrics(manifold, system)
        logger.info(f"Generation {gen} metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Log the rule that was applied in this generation
        if system.history and len(system.history) > 0:
            latest_state = system.history[-1]
            active_rules = latest_state.active_rules
            logger.info(f"Applied rule(s): {', '.join(active_rules)}")
        
        gen_time = time.time() - gen_start
        logger.info(f"Generation {gen} completed in {gen_time:.2f} seconds")
    
    # Create animation if full visualizer is available
    if VISUALIZATION_AVAILABLE and save_path:
        logger.info("Creating evolution animation")
        animation_file = os.path.join(save_path, "evolution.mp4")
        visualizer.create_animation(animation_file, fps=2)
    
    # Plot metrics
    if save_path:
        logger.info("Plotting metrics")
        metrics_tracker.plot_metrics(save_path)
    
    # Log completion
    total_time = time.time() - start_time
    logger.info(f"Thelemic evolution completed in {total_time:.2f} seconds")
    logger.info("FINAL METRICS:")
    for metric, values in metrics_tracker.history.items():
        if values:
            logger.info(f"  {metric}: {values[-1]:.4f}")
    
    logger.info("=" * 80)
    logger.info("                          \"Love is the law, love under will\"")
    logger.info("=" * 80)
    
    return system, metrics_tracker


if __name__ == "__main__":
    """
    Execute the Thelemic evolution example.
    
    This will run a complete evolution of the word manifold according to
    Thelemic principles, tracking metrics and creating visualizations.
    """
    # Create timestamped directory for results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("data", "thelemic_evolution", timestamp)
    
    # Parameters
    generations = 22  # One for each Major Arcana
    
    # Run the evolution
    logger.info(f"Running Thelemic evolution with {generations} generations")
    logger.info(f"Results will be saved to {save_dir}")
    
    system, metrics = run_thelemic_evolution(
        generations=generations,
        save_path=save_dir,
        n_cells=22,
    )
    
    logger.info(f"Evolution complete. Results saved to {save_dir}")
    
    # Print final state summary
    if system.history:
        latest = system.history[-1]
        logger.info(f"Final generation: {latest.generation}")
        logger.info(f"Active rules in final generation: {', '.join(latest.active_rules)}")
        
    # Create a summary file
    summary_file = os.path.join(save_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("THELEMIC EVOLUTION SUMMARY\n")
        f.write("=========================\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Generations: {generations}\n")
        f.write(f"Cells: {22}\n\n")
        
        f.write("FINAL METRICS:\n")
        for metric, values in metrics.history.items():
            if values:
                f.write(f"{metric}: {values[-1]:.4f}\n")
        
        f.write("\nEVOLUTION PATH:\n")
        for i, state in enumerate(system.history):
            f.write(f"Generation {i}: {', '.join(state.active_rules)}\n")
    
    logger.info(f"Summary written to {summary_file}")

