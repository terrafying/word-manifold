"""
HyperTools Visualizer

This module provides visualization capabilities for high-dimensional data using HyperTools.
It handles trajectory visualization, dimensionality reduction, and animation generation with
support for real-time interactions and multiple visualization layers.
"""

import logging
import numpy as np
import os
from pathlib import Path
import hypertools as hyp
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import warnings
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ..embeddings.word_embeddings import WordEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='umap')

class VisualizationLayer:
    """A layer in the visualization stack."""
    def __init__(self, name: str, visible: bool = True):
        self.name = name
        self.visible = visible
        self.data = {}
        self.style = {}
        
    def update(self, data: Dict[str, Any]) -> None:
        """Update layer data."""
        self.data.update(data)
        
    def set_style(self, **kwargs) -> None:
        """Set layer style properties."""
        self.style.update(kwargs)

class HyperToolsVisualizer:
    """Visualizer for high-dimensional data using HyperTools."""
    
    def __init__(self, 
                 word_embeddings: WordEmbeddings,
                 output_dir: str = "visualizations/hypertools",
                 interactive: bool = False,
                 n_dimensions: int = 3,
                 reduction_method: str = "UMAP",
                 enable_sacred_geometry: bool = True,
                 enable_audio: bool = False):
        """
        Initialize the visualizer.
        
        Args:
            word_embeddings: WordEmbeddings instance for term processing
            output_dir: Directory to save visualizations
            interactive: Whether to create interactive plots
            n_dimensions: Number of dimensions for reduction (2 or 3)
            reduction_method: Method for dimensionality reduction ('UMAP' or 'PCA')
            enable_sacred_geometry: Whether to enable sacred geometry overlays
            enable_audio: Whether to enable audio feedback
        """
        if n_dimensions not in {2, 3}:
            raise ValueError("n_dimensions must be either 2 or 3")
            
        if reduction_method not in {"UMAP", "PCA"}:
            raise ValueError("reduction_method must be either 'UMAP' or 'PCA'")
            
        self.word_embeddings = word_embeddings
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interactive = interactive
        self.n_dimensions = n_dimensions
        self.reduction_method = reduction_method
        self.enable_sacred_geometry = enable_sacred_geometry
        self.enable_audio = enable_audio
        
        # Initialize visualization layers
        self.layers = {
            'base': VisualizationLayer('Base Trajectories'),
            'sacred_geometry': VisualizationLayer('Sacred Geometry', enable_sacred_geometry),
            'emotional': VisualizationLayer('Emotional Resonance'),
            'numerological': VisualizationLayer('Numerological Significance'),
            'archetypes': VisualizationLayer('Archetypal Patterns')
        }
        
        # Setup interactive elements
        if interactive:
            self._setup_interactive_elements()
        
        logger.info(f"Initialized HyperToolsVisualizer with {n_dimensions}D visualization")
        
    def _setup_interactive_elements(self) -> None:
        """Setup interactive visualization elements."""
        self.interactive_elements = {
            'layer_toggles': {},
            'timeline_slider': None,
            'transformation_controls': {},
            'sacred_geometry_controls': {}
        }
    
    def add_sacred_geometry_overlay(self, ax: plt.Axes, data: np.ndarray) -> None:
        """Add sacred geometry patterns based on data structure."""
        if not self.enable_sacred_geometry:
            return
            
        # Calculate geometric centers and patterns
        centers = np.mean(data, axis=0)
        scale = np.std(data, axis=0)
        
        if self.n_dimensions == 3:
            # Add platonic solids
            self._add_platonic_solid(ax, centers, scale, 'tetrahedron')
            self._add_platonic_solid(ax, centers, scale, 'cube')
            self._add_platonic_solid(ax, centers, scale, 'octahedron')
        else:
            # Add sacred geometry patterns
            self._add_sacred_pattern(ax, centers, scale, 'flower_of_life')
            self._add_sacred_pattern(ax, centers, scale, 'metatrons_cube')
    
    def _add_platonic_solid(self, ax: plt.Axes, center: np.ndarray, scale: np.ndarray, solid_type: str) -> None:
        """Add a platonic solid to the visualization."""
        vertices = self._get_platonic_solid_vertices(solid_type)
        vertices = vertices * scale + center
        
        # Create faces and add to plot
        faces = self._get_platonic_solid_faces(solid_type)
        poly3d = Poly3DCollection([vertices[face] for face in faces],
                                alpha=0.2, facecolor='gold', edgecolor='black')
        ax.add_collection3d(poly3d)
    
    def _add_sacred_pattern(self, ax: plt.Axes, center: np.ndarray, scale: float, pattern_type: str) -> None:
        """Add a sacred geometry pattern to the visualization."""
        if pattern_type == 'flower_of_life':
            self._draw_flower_of_life(ax, center, scale)
        elif pattern_type == 'metatrons_cube':
            self._draw_metatrons_cube(ax, center, scale)
    
    def visualize_term_evolution(self,
                               term_trajectories: Dict[str, List[np.ndarray]],
                               phase_names: Optional[List[str]] = None,
                               title: str = "Term Evolution",
                               save_path: Optional[str] = None,
                               show_layers: Optional[List[str]] = None) -> Optional[str]:
        """
        Create a static visualization of term evolution trajectories.
        
        Args:
            term_trajectories: Dictionary mapping terms to their trajectory vectors
            phase_names: Optional list of phase names for labeling
            title: Plot title
            save_path: Optional path to save the visualization
            show_layers: Optional list of layer names to show
            
        Returns:
            Path to saved visualization if successful, None otherwise
        """
        try:
            # Prepare data
            data = []
            labels = []
            for term, trajectory in term_trajectories.items():
                data.extend(trajectory)
                labels.extend([term] * len(trajectory))
            
            data = np.array(data)
            reduced_data = self._reduce_dimensions(data)
            
            # Create plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d' if self.n_dimensions == 3 else None)
            
            # Plot base trajectories
            if 'base' in (show_layers or self.layers.keys()):
                self._plot_base_trajectories(ax, reduced_data, term_trajectories)
            
            # Add sacred geometry overlay
            if self.enable_sacred_geometry and 'sacred_geometry' in (show_layers or self.layers.keys()):
                self.add_sacred_geometry_overlay(ax, reduced_data)
            
            # Add emotional resonance layer
            if 'emotional' in (show_layers or self.layers.keys()):
                self._add_emotional_layer(ax, reduced_data, term_trajectories)
            
            # Add numerological significance
            if 'numerological' in (show_layers or self.layers.keys()):
                self._add_numerological_layer(ax, reduced_data, term_trajectories)
            
            # Add archetypal patterns
            if 'archetypes' in (show_layers or self.layers.keys()):
                self._add_archetypal_layer(ax, reduced_data, term_trajectories)
            
            # Customize appearance
            self._customize_plot_appearance(ax, title)
            
            # Save plot
            if not save_path:
                save_path = self.output_dir / f"{title.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Error creating term evolution visualization: {str(e)}")
            return None

    def create_animated_ritual(self,
                             term_trajectories: Dict[str, List[np.ndarray]],
                             phase_names: Optional[List[str]] = None,
                             title: str = "Ritual Evolution",
                             duration: float = 10.0,
                             fps: int = 30,
                             add_trails: bool = True,
                             save_path: Optional[str] = None,
                             show_layers: Optional[List[str]] = None) -> Optional[str]:
        """
        Create an animated visualization of the ritual evolution.
        
        Args:
            term_trajectories: Dictionary mapping terms to their trajectory vectors
            phase_names: Optional list of phase names for labeling
            title: Animation title
            duration: Duration of animation in seconds
            fps: Frames per second
            add_trails: Whether to add trailing effects
            save_path: Optional path to save the animation
            show_layers: Optional list of layer names to show
            
        Returns:
            Path to saved animation if successful, None otherwise
        """
        try:
            # Prepare data
            data = []
            labels = []
            for term, trajectory in term_trajectories.items():
                data.extend(trajectory)
                labels.extend([term] * len(trajectory))
            
            data = np.array(data)
            reduced_data = self._reduce_dimensions(data)
            
            # Create figure
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d' if self.n_dimensions == 3 else None)
            
            # Setup animation parameters
            n_frames = int(duration * fps)
            unique_terms = list(term_trajectories.keys())
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_terms)))
            
            def update(frame):
                ax.clear()
                progress = frame / n_frames
                
                # Update base trajectories
                if 'base' in (show_layers or self.layers.keys()):
                    self._update_trajectories(ax, reduced_data, unique_terms, colors, progress, add_trails)
                
                # Update sacred geometry
                if self.enable_sacred_geometry and 'sacred_geometry' in (show_layers or self.layers.keys()):
                    self._update_sacred_geometry(ax, reduced_data, progress)
                
                # Update emotional resonance
                if 'emotional' in (show_layers or self.layers.keys()):
                    self._update_emotional_layer(ax, reduced_data, progress)
                
                # Update numerological significance
                if 'numerological' in (show_layers or self.layers.keys()):
                    self._update_numerological_layer(ax, reduced_data, progress)
                
                # Update archetypal patterns
                if 'archetypes' in (show_layers or self.layers.keys()):
                    self._update_archetypal_layer(ax, reduced_data, progress)
                
                # Customize appearance
                self._customize_plot_appearance(ax, f"{title} - Progress: {progress:.1%}")
            
            # Create animation
            anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps)
            
            # Save animation
            if not save_path:
                save_path = self.output_dir / f"{title.lower().replace(' ', '_')}.gif"
            
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Error creating ritual animation: {str(e)}")
            return None

    def _plot_base_trajectories(self, ax: plt.Axes, reduced_data: np.ndarray, 
                              term_trajectories: Dict[str, List[np.ndarray]]) -> None:
        """Plot base term trajectories."""
        unique_terms = list(term_trajectories.keys())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_terms)))
        
        for i, term in enumerate(unique_terms):
            trajectory = reduced_data[i::len(unique_terms)]
            if self.n_dimensions == 3:
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                       color=colors[i], label=term, marker='o')
            else:
                ax.plot(trajectory[:, 0], trajectory[:, 1],
                       color=colors[i], label=term, marker='o')

    def _customize_plot_appearance(self, ax: plt.Axes, title: str) -> None:
        """Customize the appearance of the plot."""
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        if self.n_dimensions == 3:
            ax.set_zlabel('Dimension 3')
        ax.set_title(title)
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set background color
        ax.set_facecolor('black')
        ax.figure.set_facecolor('black')
        
        # Customize tick colors
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        if self.n_dimensions == 3:
            ax.zaxis.label.set_color('white')
        
        # Set title color
        ax.title.set_color('white')
        
        # Customize legend
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

    def _reduce_dimensions(self, data: np.ndarray) -> np.ndarray:
        """
        Reduce dimensionality of data using specified method.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            
        Returns:
            Reduced data array of shape (n_samples, n_dimensions)
        """
        try:
            if self.reduction_method == "UMAP":
                from umap import UMAP
                reducer = UMAP(
                    n_components=self.n_dimensions,
                    random_state=42,
                    n_neighbors=15,
                    min_dist=0.1
                )
            else:  # PCA
                from sklearn.decomposition import PCA
                reducer = PCA(
                    n_components=self.n_dimensions,
                    random_state=42
                )
            
            # Handle empty or invalid input
            if data.size == 0 or np.any(~np.isfinite(data)):
                logger.warning("Invalid input data detected")
                return np.zeros((data.shape[0], self.n_dimensions))
            
            return reducer.fit_transform(data)
            
        except Exception as e:
            logger.error(f"Error in dimension reduction: {str(e)}")
            return np.zeros((data.shape[0], self.n_dimensions))

    def process_transformation(self, data: np.ndarray) -> np.ndarray:
        """
        Process data transformation for visualization.
        
        Args:
            data: Input data array
            
        Returns:
            Transformed data array
        """
        try:
            # Apply dimensionality reduction
            if self.reduction_method == "UMAP":
                from umap import UMAP
                reducer = UMAP(n_components=self.n_dimensions)
                transformed = reducer.fit_transform(data)
            else:  # PCA
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=self.n_dimensions)
                transformed = reducer.fit_transform(data)
                
            return transformed
        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}")
            return data
    
    def calculate_energy_levels(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate energy levels for data points.
        
        Args:
            data: Input data array
            
        Returns:
            Array of energy levels
        """
        try:
            # Calculate L2 norm as energy level
            return np.linalg.norm(data, axis=1)
        except Exception as e:
            logger.error(f"Error calculating energy levels: {str(e)}")
            return np.ones(len(data))
    
    def determine_resonance(self, data: np.ndarray) -> List[bool]:
        """
        Determine resonance patterns in the data.
        
        Args:
            data: Input data array
            
        Returns:
            List of boolean resonance indicators
        """
        try:
            energy_levels = self.calculate_energy_levels(data)
            mean_energy = np.mean(energy_levels)
            return [e > mean_energy for e in energy_levels]
        except Exception as e:
            logger.error(f"Error determining resonance: {str(e)}")
            return [False] * len(data)
    
    def identify_dominant_principle(self, data: np.ndarray) -> str:
        """
        Identify the dominant hermetic principle in the data.
        
        Args:
            data: Input data array
            
        Returns:
            Name of the dominant principle
        """
        try:
            # Simplified principle identification based on data characteristics
            variance = np.var(data)
            if variance > 1.0:
                return "Vibration"
            elif np.mean(data) > 0:
                return "Polarity"
            else:
                return "Correspondence"
        except Exception as e:
            logger.error(f"Error identifying dominant principle: {str(e)}")
            return "Unknown"
    
    def generate_visualization(self, 
                             data: np.ndarray,
                             labels: Optional[List[str]] = None,
                             title: str = "Semantic Space Visualization",
                             save_path: Optional[str] = None) -> None:
        """
        Generate and save a visualization.
        
        Args:
            data: Input data array
            labels: Optional list of labels for data points
            title: Title for the visualization
            save_path: Optional path to save the visualization
        """
        try:
            # Transform data if needed
            if data.shape[1] > self.n_dimensions:
                data = self.process_transformation(data)
            
            # Create the plot
            fig = plt.figure(figsize=(12, 8))
            if self.n_dimensions == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
            
            # Plot data points
            scatter = ax.scatter(data[:, 0], data[:, 1],
                               *([data[:, 2]] if self.n_dimensions == 3 else []),
                               c=self.calculate_energy_levels(data),
                               cmap='viridis')
            
            # Add labels if provided
            if labels:
                for i, label in enumerate(labels):
                    ax.annotate(label, (data[i, 0], data[i, 1]))
            
            # Add colorbar
            plt.colorbar(scatter, label='Energy Level')
            
            # Set title and labels
            ax.set_title(title)
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            if self.n_dimensions == 3:
                ax.set_zlabel('Dimension 3')
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved visualization to {save_path}")
            elif self.interactive:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
    
    def visualize_terms(self, 
                       terms: List[str],
                       title: str = "Term Space Visualization",
                       save_path: Optional[str] = None) -> None:
        """
        Visualize a set of terms using their embeddings.
        
        Args:
            terms: List of terms to visualize
            title: Title for the visualization
            save_path: Optional path to save the visualization
        """
        try:
            # Get embeddings for terms
            embeddings = []
            valid_terms = []
            for term in terms:
                embedding = self.word_embeddings.get_embedding(term)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_terms.append(term)
            
            if not embeddings:
                logger.warning("No valid embeddings found for terms")
                return
                
            embeddings = np.array(embeddings)
            
            # Generate visualization
            self.generate_visualization(
                data=embeddings,
                labels=valid_terms,
                title=title,
                save_path=save_path or os.path.join(self.output_dir, "term_space.png")
            )
            
        except Exception as e:
            logger.error(f"Error visualizing terms: {str(e)}")
    
    def visualize_transformations(self,
                                before_terms: List[str],
                                after_terms: List[str],
                                title: str = "Transformation Visualization",
                                save_path: Optional[str] = None) -> None:
        """
        Visualize term transformations.
        
        Args:
            before_terms: Terms before transformation
            after_terms: Terms after transformation
            title: Title for the visualization
            save_path: Optional path to save the visualization
        """
        try:
            # Get embeddings for both sets of terms
            before_embeddings = []
            after_embeddings = []
            valid_before = []
            valid_after = []
            
            for term in before_terms:
                embedding = self.word_embeddings.get_embedding(term)
                if embedding is not None:
                    before_embeddings.append(embedding)
                    valid_before.append(term)
                    
            for term in after_terms:
                embedding = self.word_embeddings.get_embedding(term)
                if embedding is not None:
                    after_embeddings.append(embedding)
                    valid_after.append(term)
            
            if not before_embeddings or not after_embeddings:
                logger.warning("No valid embeddings found for transformation")
                return
                
            # Combine embeddings
            all_embeddings = np.vstack([before_embeddings, after_embeddings])
            all_labels = valid_before + valid_after
            
            # Generate visualization with arrows showing transformations
            self.generate_visualization(
                data=all_embeddings,
                labels=all_labels,
                title=title,
                save_path=save_path or os.path.join(self.output_dir, "transformation.png")
            )
            
            # Draw arrows between corresponding terms
            if len(valid_before) == len(valid_after):
                transformed = self.process_transformation(all_embeddings)
                n = len(valid_before)
                for i in range(n):
                    start = transformed[i]
                    end = transformed[i + n]
                    if self.n_dimensions == 3:
                        plt.gca().quiver(start[0], start[1], start[2],
                                       end[0] - start[0], end[1] - start[1], end[2] - start[2],
                                       color='red', alpha=0.5)
                    else:
                        plt.gca().arrow(start[0], start[1],
                                      end[0] - start[0], end[1] - start[1],
                                      color='red', alpha=0.5)
            
        except Exception as e:
            logger.error(f"Error visualizing transformations: {str(e)}")
