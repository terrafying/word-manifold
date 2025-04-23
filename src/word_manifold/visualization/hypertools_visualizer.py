"""
HyperTools Visualizer Module.

This module provides advanced high-dimensional data visualization using HyperTools,
enabling more fluid and interactive demonstrations of semantic transformations
during ritual workings.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add local hypertools to path and import
hypertools_path = Path(__file__).parent.parent.parent.parent / 'hypertools'
sys.path.insert(0, str(hypertools_path))
import hypertools as hyp

import warnings
import datetime
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.interpolate import Rbf  # For force field interpolation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import time

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Monkey patch for NumPy 2.0 compatibility
def is_string_like(obj):
    """Check if the object is a string-like object (str or bytes)"""
    return isinstance(obj, (str, bytes))

# Apply the patch to hypertools
try:
    # Get the reduce function from hypertools.tools
    if hasattr(hyp.tools, 'reduce'):
        original_reduce = hyp.tools.reduce
    else:
        # If reduce isn't directly accessible, try to get it from the module
        from hypertools.tools.reduce import reduce as original_reduce
    cache = {}

    
    # Define the patched reduce function
    # It seems the 'method' argument should be passed positionally in newer HyperTools versions
    def patched_reduce(x, *args, method=None, **kwargs): # Keep method as keyword for flexibility in receiving
        # Create a cache key from the arguments
        key = (x.tobytes(), args, tuple(sorted(kwargs.items())), method) # Include method in key

        if key in cache:
            logger.debug("Using cached reduction result")
            return cache[key]
        else:
            logger.debug("Calculating new reduction result")
            # Call the original reduce function
            # Pass the method as a positional argument if it's provided
            if method is not None:
                # Assuming the API is now reduce(data, method_name, ndims=...)
                result = original_reduce(x, method, *args, **kwargs)
            else:
                # If no method is specified, let HyperTools use its default (likely PCA)
                result = original_reduce(x, *args, **kwargs)

            cache[key] = result
            return result

    # Apply the patch
    hyp.tools.reduce = patched_reduce

    # Apply the patch
    if hasattr(hyp.tools, 'reduce'):
        hyp.tools.reduce = patched_reduce
    else:
        # If we got it from the module directly, patch that
        import hypertools.tools.reduce
        hypertools.tools.reduce.reduce = patched_reduce

    logger.info("Applied NumPy 2.0 compatibility patch to HyperTools")
except Exception as e:
    logger.warning(f"Could not apply NumPy 2.0 compatibility patch: {e}")

class HyperToolsVisualizer:
    """
    Advanced high-dimensional visualization class using HyperTools.
    
    This visualizer provides fluid, interactive visualizations of semantic
    transformations in high-dimensional spaces, with advanced animation
    capabilities for ritual evolution processes and 4D+ rotations.
    """
    
    def __init__(
        self,
        output_dir: str = "visualizations/hypertools",
        color_palette: str = "viridis",
        n_dimensions: int = 4,  # Default to 4D
        save_format: str = "html",
        interactive: bool = True
    ):
        """Initialize the HyperTools visualizer with enhanced options."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_palette = color_palette
        self.n_dimensions = n_dimensions
        self.save_format = save_format
        self.interactive = interactive
        
        # Animation settings
        self.max_frames = 1000  # Maximum number of frames to cache
        self.frame_interval = 50  # milliseconds between frames (20 fps)
        self.save_count = 500  # Number of frames to save for export
        
        # Rotation state
        self.rotation_angles = np.zeros((self.n_dimensions, self.n_dimensions))
        self.rotation_speeds = np.zeros((self.n_dimensions, self.n_dimensions))
        
        # Trail settings
        self.trail_length = 20  # Number of previous positions to show
        self.trail_fade = 0.95  # How quickly trails fade (0-1)
        
        # Force field settings
        self.force_field_resolution = 20  # Grid resolution for force field
        self.force_field_strength = 0.5  # Strength of force field visualization
        
        # Delta tracking
        self.previous_state = None
        self.transformation_type = None
        self.delta_history = []  # Store recent deltas for visualization
        self.delta_window = 50  # Number of deltas to keep in history
        
        # Set default styling for plots
        sns.set_style("darkgrid")
        plt.rcParams.update({
            'figure.facecolor': '#1E1E1E',  # Dark gray background
            'axes.facecolor': '#2D2D2D',    # Slightly lighter gray for axes
            'axes.edgecolor': '#CCCCCC',    # Light gray edges
            'axes.labelcolor': '#FFFFFF',    # White labels
            'axes.titlecolor': '#FFFFFF',    # White title
            'xtick.color': '#FFFFFF',       # White ticks
            'ytick.color': '#FFFFFF',       # White ticks
            'text.color': '#FFFFFF',        # White text
            'grid.color': '#404040',        # Dark gray grid
            'grid.alpha': 0.3,              # Semi-transparent grid
            'savefig.facecolor': '#1E1E1E', # Dark gray for saved figures
            'savefig.edgecolor': 'none',    # No edge color for saved figures
            'figure.edgecolor': 'none',     # No edge color for figure
        })
        
        # Add data conversion utilities
        self.array_dtypes = {
            'float': np.float32,
            'int': np.int32,
            'str': str
        }

        # Define known polarity pairs for semantic analysis
        self.polarity_pairs = {
            'celestial': ('sun', 'moon'),
            'vertical': ('up', 'down'),
            'horizontal': ('left', 'right'),
            'temporal': ('past', 'future'),
            'thermal': ('hot', 'cold'),
            'moral': ('good', 'evil'),
            'emotional': ('joy', 'sorrow'),
            'physical': ('light', 'heavy'),
            'spatial': ('near', 'far'),
            'energetic': ('active', 'passive')
        }

        # Cache for polarity vectors
        self.polarity_vectors = {}

        # Set default color maps with better visibility
        self.color_maps = {
            'emotion': plt.get_cmap('RdYlBu'),     # Red-Yellow-Blue for emotional valence
            'complexity': plt.get_cmap('viridis'),  # Viridis for complexity
            'abstraction': plt.get_cmap('plasma'),  # Plasma for concrete-abstract spectrum
            'energy': plt.get_cmap('magma'),        # Magma for energy/intensity
            'delta': plt.get_cmap('coolwarm')       # Coolwarm for transformation deltas
        }

        # Set default scatter plot settings
        self.scatter_settings = {
            'alpha': 0.8,           # Point transparency
            'edgecolor': '#FFFFFF', # White edges
            'linewidth': 1,         # Edge width
            's': 100,              # Point size
        }

    def _ensure_numpy_array(self, data, dtype='float'):
        """Convert input to numpy array with proper dtype."""
        if data is None:
            return None
            
        if isinstance(data, np.ndarray):
            # If already numpy array, just ensure correct dtype
            return data.astype(self.array_dtypes[dtype])
            
        try:
            # Handle lists, tuples, etc.
            if dtype == 'str':
                # For string data, convert to object array to handle variable length strings
                return np.array(data, dtype=object)
            else:
                return np.array(data, dtype=self.array_dtypes[dtype])
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert data to numpy array: {e}")
            return data

    def _prepare_plot_data(self, data, groups=None, labels=None):
        """Prepare all data for plotting by ensuring proper numpy arrays."""
        logger.info("Preparing data for plotting...")
        
        # Convert main data
        if isinstance(data, list) and len(data) > 0:
            # Check if we're dealing with a list of arrays/lists
            if isinstance(data[0], (np.ndarray, list)):
                # Stack arrays if they're the same shape
                try:
                    prepared_data = np.vstack([self._ensure_numpy_array(d, 'float') for d in data])
                except ValueError as e:
                    logger.warning(f"Could not stack arrays: {e}. Attempting direct conversion...")
                    prepared_data = self._ensure_numpy_array(data, 'float')
            else:
                prepared_data = self._ensure_numpy_array(data, 'float')
        else:
            prepared_data = self._ensure_numpy_array(data, 'float')
        
        # Convert groups/hue data if present
        if groups is not None:
            if all(isinstance(x, (int, np.integer)) for x in groups):
                prepared_groups = self._ensure_numpy_array(groups, 'int')
            else:
                prepared_groups = self._ensure_numpy_array(groups, 'str')
        else:
            prepared_groups = None
        
        # Convert labels if present
        prepared_labels = self._ensure_numpy_array(labels, 'str') if labels is not None else None
        
        # Log conversion results
        logger.info(f"Data shape after preparation: {prepared_data.shape if prepared_data is not None else 'None'}")
        logger.info(f"Groups shape after preparation: {prepared_groups.shape if prepared_groups is not None else 'None'}")
        logger.info(f"Labels shape after preparation: {prepared_labels.shape if prepared_labels is not None else 'None'}")
        
        return prepared_data, prepared_groups, prepared_labels

    def _create_rotation_matrix(self, i: int, j: int, angle: float, dim: int) -> np.ndarray:
        """Create a rotation matrix for rotating in the i-j plane."""
        matrix = np.eye(dim)
        matrix[i, i] = np.cos(angle)
        matrix[i, j] = -np.sin(angle)
        matrix[j, i] = np.sin(angle)
        matrix[j, j] = np.cos(angle)
        return matrix

    def _apply_rotations(self, points: np.ndarray) -> np.ndarray:
        """Apply all current rotations to the points."""
        rotated = points.copy()
        for i in range(self.n_dimensions):
            for j in range(i + 1, self.n_dimensions):
                if self.rotation_angles[i, j] != 0:
                    rot_matrix = self._create_rotation_matrix(
                        i, j, self.rotation_angles[i, j], self.n_dimensions
                    )
                    rotated = np.dot(rotated, rot_matrix)
        return rotated

    def _update_rotation_state(self, dt: float):
        """Update rotation angles based on speeds."""
        self.rotation_angles += self.rotation_speeds * dt
        # Normalize angles to [0, 2π]
        self.rotation_angles = np.mod(self.rotation_angles, 2 * np.pi)

    def set_rotation_speed(self, dim1: int, dim2: int, speed: float):
        """Set the rotation speed for a specific plane."""
        if 0 <= dim1 < self.n_dimensions and 0 <= dim2 < self.n_dimensions:
            self.rotation_speeds[dim1, dim2] = speed
            self.rotation_speeds[dim2, dim1] = -speed  # Keep matrix antisymmetric

    def project_to_3d(self, points: np.ndarray, projection_dims: List[int] = None) -> np.ndarray:
        """Project n-dimensional points to 3D space."""
        if projection_dims is None:
            projection_dims = [0, 1, 2]  # Default to first three dimensions
        return points[:, projection_dims]

    def _calculate_force_field(self, points: np.ndarray, projected_dims: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate force field based on point positions."""
        # Get 3D projected points
        points_3d = self.project_to_3d(points, projected_dims)
        
        # Create grid
        x_min, x_max = points_3d[:, 0].min(), points_3d[:, 0].max()
        y_min, y_max = points_3d[:, 1].min(), points_3d[:, 1].max()
        z_min, z_max = points_3d[:, 2].min(), points_3d[:, 2].max()
        
        # Add margin
        margin = 0.1
        x_margin = (x_max - x_min) * margin
        y_margin = (y_max - y_min) * margin
        z_margin = (z_max - z_min) * margin
        
        x = np.linspace(x_min - x_margin, x_max + x_margin, self.force_field_resolution)
        y = np.linspace(y_min - y_margin, y_max + y_margin, self.force_field_resolution)
        z = np.linspace(z_min - z_margin, z_max + z_margin, self.force_field_resolution)
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Calculate forces using Radial Basis Functions
        forces = np.zeros((self.force_field_resolution, self.force_field_resolution, self.force_field_resolution, 3))
        
        # Calculate pairwise distances
        grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        distances = cdist(grid_points, points_3d)
        
        # Calculate force vectors
        for i in range(3):  # For each dimension
            rbf = Rbf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                     points_3d[:, i], function='gaussian')
            forces[..., i] = rbf(X, Y, Z)
        
        # Normalize forces
        magnitudes = np.linalg.norm(forces, axis=3)
        max_magnitude = magnitudes.max()
        if max_magnitude > 0:
            forces /= max_magnitude
        
        return X, Y, Z, forces

    def _calculate_polarity_vector(self, pos_term: str, neg_term: str, embedder) -> np.ndarray:
        """Calculate a normalized polarity vector between two terms."""
        # Get embeddings for both terms
        pos_vec = embedder.get_embedding(pos_term)
        neg_vec = embedder.get_embedding(neg_term)
        
        # Calculate polarity vector (direction from negative to positive)
        polarity_vec = pos_vec - neg_vec
        
        # Normalize to unit vector
        return polarity_vec / np.linalg.norm(polarity_vec)

    def _calculate_polarity_similarity(self, vector: np.ndarray, polarity_vec: np.ndarray) -> float:
        """Calculate cosine similarity between a vector and a polarity axis."""
        # Ensure vectors are normalized
        vector_norm = vector / np.linalg.norm(vector)
        
        # Calculate cosine similarity
        return np.dot(vector_norm, polarity_vec)

    def analyze_semantic_polarities(self, vectors: np.ndarray, embedder, labels: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Analyze vectors in terms of known polarity axes.
        
        Args:
            vectors: Array of vectors to analyze
            embedder: Word embedding model to use for polarity terms
            labels: Optional labels for the vectors
            
        Returns:
            Dictionary mapping polarity names to similarity scores
        """
        # Initialize or update polarity vectors
        for polarity_name, (pos_term, neg_term) in self.polarity_pairs.items():
            if polarity_name not in self.polarity_vectors:
                self.polarity_vectors[polarity_name] = self._calculate_polarity_vector(pos_term, neg_term, embedder)

        # Calculate similarities for each vector along each polarity axis
        similarities = {}
        for polarity_name, polarity_vec in self.polarity_vectors.items():
            similarities[polarity_name] = np.array([
                self._calculate_polarity_similarity(vec, polarity_vec)
                for vec in vectors
            ])

        return similarities

    def visualize_vector_space(
        self,
        vectors: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Semantic Vector Space",
        group_by: Optional[List[int]] = None,
        legend_labels: Optional[List[str]] = None,
        reduce_method: str = 'UMAP',
        projection_dims: Optional[List[int]] = None,
        auto_rotate: bool = True,
        show_trails: bool = True,
        show_force_field: bool = True,
        embedder=None
    ) -> str:
        """Create an interactive visualization of vectors in semantic space."""
        logger.info(f"Creating HyperTools visualization with {reduce_method}")
        
        # Prepare all data with enhanced conversion
        data, groups, plot_labels = self._prepare_plot_data(vectors, group_by, labels)
        
        # Verify data preparation
        if data is None or len(data) == 0:
            raise ValueError("No valid data for visualization")
        
        # Calculate polarity similarities if embedder is provided
        similarities = {}
        if embedder is not None:
            similarities = self.analyze_semantic_polarities(vectors, embedder, labels)
            logger.info(f"Calculated similarities for {len(similarities)} polarity axes")

        # Create figure and axis with explicit background colors
        fig = plt.figure(figsize=(12, 10))
        fig.patch.set_facecolor('#1E1E1E')  # Set figure background
        
        if self.n_dimensions == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
            
        ax.set_facecolor('#2D2D2D')  # Set axes background
        
        # Set up grid with custom styling
        ax.grid(True, linestyle='--', alpha=0.3, color='#404040')
        
        # Reduce dimensionality if needed
        if data.shape[1] > self.n_dimensions:
            try:
                data = hyp.tools.reduce(data, ndims=self.n_dimensions, method=reduce_method)
                logger.info(f"Reduced data dimensions to {data.shape}")
            except Exception as e:
                logger.error(f"Dimensionality reduction failed: {e}")
                raise

        # Initialize state with prepared data and similarities
        state = {
            'auto_rotate': auto_rotate,
            'projection_dims': projection_dims or [0, 1, 2],
            'base_speed': 0.1,
            'speed_multiplier': 1.0,
            'show_trails': show_trails,
            'show_force_field': show_force_field,
            'trail_positions': [],
            'labels': plot_labels,
            'group_by': groups,
            'legend_labels': legend_labels,
            'similarities': similarities,
            'fig': fig,
            'ax': ax,
            'data': data,
            'current_frame': 0
        }

        # Create custom colormap for groups if needed
        if groups is not None:
            unique_groups = len(np.unique(groups))
            colors = plt.cm.rainbow(np.linspace(0, 1, unique_groups))
            state['group_colors'] = colors

        # Set up the plot with enhanced visibility
        try:
            hyp_plot = hyp.plot(
                data,
                '.',
                hue=groups,
                labels=plot_labels,
                legend=groups is not None,
                title=title,
                animate=False,
                ax=ax,
                size=self.scatter_settings['s'],
                alpha=self.scatter_settings['alpha']
            )
        except Exception as e:
            logger.error(f"HyperTools plotting failed: {e}")
            raise

        # Enhance legend visibility if present
        if ax.get_legend() is not None:
            leg = ax.get_legend()
            leg.set_frame_on(True)
            leg.get_frame().set_facecolor('#2D2D2D')
            leg.get_frame().set_edgecolor('#CCCCCC')
            for text in leg.get_texts():
                text.set_color('#FFFFFF')

        # Set title with enhanced visibility
        ax.set_title(title, color='#FFFFFF', fontsize=14, pad=20)

        # Add interactive controls
        self._create_interactive_controls(fig, ax, state)

        # Save with proper background colors
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath_png = self.output_dir / f"visualization_{timestamp}.png"
        
        plt.savefig(
            filepath_png,
            dpi=300,
            bbox_inches='tight',
            facecolor='#1E1E1E',
            edgecolor='none'
        )
        
        logger.info(f"Saved visualization to {filepath_png}")
        return str(filepath_png)

    def _add_trails(self, ax: plt.Axes, state: dict, projected: np.ndarray) -> None:
        """Add motion trails to the visualization."""
        if len(state['trail_positions']) > 0:
            trail_colors = plt.get_cmap('plasma')(
                np.clip(np.linspace(0, 1, len(state['trail_positions'])), 0, 1))
            
            for i, (trail_pos, color) in enumerate(zip(state['trail_positions'], trail_colors)):
                alpha = np.clip(self.trail_fade ** (len(state['trail_positions']) - i), 0, 1)
                if self.n_dimensions > 2:
                    ax.scatter(trail_pos[:, 0], trail_pos[:, 1], trail_pos[:, 2],
                             color=color, alpha=alpha * 0.3, s=2)
                else:
                    ax.scatter(trail_pos[:, 0], trail_pos[:, 1],
                             color=color, alpha=alpha * 0.3, s=2)
        
        # Store current position for trails
        state['trail_positions'].append(projected.copy())
        if len(state['trail_positions']) > self.trail_length:
            state['trail_positions'].pop(0)

    def _add_force_field(self, ax: plt.Axes, state: dict, projected: np.ndarray) -> None:
        """Add force field visualization."""
        if self.n_dimensions > 2:
            X, Y, Z, forces = self._calculate_force_field(projected, state['projection_dims'])
            stride = self.force_field_resolution // 4
            ax.quiver(
                X[::stride, ::stride, ::stride],
                Y[::stride, ::stride, ::stride],
                Z[::stride, ::stride, ::stride],
                forces[::stride, ::stride, ::stride, 0],
                forces[::stride, ::stride, ::stride, 1],
                forces[::stride, ::stride, ::stride, 2],
                color=(1, 1, 1, 0.2),
                length=self.force_field_strength,
                normalize=True
            )
        
        # Add flow lines
        flow_lines = self._create_flow_lines(projected[:, :2])
        for line in flow_lines:
            ax.plot(line[:, 0], line[:, 1],
                   color='white', alpha=0.1, linewidth=1)

    def _add_status_info(self, ax: plt.Axes, state: dict) -> None:
        """Add status information and controls help to the visualization."""
        info_text = [
            f"Frame: {state['current_frame']}",
            f"Auto-rotate: {'On' if state['auto_rotate'] else 'Off'}",
            f"Speed: {state['speed_multiplier']:.1f}x",
            f"Trails: {'On' if state['show_trails'] else 'Off'}",
            f"Force Field: {'On' if state['show_force_field'] else 'Off'}",
            "\nActive Rotations:"
        ]
        
        for i in range(self.n_dimensions):
            for j in range(i + 1, self.n_dimensions):
                if self.rotation_speeds[i, j] != 0:
                    info_text.append(
                        f"D{i+1}-D{j+1}: {self.rotation_angles[i,j]:.1f}rad"
                    )
        
        controls_text = [
            "\nControls:",
            "Space: Toggle auto-rotate",
            "R: Reset view",
            "+/-: Speed up/down",
            "Arrows: Manual rotation",
            "1-9: Select dimension",
            "Shift+1-9: Y dimension",
            "Ctrl+1-9: Z dimension",
            "T: Toggle trails",
            "F: Toggle force field"
        ]
        
        ax.text2D(
            0.02, 0.98, '\n'.join(info_text + controls_text),
            transform=ax.transAxes, color='white',
            bbox=dict(facecolor='black', alpha=0.7),
            fontsize=8, verticalalignment='top'
        )

    def visualize_term_evolution(
        self,
        term_trajectories: Dict[str, List[np.ndarray]],
        phase_names: Optional[List[str]] = None,
        title: str = "Term Evolution in Ritual Space",
        color_by_phase: bool = False,
    ) -> str:
        """
        Create a visualization showing how terms evolve through different phases.
        
        Args:
            term_trajectories: Dictionary mapping terms to lists of vectors representing states
            phase_names: Names of the phases (optional)
            title: Title for the visualization
            color_by_phase: Whether to color trajectories by phase instead of by term
            
        Returns:
            Path to the saved visualization file
        """
        logger.info("Creating term evolution visualization")
        
        # Prepare the data
        all_points = []
        labels = []
        phase_indices = []
        term_indices = []
        
        for term_idx, (term, trajectory) in enumerate(term_trajectories.items()):
            all_points.extend(trajectory)
            for phase_idx, _ in enumerate(trajectory):
                labels.append(f"{term} ({phase_names[phase_idx] if phase_names else phase_idx})")
                phase_indices.append(phase_idx)
                term_indices.append(term_idx)
        
        all_points = np.array(all_points)
        
        # Create figure with increased size for better label spacing
        plt.clf()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up colors
        n_terms = len(term_trajectories)
        n_phases = len(next(iter(term_trajectories.values())))
        
        if color_by_phase:
            colors = plt.cm.viridis(np.linspace(0, 1, n_phases))
            point_colors = colors[phase_indices]
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, n_terms))
            point_colors = colors[term_indices]
        
        # Plot points and trajectories
        scatter = ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            c=point_colors,
            alpha=0.6
        )
        
        # Draw trajectories with arrows
        for term_idx, (term, trajectory) in enumerate(term_trajectories.items()):
            points = np.array(trajectory)
            color = colors[term_idx] if not color_by_phase else 'gray'
            
            # Draw lines between consecutive points
            for i in range(len(points) - 1):
                start = points[i]
                end = points[i + 1]
                
                # Calculate arrow properties
                direction = end - start
                arrow_length = np.linalg.norm(direction) * 0.2
                
                # Draw line with arrow
                ax.quiver(
                    start[0], start[1], start[2],
                    direction[0], direction[1], direction[2],
                    color=color,
                    alpha=0.6,
                    arrow_length_ratio=0.2,
                    length=1.0
                )
        
        # Add labels with improved positioning and visibility
        for i, (point, label) in enumerate(zip(all_points, labels)):
            # Calculate offset based on phase and term
            phase = phase_indices[i]
            term_idx = term_indices[i]
            
            # Add offset to prevent overlapping
            offset_x = 0.1 * (phase + 1)
            offset_y = 0.1 * (term_idx + 1)
            offset_z = 0.1 * (phase + term_idx)
            
            # Create text with background for better visibility
            x, y, z = point[0] + offset_x, point[1] + offset_y, point[2] + offset_z
            
            # Add white background to text for better contrast
            text = ax.text(
                x, y, z, label,
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    edgecolor='none',
                    pad=1
                ),
                fontsize=8,
                ha='left',
                va='bottom'
            )
        
        # Customize the plot
        ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        
        # Add legend
        if color_by_phase and phase_names:
            legend_elements = [
                plt.Line2D([0], [0], color=colors[i], label=phase_names[i])
                for i in range(n_phases)
            ]
        else:
            legend_elements = [
                plt.Line2D([0], [0], color=colors[i], label=list(term_trajectories.keys())[i])
                for i in range(n_terms)
            ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the visualization
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"term_evolution_{timestamp}")
        
        # Save static image
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved term evolution visualization to {output_path}.png")
        
        # Save interactive version if enabled
        if self.save_format == 'html' and self.interactive:
            try:
                plt.savefig(f"{output_path}.html")
                logger.info(f"Saved interactive visualization to {output_path}.html")
            except Exception as e:
                logger.warning(f"Could not save interactive HTML: {str(e)}")
        
        plt.close()
        
        return f"{output_path}.png"

    def create_animated_ritual(
        self,
        term_trajectories: Dict[str, List[np.ndarray]],
        phase_names: Optional[List[str]] = None,
        title: str = "Ritual Evolution Animated",
        duration: float = 10.0,
        fps: int = 30,
        add_trails: bool = True,
        reduce_method: str = 'UMAP'
    ) -> str:
        """
        Create an animated visualization of the ritual evolution process.
        
        Args:
            term_trajectories: Dictionary mapping term names to lists of vector positions
            phase_names: Optional names of ritual phases
            title: Title for the visualization
            duration: Duration of animation in seconds
            fps: Frames per second
            add_trails: Whether to add trailing paths for terms
            reduce_method: Dimensionality reduction method
            
        Returns:
            Path to saved animation file
        """
        logger.info(f"Creating animated ritual visualization")
        
        # Determine max trajectory length
        max_traj_len = max(len(traj) for traj in term_trajectories.values() if traj)
        
        # Default phase names if not provided
        if phase_names is None:
            phase_names = [f"Phase {i+1}" for i in range(max_traj_len)]
        else:
            # Ensure we have enough phase names
            if len(phase_names) < max_traj_len:
                for i in range(len(phase_names), max_traj_len):
                    phase_names.append(f"Phase {i+1}")
        
        # Prepare all data for reduction
        all_vectors = []
        for term, trajectory in term_trajectories.items():
            valid_positions = [pos for pos in trajectory if pos is not None]
            all_vectors.extend(valid_positions)
            
        if not all_vectors:
            logger.warning("No valid vectors for animation")
            return None
            
        # Convert to numpy array
        data = np.array(all_vectors)
        
        # Apply dimensionality reduction once to all data points
        logger.info(f"Applying {reduce_method} to all vectors")
        if reduce_method == 'PCA':
            reduced_data = hyp.tools.reduce(data, ndims=self.n_dimensions, method='PCA')
        elif reduce_method == 'TSNE':
            reduced_data = hyp.tools.reduce(data, ndims=self.n_dimensions, method='TSNE')
        else:  # Default to UMAP
            reduced_data = hyp.tools.reduce(data, ndims=self.n_dimensions, method='UMAP')
            
        # Separate reduced data back into term trajectories
        reduced_trajectories = {}
        idx = 0
        for term, trajectory in term_trajectories.items():
            valid_positions = [pos for pos in trajectory if pos is not None]
            n_valid = len(valid_positions)
            
            if n_valid > 0:
                reduced_trajectories[term] = reduced_data[idx:idx+n_valid]
                idx += n_valid
            else:
                reduced_trajectories[term] = []
                
        # Create animation
        fig = plt.figure(figsize=(12, 10), facecolor='black')
        
        if self.n_dimensions == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('black')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('w')
            ax.yaxis.pane.set_edgecolor('w')
            ax.zaxis.pane.set_edgecolor('w')
        else:
            ax = fig.add_subplot(111)
            ax.set_facecolor('black')
            
        # Create colormap for terms
        term_colors = {}
        cmap = plt.cm.get_cmap(self.color_palette, len(term_trajectories))
        for i, term in enumerate(term_trajectories.keys()):
            term_colors[term] = cmap(i)
            
        # Setup plot elements that will be updated in animation
        scatter_plots = {}
        line_plots = {}
        trails = {}
        
        for term in reduced_trajectories.keys():
            # Initial empty scatter plot for each term
            if self.n_dimensions == 3:
                scatter, = ax.plot([], [], [], 'o', markersize=10, 
                                 color=term_colors[term], label=term)
                # Empty line for trail
                trail, = ax.plot([], [], [], '-', linewidth=2, alpha=0.5, 
                               color=term_colors[term])
            else:
                scatter, = ax.plot([], [], 'o', markersize=10, 
                                 color=term_colors[term], label=term)
                # Empty line for trail
                trail, = ax.plot([], [], '-', linewidth=2, alpha=0.5, 
                               color=term_colors[term])
                
            scatter_plots[term] = scatter
            trails[term] = trail
            
        # Add title
        ax.set_title(title, color='white', fontsize=16)
        
        # Add phase indicator text
        phase_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, 
                           fontsize=14, color='white', 
                           verticalalignment='top')
        
        # Set axis labels
        ax.set_xlabel('Dimension 1', color='white')
        ax.set_ylabel('Dimension 2', color='white')
        if self.n_dimensions == 3:
            ax.set_zlabel('Dimension 3', color='white')
            
        # Add legend
        ax.legend(loc='upper right')
        
        # Set limits - we'll adjust these based on data
        all_points = np.vstack([traj for traj in reduced_trajectories.values() if len(traj) > 0])
        
        if len(all_points) > 0:
            min_vals = np.min(all_points, axis=0)
            max_vals = np.max(all_points, axis=0)
            
            # Add margin
            margin = (max_vals - min_vals) * 0.1
            min_vals -= margin
            max_vals += margin
            
            ax.set_xlim(min_vals[0], max_vals[0])
            ax.set_ylim(min_vals[1], max_vals[1])
            if self.n_dimensions == 3:
                ax.set_zlim(min_vals[2], max_vals[2])
        
        # Animation update function
        def update(frame):
            # Calculate progress (0 to 1)
            progress = frame / (duration * fps)
            
            # Calculate current phase
            phase_idx = int(progress * (max_traj_len - 1))
            if phase_idx >= len(phase_names):
                phase_idx = len(phase_names) - 1
                
            # Update phase text
            phase_text.set_text(f"Current Phase: {phase_names[phase_idx]}")
            
            # Interpolate between phases for smooth transitions
            phase_progress = progress * (max_traj_len - 1)
            current_phase = int(phase_progress)
            phase_alpha = phase_progress - current_phase
            
            next_phase = min(current_phase + 1, max_traj_len - 1)
            
            # Update each term's position
            for term, trajectory in reduced_trajectories.items():
                if len(trajectory) <= current_phase:
                    # Term doesn't have data for this phase
                    continue
                
                # Current position
                current_pos = trajectory[current_phase]
                
                # Next position (if available)
                if next_phase < len(trajectory):
                    next_pos = trajectory[next_phase]
                    # Interpolate
                    pos = current_pos * (1 - phase_alpha) + next_pos * phase_alpha
                else:
                    pos = current_pos
                
                # Update scatter position
                if self.n_dimensions == 3:
                    scatter_plots[term].set_data_3d([pos[0]], [pos[1]], [pos[2]])
                    
                    # Update trail if enabled
                    if add_trails:
                        trail_len = min(current_phase + 1, len(trajectory))
                        trail_data = trajectory[:trail_len]
                        if phase_alpha > 0 and next_phase < len(trajectory):
                            # Add interpolated point to trail
                            trail_data = np.vstack((trail_data, pos))
                        
                        trails[term].set_data_3d(trail_data[:, 0], 
                                             trail_data[:, 1], 
                                             trail_data[:, 2])
                else:
                    scatter_plots[term].set_data([pos[0]], [pos[1]])
                    
                    # Update trail if enabled
                    if add_trails:
                        trail_len = min(current_phase + 1, len(trajectory))
                        trail_data = trajectory[:trail_len]
                        if phase_alpha > 0 and next_phase < len(trajectory):
                            # Add interpolated point to trail
                            trail_data = np.vstack((trail_data, pos))
                        
                        trails[term].set_data(trail_data[:, 0], trail_data[:, 1])
        
            return [scatter_plots[term] for term in scatter_plots] + \
                   [trails[term] for term in trails] + [phase_text]
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=int(duration * fps),
            interval=1000/fps, blit=True
        )
        
        # Save animation
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath_mp4 = self.output_dir / f"ritual_animation_{timestamp}.mp4"
        
        # Save as MP4
        anim.save(str(filepath_mp4), writer='ffmpeg', fps=fps, 
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'],
                 dpi=300)
        
        logger.info(f"Saved ritual animation to {filepath_mp4}")
        
        # Also create a GIF for wider compatibility
        filepath_gif = self.output_dir / f"ritual_animation_{timestamp}.gif"
        anim.save(str(filepath_gif), writer='pillow', fps=fps//2, dpi=150)
        
        logger.info(f"Saved ritual animation GIF to {filepath_gif}")
        
        plt.close()
        
        return str(filepath_mp4)

    def _calculate_delta(self, current_points: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate transformation delta from previous state."""
        if self.previous_state is not None:
            delta = current_points - self.previous_state
            delta_magnitude = np.linalg.norm(delta, axis=1).mean()
            delta_direction = delta / (np.linalg.norm(delta, axis=1, keepdims=True) + 1e-10)
            
            # Store in history
            self.delta_history.append((delta_magnitude, delta_direction))
            if len(self.delta_history) > self.delta_window:
                self.delta_history.pop(0)
                
            return delta_magnitude, delta_direction
        return 0.0, np.zeros_like(current_points)
        
    def _create_delta_visualization(
        self,
        points: np.ndarray,
        delta_magnitude: float,
        delta_direction: np.ndarray,
        ax: plt.Axes
    ) -> None:
        """Create visualization of transformation deltas."""
        if delta_magnitude > 0:
            # Scale arrows by delta magnitude
            arrow_scale = delta_magnitude * 2.0
            
            # Plot arrows showing transformation direction
            if points.shape[1] == 3:
                ax.quiver(
                    points[:, 0], points[:, 1], points[:, 2],
                    delta_direction[:, 0], delta_direction[:, 1], delta_direction[:, 2],
                    color='white', alpha=0.3, scale=arrow_scale,
                    label=f'Δ={delta_magnitude:.2f}'
                )
            else:
                ax.quiver(
                    points[:, 0], points[:, 1],
                    delta_direction[:, 0], delta_direction[:, 1],
                    color='white', alpha=0.3, scale=arrow_scale,
                    label=f'Δ={delta_magnitude:.2f}'
                )
            
            # Add delta magnitude text
            ax.text2D(
                0.02, 0.92, f'Δ Magnitude: {delta_magnitude:.2f}',
                transform=ax.transAxes, color='white',
                bbox=dict(facecolor='black', alpha=0.7)
            )
            
    def _create_transformation_overlay(
        self,
        points: np.ndarray,
        ax: plt.Axes
    ) -> None:
        """Create visualization overlay for current transformation type."""
        if self.transformation_type:
            # Add transformation type indicator
            ax.text2D(
                0.02, 0.86, f'Transform: {self.transformation_type}',
                transform=ax.transAxes, color='white',
                bbox=dict(facecolor='black', alpha=0.7)
            )
            
            # Add transformation-specific visualization
            if self.transformation_type == 'numerological':
                # Show numerological patterns with concentric circles
                center = np.mean(points, axis=0)
                radii = np.linspace(0, 1, 7)  # 7 circles for numerological significance
                for r in radii:
                    circle = plt.Circle(
                        (center[0], center[1]), r,
                        fill=False, color='white', alpha=0.1
                    )
                    ax.add_patch(circle)
                    
            elif self.transformation_type == 'align':
                # Show alignment forces with gradient lines
                x = np.linspace(points[:, 0].min(), points[:, 0].max(), 20)
                y = np.linspace(points[:, 1].min(), points[:, 1].max(), 20)
                X, Y = np.meshgrid(x, y)
                ax.contour(X, Y, X*0, colors='white', alpha=0.1, levels=10)
                
            elif self.transformation_type == 'contrast':
                # Show contrast forces with radial gradient
                center = np.mean(points, axis=0)
                r = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
                ax.scatter(points[:, 0], points[:, 1], c=r, 
                         cmap='coolwarm', alpha=0.2, s=50)
                
            elif self.transformation_type == 'repel':
                # Show repulsion forces with arrows pointing outward
                center = np.mean(points, axis=0)
                directions = points - center
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
                ax.quiver(points[:, 0], points[:, 1],
                         directions[:, 0], directions[:, 1],
                         color='white', alpha=0.1)
                
    def set_transformation_type(self, transform_type: str):
        """Set the current transformation type."""
        self.transformation_type = transform_type

    def _create_glow_effect(self, color: str, n_layers: int = 5) -> List[Tuple[str, float]]:
        """Create a glowing effect by layering colors with different alphas."""
        base_color = mcolors.to_rgb(color)
        glow_layers = []
        
        # Create inner glow
        for i in range(n_layers):
            alpha = 0.8 * (1 - i/n_layers)
            # Shift color slightly towards white for inner layers
            inner_color = [
                c + (1 - c) * (1 - i/n_layers) * 0.3
                for c in base_color
            ]
            glow_layers.append((mcolors.rgb2hex(inner_color), alpha))
        
        # Create outer glow
        for i in range(n_layers):
            alpha = 0.4 * (1 - i/n_layers)
            # Shift color slightly towards black for outer layers
            outer_color = [
                c * (1 - i/n_layers) * 0.7
                for c in base_color
            ]
            glow_layers.append((mcolors.rgb2hex(outer_color), alpha))
        
        return glow_layers

    def _create_flow_lines(self, points: np.ndarray, n_lines: int = 50) -> np.ndarray:
        """Create flow lines showing semantic direction."""
        # Generate starting points in a spiral pattern
        theta = np.linspace(0, 4*np.pi, n_lines)
        r = np.linspace(0.1, 1.0, n_lines)
        start_points = np.array([
            [r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i])]
            for i in range(n_lines)
        ])
        
        # Create flow lines
        lines = []
        for start_point in start_points:
            line = [start_point]
            point = start_point.copy()
            
            # Get nearest points and their directions
            distances = np.linalg.norm(points - point, axis=1)
            nearest_idx = np.argsort(distances)[:3]
            flow = np.mean(points[nearest_idx] - point, axis=0)
            flow = flow / (np.linalg.norm(flow) + 1e-10)
            
            # Create line segments
            for _ in range(10):
                point = point + flow * 0.1
                line.append(point.copy())
            
            lines.append(line)
            
        return np.array(lines)

    def _add_enhanced_point_labels(
        self,
        ax: plt.Axes,
        points: np.ndarray,
        labels: List[str],
        significant_keywords: List[str]
    ) -> None:
        """Add enhanced labels with glow effects for significant points."""
        for i, (point, label) in enumerate(zip(points, labels)):
            # Check if this is a significant point
            is_significant = any(keyword in str(label).lower() 
                               for keyword in significant_keywords)
            
            if is_significant:
                # Create glow effect
                glow_layers = self._create_glow_effect('yellow')
                
                # Add glow layers
                for color, alpha in glow_layers:
                    if self.n_dimensions == 3:
                        ax.scatter(point[0], point[1], point[2],
                                 color=color, alpha=alpha, s=100)
                    else:
                        ax.scatter(point[0], point[1],
                                 color=color, alpha=alpha, s=100)
                
                # Add label with enhanced styling
                if self.n_dimensions == 3:
                    ax.text(point[0], point[1], point[2],
                           f' {label}', color='yellow', fontsize=10,
                           bbox=dict(facecolor='black', alpha=0.7,
                                   edgecolor='yellow', boxstyle='round'))
                else:
                    ax.text(point[0], point[1],
                           f' {label}', color='yellow', fontsize=10,
                           bbox=dict(facecolor='black', alpha=0.7,
                                   edgecolor='yellow', boxstyle='round'))

    def _add_polarity_visualization(
        self,
        ax: plt.Axes,
        points: np.ndarray,
        similarities: Dict[str, np.ndarray],
        labels: Optional[List[str]] = None
    ) -> None:
        """Add polarity visualization to the plot."""
        if not similarities:
            return

        # Create polarity legend
        polarity_text = []
        for polarity_name, scores in similarities.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            polarity_text.append(f"{polarity_name}: {mean_score:.2f}±{std_score:.2f}")

            # Find most extreme points for this polarity
            max_idx = np.argmax(scores)
            min_idx = np.argmin(scores)
            
            # Add connecting line for polarity axis
            if points.shape[1] >= 3:
                ax.plot([points[min_idx, 0], points[max_idx, 0]],
                       [points[min_idx, 1], points[max_idx, 1]],
                       [points[min_idx, 2], points[max_idx, 2]],
                       '--', color='white', alpha=0.2, linewidth=1)
            else:
                ax.plot([points[min_idx, 0], points[max_idx, 0]],
                       [points[min_idx, 1], points[max_idx, 1]],
                       '--', color='white', alpha=0.2, linewidth=1)

            # Label extreme points if labels are provided
            if labels is not None:
                for idx, sign in [(max_idx, '+'), (min_idx, '-')]:
                    if points.shape[1] >= 3:
                        ax.text(points[idx, 0], points[idx, 1], points[idx, 2],
                               f"{sign}{polarity_name}: {labels[idx]}",
                               color='yellow', fontsize=8,
                               bbox=dict(facecolor='black', alpha=0.7))
                    else:
                        ax.text(points[idx, 0], points[idx, 1],
                               f"{sign}{polarity_name}: {labels[idx]}",
                               color='yellow', fontsize=8,
                               bbox=dict(facecolor='black', alpha=0.7))

        # Add polarity information to plot
        ax.text2D(0.02, 0.02, '\n'.join(polarity_text),
                 transform=ax.transAxes, color='white',
                 bbox=dict(facecolor='black', alpha=0.7),
                 fontsize=8)

    def _create_interactive_controls(self, fig: plt.Figure, ax: plt.Axes, state: dict) -> None:
        """Create interactive control buttons for the visualization."""
        # Create button axes
        button_width = 0.1
        button_height = 0.04
        button_spacing = 0.02
        
        # Auto-rotate toggle
        rotate_ax = plt.axes([0.1, 0.025, button_width, button_height])
        rotate_button = plt.Button(
            rotate_ax,
            'Auto-Rotate',
            color='lightgray' if state['auto_rotate'] else 'white'
        )
        rotate_button.on_clicked(lambda event: self._toggle_auto_rotate(state))
        
        # Trails toggle
        trails_ax = plt.axes([0.2 + button_spacing, 0.025, button_width, button_height])
        trails_button = plt.Button(
            trails_ax,
            'Trails',
            color='lightgray' if state['show_trails'] else 'white'
        )
        trails_button.on_clicked(lambda event: self._toggle_trails(state))
        
        # Force field toggle
        force_ax = plt.axes([0.3 + 2 * button_spacing, 0.025, button_width, button_height])
        force_button = plt.Button(
            force_ax,
            'Force Field',
            color='lightgray' if state['show_force_field'] else 'white'
        )
        force_button.on_clicked(lambda event: self._toggle_force_field(state))
        
        # Reset view button
        reset_ax = plt.axes([0.4 + 3 * button_spacing, 0.025, button_width, button_height])
        reset_button = plt.Button(reset_ax, 'Reset View', color='lightgray')
        reset_button.on_clicked(lambda event: self._reset_view(state))
        
        # Speed control slider
        speed_ax = plt.axes([0.6 + 4 * button_spacing, 0.025, 0.2, button_height])
        speed_slider = plt.Slider(
            speed_ax,
            'Speed',
            0.1,
            2.0,
            valinit=state['speed_multiplier'],
            color='lightgray'
        )
        speed_slider.on_changed(lambda val: self._update_speed(state, val))
        
        # Store controls in state for updates
        state['controls'] = {
            'rotate_button': rotate_button,
            'trails_button': trails_button,
            'force_button': force_button,
            'reset_button': reset_button,
            'speed_slider': speed_slider
        }

    def _toggle_auto_rotate(self, state: dict) -> None:
        """Toggle auto-rotation state."""
        state['auto_rotate'] = not state['auto_rotate']
        state['controls']['rotate_button'].color = 'lightgray' if state['auto_rotate'] else 'white'
        logger.info(f"Auto-rotation {'enabled' if state['auto_rotate'] else 'disabled'}")

    def _toggle_trails(self, state: dict) -> None:
        """Toggle trails visibility."""
        state['show_trails'] = not state['show_trails']
        state['controls']['trails_button'].color = 'lightgray' if state['show_trails'] else 'white'
        if not state['show_trails']:
            state['trail_positions'].clear()
        logger.info(f"Trails {'enabled' if state['show_trails'] else 'disabled'}")

    def _toggle_force_field(self, state: dict) -> None:
        """Toggle force field visibility."""
        state['show_force_field'] = not state['show_force_field']
        state['controls']['force_button'].color = 'lightgray' if state['show_force_field'] else 'white'
        logger.info(f"Force field {'enabled' if state['show_force_field'] else 'disabled'}")

    def _reset_view(self, state: dict) -> None:
        """Reset view to initial state."""
        self.rotation_angles.fill(0)
        self.rotation_speeds.fill(0)
        state['projection_dims'] = [0, 1, 2]
        state['speed_multiplier'] = 1.0
        state['controls']['speed_slider'].set_val(1.0)
        logger.info("View reset to initial state")

    def _update_speed(self, state: dict, value: float) -> None:
        """Update animation speed."""
        state['speed_multiplier'] = value
        if state['auto_rotate']:
            for i in range(self.n_dimensions):
                for j in range(i + 1, self.n_dimensions):
                    self.rotation_speeds[i, j] *= value
        logger.info(f"Speed updated to {value:.1f}x")
