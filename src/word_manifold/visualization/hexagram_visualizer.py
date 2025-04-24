"""
Hexagram Visualization Module.

This module provides visualization tools for I Ching hexagrams and their
transformations in the word manifold space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, PathPatch
from matplotlib.path import Path
import matplotlib.colors as mcolors
from typing import List, Dict, Optional, Tuple, Union
import logging

from ..automata.hexagram_rules import Hexagram, Line, HexagramRule
from ..manifold.vector_manifold import VectorManifold

logger = logging.getLogger(__name__)

class HexagramVisualizer:
    """Visualizer for hexagram-based transformations."""
    
    # Visual constants
    LINE_LENGTH = 1.0
    LINE_SPACING = 0.3
    YIN_GAP = 0.2
    LINE_THICKNESS = 0.1
    
    # Colors for different aspects
    COLORS = {
        'yang': '#E74C3C',  # Red for yang energy
        'yin': '#3498DB',   # Blue for yin energy
        'neutral': '#95A5A6',  # Gray for neutral elements
        'transform': '#2ECC71',  # Green for transformations
        'nuclear': '#9B59B6',  # Purple for nuclear hexagram
        'background': '#ECF0F1'  # Light gray background
    }
    
    def __init__(self, size: Tuple[int, int] = (800, 600), dpi: int = 100):
        """Initialize the visualizer."""
        self.size = size
        self.dpi = dpi
        self.fig = None
        self.ax = None
    
    def draw_line(self, y: float, line_type: Line, changing: bool = False) -> None:
        """Draw a single hexagram line."""
        if line_type == Line.YANG:
            # Solid line
            self.ax.add_patch(Rectangle(
                (-self.LINE_LENGTH/2, y - self.LINE_THICKNESS/2),
                self.LINE_LENGTH, self.LINE_THICKNESS,
                facecolor=self.COLORS['yang'],
                alpha=0.8 if not changing else 0.4
            ))
            if changing:
                # Add transformation indicator
                self.ax.add_patch(Circle(
                    (0, y),
                    self.LINE_THICKNESS,
                    facecolor=self.COLORS['transform'],
                    alpha=0.6
                ))
        else:
            # Broken line
            gap = self.YIN_GAP
            for x in [-self.LINE_LENGTH/2, gap/2]:
                self.ax.add_patch(Rectangle(
                    (x, y - self.LINE_THICKNESS/2),
                    (self.LINE_LENGTH - gap)/2, self.LINE_THICKNESS,
                    facecolor=self.COLORS['yin'],
                    alpha=0.8 if not changing else 0.4
                ))
            if changing:
                # Add transformation indicators
                for x in [-self.LINE_LENGTH/4, self.LINE_LENGTH/4]:
                    self.ax.add_patch(Circle(
                        (x, y),
                        self.LINE_THICKNESS,
                        facecolor=self.COLORS['transform'],
                        alpha=0.6
                    ))
    
    def draw_hexagram(
        self,
        hexagram: Hexagram,
        position: Tuple[float, float] = (0, 0),
        scale: float = 1.0,
        changing_lines: Optional[List[int]] = None,
        show_nuclear: bool = False
    ) -> None:
        """Draw a complete hexagram."""
        if changing_lines is None:
            changing_lines = []
            
        # Draw main hexagram
        for i, line in enumerate(hexagram.lines):
            y = position[1] + i * self.LINE_SPACING * scale
            self.draw_line(y, line, i in changing_lines)
            
        if show_nuclear:
            # Draw nuclear hexagram with reduced opacity
            nuclear = hexagram.get_nuclear_hexagram()
            for i, line in enumerate(nuclear.lines):
                y = position[1] + (i + 0.15) * self.LINE_SPACING * scale
                x = position[0] + self.LINE_LENGTH * 0.75
                self.ax.add_patch(Rectangle(
                    (x - self.LINE_LENGTH/2, y - self.LINE_THICKNESS/2),
                    self.LINE_LENGTH * 0.8, self.LINE_THICKNESS * 0.8,
                    facecolor=self.COLORS['nuclear'],
                    alpha=0.4
                ))
    
    def draw_transformation(
        self,
        from_hexagram: Hexagram,
        to_hexagram: Hexagram,
        changing_lines: List[int],
        phase: float = 0.0
    ) -> None:
        """Draw a hexagram transformation sequence."""
        # Clear previous figure
        if self.fig is not None:
            plt.close(self.fig)
        
        self.fig, self.ax = plt.subplots(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
        self.ax.set_facecolor(self.COLORS['background'])
        
        # Draw initial hexagram
        self.draw_hexagram(
            from_hexagram,
            position=(-2, 0),
            changing_lines=changing_lines
        )
        
        # Draw final hexagram
        self.draw_hexagram(
            to_hexagram,
            position=(2, 0),
            changing_lines=[]
        )
        
        # Draw transformation arrows
        for i in changing_lines:
            y = i * self.LINE_SPACING
            self.ax.arrow(
                -1, y,
                2, 0,
                head_width=0.1,
                head_length=0.2,
                fc=self.COLORS['transform'],
                ec=self.COLORS['transform'],
                alpha=0.6
            )
        
        # Add labels
        self.ax.text(-2, -1, f"Hexagram {from_hexagram.number}\n{from_hexagram.name}",
                     ha='center', va='top')
        self.ax.text(2, -1, f"Hexagram {to_hexagram.number}\n{to_hexagram.name}",
                     ha='center', va='top')
        
        # Set limits and remove axes
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-1.5, 2)
        self.ax.axis('off')
    
    def visualize_rule_application(
        self,
        rule: HexagramRule,
        manifold: VectorManifold,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize the application of a hexagram rule to a manifold."""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Draw hexagram and its properties
        self.ax = ax1
        self.draw_hexagram(rule.hexagram, show_nuclear=True)
        
        # Add rule information
        info_text = (
            f"Rule: {rule.name}\n"
            f"Transformation: {rule.vector_transformation}\n"
            f"Magnitude: {rule.parameters.magnitude:.2f}\n"
            f"Direction: {rule.parameters.vibration_direction.name}"
        )
        ax1.text(2, 0, info_text, fontsize=10, va='center')
        
        # Visualize manifold transformation
        self._plot_manifold_transformation(ax2, rule, manifold)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_manifold_transformation(
        self,
        ax: plt.Axes,
        rule: HexagramRule,
        manifold: VectorManifold
    ) -> None:
        """Plot the transformation effect on the manifold."""
        # Get cell positions before and after transformation
        original_positions = np.array([cell.centroid for cell in manifold.cells.values()])
        
        # Apply transformation
        transformed_manifold = rule.apply(manifold)
        transformed_positions = np.array([
            cell.centroid for cell in transformed_manifold.cells.values()
        ])
        
        # Project to 2D if needed
        if original_positions.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            original_positions = pca.fit_transform(original_positions)
            transformed_positions = pca.transform(transformed_positions)
        
        # Plot original positions
        ax.scatter(
            original_positions[:, 0],
            original_positions[:, 1],
            c=self.COLORS['neutral'],
            alpha=0.5,
            label='Original'
        )
        
        # Plot transformed positions
        ax.scatter(
            transformed_positions[:, 0],
            transformed_positions[:, 1],
            c=self.COLORS['transform'],
            alpha=0.7,
            label='Transformed'
        )
        
        # Draw transformation arrows
        for start, end in zip(original_positions, transformed_positions):
            ax.arrow(
                start[0], start[1],
                end[0] - start[0], end[1] - start[1],
                head_width=0.05,
                head_length=0.1,
                fc=self.COLORS['transform'],
                ec=self.COLORS['transform'],
                alpha=0.3
            )
        
        ax.legend()
        ax.set_title('Manifold Transformation')
        ax.grid(True, alpha=0.3) 