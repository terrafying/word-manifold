"""
Magic Structure Visualizer Module

This module implements visualization of magic squares, cubes, and higher-dimensional
magic structures. It supports arbitrary dimensionality and integrates semantic
meaning with numerological patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from dataclasses import dataclass
import logging
from pathlib import Path
import networkx as nx
from ..embeddings.word_embeddings import WordEmbeddings

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MagicStructure:
    """Represents a magic structure of arbitrary dimension."""
    dimension: int  # Number of dimensions
    size: int      # Size in each dimension
    values: np.ndarray  # n-dimensional array of values
    terms: Optional[List[str]] = None  # Associated terms
    semantic_weights: Optional[np.ndarray] = None  # Semantic influence weights
    
    @property
    def magic_constant(self) -> float:
        """Calculate the magic constant for this structure."""
        return np.sum(self.values) / self.dimension
    
    def is_magic(self, tolerance: float = 1e-6) -> bool:
        """Check if structure satisfies magic properties."""
        constant = self.magic_constant
        
        # Check all axes sums
        for axis in range(self.dimension):
            sums = np.sum(self.values, axis=axis)
            if not np.allclose(sums, constant, rtol=tolerance):
                return False
        
        # Check diagonals for 2D and 3D
        if self.dimension <= 3:
            if self.dimension == 2:
                diag1 = np.trace(self.values)
                diag2 = np.trace(np.fliplr(self.values))
                return np.allclose([diag1, diag2], constant, rtol=tolerance)
            else:  # 3D
                # Check space diagonals
                diags = []
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            diag = np.diagonal(
                                np.flip(self.values, axis=i) if i else self.values,
                                axis1=1 if j else 0,
                                axis2=2 if k else 0
                            )
                            diags.append(np.sum(diag))
                return np.allclose(diags, constant, rtol=tolerance)
        
        return True

class MagicVisualizer:
    """Visualizer for magic squares, cubes, and higher-dimensional structures."""
    
    def __init__(
        self,
        word_embeddings: Optional[WordEmbeddings] = None,
        output_dir: str = "visualizations/magic",
        enable_semantic_weighting: bool = True,
        color_scheme: str = "viridis"
    ):
        """
        Initialize the magic structure visualizer.
        
        Args:
            word_embeddings: Optional WordEmbeddings for semantic analysis
            output_dir: Directory to save visualizations
            enable_semantic_weighting: Whether to use semantic weights
            color_scheme: Color scheme for visualizations
        """
        self.word_embeddings = word_embeddings
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_semantic_weighting = enable_semantic_weighting
        self.color_scheme = color_scheme
        
        logger.info(f"Initialized MagicVisualizer with output dir: {output_dir}")
    
    def generate_magic_structure(
        self,
        dimension: int,
        size: int,
        terms: Optional[List[str]] = None
    ) -> MagicStructure:
        """
        Generate a magic structure of specified dimension and size.
        
        Args:
            dimension: Number of dimensions
            size: Size in each dimension
            terms: Optional list of terms for semantic weighting
            
        Returns:
            MagicStructure instance
        """
        # Generate base magic structure
        total_cells = size ** dimension
        base_values = np.arange(1, total_cells + 1)
        
        if dimension == 2:
            values = self._generate_magic_square(size)
        elif dimension == 3:
            values = self._generate_magic_cube(size)
        else:
            values = self._generate_nd_magic(dimension, size)
        
        # Apply semantic weighting if terms provided
        semantic_weights = None
        if terms and self.word_embeddings and self.enable_semantic_weighting:
            semantic_weights = self._calculate_semantic_weights(terms, total_cells)
            values = values * semantic_weights.reshape(values.shape)
        
        return MagicStructure(
            dimension=dimension,
            size=size,
            values=values,
            terms=terms,
            semantic_weights=semantic_weights
        )
    
    def _generate_magic_square(self, n: int) -> np.ndarray:
        """Generate a magic square of size n x n."""
        if n < 3:
            raise ValueError("Magic square size must be at least 3")
            
        if n % 2 == 1:  # Odd order
            return self._generate_odd_magic_square(n)
        elif n % 4 == 0:  # Doubly even order
            return self._generate_doubly_even_magic_square(n)
        else:  # Singly even order
            return self._generate_singly_even_magic_square(n)
    
    def _generate_odd_magic_square(self, n: int) -> np.ndarray:
        """Generate odd-ordered magic square using Siamese method."""
        square = np.zeros((n, n), dtype=int)
        i, j = 0, n//2  # Start position
        
        for num in range(1, n**2 + 1):
            square[i, j] = num
            new_i, new_j = (i-1) % n, (j+1) % n
            
            if square[new_i, new_j]:
                i = (i + 1) % n
            else:
                i, j = new_i, new_j
                
        return square
    
    def _generate_doubly_even_magic_square(self, n: int) -> np.ndarray:
        """Generate doubly even magic square (n divisible by 4)."""
        square = np.arange(1, n**2 + 1).reshape(n, n)
        
        # Create pattern of 1s and 0s
        pattern = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if (i % 4 == 0 or i % 4 == 3) == (j % 4 == 0 or j % 4 == 3):
                    pattern[i, j] = True
                    
        # Apply pattern
        square[pattern] = n**2 + 1 - square[pattern]
        return square
    
    def _generate_singly_even_magic_square(self, n: int) -> np.ndarray:
        """Generate singly even magic square (n divisible by 2 but not 4)."""
        # Use LUX method
        m = n // 2
        quadrant = self._generate_odd_magic_square(m)
        
        # Create the four quadrants
        square = np.zeros((n, n), dtype=int)
        square[:m, :m] = quadrant
        square[m:, m:] = quadrant + m**2
        square[:m, m:] = quadrant + 2*m**2
        square[m:, :m] = quadrant + 3*m**2
        
        # Swap elements
        k = (n - 2) // 4
        for i in range(m):
            for j in range(m):
                if i == m//2:
                    if j < k:
                        square[i][j], square[i+m][j] = square[i+m][j], square[i][j]
                elif j == m//2:
                    if i < k:
                        square[i][j], square[i][j+m] = square[i][j+m], square[i][j]
                        
        return square
    
    def _generate_magic_cube(self, n: int) -> np.ndarray:
        """Generate a magic cube of size n x n x n."""
        # Use perfect magic cube construction
        cube = np.zeros((n, n, n), dtype=int)
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    cube[i,j,k] = n**2 * ((i + j + k) % n) + n * ((i + 2*j + 3*k) % n) + ((i + 4*j + 5*k) % n) + 1
                    
        return cube
    
    def _generate_nd_magic(self, dimension: int, size: int) -> np.ndarray:
        """Generate n-dimensional magic structure."""
        # Use generalized perfect magic hypercube construction
        shape = (size,) * dimension
        values = np.zeros(shape, dtype=int)
        
        # Generate indices for all cells
        indices = np.indices(shape)
        
        # Apply generalized formula
        for idx in np.ndindex(shape):
            value = 1
            for d in range(dimension):
                value += size**(dimension-d-1) * ((sum(idx) + (d+2)*idx[d]) % size)
            values[idx] = value
            
        return values
    
    def _calculate_semantic_weights(
        self,
        terms: List[str],
        total_cells: int
    ) -> np.ndarray:
        """Calculate semantic weights for structure values."""
        if not self.word_embeddings:
            return np.ones(total_cells)
            
        # Get embeddings for terms
        embeddings = []
        for term in terms:
            try:
                emb = self.word_embeddings.get_embedding(term)
                embeddings.append(emb)
            except:
                logger.warning(f"Could not get embedding for term: {term}")
                
        if not embeddings:
            return np.ones(total_cells)
            
        # Calculate weights based on semantic similarity
        embeddings = np.array(embeddings)
        weights = np.zeros(total_cells)
        
        for i in range(total_cells):
            # Use position in structure to weight semantic influence
            pos_weight = 1 - (i / total_cells)
            sem_weight = np.mean([np.dot(emb, embeddings.mean(axis=0)) 
                                for emb in embeddings])
            weights[i] = pos_weight * sem_weight
            
        # Normalize weights
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        return weights + 0.5  # Ensure weights are positive
    
    def visualize(
        self,
        structure: MagicStructure,
        title: str = "Magic Structure",
        show_values: bool = True,
        show_connections: bool = True,
        interactive: bool = False,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize a magic structure.
        
        Args:
            structure: MagicStructure to visualize
            title: Plot title
            show_values: Whether to show numeric values
            show_connections: Whether to show connections between cells
            interactive: Whether to create interactive visualization
            save_path: Optional path to save visualization
        """
        if structure.dimension > 3 and not interactive:
            logger.warning("Static visualization not supported for dimension > 3")
            return
            
        if interactive:
            self._visualize_interactive(structure, title, show_values, show_connections, save_path)
        else:
            self._visualize_static(structure, title, show_values, show_connections, save_path)
    
    def _visualize_static(
        self,
        structure: MagicStructure,
        title: str,
        show_values: bool,
        show_connections: bool,
        save_path: Optional[str]
    ) -> None:
        """Create static visualization."""
        fig = plt.figure(figsize=(10, 10))
        
        if structure.dimension == 2:
            ax = fig.add_subplot(111)
            im = ax.imshow(structure.values, cmap=self.color_scheme)
            
            if show_values:
                for i in range(structure.size):
                    for j in range(structure.size):
                        ax.text(j, i, str(int(structure.values[i, j])),
                               ha='center', va='center')
                        
            if show_connections:
                self._add_2d_connections(ax, structure)
                
        elif structure.dimension == 3:
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot cube faces
            for z in range(structure.size):
                xs = np.arange(structure.size)
                ys = np.arange(structure.size)
                X, Y = np.meshgrid(xs, ys)
                Z = np.full_like(X, z)
                
                colors = plt.cm.get_cmap(self.color_scheme)(
                    structure.values[:,:,z] / structure.values.max()
                )
                
                ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.7)
                
            if show_connections:
                self._add_3d_connections(ax, structure)
                
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close()
    
    def _visualize_interactive(
        self,
        structure: MagicStructure,
        title: str,
        show_values: bool,
        show_connections: bool,
        save_path: Optional[str]
    ) -> None:
        """Create interactive visualization using plotly."""
        # Create graph representation
        G = nx.Graph()
        
        # Add nodes
        pos = {}
        node_colors = []
        node_text = []
        
        for idx in np.ndindex(structure.values.shape):
            G.add_node(idx)
            pos[idx] = np.array(idx)
            value = structure.values[idx]
            node_colors.append(value)
            node_text.append(f"Value: {value:.2f}")
            
        # Add edges for adjacent cells
        if show_connections:
            for node in G.nodes():
                for dim in range(structure.dimension):
                    for offset in [-1, 1]:
                        neighbor = list(node)
                        neighbor[dim] += offset
                        if all(0 <= x < structure.size for x in neighbor):
                            G.add_edge(node, tuple(neighbor))
        
        # Create plotly figure
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]][0], pos[edge[0]][1]
            x1, y1 = pos[edge[1]][0], pos[edge[1]][1]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            if structure.dimension > 2:
                z0, z1 = pos[edge[0]][2], pos[edge[1]][2]
                edge_z.extend([z0, z1, None])
        
        # Create visualization
        fig = go.Figure()
        
        # Add edges
        if show_connections:
            if structure.dimension > 2:
                fig.add_trace(go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(color='#888', width=1),
                    hoverinfo='none'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(color='#888', width=1),
                    hoverinfo='none'
                ))
        
        # Add nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        if structure.dimension > 2:
            node_z = [pos[node][2] for node in G.nodes()]
            fig.add_trace(go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers',
                marker=dict(
                    size=10,
                    color=node_colors,
                    colorscale=self.color_scheme,
                    showscale=True
                ),
                text=node_text,
                hoverinfo='text'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(
                    size=20,
                    color=node_colors,
                    colorscale=self.color_scheme,
                    showscale=True
                ),
                text=node_text,
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40)
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def _add_2d_connections(self, ax: plt.Axes, structure: MagicStructure) -> None:
        """Add connection lines for 2D visualization."""
        # Add row/column lines
        for i in range(structure.size):
            ax.axhline(y=i-0.5, color='gray', alpha=0.3)
            ax.axvline(x=i-0.5, color='gray', alpha=0.3)
            
        # Add diagonals
        ax.plot([-0.5, structure.size-0.5], [-0.5, structure.size-0.5],
                color='red', alpha=0.3)
        ax.plot([-0.5, structure.size-0.5], [structure.size-0.5, -0.5],
                color='red', alpha=0.3)
    
    def _add_3d_connections(self, ax: Axes3D, structure: MagicStructure) -> None:
        """Add connection lines for 3D visualization."""
        n = structure.size
        
        # Add edges
        for i in range(n):
            for j in range(n):
                ax.plot([i,i], [j,j], [0,n-1], color='gray', alpha=0.3)
                ax.plot([i,i], [0,n-1], [j,j], color='gray', alpha=0.3)
                ax.plot([0,n-1], [i,i], [j,j], color='gray', alpha=0.3)
        
        # Add space diagonals
        ax.plot([0,n-1], [0,n-1], [0,n-1], color='red', alpha=0.3)
        ax.plot([0,n-1], [0,n-1], [n-1,0], color='red', alpha=0.3)
        ax.plot([0,n-1], [n-1,0], [0,n-1], color='red', alpha=0.3)
        ax.plot([0,n-1], [n-1,0], [n-1,0], color='red', alpha=0.3) 