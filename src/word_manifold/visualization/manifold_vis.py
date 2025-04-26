"""Concrete manifold visualization implementations."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # Import 3D toolkit
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
import requests

from .base import VisualizationRenderer, VisualizationData
from .utils import (
    create_color_gradient, 
    scale_coordinates, 
    create_subplot_grid,
    add_colorbar,
    create_legend
)

@dataclass
class ManifoldPlotData:
    """Container for manifold visualization data."""
    points: np.ndarray
    embeddings: np.ndarray
    terms: List[List[str]]
    coherence: Optional[np.ndarray] = None
    
    def to_plot_data(self) -> Dict[str, Any]:
        """Convert to plottable format."""
        return {
            'points': self.points,
            'embeddings': self.embeddings,
            'coherence': self.coherence
        }
    
    def get_color_map(self) -> Dict[str, str]:
        """Generate color mapping for terms."""
        unique_terms = set().union(*[set(terms) for terms in self.terms])
        return dict(zip(unique_terms, sns.color_palette('husl', len(unique_terms))))
    
    def get_labels(self) -> Dict[int, str]:
        """Generate labels for points."""
        return {i: ','.join(terms[:3]) for i, terms in enumerate(self.terms)}

class ManifoldVisualizer(VisualizationRenderer):
    """Visualizer for manifold structures."""
    
    def __init__(self, embeddings: np.ndarray, terms: List[List[str]], n_components: int = 2):
        """Initialize visualizer.
        
        Args:
            embeddings: Matrix of embeddings
            terms: List of term lists for each embedding
            n_components: Number of visualization dimensions (2 or 3)
        """
        super().__init__()
        if n_components not in [2, 3]:
            raise ValueError("n_components must be 2 or 3")
        self.embeddings = embeddings
        self.terms = terms
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        
    def prepare_data(self) -> ManifoldPlotData:
        """Prepare manifold data for visualization."""
        # Reduce dimensionality if needed
        if self.embeddings.shape[1] > self.n_components:
            points = self.pca.fit_transform(self.embeddings)
        else:
            points = self.embeddings.copy()
            
        # Scale coordinates to [-1, 1] range
        points = scale_coordinates(points)
        
        # Calculate coherence matrix if embeddings available
        coherence = None
        if self.embeddings is not None:
            coherence = np.corrcoef(self.embeddings)
        
        return ManifoldPlotData(
            points=points,
            embeddings=self.embeddings,
            terms=self.terms,
            coherence=coherence
        )
    
    def _create_plot(self, data: ManifoldPlotData, title: Optional[str] = None, figure_size: tuple = (15, 5)) -> Figure:
        """Create manifold visualization."""
        plot_data = data.to_plot_data()
        points = plot_data['points']
        
        # Create figure with subplots
        fig = plt.figure(figsize=figure_size)
        
        # Create manifold plot (2D or 3D)
        if self.n_components == 2:
            ax1 = fig.add_subplot(131)
            scatter = ax1.scatter(points[:, 0], points[:, 1])
            ax1.set_xlabel('Component 1')
            ax1.set_ylabel('Component 2')
        else:  # 3D plot
            ax1 = fig.add_subplot(131, projection='3d')
            scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2])
            ax1.set_xlabel('Component 1')
            ax1.set_ylabel('Component 2')
            ax1.set_zlabel('Component 3')
        
        ax1.set_title(title or 'Manifold Structure')
        
        # Add labels and legend
        labels = data.get_labels()
        colors = data.get_color_map()
        for i, label in labels.items():
            if self.n_components == 2:
                ax1.annotate(label, (points[i, 0], points[i, 1]))
            else:
                ax1.text(points[i, 0], points[i, 1], points[i, 2], label)
        
        # Plot embeddings structure
        ax2 = fig.add_subplot(132)
        if plot_data['embeddings'] is not None:
            embeddings = plot_data['embeddings']
            dist_matrix = np.linalg.norm(embeddings[:, None] - embeddings, axis=2)
            sns.heatmap(dist_matrix, ax=ax2, cmap='viridis')
            ax2.set_title('Embedding Distances')
            add_colorbar(fig, ax2.collections[0], 'Distance')
        
        # Plot coherence heatmap if available
        ax3 = fig.add_subplot(133)
        if plot_data['coherence'] is not None:
            sns.heatmap(plot_data['coherence'], ax=ax3, cmap='RdBu_r')
            ax3.set_title('Coherence Matrix')
            add_colorbar(fig, ax3.collections[0], 'Coherence')
        
        plt.tight_layout()
        self._figure = fig
        return fig

    def render_local(
        self,
        data: Dict[str, Any],
        output_path: Path,
        title: Optional[str] = None,
        figure_size: tuple = (15, 10)
    ) -> Path:
        """Render visualization locally.
        
        Args:
            data: Visualization data dictionary
            output_path: Path to save the visualization
            title: Optional title for the visualization
            figure_size: Size of the figure (width, height)
            
        Returns:
            Path to the saved visualization file
        """
        plot_data = ManifoldPlotData(**data)
        fig = self._create_plot(plot_data, title, figure_size)
        fig.savefig(output_path)
        return output_path

    def render_server(
        self,
        data: Dict[str, Any],
        server_url: str,
        endpoint: str = '/api/visualize'
    ) -> Dict[str, Any]:
        """Render visualization using server.
        
        Args:
            data: Visualization data dictionary
            server_url: URL of the visualization server
            endpoint: API endpoint for visualization
            
        Returns:
            Server response containing visualization data
        """
        plot_data = ManifoldPlotData(**data)
        
        # Convert plot data to server format
        server_data = {
            'type': 'manifold',
            'data': plot_data.to_plot_data(),
            'metadata': {
                'n_components': self.n_components,
                'labels': plot_data.get_labels(),
                'colors': plot_data.get_color_map()
            }
        }
        
        # Send data to server
        response = requests.post(f"{server_url.rstrip('/')}{endpoint}", json=server_data)
        response.raise_for_status()
        
        return response.json() 