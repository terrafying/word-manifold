"""Interactive manifold visualization with parameter controls."""

from typing import Dict, Any, Callable, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.figure import Figure

from .base import InteractiveVisualizer, VisualizationData
from .utils import scale_coordinates, create_color_gradient
from .manifold_vis import ManifoldPlotData

class InteractiveManifoldVisualizer(InteractiveVisualizer):
    """Interactive visualizer for manifold structures with parameter controls."""
    
    def __init__(self, embeddings: np.ndarray, terms: List[List[str]]):
        """Initialize interactive visualizer.
        
        Args:
            embeddings: Matrix of embeddings
            terms: List of term lists for each embedding
        """
        super().__init__()
        self.embeddings = embeddings
        self.terms = terms
        self.controls: Dict[str, Any] = {}
        self.callbacks: Dict[str, float] = {}
        self.scatter = None
        
    def prepare_data(self) -> ManifoldPlotData:
        """Prepare initial visualization data."""
        # Scale coordinates to [-1, 1] range
        points = scale_coordinates(self.embeddings)
        
        return ManifoldPlotData(
            points=points,
            embeddings=self.embeddings,
            terms=self.terms
        )
        
    def plot(self, data: VisualizationData) -> Figure:
        """Create interactive visualization with controls."""
        # Create main figure with extra space for controls
        fig = plt.figure(figsize=(15, 8))
        
        # Add main plot area
        plot_ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
        points = data.to_plot_data()['points']
        self.scatter = plot_ax.scatter(points[:, 0], points[:, 1])
        plot_ax.set_title('Interactive Manifold Visualization')
        
        # Add labels and legend
        labels = data.get_labels()
        colors = data.get_color_map()
        for i, (label, pos) in enumerate(zip(labels.values(), points)):
            plot_ax.annotate(label, pos, fontsize=8)
        
        # Add parameter sliders
        self._add_slider('Scale', 0.1, 2.0, 1.0, 0.15)
        self._add_slider('Rotation', -180, 180, 0, 0.10)
        self._add_slider('Point Size', 10, 100, 50, 0.05)
        
        # Add control buttons
        ax_reset = fig.add_axes([0.8, 0.02, 0.1, 0.04])
        self.controls['reset'] = Button(ax_reset, 'Reset')
        self.controls['reset'].on_clicked(self._reset_parameters)
        
        # Add view toggles
        ax_check = fig.add_axes([0.1, 0.02, 0.2, 0.1])
        self.controls['toggles'] = CheckButtons(
            ax_check, 
            ['Show Labels', 'Show Points', 'Show Grid'],
            [True, True, False]
        )
        self.controls['toggles'].on_clicked(self._update_view)
        
        return fig
        
    def _add_slider(self, name: str, vmin: float, vmax: float, 
                   vinit: float, position: float) -> None:
        """Add a parameter slider to the visualization."""
        ax = self._figure.add_axes([0.2, position, 0.6, 0.03])
        slider = Slider(
            ax=ax,
            label=name,
            valmin=vmin,
            valmax=vmax,
            valinit=vinit
        )
        slider.on_changed(self._create_update_callback(name))
        self.controls[name.lower()] = slider
        
    def _create_update_callback(self, param_name: str) -> Callable:
        """Create callback function for parameter updates."""
        def update(val):
            self.callbacks[param_name] = val
            self.update()
        return update
        
    def _reset_parameters(self, event) -> None:
        """Reset all parameters to default values."""
        for name, control in self.controls.items():
            if isinstance(control, Slider):
                control.reset()
        self.update()
        
    def _update_view(self, label: str) -> None:
        """Update visualization based on view toggles."""
        if self._figure is None:
            return
            
        visibility = self.controls['toggles'].get_status()
        
        # Update visibility of different visualization elements
        if label == 'Show Labels':
            for text in self._figure.axes[0].texts:
                text.set_visible(visibility[0])
        elif label == 'Show Points':
            if self.scatter is not None:
                self.scatter.set_visible(visibility[1])
        elif label == 'Show Grid':
            self._figure.axes[0].grid(visibility[2])
            
        self._figure.canvas.draw_idle()
        
    def update(self) -> None:
        """Update visualization based on current parameters."""
        if self._figure is None or not self.callbacks or self.scatter is None:
            return
            
        # Get current parameter values
        scale = self.controls['scale'].val
        rotation = self.controls['rotation'].val
        point_size = self.controls['point size'].val
        
        # Get base points
        data = self.prepare_data()
        points = data.to_plot_data()['points']
        
        # Apply transformations
        theta = np.radians(rotation)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        transformed_points = scale * (points @ rotation_matrix)
        
        # Update visualization
        self.scatter.set_offsets(transformed_points)
        self.scatter.set_sizes([point_size] * len(points))
        
        # Update label positions
        if self.controls['toggles'].get_status()[0]:  # If labels are visible
            for text, pos in zip(self._figure.axes[0].texts, transformed_points):
                text.set_position(pos)
        
        self._figure.canvas.draw_idle()
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state of visualization parameters."""
        return {
            'scale': self.controls['scale'].val if 'scale' in self.controls else 1.0,
            'rotation': self.controls['rotation'].val if 'rotation' in self.controls else 0.0,
            'point_size': self.controls['point size'].val if 'point size' in self.controls else 50.0,
            'view_options': self.controls['toggles'].get_status() if 'toggles' in self.controls else [True, True, False]
        } 