"""Base classes for visualization components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol
import numpy as np
from matplotlib.figure import Figure

class VisualizationData(Protocol):
    """Protocol for visualization data containers."""
    def to_plot_data(self) -> Dict[str, Any]: ...
    def get_color_map(self) -> Dict[str, str]: ...
    def get_labels(self) -> Dict[int, str]: ...

class Visualizer(ABC):
    """Abstract base class for visualizers."""
    
    def __init__(self):
        """Initialize visualizer."""
        self._figure: Optional[Figure] = None
        
    @abstractmethod
    def prepare_data(self) -> VisualizationData:
        """Prepare data for visualization."""
        pass
        
    @abstractmethod
    def plot(self, data: VisualizationData) -> Figure:
        """Create visualization."""
        pass
        
    def show(self) -> None:
        """Display the visualization."""
        if self._figure is None:
            data = self.prepare_data()
            self._figure = self.plot(data)
        self._figure.show()
        
    def close(self) -> None:
        """Close the visualization."""
        if self._figure is not None:
            self._figure.clf()
            self._figure = None

class InteractiveVisualizer(Visualizer):
    """Base class for interactive visualizers."""
    
    def __init__(self):
        """Initialize interactive visualizer."""
        super().__init__()
        self._is_interactive = True
        
    @abstractmethod
    def update(self) -> None:
        """Update visualization based on current state."""
        pass
        
    def show(self) -> None:
        """Display interactive visualization."""
        super().show()
        if self._figure is not None:
            # Enable interactive mode for real-time updates
            self._figure.canvas.draw()
            
    def save(self, filepath: str, **kwargs) -> None:
        """Save current visualization state."""
        if self._figure is not None:
            self._figure.savefig(filepath, **kwargs)
            
    def get_state(self) -> Dict[str, Any]:
        """Get current state of visualization parameters."""
        return {} 