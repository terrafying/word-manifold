"""Base classes for visualization components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, List, Tuple
import numpy as np
from matplotlib.figure import Figure
from pathlib import Path

class VisualizationData(Protocol):
    """Protocol for visualization data containers."""
    def to_plot_data(self) -> Dict[str, Any]: ...
    def get_color_map(self) -> Dict[str, str]: ...
    def get_labels(self) -> Dict[int, str]: ...

class VisualizationEngine(ABC):
    """Abstract base class for visualization engines."""
    
    def __init__(self):
        """Initialize engine."""
        self._data: Optional[Dict[str, Any]] = None
        
    @abstractmethod
    def process_data(self, *args, **kwargs) -> Dict[str, Any]:
        """Process input data into visualization-ready format."""
        pass
        
    @abstractmethod
    def generate_patterns(self, *args, **kwargs) -> Dict[str, Any]:
        """Generate patterns or transformations from processed data."""
        pass
        
    def get_data(self) -> Optional[Dict[str, Any]]:
        """Get the currently processed data."""
        return self._data
        
    def clear_data(self) -> None:
        """Clear the currently processed data."""
        self._data = None

class VisualizationRenderer(ABC):
    """Abstract base class for visualization renderers."""
    
    def __init__(self):
        """Initialize renderer."""
        self._figure: Optional[Figure] = None
        
    @abstractmethod
    def render_local(
        self,
        data: Dict[str, Any],
        output_path: Path,
        title: Optional[str] = None,
        figure_size: tuple = (15, 10)
    ) -> Path:
        """Render visualization locally."""
        pass
        
    @abstractmethod
    def render_server(
        self,
        data: Dict[str, Any],
        server_url: str,
        endpoint: str = '/api/visualize'
    ) -> Dict[str, Any]:
        """Render visualization using server."""
        pass
        
    def close(self) -> None:
        """Close the current figure."""
        if self._figure is not None:
            self._figure.clf()
            self._figure = None

class InteractiveVisualizer(VisualizationRenderer):
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