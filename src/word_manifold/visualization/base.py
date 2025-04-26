"""Base visualization components."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import ray
from functools import lru_cache
import torch
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

class VisualizationData:
    """Base class for visualization data."""
    
    def __init__(self, **kwargs):
        """Initialize with any data attributes."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_plot_data(self) -> Dict[str, Any]:
        """Convert to plottable format."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def get_color_map(self) -> Dict[str, str]:
        """Get color mapping for visualization."""
        return {}
    
    def get_labels(self) -> Dict[int, str]:
        """Get labels for visualization elements."""
        return {}

class VisualizationEngine(ABC):
    """Abstract base class for visualization engines."""
    
    def __init__(self, use_gpu: bool = True, max_workers: int = 4):
        """Initialize engine.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            max_workers: Maximum number of worker threads for parallel processing
        """
        self._data: Optional[Dict[str, Any]] = None
        self._use_gpu = use_gpu and torch.cuda.is_available()
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        if self._use_gpu:
            logger.info("GPU acceleration enabled")
            torch.cuda.empty_cache()
    
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
        if self._use_gpu:
            torch.cuda.empty_cache()
    
    @lru_cache(maxsize=128)
    def _cached_process(self, data_key: str) -> Dict[str, Any]:
        """Cached version of process_data for frequently accessed data."""
        return self.process_data(data_key)
    
    def parallel_process(self, data_list: List[Any]) -> List[Dict[str, Any]]:
        """Process multiple data items in parallel.
        
        Args:
            data_list: List of data items to process
            
        Returns:
            List of processed data dictionaries
        """
        futures = [self._executor.submit(self.process_data, data) for data in data_list]
        return [future.result() for future in futures]
    
    def cleanup(self):
        """Clean up resources."""
        self._executor.shutdown()
        if self._use_gpu:
            torch.cuda.empty_cache()

class VisualizationRenderer(ABC):
    """Abstract base class for visualization renderers."""
    
    def __init__(self, size: Tuple[int, int] = (800, 600), dpi: int = 100):
        """Initialize renderer.
        
        Args:
            size: Figure size in pixels (width, height)
            dpi: Dots per inch for output
        """
        self.size = size
        self.dpi = dpi
        self._figure = None
        self._engine = None
    
    @abstractmethod
    def _create_plot(self, data: VisualizationData, title: Optional[str] = None) -> plt.Figure:
        """Create the visualization plot."""
        pass
    
    @abstractmethod
    def render_local(
        self,
        data: Dict[str, Any],
        output_path: Path,
        title: Optional[str] = None
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
    
    def close(self):
        """Close the current figure."""
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None

class InteractiveVisualizer(VisualizationRenderer):
    """Base class for interactive visualizations."""
    
    def __init__(self, *args, **kwargs):
        """Initialize interactive visualizer."""
        super().__init__(*args, **kwargs)
        self._animation = None
        self._controls = {}
    
    def add_control(self, name: str, control_type: str, **kwargs):
        """Add an interactive control.
        
        Args:
            name: Name of the control
            control_type: Type of control (slider, button, etc.)
            **kwargs: Additional control parameters
        """
        self._controls[name] = {
            'type': control_type,
            'params': kwargs
        }
    
    def update_visualization(self, **kwargs):
        """Update visualization with new parameters."""
        pass
    
    def start_animation(self, duration: float = 5.0, fps: int = 30):
        """Start animation sequence.
        
        Args:
            duration: Animation duration in seconds
            fps: Frames per second
        """
        pass
    
    def stop_animation(self):
        """Stop current animation."""
        if self._animation is not None:
            self._animation.event_source.stop()
            self._animation = None
    
    def export_animation(self, output_path: Path, format: str = 'mp4'):
        """Export animation to file.
        
        Args:
            output_path: Path to save animation
            format: Output format (mp4, gif, etc.)
        """
        pass 