"""Utility functions for visualization components."""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def create_color_gradient(n_colors: int, start_color: str = '#1f77b4', end_color: str = '#ff7f0e') -> Colormap:
    """Create a continuous color gradient between two colors.
    
    Args:
        n_colors: Number of colors in gradient
        start_color: Starting hex color code
        end_color: Ending hex color code
        
    Returns:
        matplotlib colormap
    """
    return LinearSegmentedColormap.from_list('custom', [start_color, end_color], N=n_colors)

def scale_coordinates(coords: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Scale coordinates to specified range while preserving relative distances.
    
    Args:
        coords: Array of coordinates to scale
        scale: Scale factor to apply
        
    Returns:
        Scaled coordinates array
    """
    scaler = MinMaxScaler(feature_range=(-scale, scale))
    return scaler.fit_transform(coords)

def calculate_marker_sizes(
    values: np.ndarray,
    min_size: float = 20.0,
    max_size: float = 200.0
) -> np.ndarray:
    """Calculate marker sizes based on values.
    
    Args:
        values: Array of values to map to sizes
        min_size: Minimum marker size
        max_size: Maximum marker size
        
    Returns:
        Array of marker sizes
    """
    if len(values) == 0:
        return np.array([])
    normalized = (values - values.min()) / (values.max() - values.min())
    return min_size + normalized * (max_size - min_size)

def create_subplot_grid(
    n_plots: int,
    max_cols: int = 3
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a grid of subplots.
    
    Args:
        n_plots: Number of plots needed
        max_cols: Maximum number of columns
        
    Returns:
        Figure and array of axes
    """
    n_cols = min(n_plots, max_cols)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig_width = 5 * n_cols
    fig_height = 4 * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_plots == 1:
        axes = np.array([axes])
    return fig, axes.flatten()

def add_colorbar(
    fig: plt.Figure,
    mappable: Any,
    label: str,
    orientation: str = 'vertical'
) -> None:
    """Add a colorbar to the figure.
    
    Args:
        fig: Figure to add colorbar to
        mappable: The mappable object to create colorbar from
        label: Label for the colorbar
        orientation: Orientation of colorbar ('vertical' or 'horizontal')
    """
    fig.colorbar(mappable, label=label, orientation=orientation)

def create_legend(
    ax: plt.Axes,
    labels: Dict[Any, str],
    markers: Optional[Dict[Any, str]] = None,
    colors: Optional[Dict[Any, str]] = None,
    title: Optional[str] = None
) -> None:
    """Add a legend to the axes.
    
    Args:
        ax: Axes to add legend to
        labels: Mapping of ids to labels
        markers: Optional mapping of ids to marker styles
        colors: Optional mapping of ids to colors
        title: Optional legend title
    """
    handles = []
    for id_ in labels:
        style = {}
        if markers and id_ in markers:
            style['marker'] = markers[id_]
        if colors and id_ in colors:
            style['color'] = colors[id_]
        handle = plt.Line2D([0], [0], label=labels[id_], **style)
        handles.append(handle)
    
    if handles:
        ax.legend(handles=handles, title=title) 