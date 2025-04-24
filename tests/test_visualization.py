"""Tests for visualization components."""

import pytest
import numpy as np
from matplotlib.figure import Figure

from word_manifold.visualization.base import Visualizer, InteractiveVisualizer, VisualizationData
from word_manifold.visualization.manifold_vis import ManifoldVisualizer, ManifoldPlotData
from word_manifold.visualization.interactive import InteractiveManifoldVisualizer
from word_manifold.visualization.utils import (
    create_color_gradient,
    scale_coordinates,
    calculate_marker_sizes,
    create_subplot_grid
)

class MockVisualizationData:
    """Mock visualization data for testing."""
    def __init__(self, points, labels, colors):
        self.points = points
        self.labels = labels
        self.colors = colors
        
    def to_plot_data(self):
        return {'points': self.points}
        
    def get_color_map(self):
        return self.colors
        
    def get_labels(self):
        return self.labels

class MockVisualizer(Visualizer):
    """Mock visualizer for testing."""
    def prepare_data(self):
        points = np.array([[1, 2], [3, 4]])
        labels = {0: 'A', 1: 'B'}
        colors = {'A': 'red', 'B': 'blue'}
        return MockVisualizationData(points, labels, colors)
        
    def plot(self, data):
        fig = Figure()
        ax = fig.add_subplot(111)
        points = data.to_plot_data()['points']
        ax.scatter(points[:, 0], points[:, 1])
        return fig

def test_visualizer_base():
    """Test base visualizer functionality."""
    vis = MockVisualizer()
    
    # Test data preparation
    data = vis.prepare_data()
    assert isinstance(data, VisualizationData)
    assert 'points' in data.to_plot_data()
    
    # Test plotting
    fig = vis.plot(data)
    assert isinstance(fig, Figure)
    
    # Test show and close
    vis.show()
    assert vis._figure is not None
    vis.close()
    assert vis._figure is None

def test_manifold_visualizer():
    """Test manifold visualizer functionality."""
    # Create test data
    embeddings = np.random.randn(10, 5)
    terms = [['cat', 'feline'], ['dog', 'canine'], ['bird', 'avian']] * 3 + ['fish']
    
    vis = ManifoldVisualizer(embeddings, terms)
    
    # Test data preparation
    data = vis.prepare_data()
    assert isinstance(data, ManifoldPlotData)
    plot_data = data.to_plot_data()
    assert 'points' in plot_data
    assert plot_data['points'].shape[1] == 2  # Default n_components=2
    
    # Test color mapping
    colors = data.get_color_map()
    assert len(colors) == len(set().union(*[set(t) if isinstance(t, list) else {t} for t in terms]))
    
    # Test labels
    labels = data.get_labels()
    assert len(labels) == len(terms)

def test_interactive_visualizer():
    """Test interactive visualizer functionality."""
    embeddings = np.random.randn(5, 3)
    terms = [['term1'], ['term2'], ['term3'], ['term4'], ['term5']]
    
    vis = InteractiveManifoldVisualizer(embeddings, terms)
    
    # Test data preparation
    data = vis.prepare_data()
    assert isinstance(data, ManifoldPlotData)
    
    # Test initial state
    state = vis.get_state()
    assert 'scale' in state
    assert 'rotation' in state
    assert 'point_size' in state
    assert 'view_options' in state
    
    # Test plotting
    fig = vis.plot(data)
    assert isinstance(fig, Figure)
    assert len(fig.axes) > 0

def test_visualization_utils():
    """Test visualization utility functions."""
    # Test color gradient
    cmap = create_color_gradient(10)
    assert len(cmap.colors) == 10
    
    # Test coordinate scaling
    coords = np.random.randn(10, 2)
    scaled = scale_coordinates(coords)
    assert np.all(scaled >= -1) and np.all(scaled <= 1)
    
    # Test marker size calculation
    values = np.array([1, 2, 3, 4, 5])
    sizes = calculate_marker_sizes(values)
    assert len(sizes) == len(values)
    assert np.all(sizes >= 20) and np.all(sizes <= 200)
    
    # Test subplot grid creation
    fig, axes = create_subplot_grid(4)
    assert isinstance(fig, Figure)
    assert len(axes) >= 4 