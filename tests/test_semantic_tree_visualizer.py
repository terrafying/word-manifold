"""Tests for the SemanticTreeVisualizer class."""

import os
import numpy as np
import pytest
from word_manifold.visualization.semantic_tree_visualizer import (
    SemanticNode,
    SemanticTreeVisualizer
)

@pytest.fixture
def visualizer():
    """Create a test visualizer instance."""
    return SemanticTreeVisualizer(
        output_dir="test_outputs/semantic_trees",
        color_scheme="viridis",
        node_size_base=800,
        min_similarity=0.3
    )

@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    return np.random.rand(768)  # Common embedding dimension

def test_semantic_node_creation(sample_embedding):
    """Test creation of semantic nodes."""
    node = SemanticNode("test", sample_embedding)
    assert node.text == "test"
    assert node.level == 0
    assert node.parent is None
    assert len(node.children) == 0
    assert node.similarity_to_parent == 1.0
    assert node.semantic_weight == 1.0

def test_semantic_node_child_addition(sample_embedding):
    """Test adding child nodes."""
    parent = SemanticNode("parent", sample_embedding)
    child = SemanticNode("child", sample_embedding)
    parent.add_child(child, similarity=0.8)
    
    assert len(parent.children) == 1
    assert child.parent == parent
    assert child.level == 1
    assert child.similarity_to_parent == 0.8

def test_semantic_weight_calculation(sample_embedding):
    """Test semantic weight calculation."""
    root = SemanticNode("root", sample_embedding)
    child = SemanticNode("child", sample_embedding)
    grandchild = SemanticNode("grandchild", sample_embedding)
    
    root.add_child(child, similarity=0.8)
    child.add_child(grandchild, similarity=0.7)
    
    assert root.calculate_semantic_weight() == 1.0
    assert abs(child.calculate_semantic_weight() - 0.8 * 0.8) < 1e-6
    assert abs(grandchild.calculate_semantic_weight() - 0.8 * 0.7 * 0.8 * 0.8) < 1e-6

def test_visualizer_initialization(visualizer):
    """Test visualizer initialization."""
    assert os.path.exists(visualizer.output_dir)
    assert visualizer.color_scheme == "viridis"
    assert visualizer.node_size_base == 800
    assert visualizer.min_similarity == 0.3

def test_tree_building(visualizer):
    """Test building a semantic tree."""
    root = visualizer.build_semantic_tree(
        root_text="machine learning",
        related_terms=[
            "neural networks",
            "deep learning",
            "artificial intelligence",
            "data science"
        ],
        max_depth=2,
        branching_factor=3
    )
    
    assert root.text == "machine learning"
    assert len(root.children) > 0
    assert all(child.level == 1 for child in root.children)

def test_tree_visualization(visualizer):
    """Test tree visualization."""
    root = visualizer.build_semantic_tree(
        root_text="python",
        related_terms=[
            "programming",
            "coding",
            "software",
            "development"
        ],
        max_depth=2,
        branching_factor=2
    )
    
    output_path = visualizer.visualize_tree(
        root,
        title="Test Tree",
        show_weights=True,
        show_similarities=True
    )
    
    assert os.path.exists(output_path)
    assert output_path.endswith(".png")

def test_minimum_similarity_threshold(visualizer):
    """Test minimum similarity threshold."""
    visualizer.min_similarity = 0.9  # Set very high threshold
    root = visualizer.build_semantic_tree(
        root_text="test",
        related_terms=["unrelated1", "unrelated2"],
        max_depth=2,
        branching_factor=2
    )
    
    # Should have no children due to high similarity threshold
    assert len(root.children) == 0

def test_invalid_terms_handling(visualizer):
    """Test handling of invalid terms."""
    root = visualizer.build_semantic_tree(
        root_text="test",
        related_terms=["", "   ", None],  # Invalid terms
        max_depth=2,
        branching_factor=2
    )
    
    # Should create root node without children
    assert root.text == "test"
    assert len(root.children) == 0 