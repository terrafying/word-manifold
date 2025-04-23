"""Word Manifold - A system for semantic transformations in vector space."""

from .embeddings.word_embeddings import WordEmbeddings
from .manifold.vector_manifold import VectorManifold
from .automata.cellular_rules import CellularRule, RuleSequence
from .visualization.hypertools_visualizer import HyperToolsVisualizer
from .visualization.shape_visualizer import ShapeVisualizer

__version__ = "0.1.0"

__all__ = [
    'WordEmbeddings',
    'VectorManifold',
    'CellularRule',
    'RuleSequence',
    'HyperToolsVisualizer',
    'ShapeVisualizer'
]
