"""
Common test fixtures and configuration for word-manifold tests.
"""

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np
from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.automata.system import AutomataSystem, EvolutionPattern
from word_manifold.types import CellType

@pytest.fixture
def mock_manifold():
    """Create a mock manifold for testing automata systems."""
    mock = Mock()
    # Mock the cells attribute with some test data
    mock.cells = {
        0: Mock(centroid=np.zeros(10)),
        1: Mock(centroid=np.zeros(10))
    }
    # Mock transform to return the same vectors
    mock.transform = MagicMock(return_value=np.zeros((2, 10)))
    # Mock evolve_manifold method
    mock.evolve_manifold = MagicMock()
    # Mock get_manifold_state to return a valid state
    mock.get_manifold_state = MagicMock(return_value={
        "cell_count": 2,
        "term_count": 0,
        "type_counts": {},
        "num_value_counts": {},
        "avg_connectivity": 0.0,
        "reduced_representation_available": False
    })
    return mock

@pytest.fixture
def base_embeddings():
    """Create a WordEmbeddings instance with test terms."""
    embeddings = WordEmbeddings()
    test_terms = {
        "thelema", "will", "love", "magick", "ritual",
        "knowledge", "wisdom", "power", "light", "dark"
    }
    embeddings.load_terms(test_terms)
    return embeddings

@pytest.fixture
def base_system(mock_manifold):
    """Create a base AutomataSystem for testing."""
    from word_manifold.automata.cellular_rules import create_predefined_rules
    rules = create_predefined_rules()
    return AutomataSystem(
        manifold=mock_manifold,
        rules_dict=rules,
        sequences_dict={},
        evolution_pattern=EvolutionPattern.THELEMIC,
        save_path="test_outputs"
    )