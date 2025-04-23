"""
Tests for core components of the word-manifold package.

This includes tests for:
1. Cell data structures
2. Vector manifold operations
3. Word embeddings functionality
4. Distance calculations
"""

import numpy as np
import pytest
from word_manifold.manifold.vector_manifold import VectorManifold, Cell
from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.types import CellType, DistanceType

class TestCell:
    """Tests for the Cell data structure."""
    
    def test_cell_creation(self):
        """Test Cell dataclass creation and basic properties."""
        cell = Cell(
            id=1,
            terms=["thelema", "will"],
            centroid=np.array([0.1, 0.2, 0.3]),
            type=CellType.TAROT,
            numerological_value=93
        )
        assert cell.id == 1
        assert "thelema" in cell.terms
        assert cell.type == CellType.TAROT
        assert cell.numerological_value == 93
        assert cell.centroid.shape == (3,)

class TestVectorManifold:
    """Tests for the VectorManifold class."""
    
    def test_initialization(self, base_embeddings):
        """Test VectorManifold initialization and basic properties."""
        manifold = VectorManifold(base_embeddings)
        
        # Check basic properties
        assert manifold.term_to_index is not None
        assert len(manifold.term_to_index) == len(base_embeddings.terms)
        assert manifold.n_cells == 22  # Default value
        assert manifold.reduction_dims == 3  # Default value
        
    def test_cell_creation(self, base_embeddings):
        """Test cell creation and properties in the manifold."""
        manifold = VectorManifold(base_embeddings)
        
        # Check that cells were created
        assert len(manifold.cells) > 0
        
        # Check cell properties
        for cell in manifold.cells.values():
            assert isinstance(cell, Cell)
            assert cell.centroid is not None
            assert len(cell.terms) > 0
            
    def test_add_cell(self, base_embeddings):
        """Test adding a new cell to the manifold."""
        manifold = VectorManifold(base_embeddings)
        initial_cell_count = len(manifold.cells)
        
        # Add a new cell with some terms
        test_terms = ["light", "sun", "fire"]  # Assuming these exist in base_embeddings
        cell_id = manifold.add_cell(test_terms, CellType.ELEMENTAL, numerological_value=5)
        
        # Check cell was added
        assert cell_id is not None
        assert len(manifold.cells) == initial_cell_count + 1
        
        # Check cell properties
        new_cell = manifold.cells[cell_id]
        assert isinstance(new_cell, Cell)
        assert new_cell.type == CellType.ELEMENTAL
        assert new_cell.numerological_value == 5
        assert all(term in new_cell.terms for term in test_terms)
        
        # Check term mappings
        for term in test_terms:
            assert manifold.term_to_cell[term] == cell_id
            
        # Test adding invalid terms
        invalid_cell_id = manifold.add_cell(["nonexistent_term"])
        assert invalid_cell_id is None
        
        # Test automatic numerological value calculation
        auto_cell_id = manifold.add_cell(["moon", "star"])  # Assuming these exist
        assert auto_cell_id is not None
        auto_cell = manifold.cells[auto_cell_id]
        assert auto_cell.numerological_value > 0
            
    def test_distance_calculations(self, base_embeddings):
        """Test different distance metrics between cells."""
        manifold = VectorManifold(base_embeddings)
        
        # Get first two cell IDs
        cell_ids = list(manifold.cells.keys())[:2]
        if len(cell_ids) >= 2:
            # Test Euclidean distance
            d1 = manifold.compute_cell_distance(
                cell_ids[0], cell_ids[1], 
                DistanceType.EUCLIDEAN
            )
            assert d1 >= 0
            
            # Test Cosine distance
            d2 = manifold.compute_cell_distance(
                cell_ids[0], cell_ids[1], 
                DistanceType.COSINE
            )
            assert 0 <= d2 <= 2  # Cosine distance is between 0 and 2
            
            # Test Numerological distance
            d3 = manifold.compute_cell_distance(
                cell_ids[0], cell_ids[1], 
                DistanceType.NUMEROLOGICAL
            )
            assert d3 >= 0

class TestWordEmbeddings:
    """Tests for the WordEmbeddings class."""
    
    def test_initialization(self):
        """Test WordEmbeddings initialization."""
        embeddings = WordEmbeddings()
        assert embeddings.model is not None
        assert embeddings.tokenizer is not None
        assert len(embeddings.terms) == 0
        
    def test_term_loading(self, base_embeddings):
        """Test loading terms and computing embeddings."""
        test_terms = {"thelema", "magick"}
        base_embeddings.load_terms(test_terms)
        
        # Check that terms were loaded
        assert test_terms.issubset(base_embeddings.terms)
        
        # Check embeddings
        for term in test_terms:
            embedding = base_embeddings.get_embedding(term)
            assert isinstance(embedding, np.ndarray)
            assert embedding.ndim == 1
            
    def test_numerological_values(self, base_embeddings):
        """Test numerological value calculations."""
        # Test known occult number
        assert base_embeddings.find_numerological_significance("thelema") == 93
        
        # Test regular term
        value = base_embeddings.find_numerological_significance("test")
        assert isinstance(value, int)
        assert 1 <= value <= 9 or value in {11, 22, 33}  # Single digit or master number
        
    def test_similarity_search(self, base_embeddings):
        """Test FAISS-based similarity search."""
        query = "thelema"
        k = 3
        results = base_embeddings.find_similar_terms(query, k=k)
        
        # Check results format and properties
        assert len(results) <= k
        for term, distance in results:
            assert isinstance(term, str)
            assert isinstance(distance, (float, np.floating))  # Accept both Python float and numpy float types
            assert float(distance) >= 0  # Convert to Python float for comparison 