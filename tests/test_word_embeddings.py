import pytest
from word_manifold.embeddings.word_embeddings import WordEmbeddings
import numpy as np
from sentence_transformers import SentenceTransformer

@pytest.fixture
def embeddings():
    """Create a WordEmbeddings instance for testing."""
    return WordEmbeddings()

def test_initialization():
    """Test basic initialization with default model."""
    we = WordEmbeddings()
    assert we.model_name == WordEmbeddings.DEFAULT_MODEL
    assert we.cache_size == 1024
    assert isinstance(we.embeddings, dict)

def test_model_fallback():
    """Test fallback to backup model with invalid model name."""
    we = WordEmbeddings(model_name="invalid_model_name")
    assert we.model_name == WordEmbeddings.BACKUP_MODEL

def test_embedding_generation(embeddings):
    """Test embedding generation for single and multiple terms."""
    # Single term
    term = "test"
    embedding = embeddings.get_embedding(term)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embeddings.get_embedding_dim(),)
    
    # Multiple terms
    terms = ["test", "example", "word"]
    batch_embeddings = embeddings.get_embeddings(terms)
    assert isinstance(batch_embeddings, dict)
    assert len(batch_embeddings) == len(terms)
    assert all(isinstance(e, np.ndarray) for e in batch_embeddings.values())

def test_embedding_caching(embeddings):
    """Test that embeddings are properly cached."""
    term = "cache_test"
    
    # First call should compute embedding
    first_embedding = embeddings.get_embedding(term)
    
    # Second call should return cached value
    second_embedding = embeddings.get_embedding(term)
    
    assert np.array_equal(first_embedding, second_embedding)
    assert term in embeddings.embeddings

def test_similar_terms(embeddings):
    """Test finding similar terms."""
    # Load some test terms
    test_terms = ["king", "queen", "prince", "princess", "castle"]
    embeddings.load_terms(test_terms)
    
    similar = embeddings.find_similar_terms("king", n=2)
    assert isinstance(similar, list)
    assert len(similar) == 2
    assert all(t in test_terms for t in similar)
    assert "king" not in similar  # Should not include the query term

def test_numerological_value(embeddings):
    """Test numerological value calculation."""
    # Test basic calculation
    assert embeddings.calculate_numerological_value("test") <= 21
    assert embeddings.calculate_numerological_value("") == 0
    
    # Test reduction to Tarot range
    value = embeddings.calculate_numerological_value("pneumonoultramicroscopicsilicovolcanoconiosis")
    assert 0 <= value <= 21

def test_term_info(embeddings):
    """Test comprehensive term information retrieval."""
    info = embeddings.get_term_info("testing")
    
    assert isinstance(info, dict)
    assert "embedding" in info
    assert "numerological_value" in info
    assert "length" in info
    assert info["length"] == 7

def test_error_handling(embeddings):
    """Test error handling for various edge cases."""
    # Empty string
    embedding = embeddings.get_embedding("")
    assert isinstance(embedding, np.ndarray)
    assert not np.any(embedding)  # Should be zero vector
    
    # Very long input
    long_text = "a" * 1000
    embedding = embeddings.get_embedding(long_text)
    assert isinstance(embedding, np.ndarray)
    
    # Non-string input should raise TypeError
    with pytest.raises(TypeError):
        embeddings.get_embedding(123)

def test_batch_processing(embeddings):
    """Test batch processing efficiency."""
    # Generate a large batch of terms
    terms = [f"term_{i}" for i in range(100)]
    
    # Process in batch
    batch_results = embeddings.get_embeddings(terms)
    
    assert len(batch_results) == len(terms)
    assert all(isinstance(e, np.ndarray) for e in batch_results.values())
    
    # Verify dimensions
    dim = embeddings.get_embedding_dim()
    assert all(e.shape == (dim,) for e in batch_results.values())

def test_word_embeddings_initialization():
    """Test that WordEmbeddings initializes with default model."""
    embeddings = WordEmbeddings()
    assert embeddings.model is not None
    assert isinstance(embeddings.model, SentenceTransformer)

def test_word_embeddings_fallback():
    """Test that WordEmbeddings falls back to backup model."""
    embeddings = WordEmbeddings(model_name="invalid_model_name")
    assert embeddings.model is not None
    assert isinstance(embeddings.model, SentenceTransformer)

def test_embedding_generation():
    """Test that embeddings are generated correctly."""
    embeddings = WordEmbeddings()
    terms = ["test", "example", "word"]
    embeddings.load_terms(terms)
    
    # Test single term embedding
    emb = embeddings.get_embedding("test")
    assert isinstance(emb, np.ndarray)
    assert len(emb.shape) == 1
    assert emb.shape[0] == embeddings.get_embedding_dim()
    
    # Test batch embeddings
    embs = embeddings.get_embeddings(terms)
    assert isinstance(embs, dict)
    assert all(isinstance(v, np.ndarray) for v in embs.values())
    assert all(v.shape[0] == embeddings.get_embedding_dim() for v in embs.values())

def test_similarity_search():
    """Test finding similar terms."""
    embeddings = WordEmbeddings()
    terms = ["king", "queen", "prince", "princess", "duke", "duchess"]
    embeddings.load_terms(terms)
    
    similar = embeddings.find_similar_terms("king", n=2)
    assert len(similar) == 2
    assert "king" not in similar  # Should not include the query term
    assert all(isinstance(term, str) for term in similar)

def test_numerological_values():
    """Test numerological value calculation."""
    embeddings = WordEmbeddings()
    value = embeddings.calculate_numerological_value("test")
    assert isinstance(value, int)
    assert 0 <= value <= 21  # For Tarot-based numerology 