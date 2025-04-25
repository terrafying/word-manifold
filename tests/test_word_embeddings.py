"""Tests for WordEmbeddings class with Replicate integration."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import os
import json
import tempfile
from pathlib import Path
import requests
import logging
import multiprocessing as mp
import threading
import weakref

from word_manifold.embeddings.word_embeddings import WordEmbeddings, EmbeddingMode

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test data
TEST_TERMS = ["wisdom", "understanding", "beauty", "strength"]
TEST_EMBEDDING = np.random.rand(384)  # Common embedding size
MOCK_REPLICATE_RESPONSE = {
    "embeddings": [TEST_EMBEDDING.tolist() for _ in range(len(TEST_TERMS))],
    "similar_terms": [("related_term", 0.8) for _ in range(5)]
}

class MockTermManager:
    """Mock TermManager for testing."""
    def __init__(self, model_name=None):
        self.embeddings = {term: TEST_EMBEDDING for term in TEST_TERMS}
        self.running = True
        
    def get_embedding(self, term):
        return TEST_EMBEDDING
        
    def shutdown(self):
        self.running = False

@pytest.fixture(scope="function")
def mock_term_manager():
    """Mock TermManager to avoid multiprocessing issues in tests."""
    with patch("word_manifold.embeddings.term_manager.TermManager", MockTermManager):
        yield

@pytest.fixture(scope="function")
def mock_replicate():
    """Mock Replicate API responses."""
    with patch("replicate.models.get") as mock_get:
        mock_model = Mock()
        mock_model.predict.return_value = MOCK_REPLICATE_RESPONSE
        mock_get.return_value = mock_model
        yield mock_get

@pytest.fixture(scope="function")
def mock_server():
    """Mock remote server responses."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "terms": {term: TEST_EMBEDDING.tolist() for term in TEST_TERMS},
            "similar_terms": [("related_term", 0.8) for _ in range(5)]
        }
        yield (mock_get, mock_post)

@pytest.fixture(scope="function")
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.mark.parallel
def test_init_replicate_mode(mock_replicate, mock_term_manager):
    """Test initialization in Replicate mode."""
    # Test successful initialization
    embeddings = WordEmbeddings(
        mode="replicate",
        replicate_api_token="test_token",
        replicate_model="test/model:v1"
    )
    assert embeddings.mode == EmbeddingMode.REPLICATE
    
    # Test missing API token
    with pytest.raises(ValueError, match="Replicate API token required"):
        WordEmbeddings(mode="replicate")
    
    # Test failed connection with fallback
    mock_replicate.side_effect = Exception("Connection failed")
    embeddings = WordEmbeddings(
        mode="replicate",
        replicate_api_token="test_token",
        fallback_mode="local"
    )
    assert embeddings.mode == EmbeddingMode.LOCAL

@pytest.mark.parallel
def test_init_remote_mode(mock_server, mock_term_manager):
    """Test initialization in remote mode."""
    mock_get, _ = mock_server
    
    # Test successful initialization
    embeddings = WordEmbeddings(
        mode="remote",
        server_url="http://test-server"
    )
    assert embeddings.mode == EmbeddingMode.REMOTE
    assert embeddings.is_remote
    
    # Test missing server URL
    with pytest.raises(ValueError, match="Server URL required"):
        WordEmbeddings(mode="remote")
    
    # Test failed connection with fallback
    mock_get.side_effect = requests.exceptions.ConnectionError
    embeddings = WordEmbeddings(
        mode="remote",
        server_url="http://test-server",
        fallback_mode="local"
    )
    assert embeddings.mode == EmbeddingMode.LOCAL

@pytest.mark.parallel
def test_load_terms_replicate(mock_replicate, mock_term_manager):
    """Test loading terms using Replicate."""
    embeddings = WordEmbeddings(
        mode="replicate",
        replicate_api_token="test_token"
    )
    
    # Test batch loading
    embeddings.load_terms(TEST_TERMS)
    assert all(term in embeddings.terms for term in TEST_TERMS)
    assert all(isinstance(emb, np.ndarray) for emb in embeddings.terms.values())
    
    # Test error handling
    mock_replicate.return_value.predict.side_effect = Exception("API error")
    with pytest.raises(Exception, match="API error"):
        embeddings.load_terms(["new_term"])

@pytest.mark.parallel
def test_load_terms_remote(mock_server, mock_term_manager):
    """Test loading terms from remote server."""
    _, mock_post = mock_server
    embeddings = WordEmbeddings(
        mode="remote",
        server_url="http://test-server"
    )
    
    # Test successful loading
    embeddings.load_terms(TEST_TERMS)
    assert all(term in embeddings.terms for term in TEST_TERMS)
    
    # Test error handling
    mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
    with pytest.raises(requests.exceptions.RequestException):
        embeddings.load_terms(["new_term"])

@pytest.mark.parallel
def test_load_terms_local(tmp_path):
    """Test loading terms in local mode."""
    # Create embeddings instance with local cache
    embeddings = WordEmbeddings(
        mode=EmbeddingMode.LOCAL,
        cache_dir=tmp_path
    )
    
    # Test terms to load
    test_terms = ["test", "example", "word"]
    
    # Load terms
    embeddings.load_terms(test_terms)
    
    # Verify terms were loaded
    assert all(term in embeddings.terms for term in test_terms)
    assert all(isinstance(emb, np.ndarray) for emb in embeddings.terms.values())
    
    # Test force reload
    old_embeddings = {term: emb.copy() for term, emb in embeddings.terms.items()}
    embeddings.load_terms(test_terms, force_reload=True)
    
    # Verify embeddings were reloaded
    assert all(term in embeddings.terms for term in test_terms)
    assert any(not np.array_equal(old_embeddings[term], embeddings.terms[term]) 
              for term in test_terms)
    
    # Test state persistence
    state_file = tmp_path / "embeddings_state.json"
    assert state_file.exists()
    
    # Create new instance and verify state loads
    new_embeddings = WordEmbeddings(
        mode=EmbeddingMode.LOCAL,
        cache_dir=tmp_path
    )
    assert all(term in new_embeddings.terms for term in test_terms)
    assert all(np.array_equal(embeddings.terms[term], new_embeddings.terms[term]) 
              for term in test_terms)

@pytest.mark.parallel
def test_caching(temp_cache_dir, mock_replicate, mock_term_manager):
    """Test embedding caching functionality."""
    embeddings = WordEmbeddings(
        mode="replicate",
        replicate_api_token="test_token",
        cache_dir=temp_cache_dir
    )
    
    # Test cache creation
    embeddings.load_terms(TEST_TERMS)
    cache_file = temp_cache_dir / "embeddings.json"
    assert cache_file.exists()
    
    # Test cache loading
    with open(cache_file) as f:
        cache_data = json.load(f)
    assert all(term in cache_data for term in TEST_TERMS)
    
    # Test cache update
    embeddings.load_terms(["new_term"])
    with open(cache_file) as f:
        updated_cache = json.load(f)
    assert "new_term" in updated_cache

@pytest.mark.parallel
def test_find_similar_terms_replicate(mock_replicate, mock_term_manager):
    """Test finding similar terms using Replicate."""
    embeddings = WordEmbeddings(
        mode="replicate",
        replicate_api_token="test_token"
    )
    
    # Test with term
    similar = embeddings.find_similar_terms("wisdom", k=5)
    assert len(similar) == 5
    assert all(isinstance(s, tuple) and len(s) == 2 for s in similar)
    
    # Test with embedding
    similar = embeddings.find_similar_terms(
        "test",
        k=3,
        embedding=np.random.rand(384)
    )
    assert len(similar) == 3

@pytest.mark.parallel
def test_find_similar_terms_remote(mock_server, mock_term_manager):
    """Test finding similar terms from remote server."""
    embeddings = WordEmbeddings(
        mode="remote",
        server_url="http://test-server"
    )
    
    # Test successful search
    similar = embeddings.find_similar_terms("wisdom", k=5)
    assert len(similar) == 5
    
    # Test with embedding
    similar = embeddings.find_similar_terms(
        "test",
        k=3,
        embedding=np.random.rand(384)
    )
    assert len(similar) == 3

@pytest.mark.parallel
def test_find_similar_terms_local(mock_term_manager):
    """Test finding similar terms locally."""
    embeddings = WordEmbeddings(mode="local")
    embeddings.load_terms(TEST_TERMS)
    
    # Test local similarity search
    similar = embeddings.find_similar_terms(TEST_TERMS[0], k=2)
    assert len(similar) == 2
    assert all(isinstance(s, tuple) and len(s) == 2 for s in similar)

@pytest.mark.parallel
def test_cleanup(mock_term_manager):
    """Test resource cleanup."""
    embeddings = WordEmbeddings(mode="local")
    embeddings.load_terms(TEST_TERMS)
    
    # Test term manager cleanup
    assert hasattr(embeddings, 'term_manager')
    assert embeddings.term_manager.running
    del embeddings
    # Cleanup is handled by the mock

@pytest.mark.parallel
def test_error_handling(mock_term_manager):
    """Test error handling and logging."""
    # Test invalid mode
    with pytest.raises(ValueError):
        WordEmbeddings(mode="invalid")
    
    # Test missing dependencies
    with patch.dict('sys.modules', {'replicate': None}):
        with pytest.raises(ImportError):
            embeddings = WordEmbeddings(mode="replicate")
    
    # Test invalid server URL
    with pytest.raises(ValueError):
        WordEmbeddings(mode="remote", server_url="invalid-url")

def pytest_configure(config):
    """Register parallel marker."""
    config.addinivalue_line(
        "markers",
        "parallel: mark test to run in parallel"
    )

if __name__ == "__main__":
    pytest.main(["-n", "auto", "--dist=loadfile", __file__, "-v", "--tb=short"]) 