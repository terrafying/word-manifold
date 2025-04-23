"""
Word Embeddings Module for handling word vector representations.

This module provides functionality for loading, managing, and manipulating
word embeddings, including support for numerological calculations and
semantic similarity operations.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import torch
from functools import lru_cache
from typing import List, Set, Dict, Optional, Union, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default models
DEFAULT_MODEL = 'all-MiniLM-L6-v2'
BACKUP_MODEL = 'paraphrase-MiniLM-L3-v2'

class WordEmbeddings:
    """
    A class for managing word embeddings and related operations.
    
    This class handles loading and caching of embeddings, numerological
    calculations, and semantic similarity operations.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_size: int = 10000
    ):
        """
        Initialize the WordEmbeddings class.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            cache_size: Size of the LRU cache for embeddings
        """
        self._model_name = model_name
        self._cache_size = cache_size
        self._terms: Set[str] = set()
        self._initialize_model()
        
        # Configure embedding cache
        self.get_embedding = lru_cache(maxsize=cache_size)(self._get_embedding_uncached)
    
    def _initialize_model(self):
        """Initialize the embedding model with fallback options."""
        try:
            logger.info(f"Loading model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
        except Exception as e:
            logger.warning(f"Failed to load {self._model_name}: {e}")
            logger.info(f"Attempting to load backup model: {BACKUP_MODEL}")
            try:
                self._model = SentenceTransformer(BACKUP_MODEL)
                self._model_name = BACKUP_MODEL
            except Exception as e2:
                logger.error(f"Failed to load backup model: {e2}")
                raise RuntimeError("Could not initialize any embedding model")
    
    def get_terms(self) -> Set[str]:
        """Get the set of loaded terms.
        
        Returns:
            Set of terms that have been loaded into the embeddings
        """
        return self._terms
    
    def _get_embedding_uncached(self, term: str) -> np.ndarray:
        """Get the embedding for a term without caching.
        
        Args:
            term: The term to get the embedding for
            
        Returns:
            Numpy array containing the embedding vector
        """
        # Ensure the term is in our vocabulary
        if term not in self._terms:
            raise KeyError(f"Term '{term}' not found in loaded terms")
            
        # Get embedding from model
        with torch.no_grad():
            embedding = self._model.encode([term], convert_to_numpy=True)[0]
        return embedding
    
    def load_terms(self, terms: Union[List[str], Set[str]]) -> None:
        """Load terms into the embeddings.
        
        Args:
            terms: List or set of terms to load
        """
        # Convert to set for uniqueness
        terms_set = set(terms)
        
        # Add new terms
        self._terms.update(terms_set)
        
        logger.info(f"Loaded {len(terms_set)} terms")
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings.
        
        Returns:
            Integer dimension of the embedding vectors
        """
        # Get embedding for a test term
        test_term = next(iter(self._terms)) if self._terms else "test"
        return len(self._get_embedding_uncached(test_term))
    
    def find_similar_terms(
        self,
        term: str,
        n: int = 10,
        min_similarity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Find terms similar to the given term.
        
        Args:
            term: The term to find similar terms for
            n: Maximum number of similar terms to return
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of (term, similarity) tuples
        """
        if term not in self._terms:
            raise KeyError(f"Term '{term}' not found in loaded terms")
            
        # Get embedding for the query term
        query_embedding = self.get_embedding(term)
        
        # Calculate similarities with all terms
        similarities = []
        for other_term in self._terms:
            if other_term != term:
                other_embedding = self.get_embedding(other_term)
                similarity = np.dot(query_embedding, other_embedding)
                if similarity >= min_similarity:
                    similarities.append((other_term, float(similarity)))
        
        # Sort by similarity and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    def find_numerological_significance(self, term: str) -> int:
        """Calculate the numerological value of a term.
        
        This implements a basic numerological system where:
        A=1, B=2, ..., Z=26, then sum digits until single digit
        (unless it's a master number: 11, 22, 33)
        
        Args:
            term: Term to calculate value for
            
        Returns:
            Numerological value (1-9, or 11, 22, 33)
        """
        # Basic letter to number mapping
        letter_values = {chr(i): i-96 for i in range(97, 123)}
        
        # Sum letter values
        total = sum(letter_values.get(c.lower(), 0) for c in term)
        
        # Check for master numbers
        if total in {11, 22, 33}:
            return total
            
        # Reduce to single digit
        while total > 9:
            total = sum(int(d) for d in str(total))
            
        return total
    
    def get_term_info(self, term: str) -> Dict[str, Any]:
        """Get comprehensive information about a term.
        
        Args:
            term: The term to get information for
            
        Returns:
            Dictionary containing:
            - embedding: The term's embedding vector
            - numerological_value: The term's numerological value
            - similar_terms: List of similar terms
        """
        return {
            'embedding': self.get_embedding(term),
            'numerological_value': self.find_numerological_significance(term),
            'similar_terms': self.find_similar_terms(term, n=5)
        }
