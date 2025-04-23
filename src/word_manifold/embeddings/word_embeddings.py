"""
Word Embeddings Module for handling word vector representations.

This module provides functionality for loading, managing, and manipulating
word embeddings, including support for numerological calculations and
semantic similarity operations. Uses state-of-the-art GTE models.
"""

import numpy as np
import torch
import logging
import os
from typing import List, Set, Dict, Optional, Union, Tuple, Any
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Set environment variable to handle tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default models
DEFAULT_MODEL = 'Alibaba-NLP/gte-Qwen2-7B-instruct'
BACKUP_MODEL = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'

class WordEmbeddings:
    """
    A class for managing word embeddings and related operations.
    
    Uses state-of-the-art GTE models for high-quality embeddings with
    support for long sequences and multilingual content.
    """
    
    DEFAULT_MODEL = DEFAULT_MODEL
    BACKUP_MODEL = BACKUP_MODEL
    
    def __init__(self, model_name: str = DEFAULT_MODEL, cache_size: int = 1000):
        """
        Initialize the embeddings manager.
        
        Args:
            model_name: Name of the transformer model to use
            cache_size: Size of the LRU cache for embeddings
        """
        self.model_name = model_name
        self.terms: Set[str] = set()  # Initialize empty set of terms
        self.embeddings = {}  # Cache for embeddings
        self._initialize_model()
        
        # Setup caching
        self._get_embedding = lru_cache(maxsize=cache_size)(self._get_embedding_uncached)
    
    def _initialize_model(self) -> None:
        """Initialize the transformer model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.tokenizer = self.model.tokenizer
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            logger.info(f"Falling back to {self.BACKUP_MODEL}")
            self.model_name = self.BACKUP_MODEL
            self.model = SentenceTransformer(self.BACKUP_MODEL)
            self.tokenizer = self.model.tokenizer
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def load_terms(self, terms: Union[List[str], Set[str]]) -> None:
        """
        Load terms into the embeddings manager.
        
        Args:
            terms: List or set of terms to load
        """
        if isinstance(terms, list):
            terms = set(terms)
        self.terms.update(terms)
        logger.info(f"Loaded {len(terms)} terms")
    
    def get_terms(self) -> Set[str]:
        """Get the set of loaded terms."""
        return self.terms.copy()
    
    def _get_embedding_uncached(self, term: str) -> Optional[np.ndarray]:
        """
        Get embedding for a term without caching.
        
        Args:
            term: Term to get embedding for
            
        Returns:
            Embedding vector or None if term not found
        """
        if term not in self.terms:
            logger.warning(f"Term '{term}' not found in loaded terms")
            return None
        
        try:
            embedding = self.model.encode([term], convert_to_numpy=True)[0]
            self.embeddings[term] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for '{term}': {str(e)}")
            return None
    
    def get_embedding(self, term: str) -> Optional[np.ndarray]:
        """
        Get embedding for a term.
        
        Args:
            term: Term to get embedding for
            
        Returns:
            Embedding vector or None if term not found
        """
        return self._get_embedding(term)
    
    def get_embeddings(self, terms: List[str]) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple terms.
        
        Args:
            terms: List of terms to get embeddings for
            
        Returns:
            Dictionary mapping terms to their embeddings
        """
        valid_terms = [t for t in terms if t in self.terms]
        if not valid_terms:
            return {}
            
        try:
            embeddings = self.model.encode(valid_terms, convert_to_numpy=True)
            return {term: emb for term, emb in zip(valid_terms, embeddings)}
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {str(e)}")
            return {}
    
    def get_embeddings_batch(self, terms: List[str]) -> Dict[str, np.ndarray]:
        """Alias for get_embeddings for backward compatibility."""
        return self.get_embeddings(terms)
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.embedding_dim
    
    def find_similar_terms(self, term: str, n: int = 5, k: Optional[int] = None) -> List[Union[str, Tuple[str, float]]]:
        """
        Find similar terms to the given term.
        
        Args:
            term: Term to find similar terms for
            n: Number of similar terms to return (deprecated, use k instead)
            k: Number of similar terms to return
            
        Returns:
            List of (term, similarity) tuples
        """
        k = k or n  # Support both n and k for backward compatibility
        
        if term not in self.terms:
            return []
            
        query_embedding = self.get_embedding(term)
        if query_embedding is None:
            return []
            
        similarities = []
        for other_term in self.terms:
            if other_term == term:
                continue
            other_embedding = self.get_embedding(other_term)
            if other_embedding is not None:
                similarity = np.dot(query_embedding, other_embedding)
                similarities.append((other_term, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    
    def calculate_numerological_value(self, term: str) -> int:
        """Alias for find_numerological_significance."""
        return self.find_numerological_significance(term)
    
    def find_numerological_significance(self, term: str) -> int:
        """Calculate the numerological value of a term.
        
        This implements a basic numerological system where:
        A=1, B=2, ..., Z=26, then sum digits until single digit
        (unless it's a master number: 11, 22, 33)
        
        Special cases:
        - "thelema" = 93 (special occult significance)
        
        Args:
            term: Term to calculate value for
            
        Returns:
            Numerological value (1-9, or 11, 22, 33, 93)
        """
        # Special case for thelema
        if term.lower() == "thelema":
            return 93
            
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
            - length: Length of the term
        """
        return {
            'embedding': self.get_embedding(term),
            'numerological_value': self.find_numerological_significance(term),
            'similar_terms': self.find_similar_terms(term, n=5),
            'length': len(term)
        }
