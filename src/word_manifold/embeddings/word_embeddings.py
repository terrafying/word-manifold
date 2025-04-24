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
import gc
from .term_manager import TermManager
from .distributed_term_manager import DistributedTermManager

# Set environment variable to handle tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default models - using smaller models by default
DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'  # Much smaller model
BACKUP_MODEL = 'sentence-transformers/paraphrase-MiniLM-L3-v2'  # Even smaller backup

class WordEmbeddings:
    """
    Manages word embeddings with background processing support.
    """
    
    DEFAULT_MODEL = DEFAULT_MODEL
    BACKUP_MODEL = BACKUP_MODEL
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_size: int = 10000,
        distributed: bool = False,
        ray_address: Optional[str] = None,
        num_workers: int = 2
    ):
        """
        Initialize WordEmbeddings with background processing.
        
        Args:
            model_name: Name of the transformer model to use
            cache_size: Maximum number of terms to cache
            distributed: Whether to use distributed processing
            ray_address: Optional Ray cluster address for distributed processing
            num_workers: Number of workers for distributed processing
        """
        self.model_name = model_name
        
        if distributed:
            logger.info(f"Initializing distributed term manager with Ray{'@'+ray_address if ray_address else ''}")
            self.term_manager = DistributedTermManager(
                model_name=model_name,
                cache_size=cache_size,
                num_workers=num_workers,
                ray_address=ray_address
            )
        else:
            logger.info("Initializing local term manager")
            self.term_manager = TermManager(
                model_name=model_name,
                cache_size=cache_size
            )
            
    def load_terms(self, terms: List[str]):
        """
        Load terms into the embedding space.
        
        Args:
            terms: List of terms to load
        """
        # Add terms to background processor
        self.term_manager.add_terms(terms)
        
        # Store terms locally
        self.term_manager.add_term_set('loaded_terms', set(terms))
        
    def get_embedding(self, term: str, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get embedding for a term.
        
        Args:
            term: Term to get embedding for
            timeout: Optional timeout for distributed processing
            
        Returns:
            Numpy array of embedding or None if not found
        """
        return self.term_manager.get_embedding(term, timeout=timeout)
        
    def get_terms(self) -> Set[str]:
        """
        Get all loaded terms.
        
        Returns:
            Set of loaded terms
        """
        return self.term_manager.get_term_set('loaded_terms')
        
    def __del__(self):
        """Cleanup when object is deleted."""
        if hasattr(self, 'term_manager'):
            self.term_manager.shutdown()

    def _initialize_model(self) -> None:
        """Initialize the transformer model with memory optimization."""
        try:
            logger.info(f"Loading model: {self.DEFAULT_MODEL}")
            self.model = SentenceTransformer(self.DEFAULT_MODEL)
            self.tokenizer = self.model.tokenizer
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            # Force garbage collection after model loading
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Error loading model {self.DEFAULT_MODEL}: {str(e)}")
            logger.info(f"Falling back to {self.BACKUP_MODEL}")
            self.model = SentenceTransformer(self.BACKUP_MODEL)
            self.tokenizer = self.model.tokenizer
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is above threshold."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            if memory_allocated / memory_reserved > 0.8:
                return True
        return False
    
    def _clear_caches(self) -> None:
        """Clear various caches to free memory."""
        self.embeddings.clear()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def get_embeddings(
        self,
        terms: List[str],
        timeout: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple terms.
        
        Args:
            terms: List of terms to get embeddings for
            timeout: Optional timeout for distributed processing
            
        Returns:
            Dictionary mapping terms to their embeddings
        """
        if isinstance(self.term_manager, DistributedTermManager):
            return self.term_manager.get_embeddings_batch(terms, timeout=timeout or 60.0)
        else:
            results = {}
            for term in terms:
                embedding = self.get_embedding(term)
                if embedding is not None:
                    results[term] = embedding
            return results
    
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
