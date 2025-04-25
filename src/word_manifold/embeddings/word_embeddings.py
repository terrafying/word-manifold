"""
Word Embeddings Module with support for local, remote, and Replicate-based operations.

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
import requests
from pathlib import Path
import json
import time
from enum import Enum
import sys
from ..core.model_host import ModelHost

# Set environment variable to handle tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default models - using smaller models by default
DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'  # Much smaller model
BACKUP_MODEL = 'sentence-transformers/paraphrase-MiniLM-L3-v2'  # Even smaller backup

class EmbeddingMode(Enum):
    """Mode for embedding operations."""
    LOCAL = "local"
    REMOTE = "remote"
    REPLICATE = "replicate"

class WordEmbeddings:
    """Manages word embeddings with distributed computation support."""
    
    DEFAULT_MODEL = DEFAULT_MODEL
    BACKUP_MODEL = BACKUP_MODEL
    
    # Default terms for initialization
    DEFAULT_TERMS = [
        # Core concepts
        "wisdom", "understanding", "knowledge", "truth",
        "beauty", "harmony", "balance", "unity",
        
        # Fundamental pairs
        "light", "dark",
        "above", "below",
        "inner", "outer",
        "spirit", "matter",
        
        # Elements
        "fire", "water", "air", "earth",
        
        # Temporal concepts
        "time", "space", "motion", "change",
        "past", "present", "future",
        
        # States of being
        "existence", "consciousness", "awareness",
        "transformation", "evolution", "growth"
    ]
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_workers: int = 2,
        ray_address: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize word embeddings manager."""
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize model host
        self.model_host = ModelHost(
            model_name=model_name,
            num_workers=num_workers,
            ray_address=ray_address
        )
        
        # Initialize storage
        self.terms: Dict[str, np.ndarray] = {}
        
        # Load cached state if available
        self._load_state()
        
        # Load default terms
        self.load_terms(self.DEFAULT_TERMS)
    
    def load_terms(self, terms: List[str]) -> None:
        """Load embeddings for terms."""
        # Filter out already loaded terms
        new_terms = [t for t in terms if t not in self.terms]
        if not new_terms:
            return
            
        # Get embeddings from model host
        embeddings = self.model_host.get_embeddings(new_terms)
        self.terms.update(embeddings)
        
        # Save state if cache directory is set
        if self.cache_dir:
            self._save_state()
    
    def get_embedding(self, term: str) -> Optional[np.ndarray]:
        """Get embedding for a term."""
        if term not in self.terms:
            self.load_terms([term])
        return self.terms.get(term)
    
    def find_similar_terms(
        self,
        term: str,
        k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Find terms similar to the given term."""
        if not self.terms:
            return []
            
        # Get query embedding
        query_embedding = self.get_embedding(term)
        if query_embedding is None:
            return []
            
        # Calculate similarities
        similarities = []
        for other_term, other_embedding in self.terms.items():
            if other_term != term:
                similarity = np.dot(query_embedding, other_embedding)
                if similarity >= min_similarity:
                    similarities.append((other_term, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _load_state(self) -> None:
        """Load cached state."""
        if not self.cache_dir:
            return
            
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.npz"
        if not cache_file.exists():
            return
            
        try:
            data = np.load(str(cache_file), allow_pickle=True)
            self.terms = dict(data['terms'].item())
            logger.info(f"Loaded {len(self.terms)} terms from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    def _save_state(self) -> None:
        """Save current state to cache."""
        if not self.cache_dir:
            return
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.npz"
        
        try:
            np.savez(
                str(cache_file),
                terms=np.array(self.terms, dtype=object)
            )
            logger.info(f"Saved {len(self.terms)} terms to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the embeddings and workers."""
        stats = {
            "model_name": self.model_name,
            "num_terms": len(self.terms),
            "workers": self.model_host.get_worker_stats()
        }
        return stats
    
    def shutdown(self) -> None:
        """Shutdown the embeddings manager."""
        self._save_state()
        self.model_host.shutdown()

    def _initialize_model(self) -> None:
        """Initialize the transformer model with memory optimization."""
        try:
            if not hasattr(self, 'term_manager') or self.term_manager is None:
                self._initialize_local()
                
            # Model initialization is handled by term_manager
            self.embedding_dim = 384  # Standard dimension for the default model
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
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
        self.terms.clear()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def get_terms(self) -> Set[str]:
        """Get all currently loaded terms.
        
        Returns:
            Set of all terms that have been loaded
        """
        return set(self.terms.keys())
        
    def get_embeddings(
        self,
        terms: List[str],
        timeout: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple terms.
        
        Args:
            terms: List of terms to get embeddings for
            timeout: Optional timeout for term processing
            
        Returns:
            Dictionary mapping terms to their embeddings
        """
        # Load any terms we don't have yet
        missing_terms = [t for t in terms if t not in self.terms]
        if missing_terms:
            self.load_terms(missing_terms)
            
        # Return all requested terms that we have embeddings for
        return {term: self.terms[term] for term in terms if term in self.terms}
        
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings."""
        if not hasattr(self, 'embedding_dim'):
            self._initialize_model()
        return self.embedding_dim
    
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
            'similar_terms': self.find_similar_terms(term, k=5),
            'length': len(term)
        }
