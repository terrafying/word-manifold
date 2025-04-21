"""
Word Embeddings Module for Cellular Automata in Word Vector Space.

This module provides functionality for loading, processing, and manipulating
word embeddings with a focus on occult terminology. It serves as the foundation
for the cellular automata system operating in word embedding space.
"""

import os
import json
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from pathlib import Path
import pickle
import logging
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Basic set of occult terms for initial testing
OCCULT_TERMS = {
    # Thelema/Crowley related
    "thelema", "crowley", "aiwass", "liber", "babalon", "hoor", "nuit", "hadit", 
    "magick", "aeon", "abrahadabra", "aethyr", "choronzon",
    
    # Tarot
    "tarot", "arcana", "pentacles", "wands", "cups", "swords", "hierophant", 
    "hermit", "magician", "priestess", "empress", "emperor", "chariot",
    
    # Kabbalah/Numerology
    "kabbalah", "sephiroth", "gematria", "ain", "soph", "kether", "chokmah", 
    "binah", "chesed", "geburah", "tiphareth", "netzach", "hod", "yesod", "malkuth",
    
    # General occult
    "alchemy", "hermeticism", "grimoire", "ritual", "invocation", "evocation", 
    "astral", "thaumaturgy", "theurgy", "enochian", "goetia", "pentagram",
    
    # Numerology
    "numerology", "pythagoras", "tetraktys", "sacred", "geometry", "abramelin",
    
    # Elements/Directions
    "earth", "air", "fire", "water", "spirit", "east", "west", "north", "south",
}

class WordEmbeddings:
    """
    A class for handling word embeddings with a focus on occult terminology.
    
    This class provides methods to load, process, and manipulate word embeddings
    using transformer models. It includes functionality for similarity calculations
    and neighborhood operations necessary for cellular automata in word vector space.
    """
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased", 
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the WordEmbeddings class.
        
        Args:
            model_name: Name of the transformer model to use for embeddings
            cache_dir: Directory to cache embeddings
            device: Device to use for computation ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'data', 'embeddings_cache')
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if GPU is available, otherwise use CPU
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Dictionary to store word embeddings (word -> embedding vector)
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Set of terms we're working with
        self.terms: Set[str] = set()
        
    def load_terms(self, terms: Optional[Set[str]] = None) -> None:
        """
        Load a set of terms and compute/retrieve their embeddings.
        
        Args:
            terms: Set of terms to load. If None, uses the default OCCULT_TERMS.
        """
        if terms is None:
            terms = OCCULT_TERMS
        
        self.terms = terms
        logger.info(f"Loading {len(terms)} terms")
        
        # First check if we have cached embeddings
        cache_file = os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_embeddings.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    cached_words = set(cached_data.keys())
                    # Get intersection of our terms and cached terms
                    available_terms = terms.intersection(cached_words)
                    if available_terms:
                        logger.info(f"Loaded {len(available_terms)} embeddings from cache")
                        # Load available embeddings from cache
                        for term in available_terms:
                            self.embeddings[term] = cached_data[term]
                    
                    # Calculate embeddings for terms not in cache
                    missing_terms = terms - available_terms
                    if missing_terms:
                        logger.info(f"Computing embeddings for {len(missing_terms)} new terms")
                        self._compute_embeddings(missing_terms)
            except Exception as e:
                logger.error(f"Error loading cache: {e}. Computing embeddings from scratch.")
                self._compute_embeddings(terms)
        else:
            logger.info("No cache found. Computing embeddings from scratch.")
            self._compute_embeddings(terms)
            
        # Save updated cache
        with open(cache_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    def _compute_embeddings(self, terms: Set[str]) -> None:
        """
        Compute embeddings for the given terms using the transformer model.
        
        Args:
            terms: Set of terms to compute embeddings for
        """
        self.model.eval()  # Set model to evaluation mode
        
        # Process terms in batches to avoid memory issues
        batch_size = 32
        terms_list = list(terms)
        
        for i in range(0, len(terms_list), batch_size):
            batch_terms = terms_list[i:i+batch_size]
            
            # Tokenize the terms
            encoded_input = self.tokenizer(batch_terms, padding=True, truncation=True, 
                                          return_tensors='pt').to(self.device)
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            # We use the CLS token embedding as the sentence embedding
            sentence_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Store embeddings
            for j, term in enumerate(batch_terms):
                self.embeddings[term] = sentence_embeddings[j]
                
        logger.info(f"Computed embeddings for {len(terms)} terms")
    
    def get_embedding(self, term: str) -> np.ndarray:
        """
        Get the embedding for a specific term.
        
        Args:
            term: The term to get the embedding for
            
        Returns:
            The embedding vector for the term
            
        Raises:
            KeyError: If the term has no computed embedding
        """
        if term in self.embeddings:
            return self.embeddings[term]
        
        # If we don't have this term, compute its embedding
        logger.info(f"Computing embedding for new term: {term}")
        self._compute_embeddings({term})
        return self.embeddings[term]
    
    def get_similarity(self, term1: str, term2: str) -> float:
        """
        Calculate the cosine similarity between two terms.
        
        Args:
            term1: First term
            term2: Second term
            
        Returns:
            Cosine similarity between the term embeddings (0-1)
        """
        vec1 = self.get_embedding(term1).reshape(1, -1)
        vec2 = self.get_embedding(term2).reshape(1, -1)
        return float(cosine_similarity(vec1, vec2)[0][0])
    
    def get_nearest_neighbors(self, term: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        Find the n nearest neighbors of a term in the embedding space.
        
        Args:
            term: The term to find neighbors for
            n: Number of neighbors to retrieve
            
        Returns:
            List of (term, similarity) tuples for the nearest neighbors
        """
        if not self.terms:
            raise ValueError("No terms loaded. Call load_terms() first.")
            
        query_vec = self.get_embedding(term).reshape(1, -1)
        
        # Calculate similarities with all terms
        similarities = []
        for other_term in self.terms:
            if other_term == term:
                continue
                
            other_vec = self.get_embedding(other_term).reshape(1, -1)
            sim = float(cosine_similarity(query_vec, other_vec)[0][0])
            similarities.append((other_term, sim))
            
        # Sort by similarity (descending) and return top n
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]
    
    def get_region_centroid(self, terms: List[str]) -> np.ndarray:
        """
        Calculate the centroid of a region defined by multiple terms.
        
        Args:
            terms: List of terms defining the region
            
        Returns:
            Centroid vector of the region
        """
        embeddings = np.array([self.get_embedding(term) for term in terms])
        return np.mean(embeddings, axis=0)
    
    def apply_vector_operation(
        self, 
        base_term: str, 
        operation: str, 
        magnitude: float = 1.0,
        direction_term: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply a vector operation to a term's embedding.
        
        This is useful for cellular automata rules that transform embeddings.
        
        Args:
            base_term: Term to apply the operation to
            operation: Operation type ('shift', 'amplify', 'contrast')
            magnitude: Magnitude of the operation
            direction_term: For 'shift' operations, term defining the direction
            
        Returns:
            The resulting vector after applying the operation
        """
        base_vec = self.get_embedding(base_term)
        
        if operation == 'shift' and direction_term:
            # Shift the base vector toward or away from direction_term
            direction_vec = self.get_embedding(direction_term)
            direction = direction_vec - base_vec
            # Normalize the direction vector
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            return base_vec + (direction * magnitude)
            
        elif operation == 'amplify':
            # Amplify the vector (multiply by magnitude)
            norm = np.linalg.norm(base_vec)
            if norm > 0:
                normalized = base_vec / norm
                return normalized * (norm * magnitude)
            return base_vec
            
        elif operation == 'contrast':
            # Increase contrast/distinctiveness of the vector
            # For positive magnitude, move away from the origin
            # For negative magnitude, move toward the origin
            norm = np.linalg.norm(base_vec)
            if norm > 0:
                normalized = base_vec / norm
                new_norm = norm * (1 + magnitude)
                return normalized * new_norm
            return base_vec
            
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def find_numerological_significance(self, term: str) -> int:
        """
        Calculate a numerological value for a term using English Gematria.
        
        This implements a simple English alphanumeric cipher (A=1, B=2, etc.)
        for numerological calculations in the occult tradition.
        
        Args:
            term: The term to calculate numerological value for
            
        Returns:
            Numerological value
        """
        # Simple English Gematria (A=1, B=2, ...)
        gematria_value = sum(
            ord(c.upper()) - ord('A') + 1 
            for c in term 
            if c.isalpha()
        )
        
        # Reduce to a single digit (theosophical reduction)
        while gematria_value > 9 and gematria_value not in [11, 22, 33]:  # Keep master numbers
            gematria_value = sum(int(digit) for digit in str(gematria_value))
            
        return gematria_value

