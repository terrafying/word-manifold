"""
Word Embeddings Package

Provides functionality for managing word embeddings from various sources.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

def get_embeddings(
    words: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for a list of words.
    
    Args:
        words: List of words to generate embeddings for
        model_name: Name of the sentence transformer model to use
        
    Returns:
        Dictionary mapping words to their embedding vectors
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(words)
    return {word: emb for word, emb in zip(words, embeddings)}

def load_embeddings(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from a file.
    
    Args:
        file_path: Path to the embeddings file
        
    Returns:
        Dictionary mapping words to their embedding vectors
    """
    data = np.load(file_path, allow_pickle=True)
    return data.item()

def save_embeddings(
    embeddings: Dict[str, np.ndarray],
    file_path: str
):
    """
    Save embeddings to a file.
    
    Args:
        embeddings: Dictionary mapping words to their embedding vectors
        file_path: Path to save the embeddings
    """
    np.save(file_path, embeddings)

def compute_similarity(
    word1: str,
    word2: str,
    embeddings: Dict[str, np.ndarray]
) -> float:
    """
    Compute similarity between two words.
    
    Args:
        word1: First word
        word2: Second word
        embeddings: Dictionary of word embeddings
        
    Returns:
        Cosine similarity between the words
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    vec1 = embeddings[word1].reshape(1, -1)
    vec2 = embeddings[word2].reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

def find_nearest_neighbors(
    word: str,
    embeddings: Dict[str, np.ndarray],
    k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find k nearest neighbors of a word.
    
    Args:
        word: Target word
        embeddings: Dictionary of word embeddings
        k: Number of neighbors to find
        
    Returns:
        List of (word, similarity) tuples
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    vec = embeddings[word].reshape(1, -1)
    similarities = []
    
    for other_word, other_vec in embeddings.items():
        if other_word != word:
            sim = cosine_similarity(vec, other_vec.reshape(1, -1))[0][0]
            similarities.append((other_word, sim))
            
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

__all__ = [
    "get_embeddings",
    "load_embeddings",
    "save_embeddings",
    "compute_similarity",
    "find_nearest_neighbors"
]
