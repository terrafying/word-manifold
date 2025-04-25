"""
Recursive dimensionality reduction with fractal patterns.
"""

import numpy as np
import umap
from hdbscan import HDBSCAN
from typing import Dict, Any, Optional, List
import gc
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist

class RecursiveReducer:
    """Handles recursive dimensionality reduction with fractal patterns."""
    
    def __init__(
        self,
        manifold,
        depth: int = 3,
        min_cluster_size: int = 5,
        coherence_threshold: float = 0.8,
        scale_factor: float = 0.5,
        rotation_symmetry: int = 5
    ):
        """Initialize the recursive reducer.
        
        Args:
            manifold: VectorManifold instance to reduce
            depth: Maximum recursion depth
            min_cluster_size: Minimum points per cluster
            coherence_threshold: Minimum coherence to continue recursion
            scale_factor: Controls self-similarity scale
            rotation_symmetry: Number of rotational symmetries
        """
        self.manifold = manifold
        self.depth = depth
        self.min_cluster_size = min_cluster_size
        self.coherence_threshold = coherence_threshold
        self.scale_factor = scale_factor
        self.rotation_symmetry = rotation_symmetry
        
    def reduce(self) -> Dict[str, Any]:
        """Apply recursive reduction to the manifold."""
        return self._reduce_cluster(self.manifold.vectors, 0)
        
    def _reduce_cluster(self, vectors: np.ndarray, level: int) -> Dict[str, Any]:
        """Recursively reduce a cluster of vectors."""
        if level >= self.depth or len(vectors) < self.min_cluster_size:
            return {
                'points': vectors,
                'children': [],
                'coherence': 1.0,
                'scale': self.scale_factor ** level
            }
            
        # First reduction at current level with adaptive parameters
        n_neighbors = max(min(int(15 * (0.8 ** level)), len(vectors) - 1), 2)
        min_dist = 0.1 * (self.scale_factor ** level)
        
        reducer = umap.UMAP(
            n_components=self.manifold.reduction_dims,
            random_state=self.manifold.random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            local_connectivity=2,
            repulsion_strength=2.0 * (level + 1)
        )
        
        # Apply initial reduction
        reduced = reducer.fit_transform(vectors)
        
        # Apply symmetry transformation
        reduced = self._apply_symmetry_transform(reduced, level)
        
        # Use HDBSCAN with level-specific parameters
        clusterer = HDBSCAN(
            min_cluster_size=max(self.min_cluster_size, int(len(vectors) * 0.1)),
            min_samples=max(2, int(np.log2(len(vectors)))),
            cluster_selection_epsilon=0.1 * (self.scale_factor ** level),
            metric='euclidean',
            cluster_selection_method='leaf'
        )
        cluster_labels = clusterer.fit_predict(reduced)
        
        # Process each sub-cluster recursively
        children = []
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
            
        # Sort clusters by size for more stable patterns
        label_sizes = [(label, (cluster_labels == label).sum()) 
                      for label in unique_labels]
        sorted_labels = sorted(label_sizes, key=lambda x: x[1], reverse=True)
        
        for label, size in sorted_labels:
            mask = cluster_labels == label
            sub_vectors = vectors[mask]
            sub_reduced = reduced[mask]
            
            # Calculate local coherence
            coherence = self._calculate_coherence(sub_vectors, sub_reduced)
            
            if coherence >= self.coherence_threshold * (self.scale_factor ** level):
                child_result = self._reduce_cluster(sub_vectors, level + 1)
                children.append(child_result)
        
        # Calculate global coherence for this level
        level_coherence = self._calculate_coherence(vectors, reduced)
        
        return {
            'points': reduced,
            'children': children,
            'coherence': level_coherence,
            'scale': self.scale_factor ** level,
            'n_clusters': len(children)
        }
        
    def _apply_symmetry_transform(self, points: np.ndarray, level: int) -> np.ndarray:
        """Apply symmetric transformations to create fractal patterns."""
        # Calculate level-specific rotation angle
        theta = 2 * np.pi / (self.rotation_symmetry * (level + 1))
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # Scale based on level
        scale = self.scale_factor ** level
        
        # Apply transformation
        transformed = points.copy()
        if points.shape[1] >= 2:  # Only if we have at least 2D
            # Apply rotation only to first 2 dimensions
            transformed[:, :2] = transformed[:, :2] @ rotation_matrix.T
            # Scale all dimensions
            transformed = transformed * scale
            
        return transformed
        
    def _calculate_coherence(self, original: np.ndarray, reduced: np.ndarray) -> float:
        """Calculate how well the reduced representation preserves distances."""
        # Calculate pairwise distances in both spaces
        original_dist = pdist(original, metric='cosine')
        reduced_dist = pdist(reduced, metric='euclidean')
        
        # Calculate rank correlation
        correlation, _ = spearmanr(original_dist, reduced_dist)
        
        return max(0.0, correlation)  # Ensure non-negative 