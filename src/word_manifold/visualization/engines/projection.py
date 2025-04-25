"""
Hyperdimensional projection engine for transforming high-dimensional data into 3D space.
Supports various projection methods and transformations for visualization.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ProjectionEngine:
    """Engine for projecting high-dimensional data to 3D space."""
    
    PROJECTION_TYPES = {
        "stereographic": "Stereographic projection from N-sphere to (N-1)-space",
        "orthographic": "Orthographic projection taking first 3 components",
        "pca": "Principal Component Analysis to 3D",
        "umap": "UMAP dimensionality reduction",
        "tsne": "t-SNE embedding"
    }
    
    def __init__(self, 
                 projection_type: str = "stereographic",
                 random_state: Optional[int] = None):
        """
        Initialize projection engine.
        
        Args:
            projection_type: Type of projection to use
            random_state: Random seed for reproducibility
        """
        if projection_type not in self.PROJECTION_TYPES:
            raise ValueError(f"Unknown projection type: {projection_type}")
            
        self.projection_type = projection_type
        self.random_state = random_state
        self._reducer = None
        
    def project_points(self, 
                      points: np.ndarray,
                      fit: bool = True) -> np.ndarray:
        """
        Project points from high dimensions to 3D.
        
        Args:
            points: Array of shape (n_points, n_dimensions)
            fit: Whether to fit projection parameters
            
        Returns:
            Array of shape (n_points, 3)
        """
        if points.shape[1] <= 3:
            return points
            
        if self.projection_type == "stereographic":
            return self._stereographic_projection(points)
        elif self.projection_type == "orthographic":
            return points[:, :3]
        elif self.projection_type == "pca":
            return self._pca_projection(points, fit)
        elif self.projection_type == "umap":
            return self._umap_projection(points, fit)
        else:  # t-SNE
            return self._tsne_projection(points)
            
    def _stereographic_projection(self, points: np.ndarray) -> np.ndarray:
        """
        Apply stereographic projection from N-sphere to (N-1)-space.
        
        Args:
            points: Input points
            
        Returns:
            Projected points
        """
        # Normalize to unit sphere
        norm = np.linalg.norm(points, axis=1, keepdims=True)
        points_normalized = points / norm
        
        # Project from last coordinate
        return points_normalized[:, :3] / (1 - points_normalized[:, -1:])
        
    def _pca_projection(self, 
                       points: np.ndarray,
                       fit: bool = True) -> np.ndarray:
        """
        Project using PCA.
        
        Args:
            points: Input points
            fit: Whether to fit PCA
            
        Returns:
            Projected points
        """
        from sklearn.decomposition import PCA
        
        if fit or self._reducer is None:
            self._reducer = PCA(
                n_components=3,
                random_state=self.random_state
            )
            return self._reducer.fit_transform(points)
        else:
            return self._reducer.transform(points)
            
    def _umap_projection(self,
                        points: np.ndarray,
                        fit: bool = True) -> np.ndarray:
        """
        Project using UMAP.
        
        Args:
            points: Input points
            fit: Whether to fit UMAP
            
        Returns:
            Projected points
        """
        import umap
        
        if fit or self._reducer is None:
            self._reducer = umap.UMAP(
                n_components=3,
                random_state=self.random_state
            )
            return self._reducer.fit_transform(points)
        else:
            return self._reducer.transform(points)
            
    def _tsne_projection(self, points: np.ndarray) -> np.ndarray:
        """
        Project using t-SNE.
        
        Args:
            points: Input points
            
        Returns:
            Projected points
        """
        from sklearn.manifold import TSNE
        
        tsne = TSNE(
            n_components=3,
            random_state=self.random_state
        )
        return tsne.fit_transform(points)
        
    def project_edges(self,
                     points: np.ndarray,
                     edges: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """
        Project edges to 3D space.
        
        Args:
            points: Input points
            edges: List of (start, end) vertex indices
            
        Returns:
            List of edge dictionaries with projected coordinates
        """
        projected_points = self.project_points(points, fit=False)
        
        edge_data = []
        for start_idx, end_idx in edges:
            edge = {
                "start": projected_points[start_idx].tolist(),
                "end": projected_points[end_idx].tolist()
            }
            edge_data.append(edge)
            
        return edge_data
        
    def create_projection_sequence(self,
                                 points: np.ndarray,
                                 n_steps: int = 10) -> List[np.ndarray]:
        """
        Create smooth transition between dimensions.
        
        Args:
            points: Input points
            n_steps: Number of intermediate steps
            
        Returns:
            List of projected point arrays
        """
        sequence = []
        
        # Project initial points
        projected = self.project_points(points, fit=True)
        sequence.append(projected)
        
        # Create intermediate projections
        for i in range(1, n_steps):
            alpha = i / n_steps
            intermediate = self._interpolate_projection(points, alpha)
            sequence.append(intermediate)
            
        return sequence
        
    def _interpolate_projection(self,
                              points: np.ndarray,
                              alpha: float) -> np.ndarray:
        """
        Interpolate between original and projected points.
        
        Args:
            points: Input points
            alpha: Interpolation factor (0-1)
            
        Returns:
            Interpolated points
        """
        projected = self.project_points(points, fit=False)
        
        if self.projection_type == "stereographic":
            # Ensure points have same dimensionality by padding projected points
            if points.shape[1] > projected.shape[1]:
                pad_width = points.shape[1] - projected.shape[1]
                projected_padded = np.pad(
                    projected,
                    ((0, 0), (0, pad_width)),
                    mode='constant'
                )
            else:
                projected_padded = projected
            
            # Normalize points
            norm = np.linalg.norm(points, axis=1, keepdims=True)
            points_normalized = points / norm
            projected_norm = np.linalg.norm(projected_padded, axis=1, keepdims=True)
            projected_normalized = projected_padded / projected_norm
            
            # Calculate interpolation angles
            cos_omega = np.clip(
                np.sum(points_normalized * projected_normalized, axis=1),
                -1.0, 1.0
            )
            omega = np.arccos(cos_omega)
            
            sin_omega = np.sin(omega)
            
            # Handle small angles
            mask = sin_omega > 1e-6
            factors = np.zeros((len(points), 2))
            factors[mask, 0] = np.sin((1 - alpha) * omega[mask]) / sin_omega[mask]
            factors[mask, 1] = np.sin(alpha * omega[mask]) / sin_omega[mask]
            factors[~mask, 0] = 1 - alpha
            factors[~mask, 1] = alpha
            
            # Interpolate
            result = (
                factors[:, 0, np.newaxis] * points_normalized +
                factors[:, 1, np.newaxis] * projected_normalized
            )
            
            # Project back to 3D
            return result[:, :3]
        else:
            # Linear interpolation
            return (1 - alpha) * points[:, :3] + alpha * projected 