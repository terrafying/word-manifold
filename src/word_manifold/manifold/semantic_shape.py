"""
Semantic Shape Module for Word Manifold.

This module handles the representation and transformation of semantic shapes
in the manifold, representing the structural and emotional patterns of language.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.interpolate import splprep, splev
import logging
from ..embeddings.phrase_embeddings import PhraseEmbedding
from scipy.spatial import ConvexHull
# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ShapePoint:
    """A point in the semantic shape with its associated properties."""
    position: np.ndarray
    intensity: float  # Emotional/semantic intensity
    direction: np.ndarray  # Flow direction
    properties: Dict  # Additional shape properties

class SemanticShape:
    """
    A class representing the shape of meaning in semantic space.
    
    This captures both the geometric form and the dynamic properties
    of a semantic expression (phrase, sentence, or text chunk).
    """
    
    def __init__(
        self,
        phrase_embedding: PhraseEmbedding,
        n_control_points: int = 10
    ):
        """
        Initialize a semantic shape from a phrase embedding.
        
        Args:
            phrase_embedding: The embedded phrase/sentence
            n_control_points: Number of control points for the shape
        """
        self.text = phrase_embedding.text
        self.embedding = phrase_embedding.embedding
        self.shape_params = phrase_embedding.shape_params
        self.n_control_points = n_control_points
        
        # Generate the shape's control points
        self.control_points = self._generate_control_points()
        
        # Create the shape's boundary
        self.boundary = self._create_boundary()
        
        # Calculate shape properties
        self.properties = self._calculate_properties()
    
    def _generate_control_points(self) -> List[ShapePoint]:
        """Generate control points that define the shape's form."""
        points = []
        
        # Use shape parameters to generate points
        tree_depth = self.shape_params['tree_depth']
        complexity = self.shape_params['syntax_complexity']
        emotional_valence = self.shape_params['emotional_valence']
        syllable_pattern = self.shape_params['syllable_pattern']
        
        # Create a base circle of points
        angles = np.linspace(0, 2*np.pi, self.n_control_points)
        
        for i, angle in enumerate(angles):
            # Base position on unit circle
            x = np.cos(angle)
            y = np.sin(angle)
            
            # Modify radius based on syllable pattern
            radius = 1.0 + 0.2 * np.sin(syllable_pattern[i % len(syllable_pattern)])
            
            # Create position vector
            position = np.array([x * radius, y * radius])
            
            # Calculate intensity based on tree depth and position
            intensity = (1 + np.sin(angle * tree_depth)) * complexity
            
            # Calculate direction based on emotional valence
            direction = np.array([
                np.cos(angle + emotional_valence),
                np.sin(angle + emotional_valence)
            ])
            
            # Create point properties
            properties = {
                'angle': angle,
                'radius': radius,
                'local_complexity': complexity * (1 + 0.5 * np.sin(angle * 3))
            }
            
            points.append(ShapePoint(position, intensity, direction, properties))
        
        return points
    
    def _create_boundary(self) -> np.ndarray:
        """Create a smooth boundary for the shape using spline interpolation."""
        points = np.array([p.position for p in self.control_points])
        
        # Create a closed curve using periodic spline
        tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=1)
        
        # Generate a finer sampling of points
        u_new = np.linspace(0, 1, 100)
        boundary = np.array(splev(u_new, tck)).T
        
        return boundary
    
    def _calculate_properties(self) -> Dict:
        """Calculate geometric and semantic properties of the shape."""
        points = np.array([p.position for p in self.control_points])
        
        # Calculate basic geometric properties
        hull = ConvexHull(points)
        centroid = np.mean(points, axis=0)
        
        # Calculate semantic properties
        flow = np.mean([p.direction for p in self.control_points], axis=0)
        avg_intensity = np.mean([p.intensity for p in self.control_points])
        
        return {
            'area': hull.area,
            'perimeter': hull.area,  # This is actually the convex hull perimeter
            'centroid': centroid,
            'flow_direction': flow,
            'average_intensity': avg_intensity,
            'complexity': self.shape_params['syntax_complexity'],
            'emotional_valence': self.shape_params['emotional_valence']
        }
    
    def interpolate_with(
        self,
        other: 'SemanticShape',
        t: float
    ) -> 'SemanticShape':
        """
        Interpolate between this shape and another.
        
        Args:
            other: The target shape to interpolate towards
            t: Interpolation parameter (0 to 1)
            
        Returns:
            A new SemanticShape representing the interpolated state
        """
        # Interpolate embeddings
        embedding = (1 - t) * self.embedding + t * other.embedding
        
        # Interpolate shape parameters
        shape_params = {}
        for key in self.shape_params:
            if isinstance(self.shape_params[key], (int, float)):
                shape_params[key] = (1 - t) * self.shape_params[key] + t * other.shape_params[key]
            else:
                # For non-numeric parameters, switch at t=0.5
                shape_params[key] = self.shape_params[key] if t < 0.5 else other.shape_params[key]
        
        # Create interpolated phrase embedding
        interpolated_text = f"Interpolation({self.text}, {other.text}, {t:.2f})"
        interpolated_embedding = PhraseEmbedding(
            text=interpolated_text,
            embedding=embedding,
            shape_params=shape_params
        )
        
        return SemanticShape(interpolated_embedding, self.n_control_points)
    
    def get_shape_at_time(self, t: float) -> np.ndarray:
        """
        Get the shape boundary at a specific time point,
        allowing for dynamic shape transformations.
        
        Args:
            t: Time parameter (0 to 1)
            
        Returns:
            Array of boundary points
        """
        # Create time-varying perturbation
        frequencies = np.array([1, 2, 3, 5, 8])  # Fibonacci frequencies
        amplitudes = 0.1 / frequencies
        
        # Apply perturbation to boundary points
        perturbed = self.boundary.copy()
        for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
            angle = 2 * np.pi * freq * t
            perturbed += amp * np.array([
                np.cos(angle + self.boundary[:, 0]),
                np.sin(angle + self.boundary[:, 1])
            ]).T
        
        return perturbed
    
    def get_flow_field(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate the semantic flow field at given points.
        
        Args:
            points: Array of points to calculate flow at
            
        Returns:
            Array of flow vectors
        """
        flow = np.zeros_like(points)
        
        for control_point in self.control_points:
            # Calculate influence of each control point
            diff = points - control_point.position[None, :]
            dist = np.linalg.norm(diff, axis=1)[:, None]
            
            # Avoid division by zero
            dist = np.maximum(dist, 1e-6)
            
            # Add weighted contribution
            weight = control_point.intensity / (dist ** 2)
            flow += weight * control_point.direction[None, :]
        
        # Normalize flow vectors
        norms = np.linalg.norm(flow, axis=1)[:, None]
        norms = np.maximum(norms, 1e-6)
        flow /= norms
        
        return flow 