"""
Shape Engine for Geometric Visualizations

This module provides the core engine for generating geometric shape visualizations,
including shape fields, transformations, and semantic geometry. Now with support
for hyperdimensional projections using Three.js.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Shape:
    """Represents a geometric shape with properties."""
    type: str  # circle, square, triangle, etc.
    center: Tuple[float, float, float]  # Now supports 3D coordinates
    size: float
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 3D rotation
    color: str = 'blue'
    alpha: float = 0.7
    properties: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert shape to dictionary format for Three.js renderer."""
        return {
            'type': self.type,
            'center': self.center,
            'size': self.size,
            'rotation': self.rotation,
            'color': self.color,
            'alpha': self.alpha,
            'properties': self.properties or {}
        }

class ShapeField:
    """Represents a field of shapes with semantic properties."""
    
    def __init__(self, dimensions: int = 3):
        self.shapes: List[Shape] = []
        self.dimensions = dimensions
        self.width: float = 10.0
        self.height: float = 10.0
        self.depth: float = 10.0  # Added for 3D support
        self.properties: Dict[str, Any] = {}
    
    def add_shape(self, shape: Shape) -> None:
        """Add a shape to the field."""
        self.shapes.append(shape)
    
    def get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Get field bounds (xmin, xmax, ymin, ymax, zmin, zmax)."""
        if not self.shapes:
            return (0, self.width, 0, self.height, 0, self.depth)
        
        centers = np.array([s.center for s in self.shapes])
        sizes = np.array([s.size for s in self.shapes])
        
        xmin = np.min(centers[:, 0] - sizes/2)
        xmax = np.max(centers[:, 0] + sizes/2)
        ymin = np.min(centers[:, 1] - sizes/2)
        ymax = np.max(centers[:, 1] + sizes/2)
        zmin = np.min(centers[:, 2] - sizes/2)
        zmax = np.max(centers[:, 2] + sizes/2)
        
        return (xmin, xmax, ymin, ymax, zmin, zmax)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert field to dictionary format for Three.js renderer."""
        return {
            'shapes': [s.to_dict() for s in self.shapes],
            'dimensions': self.dimensions,
            'bounds': self.get_bounds(),
            'properties': self.properties
        }

class ShapeEngine:
    """Engine for generating geometric shape visualizations."""
    
    def __init__(self, dimensions: int = 3):
        self.dimensions = dimensions
        
        # Shape type mappings for semantic properties
        self.semantic_shapes = {
            'abstract': 'sphere',  # Updated for 3D
            'concrete': 'cube',
            'dynamic': 'cone',
            'static': 'cylinder'
        }
        
        # Color mappings for semantic properties
        self.semantic_colors = {
            'positive': '#4CAF50',  # Material Design colors
            'negative': '#F44336',
            'neutral': '#9E9E9E',
            'transformative': '#9C27B0'
        }
    
    def generate_shape_field(
        self,
        text: str,
        resolution: int = 50,
        complexity: float = 0.8
    ) -> ShapeField:
        """Generate a shape field based on text properties."""
        field = ShapeField(dimensions=self.dimensions)
        words = text.split()
        
        # Create base grid in 3D
        n_shapes = int(len(words) * complexity * 2)
        grid_size = int(np.ceil(np.cbrt(n_shapes)))  # Cube root for 3D grid
        
        for i in range(n_shapes):
            # Calculate 3D position
            x = (i % grid_size + 0.5) * field.width / grid_size
            y = ((i // grid_size) % grid_size + 0.5) * field.height / grid_size
            z = (i // (grid_size * grid_size) + 0.5) * field.depth / grid_size
            
            # Get word properties
            word = words[i % len(words)]
            word_len = len(word)
            
            # Determine shape properties based on word
            shape_type = self.semantic_shapes['abstract']  # Default
            for key, shape in self.semantic_shapes.items():
                if key in word.lower():
                    shape_type = shape
                    break
            
            # Determine color based on word
            color = self.semantic_colors['neutral']  # Default
            for key, col in self.semantic_colors.items():
                if key in word.lower():
                    color = col
                    break
            
            # Create shape with 3D properties
            shape = Shape(
                type=shape_type,
                center=(x, y, z),
                size=word_len / 5,  # Scale size with word length
                rotation=(
                    np.random.uniform(0, 2*np.pi),
                    np.random.uniform(0, 2*np.pi),
                    np.random.uniform(0, 2*np.pi)
                ),
                color=color,
                alpha=0.7,
                properties={'word': word}
            )
            
            field.add_shape(shape)
        
        return field
    
    def create_transformation_sequence(
        self,
        source_shape: ShapeField,
        target_shape: ShapeField,
        n_steps: int = 20,
        add_intermediates: bool = True
    ) -> List[ShapeField]:
        """Create a sequence of shape fields transforming from source to target."""
        sequence = []
        
        # Ensure equal number of shapes
        max_shapes = max(len(source_shape.shapes), len(target_shape.shapes))
        
        # Pad source shapes if needed
        while len(source_shape.shapes) < max_shapes:
            # Add invisible shapes
            shape = source_shape.shapes[0]
            new_shape = Shape(
                type=shape.type,
                center=shape.center,
                size=shape.size,
                rotation=shape.rotation,
                color=shape.color,
                alpha=0
            )
            source_shape.shapes.append(new_shape)
        
        # Pad target shapes if needed
        while len(target_shape.shapes) < max_shapes:
            shape = target_shape.shapes[0]
            new_shape = Shape(
                type=shape.type,
                center=shape.center,
                size=shape.size,
                rotation=shape.rotation,
                color=shape.color,
                alpha=0
            )
            target_shape.shapes.append(new_shape)
        
        # Create interpolated fields
        for step in range(n_steps):
            t = step / (n_steps - 1)
            field = ShapeField(dimensions=self.dimensions)
            
            for s1, s2 in zip(source_shape.shapes, target_shape.shapes):
                # Interpolate shape properties
                center = tuple(
                    s1.center[i] * (1-t) + s2.center[i] * t
                    for i in range(3)
                )
                size = s1.size * (1-t) + s2.size * t
                rotation = tuple(
                    s1.rotation[i] * (1-t) + s2.rotation[i] * t
                    for i in range(3)
                )
                
                # Handle shape type transition
                shape_type = s1.type if t < 0.5 else s2.type
                
                # Create interpolated shape
                shape = Shape(
                    type=shape_type,
                    center=center,
                    size=size,
                    rotation=rotation,
                    color=s1.color if t < 0.5 else s2.color,
                    alpha=max(s1.alpha, s2.alpha)
                )
                
                field.add_shape(shape)
            
            sequence.append(field)
            
            # Add intermediate fields if requested
            if add_intermediates and step < n_steps - 1:
                t_mid = (step + 0.5) / (n_steps - 1)
                mid_field = ShapeField(dimensions=self.dimensions)
                
                for s1, s2 in zip(source_shape.shapes, target_shape.shapes):
                    center = tuple(
                        s1.center[i] * (1-t_mid) + s2.center[i] * t_mid
                        for i in range(3)
                    )
                    size = s1.size * (1-t_mid) + s2.size * t_mid
                    rotation = tuple(
                        s1.rotation[i] * (1-t_mid) + s2.rotation[i] * t_mid
                        for i in range(3)
                    )
                    
                    shape = Shape(
                        type=s1.type if t_mid < 0.5 else s2.type,
                        center=center,
                        size=size,
                        rotation=rotation,
                        color=s1.color if t_mid < 0.5 else s2.color,
                        alpha=max(s1.alpha, s2.alpha)
                    )
                    
                    mid_field.add_shape(shape)
                
                sequence.append(mid_field)
        
        return sequence
    
    def analyze_transformation(
        self,
        source_shape: ShapeField,
        target_shape: ShapeField,
        transformations: List[ShapeField]
    ) -> Dict[str, Any]:
        """Analyze the transformation sequence and generate metrics."""
        metrics = {}
        
        # Calculate shape type changes
        type_changes = sum(
            1 for s1, s2 in zip(source_shape.shapes, target_shape.shapes)
            if s1.type != s2.type
        )
        metrics['type_changes'] = type_changes
        
        # Calculate average movement distance in 3D
        distances = []
        for s1, s2 in zip(source_shape.shapes, target_shape.shapes):
            dist = np.sqrt(sum(
                (s2.center[i] - s1.center[i])**2
                for i in range(3)
            ))
            distances.append(dist)
        
        metrics['avg_movement'] = np.mean(distances)
        metrics['max_movement'] = np.max(distances)
        
        # Calculate size changes
        size_changes = [
            abs(s2.size - s1.size)
            for s1, s2 in zip(source_shape.shapes, target_shape.shapes)
        ]
        metrics['avg_size_change'] = np.mean(size_changes)
        
        # Calculate rotation changes
        rotation_changes = [
            np.sqrt(sum(
                (s2.rotation[i] - s1.rotation[i])**2
                for i in range(3)
            ))
            for s1, s2 in zip(source_shape.shapes, target_shape.shapes)
        ]
        metrics['avg_rotation_change'] = np.mean(rotation_changes)
        
        # Calculate transformation complexity
        metrics['complexity'] = len(transformations)
        
        return metrics 