"""
Distance type definitions for word manifold system.
"""

from enum import Enum, auto

class DistanceType(Enum):
    """Types of distance metrics used in the system."""
    EUCLIDEAN = auto()
    COSINE = auto()
    MANHATTAN = auto()
    HAMMING = auto()
    SEMANTIC = auto()  # For semantic-based distance calculations 