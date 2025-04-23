"""
Shared types and enums for the word manifold system.
"""

from enum import Enum, auto

class CellType(Enum):
    """Types of cells with occult correspondences."""
    ELEMENTAL = auto()   # Corresponds to the four elements
    PLANETARY = auto()   # Corresponds to planetary influences
    ZODIACAL = auto()    # Corresponds to zodiac signs
    TAROT = auto()       # Corresponds to tarot archetypes
    SEPHIROTIC = auto()  # Corresponds to Kabbalistic sephiroth
    OTHER = auto()       # Default/unclassified

class DistanceType(Enum):
    """Types of distance metrics for cell relationships."""
    EUCLIDEAN = auto()      # Standard Euclidean distance
    COSINE = auto()         # Cosine distance (semantic similarity)
    NUMEROLOGICAL = auto()  # Distance weighted by numerological values
    HYBRID = auto()         # Combination of semantic and numerological 