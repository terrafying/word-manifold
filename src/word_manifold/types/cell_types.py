"""
Cell Types

Basic cell type classification for vector manifold.
"""

from enum import Enum, auto

class CellType(Enum):
    """Basic cell types for vector manifold classification."""
    STANDARD = auto()  # Standard cell type
    BOUNDARY = auto()  # Cell on manifold boundary
    CORE = auto()      # Core/central cell
    BRIDGE = auto()    # Cell connecting different regions
    OTHER = auto()     # Unclassified/other type 