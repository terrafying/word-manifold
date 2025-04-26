"""Core types for the word manifold package."""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class CellType(Enum):
    """Types of cells in the semantic manifold."""
    ACTIVE = "active"  # Currently evolving
    STABLE = "stable"  # Reached equilibrium
    BOUNDARY = "boundary"  # Between regions
    TRANSITION = "transition"  # Undergoing change
    ANCHOR = "anchor"  # Fixed reference point

class ManifoldState:
    """Represents the state of the semantic manifold."""
    
    def __init__(
        self,
        dimension: int = 768,
        cell_types: Optional[Dict[str, CellType]] = None
    ):
        """
        Initialize the manifold state.
        
        Args:
            dimension: Dimension of the embedding space
            cell_types: Dictionary mapping words to cell types
        """
        self.dimension = dimension
        self.cell_types = cell_types or {}
        
    def update_cell_type(self, word: str, cell_type: CellType):
        """Update the type of a cell."""
        self.cell_types[word] = cell_type
        
    def get_cell_type(self, word: str) -> Optional[CellType]:
        """Get the type of a cell."""
        return self.cell_types.get(word)
        
    def get_words_by_type(self, cell_type: CellType) -> List[str]:
        """Get all words of a specific cell type."""
        return [
            word for word, type_ in self.cell_types.items()
            if type_ == cell_type
        ]

class DistanceType(Enum):
    """Types of distance metrics for cell relationships."""
    EUCLIDEAN = auto()      # Standard Euclidean distance
    COSINE = auto()         # Cosine distance (semantic similarity)
    NUMEROLOGICAL = auto()  # Distance weighted by numerological values
    HYBRID = auto()         # Combination of semantic and numerological 