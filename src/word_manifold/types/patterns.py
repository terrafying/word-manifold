"""
Pattern Types

Core pattern types for ASCII visualization.
Simplified implementation focusing on essential functionality.
"""

from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Dict, Any

@dataclass
class Pattern:
    """Base pattern class with minimal required attributes."""
    data: np.ndarray  # 2D array of characters
    symbols: str      # Available symbols for the pattern
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    def to_string(self) -> str:
        """Convert pattern to string representation."""
        return '\n'.join(''.join(row) for row in self.data)
    
    @classmethod
    def create_empty(cls, width: int, height: int, symbols: str = ' .:-=+*#%@') -> 'Pattern':
        """Create an empty pattern."""
        return cls(
            data=np.full((height, width), ' ', dtype=str),
            symbols=symbols
        )

@dataclass
class ASCIIPattern(Pattern):
    """ASCII art pattern with additional metadata."""
    metadata: Dict[str, Any] = None
    frame_index: int = 0
    
    @classmethod
    def create(cls, width: int, height: int, symbols: str = ' .:-=+*#%@', metadata: Optional[Dict[str, Any]] = None) -> 'ASCIIPattern':
        """Create a new ASCII pattern."""
        return cls(
            data=np.full((height, width), ' ', dtype=str),
            symbols=symbols,
            metadata=metadata or {},
            frame_index=0
        )
    
    def add_frame(self, frame_data: np.ndarray) -> None:
        """Add a new frame to the pattern."""
        if frame_data.shape != self.data.shape:
            raise ValueError("Frame dimensions must match pattern dimensions")
        self.data = frame_data
        self.frame_index += 1

@dataclass
class Mandala(Pattern):
    """Circular pattern with radius and rotation."""
    radius: int
    rotation: float = 0.0
    
    @classmethod
    def create(cls, radius: int, symbols: str = ' .:-=+*#%@', rotation: float = 0.0) -> 'Mandala':
        """Create a new mandala pattern."""
        size = 2 * radius + 1
        return cls(
            data=np.full((size, size), ' ', dtype=str),
            symbols=symbols,
            radius=radius,
            rotation=rotation
        )

@dataclass
class Field(Pattern):
    """Field pattern with density control."""
    density: float = 0.7
    
    @classmethod
    def create(cls, width: int, height: int, symbols: str = '♠♡♢♣★☆⚝✧✦❈❉❊❋', density: float = 0.7) -> 'Field':
        """Create a new field pattern."""
        return cls(
            data=np.full((height, width), ' ', dtype=str),
            symbols=symbols,
            density=density
        ) 