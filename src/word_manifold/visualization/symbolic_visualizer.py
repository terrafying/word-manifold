"""
Symbolic Visualizer Module

This module implements a text-based visualization system inspired by:
1. Dwarf Fortress's ASCII art representation of complex systems
2. McKenna's descriptions of machine elf communication - self-transforming, fractal language
3. Sacred geometry and numerological patterns

The visualizer creates living ASCII mandalas that transform based on semantic meaning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Union, Any
import logging
from dataclasses import dataclass
from ..embeddings.word_embeddings import WordEmbeddings

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SymbolicPattern:
    """A pattern of symbols that can transform."""
    base_symbols: str  # Core symbols that make up the pattern
    transformations: List[str]  # Sequence of transformation states
    meaning: str  # Semantic meaning of the pattern
    energy: float  # Current energy level (affects transformation rate)
    resonance: Set[str]  # Terms that resonate with this pattern

    def __repr__(self) -> str:
        """String representation of the pattern."""
        return f"SymbolicPattern(meaning='{self.meaning}', energy={self.energy:.2f}, n_transforms={len(self.transformations)})"

class SymbolicVisualizer:
    """Creates living ASCII mandalas from semantic spaces."""
    
    # Symbol sets for different semantic qualities
    ABSTRACT_SYMBOLS: str = "◇○□△▽⬡⬢⬣"  # Abstract concepts
    ORGANIC_SYMBOLS: str = "~≈≋∿☘❀❁❃"    # Natural/flowing concepts
    TECH_SYMBOLS: str = "⌘⌥⌦⌬⌸⌹⍋⍚"       # Technological concepts
    SACRED_SYMBOLS: str = "☯☮✴✵✶✷"      # Spiritual concepts
    EMOTIONAL_SYMBOLS: str = "♡♢♤♧♪♫"    # Emotional concepts
    
    def __init__(
        self, 
        word_embeddings: WordEmbeddings,
        width: int = 80,
        height: int = 40,
        frame_rate: float = 1.0
    ) -> None:
        """
        Initialize the visualizer.
        
        Args:
            word_embeddings: WordEmbeddings instance for semantic analysis
            width: Width of the visualization field
            height: Height of the visualization field
            frame_rate: Frames per second for animations
        """
        self.word_embeddings = word_embeddings
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.field = np.full((height, width), ' ', dtype=str)
        self.patterns: Dict[str, SymbolicPattern] = {}  # Active symbolic patterns
        
    def _select_base_symbols(self, term: str) -> str:
        """
        Select appropriate symbols based on term semantics.
        
        Args:
            term: The term to analyze
            
        Returns:
            String of symbols appropriate for the term's semantic meaning
        """
        embedding = self.word_embeddings.get_embedding(term)
        if embedding is None:
            return self.ABSTRACT_SYMBOLS
            
        # Calculate similarities to different concept types
        abstract_sim = self.word_embeddings.find_similar_terms("abstract", embedding)[0][1]
        nature_sim = self.word_embeddings.find_similar_terms("nature", embedding)[0][1]
        tech_sim = self.word_embeddings.find_similar_terms("technology", embedding)[0][1]
        spiritual_sim = self.word_embeddings.find_similar_terms("spiritual", embedding)[0][1]
        emotional_sim = self.word_embeddings.find_similar_terms("emotion", embedding)[0][1]
        
        # Return symbol set with highest similarity
        sims = [abstract_sim, nature_sim, tech_sim, spiritual_sim, emotional_sim]
        symbol_sets = [self.ABSTRACT_SYMBOLS, self.ORGANIC_SYMBOLS, 
                      self.TECH_SYMBOLS, self.SACRED_SYMBOLS, self.EMOTIONAL_SYMBOLS]
        return symbol_sets[np.argmax(sims)]
    
    def create_pattern(self, term: str) -> SymbolicPattern:
        """
        Create a new symbolic pattern for a term.
        
        Args:
            term: The term to create a pattern for
            
        Returns:
            SymbolicPattern instance representing the term
        """
        base_symbols = self._select_base_symbols(term)
        
        # Create transformations based on semantic neighbors
        similar_terms = self.word_embeddings.find_similar_terms(term, k=5)
        transformations = []
        for _, sim_term in similar_terms:
            symbols = self._select_base_symbols(sim_term)
            transformations.append(symbols)
            
        return SymbolicPattern(
            base_symbols=base_symbols,
            transformations=transformations,
            meaning=term,
            energy=1.0,
            resonance=set(t for t, _ in similar_terms)
        )
    
    def _generate_mandala(
        self, 
        pattern: SymbolicPattern, 
        center: Tuple[int, int], 
        radius: int
    ) -> np.ndarray:
        """
        Generate a mandala pattern centered at the given point.
        
        Args:
            pattern: The symbolic pattern to visualize
            center: (y, x) coordinates of the mandala center
            radius: Radius of the mandala in characters
            
        Returns:
            2D numpy array containing the mandala pattern
        """
        y, x = center
        mandala = np.full((2*radius+1, 2*radius+1), ' ', dtype=str)
        
        # Create concentric rings of symbols
        for r in range(radius):
            n_points = max(1, int(8 * r))  # Number of points in this ring
            for i in range(n_points):
                theta = 2 * np.pi * i / n_points
                py = int(y + r * np.sin(theta))
                px = int(x + r * np.cos(theta))
                if 0 <= py < mandala.shape[0] and 0 <= px < mandala.shape[1]:
                    symbol_idx = (r + i) % len(pattern.base_symbols)
                    mandala[py, px] = pattern.base_symbols[symbol_idx]
        
        return mandala
    
    def _apply_transformation(
        self, 
        pattern: SymbolicPattern, 
        mandala: np.ndarray, 
        phase: float
    ) -> np.ndarray:
        """
        Apply transformation to the mandala based on phase.
        
        Args:
            pattern: The pattern being transformed
            mandala: The current mandala state
            phase: Transformation phase (0.0 to 1.0)
            
        Returns:
            Transformed mandala array
        """
        transformed = mandala.copy()
        
        # Interpolate between transformation states
        t_idx = int(phase * len(pattern.transformations))
        if t_idx < len(pattern.transformations):
            new_symbols = pattern.transformations[t_idx]
            for i in range(transformed.shape[0]):
                for j in range(transformed.shape[1]):
                    if transformed[i, j] != ' ':
                        idx = (i + j) % len(new_symbols)
                        transformed[i, j] = new_symbols[idx]
        
        return transformed
    
    def visualize_term(self, term: str) -> str:
        """
        Create a static visualization for a term.
        
        Args:
            term: The term to visualize
            
        Returns:
            Multi-line string containing the ASCII visualization
        """
        pattern = self.create_pattern(term)
        center = (self.height // 2, self.width // 2)
        radius = min(self.height, self.width) // 4
        
        mandala = self._generate_mandala(pattern, center, radius)
        
        # Convert to string
        return '\n'.join(''.join(row) for row in mandala)
    
    def visualize_transformation(
        self, 
        term1: str, 
        term2: str, 
        steps: int = 10
    ) -> List[str]:
        """
        Visualize transformation between two terms.
        
        Args:
            term1: Starting term
            term2: Ending term
            steps: Number of transformation steps
            
        Returns:
            List of strings, each representing one frame of the transformation
        """
        pattern1 = self.create_pattern(term1)
        pattern2 = self.create_pattern(term2)
        center = (self.height // 2, self.width // 2)
        radius = min(self.height, self.width) // 4
        
        frames = []
        for step in range(steps):
            phase = step / (steps - 1)
            # Interpolate between patterns
            base_mandala = self._generate_mandala(pattern1, center, radius)
            transformed = self._apply_transformation(pattern2, base_mandala, phase)
            frames.append('\n'.join(''.join(row) for row in transformed))
        
        return frames
    
    def create_semantic_field(self, terms: List[str]) -> str:
        """
        Create a field of interacting patterns for multiple terms.
        
        Args:
            terms: List of terms to visualize in the field
            
        Returns:
            Multi-line string containing the ASCII visualization of the field
        
        Raises:
            ValueError: If embeddings cannot be obtained for any term
        """
        self.field.fill(' ')
        patterns = [self.create_pattern(term) for term in terms]
        
        # Position patterns based on semantic similarity
        embeddings = [self.word_embeddings.get_embedding(term) for term in terms]
        if any(e is None for e in embeddings):
            raise ValueError("Could not get embeddings for all terms")
            
        # Use PCA to position terms in 2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        positions = pca.fit_transform(embeddings)
        
        # Scale positions to field dimensions
        positions = (positions - positions.min(axis=0)) / (positions.max(axis=0) - positions.min(axis=0))
        positions[:, 0] *= (self.width - 20)
        positions[:, 1] *= (self.height - 10)
        positions = positions.astype(int)
        
        # Generate and combine mandalas
        for (y, x), pattern in zip(positions, patterns):
            radius = 5  # Smaller radius for combined field
            mandala = self._generate_mandala(pattern, (y+5, x+10), radius)
            
            # Add to field with bounds checking
            y_start = max(0, y)
            y_end = min(self.height, y + mandala.shape[0])
            x_start = max(0, x)
            x_end = min(self.width, x + mandala.shape[1])
            
            self.field[y_start:y_end, x_start:x_end] = mandala[
                :y_end-y_start, :x_end-x_start]
        
        return '\n'.join(''.join(row) for row in self.field) 