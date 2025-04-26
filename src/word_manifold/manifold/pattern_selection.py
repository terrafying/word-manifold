"""Pattern selection module for semantic manifold analysis."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum
import random

class PatternSource(Enum):
    """Sources of patterns for selection."""
    HERMETIC = "hermetic"
    DAOIST = "daoist"
    KABBALISTIC = "kabbalistic"
    I_CHING = "i_ching"
    SACRED_GEOMETRY = "sacred_geometry"

class PatternSelector:
    """Selects patterns based on various traditions when multiple valid solutions exist."""
    
    def __init__(self, source: PatternSource = PatternSource.HERMETIC):
        """
        Initialize the pattern selector.
        
        Args:
            source: Source of patterns to use
        """
        self.source = source
        self.patterns = self._load_patterns()
        
    def _load_patterns(self) -> Dict[str, List[float]]:
        """Load patterns based on the selected source."""
        if self.source == PatternSource.HERMETIC:
            return {
                "mentalism": [1.0, 0.0, 0.0],
                "correspondence": [0.0, 1.0, 0.0],
                "vibration": [0.0, 0.0, 1.0],
                "polarity": [0.5, 0.5, 0.0],
                "rhythm": [0.33, 0.33, 0.33],
                "causation": [0.25, 0.5, 0.25],
                "gender": [0.5, 0.0, 0.5]
            }
        elif self.source == PatternSource.DAOIST:
            return {
                "yin": [0.0, 1.0],
                "yang": [1.0, 0.0],
                "taiji": [0.5, 0.5],
                "wuxing": [0.2, 0.2, 0.2, 0.2, 0.2]
            }
        elif self.source == PatternSource.KABBALISTIC:
            return {
                "keter": [1.0, 0.0, 0.0, 0.0],
                "chokmah": [0.0, 1.0, 0.0, 0.0],
                "binah": [0.0, 0.0, 1.0, 0.0],
                "chesed": [0.0, 0.0, 0.0, 1.0]
            }
        elif self.source == PatternSource.I_CHING:
            return {
                "qian": [1.0, 1.0, 1.0],
                "kun": [0.0, 0.0, 0.0],
                "zhen": [1.0, 0.0, 0.0],
                "xun": [0.0, 1.0, 0.0]
            }
        else:  # SACRED_GEOMETRY
            return {
                "circle": [0.5, 0.5],
                "square": [0.0, 1.0, 0.0, 1.0],
                "triangle": [0.33, 0.33, 0.33],
                "pentagon": [0.2, 0.2, 0.2, 0.2, 0.2]
            }
            
    def select_pattern(
        self,
        valid_solutions: List[np.ndarray],
        context: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Select a pattern from valid solutions based on the chosen tradition.
        
        Args:
            valid_solutions: List of valid solution vectors
            context: Optional context for pattern selection
            
        Returns:
            Selected solution vector
        """
        if not valid_solutions:
            raise ValueError("No valid solutions provided")
            
        # If only one solution, return it
        if len(valid_solutions) == 1:
            return valid_solutions[0]
            
        # Get pattern weights based on context
        weights = self._get_pattern_weights(context)
        
        # Score each solution based on pattern similarity
        scores = []
        for solution in valid_solutions:
            score = self._calculate_pattern_score(solution, weights)
            scores.append(score)
            
        # Select solution with highest pattern score
        return valid_solutions[np.argmax(scores)]
        
    def _get_pattern_weights(self, context: Optional[Dict]) -> Dict[str, float]:
        """Get weights for different patterns based on context."""
        if not context:
            # Default to equal weights
            return {pattern: 1.0 for pattern in self.patterns}
            
        # Use context to adjust weights
        weights = {}
        for pattern in self.patterns:
            if pattern in context:
                weights[pattern] = context[pattern]
            else:
                weights[pattern] = 1.0
                
        return weights
        
    def _calculate_pattern_score(
        self,
        solution: np.ndarray,
        weights: Dict[str, float]
    ) -> float:
        """Calculate how well a solution matches the patterns."""
        score = 0.0
        for pattern_name, pattern in self.patterns.items():
            # Calculate similarity between solution and pattern
            similarity = self._calculate_similarity(solution, pattern)
            # Weight the similarity
            score += similarity * weights[pattern_name]
            
        return score
        
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate similarity between two vectors."""
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        return np.dot(vec1_norm, vec2_norm)
        
    def get_pattern_description(self, pattern_name: str) -> str:
        """Get description of a pattern."""
        descriptions = {
            PatternSource.HERMETIC: {
                "mentalism": "The principle that all is mind",
                "correspondence": "As above, so below",
                "vibration": "Everything is in motion",
                "polarity": "Everything has its opposite",
                "rhythm": "Everything flows in cycles",
                "causation": "Every cause has its effect",
                "gender": "Everything has masculine and feminine principles"
            },
            PatternSource.DAOIST: {
                "yin": "The receptive principle",
                "yang": "The active principle",
                "taiji": "The unity of opposites",
                "wuxing": "The five phases of transformation"
            },
            PatternSource.KABBALISTIC: {
                "keter": "The crown of creation",
                "chokmah": "The wisdom of understanding",
                "binah": "The understanding of wisdom",
                "chesed": "The loving kindness"
            },
            PatternSource.I_CHING: {
                "qian": "The creative force",
                "kun": "The receptive force",
                "zhen": "The arousing force",
                "xun": "The gentle force"
            },
            PatternSource.SACRED_GEOMETRY: {
                "circle": "The perfect unity",
                "square": "The foundation of order",
                "triangle": "The trinity of forces",
                "pentagon": "The harmony of five"
            }
        }
        
        return descriptions[self.source].get(pattern_name, "Unknown pattern") 