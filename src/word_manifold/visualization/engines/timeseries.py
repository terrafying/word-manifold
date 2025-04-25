"""
Time Series Visualization Engine.

This module provides the core engine for generating and processing time series data
for visualization, supporting both embedding-based and I Ching-based temporal analysis.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from enum import Enum
import pandas as pd
from scipy import signal
import logging

from ...embeddings.word_embeddings import WordEmbeddings
from ..base import VisualizationEngine

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of temporal patterns that can be generated."""
    CYCLIC = "cyclic"
    LINEAR = "linear"
    HARMONIC = "harmonic"
    SPIRAL = "spiral"
    WAVE = "wave"

class TimeSeriesEngine(VisualizationEngine):
    """Engine for generating time series patterns and analysis."""
    
    def __init__(
        self,
        word_embeddings: Optional[WordEmbeddings] = None,
        pattern_type: str = "cyclic",
        timeframe: str = "1d",
        interval: str = "1h"
    ):
        """
        Initialize the time series engine.
        
        Args:
            word_embeddings: Optional word embeddings for semantic analysis
            pattern_type: Type of pattern to generate (default: cyclic)
            timeframe: Time range to analyze (e.g. 1h, 1d, 1w)
            interval: Sampling interval (e.g. 1m, 5m, 1h)
        """
        super().__init__()
        self.word_embeddings = word_embeddings
        self.pattern_type = PatternType(pattern_type)
        self.timeframe = timeframe
        self.interval = interval
        self._data = None
        
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data to generate time series visualization data.
        
        Args:
            data: Dictionary containing:
                - terms: List of terms to analyze
                - embeddings: List of embeddings (optional)
                - timeframe: Time range (optional)
                - interval: Sampling interval (optional)
                - pattern_type: Type of pattern (optional)
                - hexagram_data: I Ching data (optional)
                
        Returns:
            Dictionary containing processed data including:
                - time_points: Array of timestamps
                - patterns: Dictionary mapping terms to pattern values
                - correlations: Matrix of pattern correlations
                - metadata: Dictionary of temporal metadata
        """
        # Update parameters if provided
        if 'timeframe' in data:
            self.timeframe = data['timeframe']
        if 'interval' in data:
            self.interval = data['interval']
        if 'pattern_type' in data:
            self.pattern_type = PatternType(data['pattern_type'])
            
        # Generate time points
        time_points, metadata = self.generate_time_points()
        
        # Get terms and embeddings
        terms = data.get('terms', [])
        embeddings = data.get('embeddings', [])
        if not embeddings and self.word_embeddings:
            embeddings = [self.word_embeddings.get_embedding(t) for t in terms]
        
        # Generate patterns
        patterns = self.generate_patterns(
            terms=terms,
            embeddings=embeddings,
            time_points=time_points,
            hexagram_data=data.get('hexagram_data')
        )
        
        # Calculate correlations
        correlations = np.corrcoef([p for p in patterns.values()])
        
        # Store processed data
        self._data = {
            'time_points': time_points.tolist(),
            'patterns': {t: p.tolist() for t, p in patterns.items()},
            'correlations': correlations.tolist(),
            'metadata': {
                'temporal': metadata,
                'pattern_type': self.pattern_type.value,
                'terms': terms
            }
        }
        
        return self._data
        
    def generate_time_points(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate time points based on timeframe and interval.
        
        Returns:
            Tuple containing:
                - Array of timestamps
                - Dictionary of temporal metadata
        """
        # Parse timeframe
        unit = self.timeframe[-1]
        value = int(self.timeframe[:-1])
        
        # Convert to timedelta
        if unit == 'h':
            delta = timedelta(hours=value)
            scale = 'hourly'
        elif unit == 'd':
            delta = timedelta(days=value)
            scale = 'daily'
        elif unit == 'w':
            delta = timedelta(weeks=value)
            scale = 'weekly'
        else:  # Default to daily
            delta = timedelta(days=value)
            scale = 'daily'
            
        # Generate time points
        end_time = datetime.now()
        start_time = end_time - delta
        
        time_points = pd.date_range(
            start=start_time,
            end=end_time,
            freq=self.interval
        )
        
        metadata = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'interval': self.interval,
            'scale': scale,
            'num_points': len(time_points)
        }
        
        return np.array(time_points), metadata
        
    def generate_patterns(
        self,
        terms: List[str],
        embeddings: List[np.ndarray],
        time_points: np.ndarray,
        hexagram_data: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate temporal patterns for terms.
        
        Args:
            terms: List of terms
            embeddings: List of term embeddings
            time_points: Array of timestamps
            hexagram_data: Optional I Ching data
            
        Returns:
            Dictionary mapping terms to pattern values
        """
        patterns = {}
        num_points = len(time_points)
        
        for i, (term, embedding) in enumerate(zip(terms, embeddings)):
            # Base pattern based on type
            if self.pattern_type == PatternType.CYCLIC:
                # Cyclic pattern with phase shift based on embedding
                phase = 2 * np.pi * (i / len(terms))
                pattern = np.sin(np.linspace(0, 4*np.pi, num_points) + phase)
                
            elif self.pattern_type == PatternType.LINEAR:
                # Linear trend with slope based on embedding
                slope = 0.5 + np.mean(embedding) * 0.5
                pattern = np.linspace(0, slope * num_points, num_points)
                
            elif self.pattern_type == PatternType.HARMONIC:
                # Harmonic pattern with frequencies from embedding
                freqs = np.abs(embedding[:3])  # Use first 3 components
                pattern = sum(
                    amp * np.sin(freq * np.linspace(0, 2*np.pi, num_points))
                    for amp, freq in zip([0.5, 0.3, 0.2], freqs)
                )
                
            elif self.pattern_type == PatternType.SPIRAL:
                # Spiral pattern with radius from embedding
                t = np.linspace(0, 8*np.pi, num_points)
                radius = 0.5 + np.mean(embedding) * 0.5
                pattern = radius * t * np.cos(t)
                
            else:  # WAVE
                # Complex waveform
                t = np.linspace(0, 4*np.pi, num_points)
                pattern = np.sin(t) + 0.5 * np.sin(2*t + np.pi/4)
            
            # Apply hexagram influence if available
            if hexagram_data:
                hexagram = hexagram_data.get('hexagram')
                if hexagram:
                    # Modify pattern based on hexagram number
                    modifier = hexagram.number / 64.0
                    pattern = pattern * (1 + modifier)
            
            # Normalize pattern
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
            patterns[term] = pattern
            
        return patterns
        
    def get_data(self) -> Optional[Dict[str, Any]]:
        """Get the last processed data."""
        return self._data
        
    def clear_data(self) -> None:
        """Clear the stored data."""
        self._data = None 