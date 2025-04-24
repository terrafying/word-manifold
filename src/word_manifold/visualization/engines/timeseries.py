"""
Time Series Visualization Engine.

This module provides the core engine for generating and processing time series data
for visualization, supporting both embedding-based and I Ching-based temporal analysis.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from enum import Enum

from ...embeddings.word_embeddings import WordEmbeddings
from ..base import VisualizationEngine

class PatternType(Enum):
    """Types of temporal patterns that can be generated."""
    CYCLIC = "cyclic"           # Cyclic patterns based on trigonometric functions
    LINEAR = "linear"           # Linear trends with embedding-based slopes
    HARMONIC = "harmonic"       # Multiple harmonics with embedding-weighted amplitudes
    SPIRAL = "spiral"           # Spiral patterns showing cyclic evolution
    WAVE = "wave"              # Complex waveforms combining multiple patterns

class TimeSeriesEngine(VisualizationEngine):
    """Engine for generating and processing time series data."""
    
    def __init__(self, embeddings: WordEmbeddings):
        """Initialize the engine with word embeddings."""
        super().__init__()
        self.embeddings = embeddings
        
    def generate_time_points(
        self,
        timeframe: str,
        interval: str,
        end_time: Optional[datetime] = None
    ) -> Tuple[List[datetime], Dict[str, Any]]:
        """Generate time points and temporal metadata."""
        end_time = end_time or datetime.now()
        
        # Parse timeframe
        if timeframe.endswith('h'):
            start_time = end_time - timedelta(hours=int(timeframe[:-1]))
            step = timedelta(minutes=int(interval[:-1]) if interval.endswith('m') else int(interval[:-1])*60)
            scale = 'hourly'
        elif timeframe.endswith('d'):
            start_time = end_time - timedelta(days=int(timeframe[:-1]))
            step = timedelta(hours=int(interval[:-1]))
            scale = 'daily'
        elif timeframe.endswith('w'):
            start_time = end_time - timedelta(weeks=int(timeframe[:-1]))
            step = timedelta(days=int(interval[:-1]) if interval.endswith('d') else 1)
            scale = 'weekly'
        else:  # months
            start_time = end_time - timedelta(days=30*int(timeframe[:-1]))
            step = timedelta(days=int(interval[:-1]) if interval.endswith('d') else 1)
            scale = 'monthly'
            
        time_points = []
        current = start_time
        while current <= end_time:
            time_points.append(current)
            current += step
            
        metadata = {
            'start_time': start_time,
            'end_time': end_time,
            'interval': step,
            'scale': scale,
            'num_points': len(time_points)
        }
            
        return time_points, metadata
        
    def generate_patterns(
        self,
        terms: List[str],
        time_points: List[datetime],
        pattern_type: str = 'cyclic'
    ) -> Dict[str, Dict[str, Any]]:
        """Generate temporal patterns for terms based on their embeddings."""
        patterns = {}
        num_points = len(time_points)
        
        for term in terms:
            embedding = self.embeddings.get_embedding(term)
            if embedding is not None:
                pattern_data = {'values': None, 'metadata': {}}
                x = np.linspace(0, 1, num_points)
                
                if pattern_type == PatternType.CYCLIC.value:
                    # Enhanced cyclic pattern with phase shift
                    phase = np.arctan2(embedding[1], embedding[0])
                    pattern = np.sin(np.linspace(0, 4*np.pi, num_points) + phase) * embedding[0] + \
                             np.cos(np.linspace(0, 2*np.pi, num_points) + phase) * embedding[1]
                    pattern_data['metadata']['phase'] = phase
                    
                elif pattern_type == PatternType.LINEAR.value:
                    # Linear trend with confidence bounds
                    slope = np.mean(embedding[:2])
                    intercept = embedding[2]
                    pattern = slope * x + intercept
                    pattern_data['metadata'].update({
                        'slope': slope,
                        'intercept': intercept,
                        'confidence': abs(embedding[3]) if len(embedding) > 3 else 0.5
                    })
                    
                elif pattern_type == PatternType.HARMONIC.value:
                    # Multiple harmonics with embedding-weighted amplitudes
                    harmonics = []
                    for i, comp in enumerate(embedding[:4]):
                        harmonic = comp * np.sin(2*np.pi*(i+1)*x)
                        harmonics.append(harmonic)
                    pattern = sum(harmonics)
                    pattern_data['metadata']['harmonics'] = len(harmonics)
                    
                elif pattern_type == PatternType.SPIRAL.value:
                    # Spiral pattern showing cyclic evolution
                    radius = 0.5 + 0.5 * np.tanh(embedding[0])
                    frequency = 1 + abs(embedding[1])
                    t = np.linspace(0, 4*np.pi, num_points)
                    pattern = radius * t * np.sin(frequency * t)
                    pattern_data['metadata'].update({
                        'radius': radius,
                        'frequency': frequency
                    })
                    
                else:  # WAVE
                    # Complex waveform combining multiple patterns
                    components = []
                    weights = np.abs(embedding[:4]) / np.sum(np.abs(embedding[:4]))
                    
                    # Fast oscillation
                    components.append(weights[0] * np.sin(8*np.pi*x))
                    # Medium oscillation
                    components.append(weights[1] * np.sin(4*np.pi*x))
                    # Slow trend
                    components.append(weights[2] * x)
                    # Random fluctuation
                    noise = weights[3] * np.random.randn(num_points) * 0.1
                    
                    pattern = sum(components) + noise
                    pattern_data['metadata']['components'] = len(components)
                
                # Normalize pattern
                pattern = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern))
                pattern_data['values'] = pattern
                pattern_data['metadata'].update({
                    'mean': float(np.mean(pattern)),
                    'std': float(np.std(pattern)),
                    'trend': 'increasing' if pattern[-1] > pattern[0] else 'decreasing',
                    'volatility': float(np.std(np.diff(pattern)))
                })
                
                patterns[term] = pattern_data
                
        return patterns
        
    def process_data(
        self,
        terms: List[str],
        timeframe: str,
        interval: str,
        pattern_type: str = PatternType.CYCLIC.value
    ) -> Dict[str, Any]:
        """Process time series data for visualization."""
        time_points, temporal_metadata = self.generate_time_points(timeframe, interval)
        patterns = self.generate_patterns(terms, time_points, pattern_type)
        
        # Calculate cross-term correlations
        correlations = {}
        if len(terms) > 1:
            for i, term1 in enumerate(terms):
                for term2 in terms[i+1:]:
                    if term1 in patterns and term2 in patterns:
                        corr = np.corrcoef(
                            patterns[term1]['values'],
                            patterns[term2]['values']
                        )[0,1]
                        correlations[f"{term1}-{term2}"] = float(corr)
        
        return {
            'time_points': time_points,
            'patterns': patterns,
            'correlations': correlations,
            'metadata': {
                'timeframe': timeframe,
                'interval': interval,
                'pattern_type': pattern_type,
                'temporal': temporal_metadata
            }
        } 