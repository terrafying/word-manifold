from typing import Dict, List, Optional, Tuple
import numpy as np
from z3 import *
import pandas as pd
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import networkx as nx

class SemanticTimeSeriesAnalyzer:
    """A class for analyzing semantic evolution over time."""
    
    def __init__(
        self,
        dimension: int = 768,
        window_size: int = 5,
        stride: int = 1
    ):
        """
        Initialize the SemanticTimeSeriesAnalyzer.
        
        Args:
            dimension: Dimension of the embedding space
            window_size: Size of sliding window for analysis
            stride: Step size for sliding window
        """
        self.dimension = dimension
        self.window_size = window_size
        self.stride = stride
        
    def analyze_passage(
        self,
        passage: str,
        embeddings: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Analyze semantic evolution in a passage.
        
        Args:
            passage: Text passage to analyze
            embeddings: Dictionary of word embeddings
            
        Returns:
            Dictionary containing analysis results
        """
        # Split passage into words
        words = passage.split()
        
        # Create time series of embeddings
        series = []
        for word in words:
            if word in embeddings:
                series.append(embeddings[word])
                
        if not series:
            return {"error": "No valid words found in passage"}
            
        # Convert to numpy array
        series = np.array(series)
        
        # Reduce dimensionality for analysis
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(series)
        
        # Analyze temporal patterns
        patterns = self._analyze_temporal_patterns(reduced)
        
        # Detect semantic shifts
        shifts = self._detect_semantic_shifts(reduced)
        
        # Build evolution graph
        graph = self._build_evolution_graph(words, reduced)
        
        return {
            "patterns": patterns,
            "shifts": shifts,
            "graph_metrics": self._analyze_graph(graph),
            "reduced_dimensions": reduced.tolist()
        }
        
    def _analyze_temporal_patterns(self, series: np.ndarray) -> Dict:
        """Analyze temporal patterns in the reduced series."""
        # Calculate velocity (rate of change)
        velocity = np.diff(series, axis=0)
        
        # Find peaks in magnitude of change
        magnitude = np.linalg.norm(velocity, axis=1)
        peaks, _ = find_peaks(magnitude, height=np.mean(magnitude))
        
        # Calculate periodicity
        fft = np.fft.fft(magnitude)
        freqs = np.fft.fftfreq(len(magnitude))
        main_freq = freqs[np.argmax(np.abs(fft[1:len(fft)//2])) + 1]
        
        return {
            "velocity_stats": {
                "mean": float(np.mean(velocity)),
                "std": float(np.std(velocity)),
                "max": float(np.max(velocity))
            },
            "peaks": peaks.tolist(),
            "periodicity": float(1/main_freq) if main_freq != 0 else float('inf')
        }
        
    def _detect_semantic_shifts(self, series: np.ndarray) -> List[Dict]:
        """Detect significant semantic shifts in the series."""
        shifts = []
        
        # Calculate cumulative change
        cumulative = np.cumsum(np.linalg.norm(np.diff(series, axis=0), axis=1))
        
        # Find points where change exceeds threshold
        threshold = np.mean(cumulative) + np.std(cumulative)
        shift_points = np.where(cumulative > threshold)[0]
        
        for point in shift_points:
            shifts.append({
                "position": int(point),
                "magnitude": float(cumulative[point]),
                "direction": self._get_shift_direction(series, point)
            })
            
        return shifts
        
    def _get_shift_direction(self, series: np.ndarray, point: int) -> str:
        """Determine the direction of semantic shift."""
        if point == 0:
            return "start"
            
        # Calculate direction vector
        direction = series[point] - series[point-1]
        
        # Determine primary direction
        if abs(direction[0]) > abs(direction[1]):
            return "horizontal"
        else:
            return "vertical"
            
    def _build_evolution_graph(
        self,
        words: List[str],
        series: np.ndarray
    ) -> nx.DiGraph:
        """Build a directed graph representing semantic evolution."""
        G = nx.DiGraph()
        
        # Add nodes
        for i, word in enumerate(words):
            G.add_node(word, position=series[i].tolist())
            
        # Add edges based on temporal proximity and similarity
        for i in range(len(words)-1):
            word1 = words[i]
            word2 = words[i+1]
            
            # Calculate similarity
            sim = np.dot(series[i], series[i+1]) / (
                np.linalg.norm(series[i]) * np.linalg.norm(series[i+1])
            )
            
            # Add edge if similarity exceeds threshold
            if sim > 0.5:  # Adjust threshold as needed
                G.add_edge(word1, word2, weight=float(sim))
                
        return G
        
    def _analyze_graph(self, graph: nx.DiGraph) -> Dict:
        """Analyze the evolution graph."""
        return {
            "density": nx.density(graph),
            "average_clustering": nx.average_clustering(graph),
            "average_shortest_path": nx.average_shortest_path_length(graph),
            "diameter": nx.diameter(graph),
            "num_communities": len(list(nx.community.greedy_modularity_communities(graph)))
        } 