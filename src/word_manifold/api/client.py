"""
API Client for Word Manifold.

This module provides a unified client for interacting with the Word Manifold API,
handling both local and remote operations consistently.
"""

import requests
from typing import Dict, List, Optional, Any, Union
import numpy as np
import logging
from pathlib import Path
import json
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class WordManifoldClient:
    """Client for interacting with Word Manifold API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        timeout: int = 30,
        verify_ssl: bool = True
    ):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL for API endpoints
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        files: Optional[Dict] = None
    ) -> Dict:
        """Make HTTP request to API endpoint."""
        url = urljoin(self.base_url, endpoint)
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                files=files,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
            
    def get_embedding(
        self,
        text: str,
        model: str = "all-MiniLM-L6-v2",
        normalize: bool = True
    ) -> np.ndarray:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            model: Model name to use
            normalize: Whether to normalize the embedding
            
        Returns:
            Embedding vector as numpy array
        """
        response = self._make_request(
            "POST",
            "/api/v1/embeddings/embed",
            {
                "text": text,
                "model_name": model,
                "normalize": normalize
            }
        )
        return np.array(response["embedding"])
        
    def get_embeddings(
        self,
        texts: List[str],
        model: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            model: Model name to use
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            Array of embedding vectors
        """
        response = self._make_request(
            "POST",
            "/api/v1/embeddings/embed_batch",
            {
                "texts": texts,
                "model_name": model,
                "batch_size": batch_size,
                "normalize": normalize
            }
        )
        return np.array(response["embeddings"])
        
    def get_similarity(
        self,
        text1: str,
        text2: str,
        model: str = "all-MiniLM-L6-v2",
        metric: str = "cosine"
    ) -> float:
        """
        Get similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            model: Model name to use
            metric: Similarity metric (cosine or euclidean)
            
        Returns:
            Similarity score
        """
        response = self._make_request(
            "POST",
            "/api/v1/embeddings/similarity",
            {
                "text1": text1,
                "text2": text2,
                "model_name": model,
                "metric": metric
            }
        )
        return float(response["similarity"])
        
    def create_timeseries(
        self,
        terms: List[str],
        timeframe: str = "1d",
        interval: str = "1h",
        pattern_type: str = "cyclic",
        interactive: bool = False,
        show_values: bool = True,
        show_connections: bool = True,
        hexagram_data: Optional[Dict] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Create time series visualization.
        
        Args:
            terms: Terms to analyze
            timeframe: Time range to analyze
            interval: Sampling interval
            pattern_type: Type of pattern to generate
            interactive: Whether to create interactive visualization
            show_values: Whether to show numeric values
            show_connections: Whether to show connections
            hexagram_data: Optional I Ching data
            output_path: Optional path to save visualization
            
        Returns:
            Dictionary containing visualization data and metadata
        """
        response = self._make_request(
            "POST",
            "/api/v1/visualizations/timeseries",
            {
                "terms": terms,
                "timeframe": timeframe,
                "interval": interval,
                "pattern_type": pattern_type,
                "interactive": interactive,
                "show_values": show_values,
                "show_connections": show_connections,
                "hexagram_data": hexagram_data
            }
        )
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            if response["format"] == "html":
                output_path.write_text(response["visualization"])
            else:
                import base64
                img_data = base64.b64decode(response["visualization"])
                output_path.write_bytes(img_data)
                
            # Save insights if available
            if response.get("insights"):
                insights_path = output_path.parent / "insights.json"
                with open(insights_path, "w") as f:
                    json.dump(response["insights"], f, indent=2)
                    
        return response
        
    def create_magic_structure(
        self,
        dimension: int = 2,
        size: int = 3,
        terms: Optional[List[str]] = None,
        interactive: bool = False,
        show_values: bool = True,
        show_connections: bool = True,
        color_scheme: str = "viridis",
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Create magic structure visualization.
        
        Args:
            dimension: Number of dimensions
            size: Size in each dimension
            terms: Optional terms for semantic weighting
            interactive: Whether to create interactive visualization
            show_values: Whether to show numeric values
            show_connections: Whether to show connections
            color_scheme: Color scheme for visualization
            output_path: Optional path to save visualization
            
        Returns:
            Dictionary containing visualization data and metadata
        """
        response = self._make_request(
            "POST",
            "/api/v1/visualizations/magic",
            {
                "dimension": dimension,
                "size": size,
                "terms": terms,
                "interactive": interactive,
                "show_values": show_values,
                "show_connections": show_connections,
                "color_scheme": color_scheme
            }
        )
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            if response["format"] == "html":
                output_path.write_text(response["visualization"])
            else:
                import base64
                img_data = base64.b64decode(response["visualization"])
                output_path.write_bytes(img_data)
                
        return response 