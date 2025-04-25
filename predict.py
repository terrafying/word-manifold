"""
Prediction interface for Replicate.
"""

from typing import List, Dict, Any, Optional
import os
from cog import BasePredictor, Input, Path
import torch
import numpy as np
from word_manifold.api.client import WordManifoldClient
from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.visualization.engines.timeseries import TimeSeriesEngine
from word_manifold.visualization.renderers.timeseries import TimeSeriesRenderer
from word_manifold.visualization.magic_visualizer import MagicVisualizer

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient."""
        self.embeddings = WordEmbeddings()
        self.client = WordManifoldClient()
        
    def predict(
        self,
        task: str = Input(
            description="Task to perform",
            choices=["embed", "timeseries", "magic", "similarity"],
            default="embed"
        ),
        text: str = Input(
            description="Text to process (for embedding or similarity tasks)",
            default=None
        ),
        terms: List[str] = Input(
            description="Terms to analyze (for timeseries or magic visualization)",
            default=None
        ),
        compare_text: str = Input(
            description="Second text for similarity comparison",
            default=None
        ),
        timeframe: str = Input(
            description="Time range for analysis (e.g. 1h, 1d, 1w)",
            default="1d"
        ),
        pattern_type: str = Input(
            description="Type of temporal pattern",
            choices=["cyclic", "linear", "harmonic", "spiral", "wave"],
            default="cyclic"
        ),
        dimension: int = Input(
            description="Number of dimensions for magic structure",
            ge=2,
            le=5,
            default=2
        ),
        size: int = Input(
            description="Size in each dimension for magic structure",
            ge=3,
            default=3
        ),
        interactive: bool = Input(
            description="Whether to create interactive visualization",
            default=True
        )
    ) -> Dict[str, Any]:
        """Run a single prediction on the model."""
        try:
            if task == "embed":
                if not text:
                    raise ValueError("Text is required for embedding task")
                    
                # Get embedding
                embedding = self.client.get_embedding(text)
                return {
                    "embedding": embedding.tolist(),
                    "dimensions": len(embedding)
                }
                
            elif task == "similarity":
                if not text or not compare_text:
                    raise ValueError("Both texts are required for similarity task")
                    
                # Calculate similarity
                similarity = self.client.get_similarity(text, compare_text)
                return {
                    "similarity": similarity,
                    "text1": text,
                    "text2": compare_text
                }
                
            elif task == "timeseries":
                if not terms:
                    raise ValueError("Terms are required for timeseries task")
                    
                # Create visualization
                output_path = Path(os.path.join("/tmp", "timeseries.html" if interactive else "timeseries.png"))
                result = self.client.create_timeseries(
                    terms=terms,
                    timeframe=timeframe,
                    pattern_type=pattern_type,
                    interactive=interactive,
                    output_path=output_path
                )
                
                return {
                    "visualization": output_path,
                    "metadata": result["metadata"],
                    "insights": result.get("insights", [])
                }
                
            elif task == "magic":
                if not terms:
                    terms = ["wisdom", "understanding", "knowledge"]
                    
                # Create visualization
                output_path = Path(os.path.join("/tmp", "magic.html" if interactive else "magic.png"))
                result = self.client.create_magic_structure(
                    dimension=dimension,
                    size=size,
                    terms=terms,
                    interactive=interactive,
                    output_path=output_path
                )
                
                return {
                    "visualization": output_path,
                    "magic_constant": result["magic_constant"],
                    "is_magic": result["is_magic"]
                }
                
        except Exception as e:
            return {"error": str(e)} 