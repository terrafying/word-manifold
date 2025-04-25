"""
Model Host

Manages ML models across machines using Ray for distributed computation.
Provides simple interface for model loading and inference.
"""

import ray
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class ModelWorker:
    """Ray worker for hosting models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """Initialize model worker."""
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode texts to embeddings."""
        try:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                device=self.device
            )
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        stats = {
            "model_name": self.model_name,
            "device": self.device,
            "memory_used": None
        }
        
        if torch.cuda.is_available():
            try:
                memory = torch.cuda.memory_allocated() / 1024**2  # MB
                stats["memory_used"] = f"{memory:.1f}MB"
            except:
                pass
                
        return stats

class ModelHost:
    """Manages distributed model workers."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_workers: int = 2,
        ray_address: Optional[str] = None
    ):
        """Initialize model host."""
        self.model_name = model_name
        self.num_workers = num_workers
        
        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(address=ray_address)
            logger.info(f"Connected to Ray cluster at {ray_address}" if ray_address else "Initialized Ray locally")
        
        # Create workers
        self.workers = [
            ModelWorker.remote(model_name)
            for _ in range(num_workers)
        ]
        logger.info(f"Created {num_workers} model workers")
        
        # Track worker loads
        self.worker_loads = [0] * num_workers
    
    def get_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """Get embeddings for texts using least loaded worker."""
        if not texts:
            return {}
            
        # Find least loaded worker
        worker_idx = self.worker_loads.index(min(self.worker_loads))
        worker = self.workers[worker_idx]
        
        try:
            # Update load counter
            self.worker_loads[worker_idx] += len(texts)
            
            # Get embeddings
            embeddings = ray.get(worker.encode.remote(texts, batch_size))
            
            # Create result dictionary
            results = {
                text: embedding
                for text, embedding in zip(texts, embeddings)
                if embedding is not None
            }
            
            return results
            
        finally:
            # Update load counter
            self.worker_loads[worker_idx] -= len(texts)
    
    def get_worker_stats(self) -> List[Dict[str, Any]]:
        """Get statistics from all workers."""
        return ray.get([worker.get_stats.remote() for worker in self.workers])
    
    def shutdown(self) -> None:
        """Shutdown the model host."""
        # Ray will handle worker cleanup
        logger.info("Shutting down model host") 