"""
Distributed Term Manager using Ray for parallel processing across machines.
"""

import ray
import numpy as np
from typing import Dict, List, Set, Optional, Any
import logging
from sentence_transformers import SentenceTransformer
import time
from collections import OrderedDict
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote
class TermWorker:
    """Worker for processing terms in parallel."""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
            
    def process_batch(self, terms: List[str]) -> Dict[str, np.ndarray]:
        """Process a batch of terms to get embeddings."""
        try:
            embeddings = self.model.encode(terms, convert_to_numpy=True)
            return {term: emb for term, emb in zip(terms, embeddings)}
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return {}

class DistributedTermManager:
    """
    Distributed term manager using Ray for parallel processing.
    Coordinates multiple workers across machines for term embedding.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_size: int = 10000,
        batch_size: int = 32,
        num_workers: int = 2,
        ray_address: Optional[str] = None
    ):
        """
        Initialize the distributed term manager.
        
        Args:
            model_name: Name of the transformer model
            cache_size: Maximum number of terms to cache
            batch_size: Size of batches for processing
            num_workers: Number of parallel workers
            ray_address: Optional Ray cluster address (e.g., "ray://192.168.1.100:10001")
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            if ray_address:
                ray.init(address=ray_address)
                logger.info(f"Connected to Ray cluster at {ray_address}")
            else:
                ray.init()
                logger.info("Initialized Ray locally")
        
        # Create term workers
        self.workers = [
            TermWorker.remote(model_name)
            for _ in range(num_workers)
        ]
        
        # Initialize cache with LRU tracking
        self.cache: Dict[str, np.ndarray] = OrderedDict()
        self.processing: Set[str] = set()
        
        logger.info(f"Initialized DistributedTermManager with {num_workers} workers")
        
    def _manage_cache(self):
        """Manage cache size using LRU policy."""
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
            
    async def get_embedding(self, term: str, timeout: float = 30.0) -> Optional[np.ndarray]:
        """
        Get embedding for a term, processing it if needed.
        
        Args:
            term: Term to get embedding for
            timeout: Maximum time to wait for result
            
        Returns:
            Numpy array of embedding or None if not found/timeout
        """
        # Check cache first
        if term in self.cache:
            # Move to end (most recently used)
            embedding = self.cache.pop(term)
            self.cache[term] = embedding
            return embedding
            
        # Process if not in cache
        if term not in self.processing:
            self.processing.add(term)
            # Get least busy worker
            worker = min(self.workers, key=lambda w: ray.get(w.get_queue_length.remote()))
            future = worker.process_batch.remote([term])
            
            try:
                result = await ray.wait([future], timeout=timeout)
                if result:
                    embeddings = ray.get(future)
                    if term in embeddings:
                        self.cache[term] = embeddings[term]
                        self._manage_cache()
                        return embeddings[term]
            except Exception as e:
                logger.error(f"Error getting embedding for {term}: {e}")
            finally:
                self.processing.remove(term)
                
        return None
        
    def get_embeddings_batch(
        self,
        terms: List[str],
        timeout: float = 60.0
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple terms in parallel.
        
        Args:
            terms: List of terms to process
            timeout: Maximum time to wait for results
            
        Returns:
            Dictionary mapping terms to their embeddings
        """
        results = {}
        pending_terms = []
        
        # Check cache first
        for term in terms:
            if term in self.cache:
                results[term] = self.cache[term]
            else:
                pending_terms.append(term)
                
        if not pending_terms:
            return results
            
        # Process pending terms in parallel
        futures = []
        for i in range(0, len(pending_terms), self.batch_size):
            batch = pending_terms[i:i + self.batch_size]
            # Round-robin assignment to workers
            worker = self.workers[len(futures) % len(self.workers)]
            futures.append(worker.process_batch.remote(batch))
            
        # Wait for results
        try:
            done_refs = ray.get(futures, timeout=timeout)
            for batch_results in done_refs:
                results.update(batch_results)
                # Update cache
                for term, embedding in batch_results.items():
                    self.cache[term] = embedding
                self._manage_cache()
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            
        return results
        
    def shutdown(self):
        """Shutdown the distributed term manager."""
        # Ray will handle worker cleanup
        logger.info("Shutting down DistributedTermManager")
        self.cache.clear()
        self.processing.clear() 