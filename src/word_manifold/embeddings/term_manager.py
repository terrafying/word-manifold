"""
Term Manager Module - Handles term processing and caching in a background process.
"""

import multiprocessing as mp
from typing import Dict, List, Set, Optional, Any, Tuple
import numpy as np
import logging
from queue import Empty
import time
import threading
import torch
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TermManager:
    """
    Manages terms and their embeddings in a background process.
    Provides caching and asynchronous processing capabilities with proper synchronization.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_size: int = 10000,
        batch_size: int = 32,
        processing_timeout: float = 30.0
    ):
        """
        Initialize the TermManager.
        
        Args:
            model_name: Name of the transformer model to use
            cache_size: Maximum number of terms to cache
            batch_size: Size of batches for processing
            processing_timeout: Maximum time to wait for term processing (seconds)
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.processing_timeout = processing_timeout
        
        # Initialize multiprocessing components
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        manager = mp.Manager()
        self.cache = manager.dict()
        self.term_sets = manager.dict()
        self.processing_terms = manager.dict()  # Track terms being processed
        self.term_locks = manager.dict()  # Locks for term synchronization
        self.cache_access_times = manager.dict()  # Track when terms were last accessed
        
        # Start the background process
        self.process = mp.Process(target=self._run_background_process)
        self.process.start()
        
        # Start result processing thread
        self.running = True
        self.result_thread = threading.Thread(target=self._process_results)
        self.result_thread.daemon = True
        self.result_thread.start()
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"TermManager initialized with model {model_name}")
        
    def _process_results(self):
        """Process results from the background process."""
        while self.running:
            try:
                # Get result from queue with timeout
                result = self.result_queue.get(timeout=1)
                
                if result['type'] == 'shutdown':
                    logger.info("Result processor received shutdown signal")
                    break
                    
                elif result['type'] == 'batch_complete':
                    # Log batch completion
                    batch_size = result.get('batch_size', 0)
                    logger.debug(f"Completed processing batch of {batch_size} terms")
                    
                elif result['type'] == 'error':
                    # Log any errors from background process
                    error_msg = result.get('error', 'Unknown error')
                    logger.error(f"Background process error: {error_msg}")
                    
            except Empty:
                # No results to process, continue waiting
                continue
            except Exception as e:
                logger.error(f"Error in result processor: {e}")
                continue
                
        logger.info("Result processor shutting down")
        
    def _run_background_process(self):
        """Background process for term processing."""
        try:
            # Initialize model in the background process
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SentenceTransformer(self.model_name)
            model.to(device)
            
            while True:
                try:
                    # Get task from queue
                    task = self.task_queue.get(timeout=1)
                    
                    if task['type'] == 'shutdown':
                        self.result_queue.put({'type': 'shutdown'})
                        break
                        
                    elif task['type'] == 'process_terms':
                        terms = task['terms']
                        batch = []
                        
                        # Process terms in batches
                        for term in terms:
                            if term not in self.cache and term not in self.processing_terms:
                                self.processing_terms[term] = time.time()
                                batch.append(term)
                                
                                if len(batch) >= self.batch_size:
                                    self._process_batch(batch, model)
                                    # Notify about batch completion
                                    self.result_queue.put({
                                        'type': 'batch_complete',
                                        'batch_size': len(batch)
                                    })
                                    batch = []
                                    
                        # Process remaining terms
                        if batch:
                            self._process_batch(batch, model)
                            self.result_queue.put({
                                'type': 'batch_complete',
                                'batch_size': len(batch)
                            })
                            
                    elif task['type'] == 'add_term_set':
                        name, terms = task['name'], task['terms']
                        self.term_sets[name] = set(terms)
                        # Pre-process terms in the set
                        self.task_queue.put({
                            'type': 'process_terms',
                            'terms': list(terms)
                        })
                        
                except Empty:
                    # Clean up stale processing terms
                    current_time = time.time()
                    stale_terms = [
                        term for term, start_time in self.processing_terms.items()
                        if current_time - start_time > self.processing_timeout
                    ]
                    for term in stale_terms:
                        del self.processing_terms[term]
                    continue
                    
        except Exception as e:
            # Report error to main process
            self.result_queue.put({
                'type': 'error',
                'error': str(e)
            })
            logger.error(f"Background process error: {e}")
            
        finally:
            logger.info("Background process shutting down")
            
    def _process_batch(self, batch: List[str], model: SentenceTransformer):
        """Process a batch of terms to get their embeddings."""
        try:
            # Get embeddings using sentence-transformers
            embeddings = model.encode(batch, convert_to_numpy=True)
            
            # Store results
            current_time = time.time()
            for term, embedding in zip(batch, embeddings):
                self.cache[term] = embedding
                self.cache_access_times[term] = current_time
                if term in self.processing_terms:
                    del self.processing_terms[term]
                
            # Manage cache size using LRU strategy
            if len(self.cache) > self.cache_size:
                # Sort by access time and remove oldest
                sorted_terms = sorted(
                    self.cache_access_times.items(),
                    key=lambda x: x[1]
                )
                terms_to_remove = sorted_terms[:len(self.cache) - self.cache_size]
                for term, _ in terms_to_remove:
                    del self.cache[term]
                    del self.cache_access_times[term]
                    
        except Exception as e:
            # Report error to main process
            self.result_queue.put({
                'type': 'error',
                'error': str(e)
            })
            logger.error(f"Batch processing error: {e}")
            # Clean up processing status for failed terms
            for term in batch:
                if term in self.processing_terms:
                    del self.processing_terms[term]
            
    async def get_embedding_async(self, term: str, timeout: float = None) -> Optional[np.ndarray]:
        """
        Get the embedding for a term asynchronously.
        
        Args:
            term: Term to get embedding for
            timeout: Maximum time to wait for embedding (seconds)
            
        Returns:
            Numpy array of embedding or None if not found/timeout
        """
        if timeout is None:
            timeout = self.processing_timeout
            
        start_time = time.time()
        
        while True:
            # Check if term is in cache
            if term in self.cache:
                # Update access time
                self.cache_access_times[term] = time.time()
                return self.cache[term]
                
            # Check if term is being processed
            if term in self.processing_terms:
                # Wait a bit and retry
                await asyncio.sleep(0.1)
                if time.time() - start_time > timeout:
                    logger.warning(f"Timeout waiting for term: {term}")
                    return None
                continue
                
            # Term not found and not being processed
            self.add_terms([term])
            await asyncio.sleep(0.1)
            
    def get_embedding(self, term: str, timeout: float = None) -> Optional[np.ndarray]:
        """
        Get the embedding for a term (synchronous version).
        
        Args:
            term: Term to get embedding for
            timeout: Maximum time to wait for embedding (seconds)
            
        Returns:
            Numpy array of embedding or None if not found/timeout
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.get_embedding_async(term, timeout)
            )
        finally:
            loop.close()
        
    def add_terms(self, terms: List[str]):
        """
        Add terms for processing.
        
        Args:
            terms: List of terms to process
        """
        # Filter out terms already in cache or being processed
        new_terms = [
            term for term in terms
            if term not in self.cache and term not in self.processing_terms
        ]
        if new_terms:
            self.task_queue.put({
                'type': 'process_terms',
                'terms': new_terms
            })
        
    def add_term_set(self, name: str, terms: Set[str]):
        """
        Add a named set of terms.
        
        Args:
            name: Name of the term set
            terms: Set of terms
        """
        self.task_queue.put({
            'type': 'add_term_set',
            'name': name,
            'terms': terms
        })
        
    def get_term_set(self, name: str) -> Set[str]:
        """
        Get a named set of terms.
        
        Args:
            name: Name of the term set
            
        Returns:
            Set of terms
        """
        return self.term_sets.get(name, set())
        
    def shutdown(self):
        """Shutdown the background process and cleanup resources."""
        self.running = False  # Signal result thread to stop
        self.task_queue.put({'type': 'shutdown'})
        self.result_queue.put({'type': 'shutdown'})
        
        # Wait for background process and thread to finish
        if hasattr(self, 'process'):
            self.process.join()
        if hasattr(self, 'result_thread'):
            self.result_thread.join()
            
        # Cleanup thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown()
            
        logger.info("TermManager shut down") 