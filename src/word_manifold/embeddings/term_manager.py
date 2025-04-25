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
import os
import signal
import tempfile
import json
from pathlib import Path
import atexit
import sys
import weakref
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global registry of active managers
_active_managers = weakref.WeakSet()

def _cleanup_managers():
    """Clean up all active managers during interpreter shutdown."""
    for manager in _active_managers:
        try:
            manager.shutdown()
        except:
            pass

# Register cleanup function
atexit.register(_cleanup_managers)

class TermManagerProcess(mp.Process):
    """Separate process for term management to avoid pickling issues."""
    
    def __init__(self, model_name: str, task_queue: mp.Queue, result_queue: mp.Queue):
        super().__init__(daemon=True)
        self.model_name = model_name
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.running = True
        
    def run(self):
        """Run the term management process."""
        try:
            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SentenceTransformer(self.model_name)
            model.to(device)
            
            while self.running:
                try:
                    # Get task from queue with timeout
                    task = self.task_queue.get(timeout=1)
                except Empty:
                    continue
                except (EOFError, BrokenPipeError):
                    break
                
                if task['type'] == 'shutdown':
                    break
                    
                try:
                    if task['type'] == 'get_embedding':
                        embedding = model.encode([task['term']], convert_to_numpy=True)[0]
                        self.result_queue.put({
                            'type': 'embedding',
                            'term': task['term'],
                            'embedding': embedding
                        })
                except Exception as e:
                    self.result_queue.put({
                        'type': 'error',
                        'error': str(e)
                    })
                    
        except Exception as e:
            try:
                self.result_queue.put({
                    'type': 'error',
                    'error': str(e)
                })
            except:
                pass
            
        finally:
            # Clean up resources
            try:
                model.cpu()
                del model
                torch.cuda.empty_cache()
            except:
                pass

class TermManager:
    """
    Manages terms and their embeddings in a background process.
    Provides caching and asynchronous processing capabilities with proper synchronization.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        """
        Initialize the TermManager.
        
        Args:
            model_name: Name of the transformer model to use
            cache_size: Maximum number of terms to cache
            batch_size: Size of batches for processing
            processing_timeout: Maximum time to wait for term processing (seconds)
            cache_dir: Optional directory for caching embeddings
        """
        self.model_name = model_name
        self.cache_size = kwargs.get('cache_size', 10000)
        self.batch_size = kwargs.get('batch_size', 32)
        self.processing_timeout = kwargs.get('processing_timeout', 30.0)
        self.cache_dir = Path(kwargs.get('cache_dir', tempfile.gettempdir())) / "term_manager"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create state file
        self.state_file = self.cache_dir / "state.json"
        self.pid_file = self.cache_dir / "pid.txt"
        
        # Save PID
        with open(self.pid_file, "w") as f:
            f.write(str(os.getpid()))
        
        # Initialize multiprocessing components
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.manager = mp.Manager()
        self.cache = self.manager.dict()
        self.term_sets = self.manager.dict()
        self.processing_terms = self.manager.dict()
        self.term_locks = self.manager.dict()
        self.cache_access_times = self.manager.dict()
        
        # Start the background process
        self.process = TermManagerProcess(
            model_name=self.model_name,
            task_queue=self.task_queue,
            result_queue=self.result_queue
        )
        self.process.start()
        
        # Start result processing thread
        self.running = True
        self.result_thread = threading.Thread(target=self._process_results)
        self.result_thread.daemon = True
        self.result_thread.start()
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Initialize state
        self.state = {
            "embeddings": {},
            "terms": set()
        }
        self._load_state()
        
        # Register in global registry
        _active_managers.add(self)
        
        logger.info(f"TermManager initialized with model {model_name}")
        
    def _process_results(self):
        """Process results from the background process."""
        while self.running:
            try:
                # Get result from queue with timeout
                try:
                    result = self.result_queue.get(timeout=1)
                except Empty:
                    continue
                except (EOFError, BrokenPipeError):
                    break
                
                if result['type'] == 'shutdown':
                    break
                    
                elif result['type'] == 'embedding':
                    term = result['term']
                    embedding = result['embedding']
                    self.cache[term] = embedding
                    self.cache_access_times[term] = time.time()
                    if term in self.processing_terms:
                        del self.processing_terms[term]
                    
                elif result['type'] == 'error':
                    error_msg = result.get('error', 'Unknown error')
                    if self.running:  # Only log if not shutting down
                        logger.error(f"Background process error: {error_msg}")
                    
            except Exception as e:
                if self.running:  # Only log if not shutting down
                    logger.error(f"Error in result processor: {e}")
                continue
                
    def get_embedding(self, term: str) -> Optional[np.ndarray]:
        """Get embedding for a term."""
        if term in self.cache:
            self.cache_access_times[term] = time.time()
            return self.cache[term]
            
        if term in self.processing_terms:
            start_time = self.processing_terms[term]
            if time.time() - start_time > self.processing_timeout:
                del self.processing_terms[term]
            else:
                return None
                
        self.processing_terms[term] = time.time()
        self.task_queue.put({
            'type': 'get_embedding',
            'term': term
        })
        
        return None
        
    def get_embeddings(self, terms: List[str], timeout: float = 30.0) -> List[Optional[np.ndarray]]:
        """Get embeddings for a batch of terms.
        
        Args:
            terms: List of terms to get embeddings for
            timeout: Maximum time to wait for embeddings
            
        Returns:
            List of embeddings in the same order as terms. None for failed terms.
        """
        results = []
        start_time = time.time()
        
        for term in terms:
            # Check if we've exceeded timeout
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout getting embeddings after {len(results)} terms")
                break
                
            # Try to get embedding with shorter timeout for each term
            remaining_time = max(1.0, timeout - (time.time() - start_time))
            embedding = None
            
            # First check cache
            if term in self.cache:
                embedding = self.cache[term]
            else:
                # Request embedding from background process
                self.task_queue.put({
                    'type': 'get_embedding',
                    'term': term
                })
                
                # Wait for result with timeout
                wait_start = time.time()
                while time.time() - wait_start < remaining_time:
                    if term in self.cache:
                        embedding = self.cache[term]
                        break
                    time.sleep(0.1)
                    
            results.append(embedding)
            
        return results
        
    def shutdown(self):
        """Shutdown the background process and cleanup resources."""
        if not hasattr(self, 'running') or not self.running:
            return
            
        self.running = False
        
        try:
            # Save final state
            self._save_state()
            
            # Clean up thread pool
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
            
            # Signal process to stop and close queues
            try:
                self.task_queue.put_nowait({'type': 'shutdown'})
            except:
                pass
            
            # Wait for process with timeout
            if hasattr(self, 'process'):
                try:
                    self.process.join(timeout=2.0)
                except:
                    pass
                if self.process.is_alive():
                    self.process.terminate()
                    try:
                        self.process.join(timeout=1.0)
                    except:
                        pass
                    if self.process.is_alive():
                        self.process.kill()
            
            # Clean up manager
            if hasattr(self, 'manager'):
                try:
                    self.manager.shutdown()
                except:
                    pass
            
            # Clean up files
            try:
                if hasattr(self, 'pid_file'):
                    self.pid_file.unlink(missing_ok=True)
            except:
                pass
            
            # Close queues
            for queue in ['task_queue', 'result_queue']:
                if hasattr(self, queue):
                    try:
                        getattr(self, queue).close()
                        getattr(self, queue).join_thread()
                    except:
                        pass
            
        except Exception as e:
            if not sys.is_finalizing():
                logger.error(f"Error during shutdown: {e}")
        
        finally:
            # Remove from registry
            _active_managers.discard(self)
            
            # Release resources
            for attr in ['task_queue', 'result_queue', 'cache', 'term_sets',
                        'processing_terms', 'term_locks', 'cache_access_times',
                        'manager', 'process', 'result_thread']:
                if hasattr(self, attr):
                    delattr(self, attr)
    
    def _load_state(self):
        """Load state from file."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                self.state["embeddings"] = {
                    term: np.array(embedding)
                    for term, embedding in state["embeddings"].items()
                }
                self.state["terms"] = set(state["terms"])
                
    def _save_state(self):
        """Save state to file."""
        state = {
            "embeddings": {
                term: embedding.tolist()
                for term, embedding in self.state["embeddings"].items()
            },
            "terms": list(self.state["terms"])
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f)
            
    def __getstate__(self):
        """Get state for pickling."""
        state = self.__dict__.copy()
        # Remove unpicklable attributes
        for attr in ['task_queue', 'result_queue', 'manager', 'process',
                    'result_thread', 'thread_pool']:
            state.pop(attr, None)
        return state
        
    def __setstate__(self, state):
        """Set state after unpickling."""
        self.__dict__.update(state)
        # Reinitialize multiprocessing components
        self.__init__(self.model_name)
        
    @classmethod
    def connect(cls, pid: int) -> 'TermManager':
        """
        Connect to existing term manager instance.
        
        Args:
            pid: Process ID of existing term manager
            
        Returns:
            Connected TermManager instance
        """
        # Find state file
        cache_dir = Path(tempfile.gettempdir()) / "term_manager"
        pid_file = cache_dir / "pid.txt"
        
        if not pid_file.exists():
            raise ValueError("No existing term manager found")
            
        with open(pid_file) as f:
            stored_pid = int(f.read().strip())
            
        if stored_pid != pid:
            raise ValueError(f"PID mismatch: {stored_pid} != {pid}")
            
        # Create instance without starting new process
        instance = cls.__new__(cls)
        instance.cache_dir = cache_dir
        instance.state_file = cache_dir / "state.json"
        instance.pid_file = pid_file
        
        # Load state
        instance.state = {
            "embeddings": {},
            "terms": set()
        }
        instance._load_state()
        
        return instance 