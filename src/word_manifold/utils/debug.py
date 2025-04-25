"""
Debug Utilities

Helper functions for debugging and development.
"""

import sys
import os
import logging
import traceback
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from concurrent.futures import ThreadPoolExecutor, Future
from functools import wraps
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variable for generic function type
F = TypeVar('F', bound=Callable)

def log_errors(func: F) -> F:
    """Decorator to log exceptions with full traceback."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}:")
            logger.error(f"Args: {args}, Kwargs: {kwargs}")
            logger.error(f"Exception: {e}")
            logger.error(f"Traceback:\n{''.join(traceback.format_tb(sys.exc_info()[2]))}")
            raise
    return wrapper

def time_it(func: F) -> F:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

class AsyncTaskManager:
    """Manages asynchronous task execution with thread pooling."""
    
    def __init__(self, max_workers: int = None):
        """Initialize task manager.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, Future] = {}
        self._lock = threading.Lock()
        
    def submit_task(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Future:
        """Submit a task for asynchronous execution.
        
        Args:
            name: Task identifier
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Future object for the task
        """
        with self._lock:
            future = self.executor.submit(func, *args, **kwargs)
            self.tasks[name] = future
            return future
    
    def get_result(self, name: str, timeout: Optional[float] = None) -> Any:
        """Get result of a task by name.
        
        Args:
            name: Task identifier
            timeout: Maximum time to wait for result
            
        Returns:
            Task result
            
        Raises:
            KeyError: If task not found
            TimeoutError: If task doesn't complete within timeout
        """
        with self._lock:
            if name not in self.tasks:
                raise KeyError(f"No task named '{name}'")
            future = self.tasks[name]
            
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.error(f"Task {name} timed out")
            raise
        except Exception as e:
            logger.error(f"Task {name} failed: {e}")
            raise
    
    def cancel_task(self, name: str) -> bool:
        """Cancel a running task.
        
        Args:
            name: Task identifier
            
        Returns:
            True if task was cancelled, False if already complete
        """
        with self._lock:
            if name not in self.tasks:
                raise KeyError(f"No task named '{name}'")
            return self.tasks[name].cancel()
    
    def shutdown(self, wait: bool = True):
        """Shutdown the task manager.
        
        Args:
            wait: Wait for pending tasks to complete
        """
        self.executor.shutdown(wait=wait)

class DebugContext:
    """Context manager for debugging blocks of code."""
    
    def __init__(self, name: str, log_level: int = logging.DEBUG):
        self.name = name
        self.log_level = log_level
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.log(self.log_level, f"Entering {self.name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        if exc_type:
            logger.error(f"Error in {self.name}: {exc_val}")
            logger.error(f"Traceback:\n{''.join(traceback.format_tb(exc_tb))}")
        else:
            logger.log(self.log_level, f"Exiting {self.name} after {duration:.2f}s")
        return False  # Don't suppress exceptions

def memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    try:
        import psutil
        process = psutil.Process()
        mem = process.memory_info()
        return {
            'rss': mem.rss / 1024 / 1024,  # MB
            'vms': mem.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }
    except ImportError:
        logger.warning("psutil not available, memory stats disabled")
        return {}

def profile_function(func: F) -> F:
    """Decorator to profile function execution."""
    try:
        import cProfile
        import pstats
        import io
    except ImportError:
        logger.warning("cProfile not available, profiling disabled")
        return func
        
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        logger.debug(f"Profile for {func.__name__}:\n{s.getvalue()}")
        return result
        
    return wrapper

# Example usage:
if __name__ == "__main__":
    # Set up task manager
    task_mgr = AsyncTaskManager(max_workers=4)
    
    # Example async task
    @log_errors
    def slow_task(name: str, delay: float) -> str:
        time.sleep(delay)
        return f"Task {name} completed"
    
    # Submit some tasks
    task_mgr.submit_task("task1", slow_task, "one", 2)
    task_mgr.submit_task("task2", slow_task, "two", 1)
    
    try:
        # Get results
        with DebugContext("Getting results"):
            result1 = task_mgr.get_result("task1", timeout=3)
            result2 = task_mgr.get_result("task2", timeout=3)
            logger.info(f"Results: {result1}, {result2}")
            
        # Show memory usage
        mem = memory_usage()
        if mem:
            logger.info(f"Memory usage: {mem['rss']:.1f}MB RSS")
            
    finally:
        task_mgr.shutdown() 