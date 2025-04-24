"""
Base interfaces and abstract classes for embeddings.

This module defines the core protocols and abstract base classes for
embedding text into vector spaces, with proper typing and async support.
"""

from typing import Protocol, List, Dict, Any, Optional, TypeVar, Generic
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from pathlib import Path

# Type variables
T_co = TypeVar("T_co", covariant=True)
EmbeddingVector = npt.NDArray[np.float32]

@dataclass
class EmbeddingMetadata:
    """Metadata for an embedding."""
    model_name: str
    dimensions: int
    created_at: float
    version: str = "1.0.0"

@dataclass
class EmbeddingResult(Generic[T_co]):
    """Result of an embedding operation."""
    vector: EmbeddingVector
    metadata: EmbeddingMetadata
    extra: Optional[T_co] = None

class EmbeddingError(Exception):
    """Base class for embedding-related errors."""
    pass

class ModelLoadError(EmbeddingError):
    """Error loading embedding model."""
    pass

class EmbeddingProvider(Protocol):
    """Protocol defining the interface for embedding providers."""
    
    @property
    def dimensions(self) -> int:
        """Get the dimensionality of the embeddings."""
        ...
    
    @property
    def model_name(self) -> str:
        """Get the name of the embedding model."""
        ...
    
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text string."""
        ...
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed multiple texts efficiently."""
        ...
    
    async def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        ...
    
    def to_device(self, device: torch.device) -> None:
        """Move the model to specified device."""
        ...

class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        cache_dir: Optional[Path] = None,
        **kwargs: Any
    ):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self._executor = ThreadPoolExecutor(max_workers=2)  # For CPU-bound operations
        self._metadata = EmbeddingMetadata(
            model_name=model_name,
            dimensions=self.dimensions,
            created_at=asyncio.get_event_loop().time()
        )
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the dimensionality of the embeddings."""
        pass
    
    @abstractmethod
    def _embed_sync(self, text: str) -> EmbeddingVector:
        """Synchronous embedding implementation."""
        pass
    
    @abstractmethod
    def _embed_batch_sync(self, texts: List[str]) -> List[EmbeddingVector]:
        """Synchronous batch embedding implementation."""
        pass
    
    async def embed(self, text: str) -> EmbeddingResult:
        """Asynchronously embed a single text string."""
        vector = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._embed_sync, text
        )
        return EmbeddingResult(vector=vector, metadata=self._metadata)
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Asynchronously embed multiple texts."""
        vectors = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._embed_batch_sync, texts
        )
        return [
            EmbeddingResult(vector=vector, metadata=self._metadata)
            for vector in vectors
        ]
    
    async def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        results = await self.embed_batch([text1, text2])
        vec1, vec2 = results[0].vector, results[1].vector
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def to_device(self, device: torch.device) -> None:
        """Move the model to specified device."""
        self.device = device
    
    async def __aenter__(self) -> "BaseEmbeddingProvider":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self._executor.shutdown(wait=True)

class CachedEmbeddingProvider:
    """Decorator class for adding caching to embedding providers."""
    
    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_size: int = 10000,
        ttl: Optional[int] = None
    ):
        self.provider = provider
        self.cache: Dict[str, tuple[float, EmbeddingResult]] = {}
        self.cache_size = cache_size
        self.ttl = ttl
    
    async def _get_cached(self, key: str) -> Optional[EmbeddingResult]:
        """Get cached result if valid."""
        if key in self.cache:
            timestamp, result = self.cache[key]
            if self.ttl is None or (asyncio.get_event_loop().time() - timestamp) < self.ttl:
                return result
            del self.cache[key]
        return None
    
    def _cache_result(self, key: str, result: EmbeddingResult) -> None:
        """Cache a result with timestamp."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]
        self.cache[key] = (asyncio.get_event_loop().time(), result)
    
    async def embed(self, text: str) -> EmbeddingResult:
        """Get embedding with caching."""
        if cached := await self._get_cached(text):
            return cached
        result = await self.provider.embed(text)
        self._cache_result(text, result)
        return result
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Get batch embeddings with caching."""
        results: List[EmbeddingResult] = []
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []
        
        # Check cache first
        for i, text in enumerate(texts):
            if cached := await self._get_cached(text):
                results.append(cached)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Get uncached embeddings
        if uncached_texts:
            uncached_results = await self.provider.embed_batch(uncached_texts)
            for text, result in zip(uncached_texts, uncached_results):
                self._cache_result(text, result)
            
            # Merge cached and new results
            final_results = [None] * len(texts)
            cached_idx = 0
            uncached_idx = 0
            
            for i in range(len(texts)):
                if i in uncached_indices:
                    final_results[i] = uncached_results[uncached_idx]
                    uncached_idx += 1
                else:
                    final_results[i] = results[cached_idx]
                    cached_idx += 1
            
            results = final_results
        
        return results
    
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to provider."""
        return getattr(self.provider, name) 