"""
Transformer-based embedding implementations.

This module provides concrete implementations of embedding providers
using transformer models from the sentence-transformers library.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
from functools import cached_property

from .base import (
    BaseEmbeddingProvider,
    EmbeddingVector,
    ModelLoadError,
    EmbeddingResult
)

logger = logging.getLogger(__name__)

class TransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using sentence-transformers models."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
        cache_dir: Optional[Path] = None,
        max_seq_length: int = 512,
        normalize_embeddings: bool = True,
        **kwargs: Any
    ):
        """Initialize the transformer embedding provider.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            device: Device to run the model on
            cache_dir: Directory to cache models
            max_seq_length: Maximum sequence length for tokenization
            normalize_embeddings: Whether to L2-normalize embeddings
            **kwargs: Additional arguments passed to SentenceTransformer
        """
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)
        
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        
        try:
            logger.info(f"Loading transformer model: {model_name}")
            self.model = SentenceTransformer(
                model_name_or_path=model_name,
                device=str(self.device),
                cache_folder=str(cache_dir) if cache_dir else None,
                **kwargs
            )
            self.model.max_seq_length = max_seq_length
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadError(f"Failed to load model {model_name}: {e}") from e
    
    @cached_property
    def dimensions(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.model.get_sentence_embedding_dimension()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding."""
        # Add any custom preprocessing here
        return text.strip()
    
    def _postprocess_vector(self, vector: np.ndarray) -> EmbeddingVector:
        """Postprocess embedding vector."""
        if self.normalize_embeddings:
            return vector / np.linalg.norm(vector)
        return vector
    
    def _embed_sync(self, text: str) -> EmbeddingVector:
        """Synchronously embed a single text."""
        processed_text = self._preprocess_text(text)
        with torch.no_grad():
            embedding = self.model.encode(
                processed_text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings
            )
        return self._postprocess_vector(embedding)
    
    def _embed_batch_sync(self, texts: List[str]) -> List[EmbeddingVector]:
        """Synchronously embed multiple texts."""
        processed_texts = [self._preprocess_text(text) for text in texts]
        with torch.no_grad():
            embeddings = self.model.encode(
                processed_texts,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                batch_size=32  # Adjust based on available memory
            )
        return [self._postprocess_vector(emb) for emb in embeddings]
    
    def to_device(self, device: torch.device) -> None:
        """Move the model to specified device."""
        super().to_device(device)
        self.model.to(device)
    
    async def __aenter__(self) -> "TransformerEmbeddingProvider":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await super().__aexit__(exc_type, exc_val, exc_tb)
        # Clean up any model-specific resources
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class InstructEmbeddingProvider(TransformerEmbeddingProvider):
    """Embedding provider with instruction tuning support."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        instruction: str = "Represent this text for retrieval:",
        **kwargs: Any
    ):
        """Initialize the instruction-tuned embedding provider.
        
        Args:
            model_name: Name of the sentence-transformer model
            instruction: Instruction prefix for embeddings
            **kwargs: Additional arguments passed to TransformerEmbeddingProvider
        """
        super().__init__(model_name=model_name, **kwargs)
        self.instruction = instruction
    
    def _preprocess_text(self, text: str) -> str:
        """Add instruction prefix to text."""
        processed = super()._preprocess_text(text)
        return f"{self.instruction}\n{processed}"

class MultilingualEmbeddingProvider(TransformerEmbeddingProvider):
    """Embedding provider with multilingual support."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        **kwargs: Any
    ):
        """Initialize the multilingual embedding provider."""
        super().__init__(model_name=model_name, **kwargs)
        
    async def translate_and_embed(self, text: str, source_lang: str) -> EmbeddingResult:
        """Translate text if needed and then embed."""
        # TODO: Implement translation support
        return await self.embed(text) 