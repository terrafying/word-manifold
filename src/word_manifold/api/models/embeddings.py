"""
Pydantic models for embedding API requests and responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, constr
import numpy as np

class EmbeddingRequest(BaseModel):
    """Request model for text embedding."""
    text: constr(min_length=1) = Field(..., description="Text to embed")
    model_name: Optional[str] = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the model to use for embedding"
    )
    normalize: bool = Field(True, description="Whether to L2-normalize the embeddings")
    instruction: Optional[str] = Field(None, description="Optional instruction for embedding")

class BatchEmbeddingRequest(BaseModel):
    """Request model for batch text embedding."""
    texts: List[constr(min_length=1)] = Field(..., description="List of texts to embed")
    model_name: Optional[str] = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the model to use for embedding"
    )
    normalize: bool = Field(True, description="Whether to L2-normalize the embeddings")
    instruction: Optional[str] = Field(None, description="Optional instruction for embedding")
    batch_size: Optional[int] = Field(32, description="Batch size for processing")

class EmbeddingResponse(BaseModel):
    """Response model for text embedding."""
    embedding: List[float] = Field(..., description="Embedding vector")
    model_name: str = Field(..., description="Model used for embedding")
    dimensions: int = Field(..., description="Dimensionality of embedding")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
            np.float32: float,
            np.float64: float
        }

class BatchEmbeddingResponse(BaseModel):
    """Response model for batch text embedding."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model_name: str = Field(..., description="Model used for embedding")
    dimensions: int = Field(..., description="Dimensionality of embeddings")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
            np.float32: float,
            np.float64: float
        }

class SimilarityRequest(BaseModel):
    """Request model for computing text similarity."""
    text1: constr(min_length=1) = Field(..., description="First text")
    text2: constr(min_length=1) = Field(..., description="Second text")
    model_name: Optional[str] = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the model to use"
    )
    metric: str = Field("cosine", description="Similarity metric to use")

class SimilarityResponse(BaseModel):
    """Response model for text similarity."""
    similarity: float = Field(..., description="Similarity score")
    model_name: str = Field(..., description="Model used for comparison")
    metric: str = Field(..., description="Similarity metric used")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata") 