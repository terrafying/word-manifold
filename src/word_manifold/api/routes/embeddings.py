"""
FastAPI routes for text embedding services.
"""

from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, Depends
from word_manifold.api.models.embeddings import (
    EmbeddingRequest, EmbeddingResponse,
    BatchEmbeddingRequest, BatchEmbeddingResponse,
    SimilarityRequest, SimilarityResponse
)
from word_manifold.core.embeddings.transformers import (
    TransformerEmbeddingProvider,
    InstructEmbeddingProvider
)
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import torch

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

# Cache for embedding providers to avoid recreating them
_provider_cache: Dict[str, TransformerEmbeddingProvider] = {}

def get_provider(
    model_name: str,
    instruction: Optional[str] = None
) -> TransformerEmbeddingProvider:
    """Get or create an embedding provider for the specified model."""
    if model_name not in _provider_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if instruction is not None:
            provider = InstructEmbeddingProvider(
                model_name=model_name,
                device=device
            )
        else:
            provider = TransformerEmbeddingProvider(
                model_name=model_name,
                device=device
            )
        _provider_cache[model_name] = provider
    return _provider_cache[model_name]

@router.post("/embed", response_model=EmbeddingResponse)
async def embed_text(request: EmbeddingRequest):
    """Embed a single text using the specified model."""
    try:
        provider = get_provider(request.model_name, request.instruction)
        embedding = provider.embed_text(request.text)
        
        if request.normalize:
            embedding = embedding / np.linalg.norm(embedding)
            
        return EmbeddingResponse(
            embedding=embedding.tolist(),
            model_name=request.model_name,
            dimensions=len(embedding),
            metadata={
                "normalized": request.normalize,
                "instruction": request.instruction
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed_batch", response_model=BatchEmbeddingResponse)
async def embed_texts(request: BatchEmbeddingRequest):
    """Embed multiple texts using the specified model."""
    try:
        provider = get_provider(request.model_name, request.instruction)
        embeddings = provider.embed_texts(
            request.texts,
            batch_size=request.batch_size
        )
        
        if request.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
        return BatchEmbeddingResponse(
            embeddings=embeddings.tolist(),
            model_name=request.model_name,
            dimensions=embeddings.shape[1],
            metadata={
                "normalized": request.normalize,
                "instruction": request.instruction,
                "batch_size": request.batch_size
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """Compute similarity between two texts."""
    try:
        provider = get_provider(request.model_name)
        
        # Get embeddings
        emb1 = provider.embed_text(request.text1)
        emb2 = provider.embed_text(request.text2)
        
        # Normalize embeddings
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Compute similarity
        if request.metric.lower() == "cosine":
            similarity = 1 - cosine(emb1, emb2)
        elif request.metric.lower() == "euclidean":
            similarity = 1 / (1 + euclidean(emb1, emb2))
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported similarity metric: {request.metric}"
            )
            
        return SimilarityResponse(
            similarity=float(similarity),
            model_name=request.model_name,
            metric=request.metric,
            metadata={
                "embedding_dimensions": len(emb1)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 