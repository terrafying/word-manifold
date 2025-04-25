"""
Vector Manifold Module

A simplified implementation of vector manifold for word embeddings,
providing basic clustering and neighborhood operations.
"""

import numpy as np
import logging
from typing import Dict, List, Set, Optional, Union, NamedTuple
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import time

from ..embeddings.word_embeddings import WordEmbeddings
from ..types.cell_types import CellType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ManifoldReducedState(NamedTuple):
    """Represents the reduced state of a manifold for visualization."""
    points: np.ndarray  # Reduced point coordinates
    labels: List[str]   # Point labels
    edges: List[tuple]  # Edge connections between points
    metadata: Dict      # Additional visualization metadata

@dataclass
class Cell:
    """A cell in the manifold representing a region in the vector space."""
    id: int
    terms: Set[str]  # Words that belong to this cell
    centroid: np.ndarray  # Center point of the cell
    type: CellType = CellType.STANDARD  # Type of cell
    
    def __post_init__(self):
        """Validate cell data after initialization."""
        if not isinstance(self.terms, set):
            self.terms = set(self.terms)
        if not isinstance(self.centroid, np.ndarray):
            self.centroid = np.array(self.centroid)

class VectorManifold:
    """
    A simplified manifold in vector space for word embeddings.
    Handles basic clustering and neighborhood relationships.
    """
    
    def __init__(
        self,
        embeddings: WordEmbeddings,
        n_cells: int = 10,
        random_state: int = 42
    ):
        """
        Initialize the VectorManifold.
        
        Args:
            embeddings: WordEmbeddings instance to use
            n_cells: Number of cells to divide the manifold into
            random_state: Random seed for reproducibility
        """
        self.embeddings = embeddings
        self.n_cells = n_cells
        self.random_state = random_state
        
        # Wait for terms to be loaded
        terms = self.embeddings.get_terms()
        if not terms:
            time.sleep(2)  # Give background process time to load terms
            terms = self.embeddings.get_terms()
            
        if not terms:
            raise ValueError("Embeddings must have terms loaded")
        
        # Initialize data structures
        self.terms = list(terms)
        self.term_to_index = {term: i for i, term in enumerate(self.terms)}
        
        # Get vectors for all terms
        self.vectors = np.array([
            embedding for embedding in [
                self.embeddings.get_embedding(term) for term in self.terms
            ] if embedding is not None
        ])
        
        if len(self.vectors) == 0:
            raise ValueError("No valid embeddings found for terms")
        
        # Initialize manifold structures
        self.cells: Dict[int, Cell] = {}
        self.cell_neighbors: Dict[int, Set[int]] = {}
        self.term_to_cell: Dict[str, int] = {}
        
        # Define the manifold structure
        self._define_manifold()
        
    def _define_manifold(self) -> None:
        """Define the basic manifold structure."""
        logger.info("Defining manifold structure")
        
        # Cluster terms
        self._cluster_terms()
        
        # Establish neighborhoods
        self._establish_neighborhoods()
        
        logger.info(f"Manifold defined with {len(self.cells)} cells")
    
    def _cluster_terms(self) -> None:
        """Cluster terms to define cells in the manifold."""
        # Adjust number of clusters based on number of terms
        n_clusters = min(self.n_cells, len(self.terms))
        if n_clusters < self.n_cells:
            logger.warning(f"Reducing clusters from {self.n_cells} to {n_clusters}")
        
        # Use KMeans clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init="auto"
        )
        cluster_labels = kmeans.fit_predict(self.vectors)
        
        # Create cells
        for i in range(n_clusters):
            # Get terms in this cluster
            term_indices = np.where(cluster_labels == i)[0]
            cluster_terms = {self.terms[idx] for idx in term_indices}
            
            if not cluster_terms:
                logger.warning(f"Cluster {i} is empty, skipping")
                continue
            
            # Create cell with centroid from KMeans
            cell = Cell(
                id=i,
                terms=cluster_terms,
                centroid=kmeans.cluster_centers_[i]
            )
            
            self.cells[i] = cell
            
            # Map terms to this cell
            for term in cluster_terms:
                self.term_to_cell[term] = i
    
    def _establish_neighborhoods(self) -> None:
        """Establish neighborhood relationships between cells."""
        # Calculate distances between all centroids
        centroids = np.array([cell.centroid for cell in self.cells.values()])
        distances = cosine_distances(centroids)
        
        # For each cell, find its closest neighbors
        for i in range(len(self.cells)):
            # Get indices of closest cells (excluding self)
            closest = np.argsort(distances[i])[1:4]  # Get 3 nearest neighbors
            self.cell_neighbors[i] = set(closest)
            
            # Mark boundary cells (those with high average distance to neighbors)
            avg_distance = np.mean(distances[i][closest])
            if avg_distance > np.mean(distances) + np.std(distances):
                self.cells[i].type = CellType.BOUNDARY
    
    def get_cell(self, term: str) -> Optional[Cell]:
        """Get the cell containing a term."""
        cell_id = self.term_to_cell.get(term)
        return self.cells.get(cell_id) if cell_id is not None else None
    
    def get_neighbors(self, term: str) -> Set[str]:
        """Get neighboring terms for a given term."""
        cell = self.get_cell(term)
        if not cell:
            return set()
            
        # Get all terms from neighboring cells
        neighbor_terms = set()
        for neighbor_id in self.cell_neighbors.get(cell.id, set()):
            if neighbor_id in self.cells:
                neighbor_terms.update(self.cells[neighbor_id].terms)
        return neighbor_terms
    
    def add_term(self, term: str) -> Optional[int]:
        """Add a new term to the manifold."""
        # Get embedding for the term
        vector = self.embeddings.get_embedding(term)
        if vector is None:
            return None
            
        # Find closest cell
        min_distance = float('inf')
        closest_cell_id = None
        
        for cell_id, cell in self.cells.items():
            distance = cosine_distances(
                vector.reshape(1, -1),
                cell.centroid.reshape(1, -1)
            )[0][0]
            
            if distance < min_distance:
                min_distance = distance
                closest_cell_id = cell_id
        
        if closest_cell_id is not None:
            # Add term to closest cell
            cell = self.cells[closest_cell_id]
            cell.terms.add(term)
            self.term_to_cell[term] = closest_cell_id
            
            # Update cell centroid
            vectors = [self.embeddings.get_embedding(t) for t in cell.terms]
            vectors = [v for v in vectors if v is not None]
            if vectors:
                cell.centroid = np.mean(vectors, axis=0)
            
            return closest_cell_id
        return None
