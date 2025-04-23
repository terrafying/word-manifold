"""
Vector Manifold Module for Cellular Automata in Word Vector Space.

This module defines a manifold structure over word and phrase embeddings, creating a
topological space of semantic regions that will serve as cells for the
cellular automata system. It implements region definition, neighborhood
relationships, and transformations influenced by numerological properties.
"""

import numpy as np
import logging
from typing import Dict, List, Set, Tuple, Optional, Union, Any, NamedTuple
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import umap
try:
    from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
except ImportError:
    from scipy import spatial
    Voronoi = spatial.Voronoi
    ConvexHull = spatial.ConvexHull
    voronoi_plot_2d = spatial.voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import colorsys

from ..embeddings.word_embeddings import WordEmbeddings
from ..embeddings.phrase_embeddings import PhraseEmbedding

from ..types import CellType, DistanceType

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Cell:
    """
    A cell in the manifold representing a region in the vector space.
    """
    id: int
    terms: Set[str]  # Words or phrases that belong to this cell
    centroid: np.ndarray  # Center point of the cell in embedding space
    type: CellType  # Type of cell with occult significance
    numerological_value: int  # Numerological value of the cell
    boundary_points: Optional[np.ndarray] = None  # Points defining the boundary (if available)
    is_phrase: bool = False  # Whether this cell contains phrases rather than single words

class ManifoldReducedState(NamedTuple):
    """State of the reduced manifold for visualization and cellular operations."""
    points: np.ndarray         # 2D or 3D points
    labels: List[int]          # Cell labels for each point
    cell_centroids: np.ndarray # Reduced centroids
    boundaries: Any            # Boundary representations (e.g., Voronoi)

class VectorManifold:
    """
    A class representing a manifold in vector space for word and phrase embeddings.
    
    This class handles the geometric relationships between embeddings,
    including Voronoi tessellation and neighborhood calculations.
    """
    
    def __init__(
        self,
        embeddings: Union[WordEmbeddings, PhraseEmbedding],
        n_cells: int = 22,  # Default to 22 cells (major arcana)
        random_state: int = 93,  # Occult significance
        reduction_dims: int = 3
    ):
        """
        Initialize the VectorManifold.
        
        Args:
            embeddings: The WordEmbeddings or PhraseEmbeddings instance to use
            n_cells: Number of cells to divide the manifold into
            random_state: Random seed for reproducibility
            reduction_dims: Dimensions for the reduced representation
        """
        self.embeddings = embeddings
        self.n_cells = n_cells
        self.random_state = random_state
        self.reduction_dims = reduction_dims
        
        # Check if embeddings are loaded
        if not self.embeddings.get_terms():
            raise ValueError("Embeddings must have terms loaded. Call embeddings.load_terms() first.")
        
        # Create matrices for vector operations
        self.terms = list(self.embeddings.get_terms())
        self.term_to_index = {term: i for i, term in enumerate(self.terms)}
        
        # Vector representation of all terms
        self.vectors = np.array([self.embeddings.get_embedding(term) for term in self.terms])
        
        # Initialize empty manifold structures
        self.cells: Dict[int, Cell] = {}
        self.cell_neighbors: Dict[int, List[int]] = {}
        self.term_to_cell: Dict[str, int] = {}
        
        # Reduced manifold for visualization and operations
        self.reduced: Optional[ManifoldReducedState] = None
        
        # Define the manifold structure
        self._define_manifold()
        
    def _define_manifold(self) -> None:
        """
        Define the manifold structure by clustering terms and establishing relationships.
        """
        logger.info("Defining manifold structure")
        
        # 1. Cluster terms to create cells
        self._cluster_terms()
        
        # 2. Reduce dimensions for visualization and operations
        self._reduce_dimensions()
        
        # 3. Define cell boundaries in reduced space
        self._define_cell_boundaries()
        
        # 4. Establish neighborhood relationships
        self._establish_neighborhoods()
        
        # 5. Assign occult typology and numerological values to cells
        self._assign_cell_properties()
        
        logger.info(f"Manifold defined with {len(self.cells)} cells")
        
    def _cluster_terms(self) -> None:
        """Cluster terms to define cells in the manifold."""
        # Adjust number of clusters based on number of terms
        n_clusters = min(self.n_cells, len(self.terms))
        if n_clusters < self.n_cells:
            logger.warning(f"Reducing number of clusters from {self.n_cells} to {n_clusters} due to insufficient terms")
            
        # Use KMeans clustering to partition the space
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init="auto",
            # ensure_all_finite=True  # Updated from force_all_finite
        )
        cluster_labels = kmeans.fit_predict(self.vectors)
        
        # Create cell structures
        for i in range(n_clusters):
            # Get terms in this cluster
            term_indices = np.where(cluster_labels == i)[0]
            cluster_terms = [self.terms[idx] for idx in term_indices]
            
            if not cluster_terms:
                logger.warning(f"Cluster {i} is empty, skipping")
                continue
            
            # Compute centroid (we use the KMeans centroid)
            centroid = kmeans.cluster_centers_[i]
            
            # Create cell and store mappings
            cell = Cell(
                id=i,
                terms=set(cluster_terms),
                centroid=centroid,
                type=CellType.OTHER,  # Will be assigned later
                numerological_value=0  # Will be assigned later
            )
            
            self.cells[i] = cell
            
            # Map terms to this cell
            for term in cluster_terms:
                self.term_to_cell[term] = i
                
    def _reduce_dimensions(self) -> None:
        """
        Reduce the dimensionality of the vector space for visualization and operations.
        """
        logger.info(f"Reducing manifold to {self.reduction_dims} dimensions")
        
        # We use UMAP for dimension reduction as it better preserves global structure
        reducer = umap.UMAP(
            n_components=self.reduction_dims, 
            random_state=self.random_state,
            n_neighbors=12,
            min_dist=0.01
        )
        
        # Reduce the term vectors
        reduced_vectors = reducer.fit_transform(self.vectors)
        
        # Also reduce cell centroids
        cell_centroids = np.array([cell.centroid for cell in self.cells.values()])
        reduced_centroids = reducer.transform(cell_centroids)
        
        # Create labels list matching the reduced points
        labels = [self.term_to_cell[term] for term in self.terms]
        
        # Store the reduced state
        self.reduced = ManifoldReducedState(
            points=reduced_vectors,
            labels=labels,
            cell_centroids=reduced_centroids,
            boundaries=[]  # Will be filled in _define_cell_boundaries
        )
        
        logger.info(f"Reduced manifold to {self.reduction_dims} dimensions")
        
    def _define_cell_boundaries(self) -> None:
        """Define boundaries between cells in the reduced space."""
        if self.reduced is None:
            raise ValueError("Reduced representation not available")
            
        try:
            # Compute Voronoi diagram of cell centroids
            self.voronoi = Voronoi(self.reduced.cell_centroids)
            
            # Update cell objects with boundary points if applicable
            for i, (cell_id, cell) in enumerate(self.cells.items()):
                # Get the region for this cell
                region_idx = self.voronoi.point_region[i]
                region = self.voronoi.regions[region_idx]
                
                # Skip if region is empty or contains -1 (unbounded)
                if -1 in region or not region:
                    continue
                    
                # Get boundary vertices
                boundary_points = self.voronoi.vertices[region]
                cell.boundary_points = boundary_points
                
        except Exception as e:
            logger.error(f"Failed to compute Voronoi tessellation: {e}")
            self.voronoi = None
        
    def get_neighbors(self, term: str) -> Set[str]:
        """
        Get the neighboring terms in the Voronoi tessellation.
        
        Args:
            term: Term to find neighbors for
            
        Returns:
            Set of neighboring terms
        """
        if not self.voronoi or term not in self.term_to_index:
            return set()
            
        idx = self.term_to_index[term]
        neighbors = set()
        
        # Find all regions that share a ridge with the given term's region
        for ridge_vertices in self.voronoi.ridge_points:
            if idx in ridge_vertices:
                other_idx = ridge_vertices[0] if ridge_vertices[1] == idx else ridge_vertices[1]
                neighbors.add(self.terms[other_idx])
                
        return neighbors
        
    def get_region_vertices(self, term: str) -> List[np.ndarray]:
        """
        Get the vertices of a term's Voronoi region.
        
        Args:
            term: Term to get region vertices for
            
        Returns:
            List of vertex coordinates
        """
        if not self.voronoi or term not in self.term_to_index:
            return []
            
        idx = self.term_to_index[term]
        region_idx = self.voronoi.point_region[idx]
        vertex_indices = self.voronoi.regions[region_idx]
        
        # Filter out any infinite vertices
        vertices = []
        for v_idx in vertex_indices:
            if v_idx >= 0:  # -1 indicates infinite vertex
                vertices.append(self.voronoi.vertices[v_idx])
                
        return vertices
        
    def find_path(self, start_term: str, end_term: str, max_steps: int = 10) -> List[str]:
        """
        Find a path between two terms through neighboring Voronoi regions.
        
        Args:
            start_term: Starting term
            end_term: Target term
            max_steps: Maximum number of steps in the path
            
        Returns:
            List of terms forming the path (including start and end terms)
        """
        if not self.voronoi:
            return []
            
        if start_term not in self.term_to_index or end_term not in self.term_to_index:
            return []
            
        # Use A* search to find the path
        from queue import PriorityQueue
        
        def heuristic(term: str) -> float:
            """Manhattan distance between term embeddings"""
            current = self.embeddings.get_embedding(term)
            target = self.embeddings.get_embedding(end_term)
            return np.sum(np.abs(current - target))
            
        frontier = PriorityQueue()
        frontier.put((0, start_term))
        came_from = {start_term: None}
        cost_so_far = {start_term: 0}
        
        while not frontier.empty():
            current = frontier.get()[1]
            
            if current == end_term:
                break
                
            if cost_so_far[current] >= max_steps:
                continue
                
            for next_term in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                
                if next_term not in cost_so_far or new_cost < cost_so_far[next_term]:
                    cost_so_far[next_term] = new_cost
                    priority = new_cost + heuristic(next_term)
                    frontier.put((priority, next_term))
                    came_from[next_term] = current
                    
        # Reconstruct path
        if end_term not in came_from:
            return []
            
        path = [end_term]
        current = end_term
        while current != start_term:
            current = came_from[current]
            path.append(current)
            
        path.reverse()
        return path
        
    def get_region_centroid(self, terms: List[str]) -> np.ndarray:
        """
        Calculate the centroid of a region defined by multiple terms.
        
        Args:
            terms: List of terms defining the region
            
        Returns:
            Centroid vector of the region
        """
        if not terms:
            return np.array([])
            
        embeddings = []
        for term in terms:
            if term in self.terms:
                embeddings.append(self.embeddings.get_embedding(term))
                
        if not embeddings:
            return np.array([])
            
        return np.mean(embeddings, axis=0)
        
    def find_term_in_direction(
        self,
        base_term: str,
        direction_term: str,
        step_size: float = 1.0
    ) -> Optional[str]:
        """
        Find the term in the direction of another term.
        
        Args:
            base_term: Starting term
            direction_term: Term defining the direction
            step_size: Size of the step to take
            
        Returns:
            Term found in the direction, or None if no suitable term found
        """
        if not self.voronoi or base_term not in self.terms:
            return None
            
        base_vec = self.embeddings.get_embedding(base_term)
        direction_vec = self.embeddings.get_embedding(direction_term)
        
        # Calculate direction vector
        direction = direction_vec - base_vec
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            return None
            
        # Normalize and scale
        direction = (direction / direction_norm) * step_size
        target_point = base_vec + direction
        
        # Find nearest term to target point
        min_dist = float('inf')
        nearest_term = None
        
        for term in self.terms:
            if term == base_term:
                continue
                
            term_vec = self.embeddings.get_embedding(term)
            dist = np.linalg.norm(term_vec - target_point)
            
            if dist < min_dist:
                min_dist = dist
                nearest_term = term
                
        return nearest_term
        
    def get_boundary_terms(self) -> Set[str]:
        """
        Get terms that lie on the boundary of the manifold.
        
        Returns:
            Set of terms on the boundary
        """
        if not self.voronoi:
            return set()
            
        boundary_terms = set()
        
        # A term is on the boundary if its Voronoi region has an infinite vertex
        for idx, region_idx in enumerate(self.voronoi.point_region):
            region = self.voronoi.regions[region_idx]
            if -1 in region:  # -1 indicates an infinite vertex
                term = self.terms[idx]
                boundary_terms.add(term)
                
        return boundary_terms
        
    def interpolate_terms(
        self,
        term1: str,
        term2: str,
        num_steps: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find terms that lie between two given terms.
        
        Args:
            term1: First term
            term2: Second term
            num_steps: Number of interpolation steps
            
        Returns:
            List of (term, distance) tuples along the interpolation path
        """
        if not self.voronoi or term1 not in self.terms or term2 not in self.terms:
            return []
            
        vec1 = self.embeddings.get_embedding(term1)
        vec2 = self.embeddings.get_embedding(term2)
        
        # Generate interpolated points
        alphas = np.linspace(0, 1, num_steps + 2)[1:-1]  # Exclude endpoints
        interpolated = []
        
        for alpha in alphas:
            point = vec1 * (1 - alpha) + vec2 * alpha
            
            # Find nearest term to interpolated point
            min_dist = float('inf')
            nearest_term = None
            
            for term in self.terms:
                if term in {term1, term2}:
                    continue
                    
                term_vec = self.embeddings.get_embedding(term)
                dist = np.linalg.norm(term_vec - point)
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_term = term
                    
            if nearest_term:
                interpolated.append((nearest_term, float(min_dist)))
                
        return interpolated
    
    def compute_cell_distance(
        self,
        cell_id1: int,
        cell_id2: int,
        distance_type: DistanceType = DistanceType.HYBRID
    ) -> float:
        """
        Compute the distance between two cells using specified distance metric.
        
        Args:
            cell_id1: First cell ID
            cell_id2: Second cell ID
            distance_type: Type of distance metric to use
            
        Returns:
            Distance value between the cells
        """
        if cell_id1 not in self.cells or cell_id2 not in self.cells:
            logger.warning(f"Cell ID {cell_id1} or {cell_id2} not found in manifold")
            return float('inf')
        
        cell1 = self.cells[cell_id1]
        cell2 = self.cells[cell_id2]
        
        # Get cell centroids
        centroid1 = cell1.centroid
        centroid2 = cell2.centroid
        
        if distance_type == DistanceType.EUCLIDEAN:
            return float(np.linalg.norm(centroid1 - centroid2))
            
        elif distance_type == DistanceType.COSINE:
            # Compute cosine similarity
            dot_product = np.dot(centroid1, centroid2)
            norm1 = np.linalg.norm(centroid1)
            norm2 = np.linalg.norm(centroid2)
            if norm1 == 0 or norm2 == 0:
                return 1.0  # Maximum distance for zero vectors
            return 1.0 - float(dot_product / (norm1 * norm2))
            
        elif distance_type == DistanceType.NUMEROLOGICAL:
            # Use numerological values to weight the distance
            num_diff = abs(cell1.numerological_value - cell2.numerological_value)
            
            # Master numbers have special significance
            master_numbers = [11, 22, 33]
            if cell1.numerological_value in master_numbers or cell2.numerological_value in master_numbers:
                factor = 0.7  # Reduce distance if either cell has a master number
            else:
                factor = 1.0
                
            # Base distance is Euclidean, modified by numerological significance
            base_distance = float(np.linalg.norm(centroid1 - centroid2))
            return base_distance * (1.0 + (num_diff / 9.0) * factor)
            
        elif distance_type == DistanceType.HYBRID:
            # Combine semantic and numerological distances
            cosine_dist = 1.0 - float(np.dot(centroid1, centroid2) / 
                                    (np.linalg.norm(centroid1) * np.linalg.norm(centroid2)))
            num_diff = abs(cell1.numerological_value - cell2.numerological_value)
            num_factor = 1.0 + (num_diff / 9.0)
            return cosine_dist * num_factor
            
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")

    def _establish_neighborhoods(self) -> None:
        """Establish neighborhood relationships between cells."""
        # Calculate distances between cell centroids
        cell_ids = list(self.cells.keys())
        cell_centroids = np.array([self.cells[cell_id].centroid for cell_id in cell_ids])
        
        # Use cosine distance for semantic relationships
        distances = cosine_distances(cell_centroids)
        
        # For each cell, find k nearest neighbors
        k = min(5, len(cell_ids) - 1)  # At most 5 neighbors
        for i, cell_id in enumerate(cell_ids):
            # Sort distances (excluding self)
            neighbor_indices = np.argsort(distances[i])[1:k+1]
            neighbor_ids = [cell_ids[idx] for idx in neighbor_indices]
            
            # Store neighbors
            self.cell_neighbors[cell_id] = neighbor_ids
            
    def _assign_cell_properties(self) -> None:
        """Assign occult typology and numerological values to cells."""
        # Define term sets for each cell type
        element_terms = {"earth", "air", "fire", "water"}
        planet_terms = {"mercury", "venus", "mars", "jupiter", "saturn", "sun", "moon"}
        zodiac_terms = {"aries", "taurus", "gemini", "cancer", "leo", "virgo",
                      "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"}
        tarot_terms = {"magician", "priestess", "empress", "emperor", "hierophant",
                     "lovers", "chariot", "strength", "hermit", "wheel", "justice", 
                     "hanged", "death", "temperance", "devil", "tower", "star", 
                     "moon", "sun", "judgement", "world", "fool"}
        sephiroth_terms = {"kether", "chokmah", "binah", "chesed", "geburah",
                         "tiphareth", "netzach", "hod", "yesod", "malkuth"}
        
        # Map terms to their numerological values
        numerological_values = {
            "earth": 4, "air": 1, "fire": 3, "water": 2,  # Elements
            "mercury": 8, "venus": 7, "mars": 5, "jupiter": 4, "saturn": 3, "sun": 1, "moon": 2,  # Planets
            "magician": 1, "priestess": 2, "empress": 3, "emperor": 4,  # Major arcana
            "hierophant": 5, "lovers": 6, "chariot": 7, "strength": 8,
            "hermit": 9, "wheel": 10, "justice": 11, "hanged": 12,
            "death": 13, "temperance": 14, "devil": 15, "tower": 16,
            "star": 17, "eighteen": 18, "nineteen": 19, "judgement": 20, "world": 21, "space": 0
        }
        
        for cell_id, cell in self.cells.items():
            # Check what type of terms are most common in this cell
            cell_term_set = set(t.lower() for t in cell.terms)
            
            element_count = len(cell_term_set.intersection(element_terms))
            planet_count = len(cell_term_set.intersection(planet_terms))
            zodiac_count = len(cell_term_set.intersection(zodiac_terms))
            tarot_count = len(cell_term_set.intersection(tarot_terms))
            sephiroth_count = len(cell_term_set.intersection(sephiroth_terms))
            
            # Find the dominant type
            counts = [
                (element_count, CellType.ELEMENTAL),
                (planet_count, CellType.PLANETARY),
                (zodiac_count, CellType.ZODIACAL),
                (tarot_count, CellType.TAROT),
                (sephiroth_count, CellType.SEPHIROTIC)
            ]
            
            max_count, cell_type = max(counts, key=lambda x: x[0])
            
            # If no clear type is found, use OTHER
            if max_count == 0:
                cell_type = CellType.OTHER
            
            # Calculate numerological value based on contained terms
            term_values = []
            for term in cell_term_set:
                if term in numerological_values:
                    term_values.append(numerological_values[term])
            
            # Use average of term values or cell_id + 1 as fallback
            if term_values:
                num_value = int(np.mean(term_values))
            else:
                num_value = (cell_id + 1) % 22  # Keep within 0-21 range
            
            # Update cell properties
            self.cells[cell_id] = Cell(
                id=cell_id,
                terms=cell.terms,
                centroid=cell.centroid,
                type=cell_type,
                numerological_value=num_value,
                boundary_points=cell.boundary_points,
                is_phrase=cell.is_phrase
            )

    def get_term_cell(self, term: str) -> Optional[Cell]:
        """
        Get the cell that contains a specific term.
        
        Args:
            term: The term to find the cell for
            
        Returns:
            The Cell that contains this term, or None if not found
        """
        if term not in self.term_to_cell:
            logger.warning(f"Term '{term}' not found in manifold")
            return None
            
        cell_id = self.term_to_cell[term]
        return self.cells.get(cell_id)
    
    def add_cell(
        self,
        terms: List[str],
        cell_type: CellType = CellType.OTHER,
        numerological_value: Optional[int] = None,
        is_phrase: bool = False
    ) -> Optional[int]:
        """
        Add a new cell to the manifold.
        
        Args:
            terms: List of terms (words or phrases) to include in the cell
            cell_type: Type of the cell (default: OTHER)
            numerological_value: Optional numerological value for the cell
                               If None, will be calculated from terms
            is_phrase: Whether the terms are phrases rather than single words
        
        Returns:
            ID of the new cell if successful, None if failed
        """
        # Validate terms
        valid_terms = [term for term in terms if term in self.embeddings.get_terms()]
        if not valid_terms:
            logger.warning("No valid terms provided for new cell")
            return None
            
        # Calculate cell ID (use next available number)
        cell_id = max(self.cells.keys(), default=-1) + 1
        
        # Get embeddings for terms and calculate centroid
        term_vectors = [self.embeddings.get_embedding(term) for term in valid_terms]
        centroid = np.mean(term_vectors, axis=0)
        
        # Calculate numerological value if not provided
        if numerological_value is None:
            # Use average of first 5 terms (or all if less than 5)
            terms_to_use = valid_terms[:min(5, len(valid_terms))]
            num_values = [self.embeddings.find_numerological_significance(term) 
                         for term in terms_to_use]
            numerological_value = int(np.mean(num_values))
            
            # Check for master numbers
            master_numbers = [11, 22, 33]
            for term in terms_to_use:
                value = self.embeddings.find_numerological_significance(term)
                if value in master_numbers:
                    numerological_value = value
                    break
        
        # Create the new cell
        new_cell = Cell(
            id=cell_id,
            terms=set(valid_terms),
            centroid=centroid,
            type=cell_type,
            numerological_value=numerological_value,
            is_phrase=is_phrase
        )
        
        # Add cell to manifold
        self.cells[cell_id] = new_cell
        
        # Update term mappings
        for term in valid_terms:
            # If term was in another cell, remove it
            old_cell_id = self.term_to_cell.get(term)
            if old_cell_id is not None and old_cell_id != cell_id:
                old_cell = self.cells.get(old_cell_id)
                if old_cell and term in old_cell.terms:
                    old_cell.terms.remove(term)
            
            self.term_to_cell[term] = cell_id
        
        # Update reduced representation if it exists
        if self.reduced is not None:
            try:
                # Create UMAP reducer with same parameters
                reducer = umap.UMAP(
                    n_components=self.reduction_dims,
                    random_state=self.random_state,
                    n_neighbors=15,
                    min_dist=0.1
                )
                
                # Get all points including new cell
                all_points = np.array([cell.centroid for cell in self.cells.values()])
                
                # Reduce all points
                reduced_points = reducer.fit_transform(all_points)
                
                # Update reduced state
                self.reduced = ManifoldReducedState(
                    points=reduced_points,
                    labels=[cell.id for cell in self.cells.values()],
                    cell_centroids=reduced_points,
                    boundaries=[]  # Will be updated in _define_cell_boundaries
                )
                
                # Redefine cell boundaries with new cell
                self._define_cell_boundaries()
                
            except Exception as e:
                logger.error(f"Failed to update reduced representation: {e}")
        
        # Update neighborhood relationships
        self._establish_neighborhoods()
        
        logger.info(f"Added new cell {cell_id} with {len(valid_terms)} terms")
        return cell_id
    
    def get_manifold_state(self) -> Dict[str, Any]:
        """Get the current state of the manifold."""
        return {
            'cells': self.cells,
            'neighbors': self.cell_neighbors,
            'term_mapping': self.term_to_cell,
            'reduced_state': self.reduced
        }
        
    def transform(self, vectors: np.ndarray, evolution_rules: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Transform vectors through the manifold's semantic space based on evolution rules.
        
        This method applies semantic transformations to input vectors based on:
        1. Cell influence - vectors are influenced by their nearest cell's properties
        2. Numerological weighting - transformations are weighted by numerological values
        3. Neighborhood effects - nearby cells influence the transformation
        4. Hermetic principles - transformations follow occult correspondences
        5. Rule sequences - support for chained transformations with dependencies
        6. Platonic ideals - inference and attraction to ideal forms
        
        Args:
            vectors: Input vectors to transform, shape (n_vectors, n_dimensions)
            evolution_rules: Dictionary containing transformation parameters:
                - transformation: Type of transformation ('numerological', 'align', 'contrast', 'repel')
                - magnitude: Base magnitude of transformation
                - cell_type_weights: Weights for different cell types
                - numerological_weights: Weights for numerological values
                - sequence_info: Optional dict with sequence parameters:
                    - dependencies: List of rule names this transformation depends on
                    - conditions: Dict of conditions that must be met
                    - platonic_ideal: Whether to infer and use platonic ideals
            
        Returns:
            Transformed vectors with same shape as input
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
            
        if vectors.shape[1] != self.vectors.shape[1]:
            raise ValueError(f"Input vectors must have same dimensionality as manifold vectors. "
                           f"Expected {self.vectors.shape[1]}, got {vectors.shape[1]}")
            
        # Initialize transformed vectors
        transformed = vectors.copy()
        
        # Get transformation parameters
        if evolution_rules is None:
            evolution_rules = {}
            
        transform_type = evolution_rules.get('transformation', 'numerological')
        base_magnitude = evolution_rules.get('magnitude', 1.0)
        cell_type_weights = evolution_rules.get('cell_type_weights', {})
        numerological_weights = evolution_rules.get('numerological_weights', {})
        sequence_info = evolution_rules.get('sequence_info', {})
        
        # Check sequence dependencies if part of a sequence
        if sequence_info and sequence_info.get('dependencies'):
            for dep_rule in sequence_info['dependencies']:
                if not self._check_rule_applied(dep_rule):
                    logger.warning(f"Dependency {dep_rule} not met for transformation")
                    return transformed
                    
        # Check sequence conditions if specified
        if sequence_info and sequence_info.get('conditions'):
            if not self._evaluate_conditions(sequence_info['conditions']):
                logger.warning("Sequence conditions not met for transformation")
                return transformed
                
        # Infer platonic ideals if requested
        platonic_ideals = None
        if sequence_info.get('platonic_ideal', False):
            platonic_ideals = self._infer_platonic_ideals(vectors)
            
        # For each input vector
        for i in range(vectors.shape[0]):
            # Find nearest cell and neighbors
            distances = euclidean_distances(vectors[i].reshape(1, -1), 
                                         np.array([cell.centroid for cell in self.cells.values()]))
            nearest_cell_idx = np.argmin(distances)
            nearest_cell = self.cells[nearest_cell_idx]
            neighbor_cells = [self.cells[n] for n in self.cell_neighbors.get(nearest_cell_idx, [])]
            
            # Calculate numerological influence
            num_value = nearest_cell.numerological_value
            num_weight = numerological_weights.get(num_value, 1.0)
            
            # Calculate cell type influence
            type_weight = cell_type_weights.get(nearest_cell.type, 1.0)
            
            # Combined weight factor
            weight_factor = base_magnitude * num_weight * type_weight
            
            # Apply transformation based on type
            if transform_type == 'numerological':
                # Transform based on numerological correspondences
                direction = nearest_cell.centroid - vectors[i]
                scale = (num_value / 22.0) * weight_factor
                transformed[i] += direction * scale
                
            elif transform_type == 'align':
                # Move towards semantic consensus with neighbors
                if neighbor_cells:
                    neighbor_vectors = np.array([cell.centroid for cell in neighbor_cells])
                    consensus = np.mean(neighbor_vectors, axis=0)
                    direction = consensus - vectors[i]
                    transformed[i] += direction * weight_factor
                    
            elif transform_type == 'contrast':
                # Move away from semantic neighbors to increase distinction
                if neighbor_cells:
                    neighbor_vectors = np.array([cell.centroid for cell in neighbor_cells])
                    repulsion = vectors[i] - np.mean(neighbor_vectors, axis=0)
                    repulsion_norm = np.linalg.norm(repulsion)
                    if repulsion_norm > 0:
                        repulsion = repulsion / repulsion_norm
                    transformed[i] += repulsion * weight_factor
                    
            elif transform_type == 'repel':
                # Active repulsion from nearest cell
                direction = vectors[i] - nearest_cell.centroid
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                transformed[i] += direction * weight_factor
                
            # Apply platonic ideal influence if available
            if platonic_ideals is not None:
                ideal_vector = platonic_ideals[i]
                ideal_direction = ideal_vector - vectors[i]
                ideal_weight = 0.3  # Moderate influence from ideal forms
                transformed[i] += ideal_direction * ideal_weight * weight_factor
                
            # Apply neighbor influence based on numerological values
            if neighbor_cells:
                neighbor_weights = np.array([
                    numerological_weights.get(cell.numerological_value, 1.0) 
                    for cell in neighbor_cells
                ])
                neighbor_weights /= np.sum(neighbor_weights)
                
                neighbor_vectors = np.array([cell.centroid for cell in neighbor_cells])
                weighted_influence = np.average(
                    neighbor_vectors - vectors[i],
                    weights=neighbor_weights,
                    axis=0
                )
                transformed[i] += weighted_influence * weight_factor * 0.3
            
            # Normalize the vector
            transformed[i] /= np.linalg.norm(transformed[i])
            transformed[i] *= np.linalg.norm(vectors[i])
            
        return transformed

    def _check_rule_applied(self, rule_name: str) -> bool:
        """Check if a rule has been applied in the current sequence."""
        # Implementation depends on how rule application history is tracked
        # For now, return True to avoid blocking transformations
        return True
        
    def _evaluate_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Evaluate conditions for rule application."""
        # Example condition evaluation - extend based on needs
        for condition, value in conditions.items():
            if condition == "min_cells":
                if len(self.cells) < value:
                    return False
            elif condition == "max_distance":
                # Check if any cells are too far apart
                centroids = np.array([cell.centroid for cell in self.cells.values()])
                max_dist = np.max(euclidean_distances(centroids))
                if max_dist > value:
                    return False
        return True
        
    def _infer_platonic_ideals(self, vectors: np.ndarray, method: str = 'pca', n_ideals: int = 5) -> np.ndarray:
        """
        Infer platonic ideal forms for input vectors using various methods.
        
        This method attempts to find the "perfect" or "ideal" forms that the
        input vectors might be approximating, based on the assumption that
        platonic ideals exist as attractors in the semantic space.
        
        Methods:
            - 'pca': Uses PCA to find principal components as ideals
            - 'archetypal': Uses archetypal analysis to find extreme points
            - 'geometric': Uses geometric analysis to find regular polytope vertices
        
        Args:
            vectors: Input vectors to find ideals for
            method: Method to use for ideal inference
            n_ideals: Number of ideal forms to infer
            
        Returns:
            Array of inferred ideal vectors
        """
        if method == 'pca':
            return self._infer_ideals_pca(vectors, n_ideals)
        elif method == 'archetypal':
            return self._infer_ideals_archetypal(vectors, n_ideals)
        elif method == 'geometric':
            return self._infer_ideals_geometric(vectors, n_ideals)
        else:
            raise ValueError(f"Unknown ideal inference method: {method}")
            
    def _infer_ideals_pca(self, vectors: np.ndarray, n_ideals: int) -> np.ndarray:
        """Infer ideals using PCA and clustering."""
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        
        # Number of clusters (ideal forms) to infer
        n_clusters = min(n_ideals, len(vectors))
        
        # Cluster vectors to find common patterns
        kmeans = KMeans(n_clusters=n_clusters, random_state=93)
        cluster_labels = kmeans.fit_predict(vectors)
        
        # For each cluster, find the principal components
        ideal_vectors = np.zeros_like(vectors)
        
        for i in range(n_clusters):
            cluster_vectors = vectors[cluster_labels == i]
            if len(cluster_vectors) > 0:
                # Use PCA to find the dominant direction in this cluster
                pca = PCA(n_components=1)
                pca.fit(cluster_vectors)
                
                # Project the cluster centroid onto the principal component
                centroid = np.mean(cluster_vectors, axis=0)
                ideal = pca.inverse_transform(pca.transform([centroid]))[0]
                
                # Assign this ideal to all vectors in the cluster
                ideal_vectors[cluster_labels == i] = ideal
                
        return ideal_vectors
        
    def _infer_ideals_archetypal(self, vectors: np.ndarray, n_ideals: int) -> np.ndarray:
        """Infer ideals using archetypal analysis."""
        from sklearn.decomposition import NMF
        
        # Use NMF to find archetypal vectors
        n_archetypes = min(n_ideals, len(vectors))
        nmf = NMF(n_components=n_archetypes, init='nndsvdar', random_state=93)
        
        # Normalize vectors to non-negative space
        vectors_shifted = vectors - np.min(vectors)
        
        # Find archetypal components
        W = nmf.fit_transform(vectors_shifted)  # Weights
        H = nmf.components_  # Archetypal vectors
        
        # Assign each vector to nearest archetype
        assignments = np.argmax(W, axis=1)
        ideal_vectors = np.zeros_like(vectors)
        
        for i in range(len(vectors)):
            # Shift archetype back to original space
            ideal = H[assignments[i]] + np.min(vectors)
            ideal_vectors[i] = ideal
            
        return ideal_vectors
        
    def _infer_ideals_geometric(self, vectors: np.ndarray, n_ideals: int) -> np.ndarray:
        """Infer ideals using geometric analysis of vector space structure."""
        from scipy.spatial import ConvexHull
        
        # First reduce dimensionality to 3D for geometric analysis
        if vectors.shape[1] > 3:
            pca = PCA(n_components=3)
            vectors_3d = pca.fit_transform(vectors)
        else:
            vectors_3d = vectors
            
        # Find convex hull of points
        hull = ConvexHull(vectors_3d)
        
        # Use vertices of convex hull as initial ideal candidates
        vertices = vectors_3d[hull.vertices]
        
        # Find most regular arrangement of n_ideals points
        if n_ideals <= 4:
            # Use regular simplex vertices
            ideal_points = self._generate_regular_simplex(n_ideals)
        elif n_ideals <= 8:
            # Use regular polytope vertices
            ideal_points = self._generate_regular_polytope(n_ideals)
        else:
            # Use fibonacci spiral points on sphere
            ideal_points = self._generate_fibonacci_points(n_ideals)
            
        # Scale and center ideal points to match data
        scale = np.std(vectors_3d, axis=0)
        center = np.mean(vectors_3d, axis=0)
        ideal_points = (ideal_points * scale) + center
        
        # Project back to original space if needed
        if vectors.shape[1] > 3:
            ideal_points = pca.inverse_transform(ideal_points)
            
        # Assign each vector to nearest ideal point
        distances = euclidean_distances(vectors, ideal_points)
        assignments = np.argmin(distances, axis=1)
        
        ideal_vectors = ideal_points[assignments]
        return ideal_vectors
        
    def _generate_regular_simplex(self, n_points: int) -> np.ndarray:
        """Generate vertices of a regular simplex."""
        if n_points == 1:
            return np.array([[0, 0, 0]])
        elif n_points == 2:
            return np.array([[1, 0, 0], [-1, 0, 0]])
        elif n_points == 3:
            return np.array([[1, 0, 0], [-0.5, 0.866, 0], [-0.5, -0.866, 0]])
        elif n_points == 4:
            return np.array([[1, 0, 0], [-0.333, 0.943, 0],
                           [-0.333, -0.471, 0.816], [-0.333, -0.471, -0.816]])
        else:
            raise ValueError("Regular simplex only supported for 1-4 points")
            
    def _generate_regular_polytope(self, n_points: int) -> np.ndarray:
        """Generate vertices of a regular polytope."""
        if n_points == 6:
            # Octahedron vertices
            return np.array([[1,0,0], [-1,0,0], [0,1,0],
                           [0,-1,0], [0,0,1], [0,0,-1]])
        elif n_points == 8:
            # Cube vertices
            points = []
            for x in [-1, 1]:
                for y in [-1, 1]:
                    for z in [-1, 1]:
                        points.append([x, y, z])
            return np.array(points)
        else:
            raise ValueError("Regular polytope only supported for 6 or 8 points")
            
    def _generate_fibonacci_points(self, n_points: int) -> np.ndarray:
        """Generate evenly distributed points on a sphere using fibonacci spiral."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = 2 * np.pi * i / phi  # Golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
            
        return np.array(points)

    def evolve_manifold(self, vectors: np.ndarray, evolution_rules: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Alias for transform() to maintain compatibility with existing tests.
        See transform() for full documentation.
        """
        return self.transform(vectors, evolution_rules)

    def track_ideal_evolution(self, vectors: np.ndarray, n_steps: int = 10, 
                              method: str = 'pca', n_ideals: int = 5) -> Dict[str, Any]:
        """
        Track the evolution of vectors towards their platonic ideals over multiple steps.
        
        Args:
            vectors: Input vectors to track
            n_steps: Number of evolution steps
            method: Method to use for ideal inference
            n_ideals: Number of ideal forms to infer
            
        Returns:
            Dictionary containing:
                - evolution_path: Array of shape (n_steps, n_vectors, n_dims)
                - ideal_convergence: Array of shape (n_steps, n_vectors)
                - final_ideals: Array of shape (n_ideals, n_dims)
        """
        n_vectors = len(vectors)
        n_dims = vectors.shape[1]
        
        # Initialize tracking arrays
        evolution_path = np.zeros((n_steps + 1, n_vectors, n_dims))
        evolution_path[0] = vectors
        
        ideal_convergence = np.zeros((n_steps + 1, n_vectors))
        
        # Get initial ideals
        current_ideals = self._infer_platonic_ideals(vectors, method, n_ideals)
        final_ideals = current_ideals.copy()
        
        # Track evolution
        for step in range(n_steps):
            # Evolve vectors towards ideals
            evolved = self._evolve_towards_ideals(evolution_path[step], current_ideals)
            evolution_path[step + 1] = evolved
            
            # Update ideals if needed
            if step < n_steps - 1:  # Don't update on final step
                current_ideals = self._infer_platonic_ideals(evolved, method, n_ideals)
            
            # Calculate convergence metrics
            ideal_convergence[step + 1] = self._calculate_ideal_convergence(
                evolved, final_ideals)
            
        return {
            'evolution_path': evolution_path,
            'ideal_convergence': ideal_convergence,
            'final_ideals': final_ideals
        }
        
    def _evolve_towards_ideals(self, vectors: np.ndarray, 
                              ideals: np.ndarray, 
                              rate: float = 0.1) -> np.ndarray:
        """Evolve vectors one step towards their assigned ideals."""
        # Find nearest ideal for each vector
        distances = euclidean_distances(vectors, ideals)
        nearest_ideal_idx = np.argmin(distances, axis=1)
        
        # Move vectors towards their nearest ideals
        evolved = vectors.copy()
        for i in range(len(vectors)):
            direction = ideals[nearest_ideal_idx[i]] - vectors[i]
            evolved[i] += rate * direction
            
        return evolved
        
    def _calculate_ideal_convergence(self, vectors: np.ndarray, 
                                   ideals: np.ndarray) -> np.ndarray:
        """Calculate how close each vector is to its nearest ideal."""
        distances = euclidean_distances(vectors, ideals)
        min_distances = np.min(distances, axis=1)
        
        # Normalize to [0, 1] range where 1 means perfect convergence
        max_dist = np.max(distances)
        convergence = 1 - (min_distances / max_dist)
        
        return convergence
        
    def analyze_ideal_stability(self, vectors: np.ndarray, 
                              n_perturbations: int = 10,
                              perturbation_scale: float = 0.1,
                              method: str = 'pca',
                              n_ideals: int = 5) -> Dict[str, Any]:
        """
        Analyze the stability of inferred platonic ideals under perturbations.
        
        Args:
            vectors: Input vectors
            n_perturbations: Number of perturbation trials
            perturbation_scale: Scale of random perturbations
            method: Method to use for ideal inference
            n_ideals: Number of ideal forms to infer
            
        Returns:
            Dictionary containing:
                - ideal_stability: Average cosine similarity between perturbed ideals
                - perturbation_sensitivity: How much ideals change with perturbation scale
                - stable_ideals: Most stable ideal vectors found
        """
        n_dims = vectors.shape[1]
        
        # Get baseline ideals
        baseline_ideals = self._infer_platonic_ideals(vectors, method, n_ideals)
        
        # Initialize tracking arrays
        all_ideals = np.zeros((n_perturbations, n_ideals, n_dims))
        stabilities = np.zeros(n_perturbations)
        
        # Run perturbation trials
        for i in range(n_perturbations):
            # Add random perturbations to vectors
            noise = np.random.normal(0, perturbation_scale, vectors.shape)
            perturbed = vectors + noise
            
            # Get ideals for perturbed vectors
            perturbed_ideals = self._infer_platonic_ideals(perturbed, method, n_ideals)
            all_ideals[i] = perturbed_ideals
            
            # Calculate stability metric
            stabilities[i] = self._calculate_ideal_similarity(
                baseline_ideals, perturbed_ideals)
            
        # Find most stable ideals across all trials
        stable_ideals = np.mean(all_ideals, axis=0)
        
        # Calculate sensitivity to perturbation scale
        sensitivity = 1 - np.mean(stabilities)
        
        return {
            'ideal_stability': np.mean(stabilities),
            'perturbation_sensitivity': sensitivity,
            'stable_ideals': stable_ideals
        }
        
    def _calculate_ideal_similarity(self, ideals1: np.ndarray, 
                                  ideals2: np.ndarray) -> float:
        """Calculate average cosine similarity between two sets of ideal vectors."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(ideals1, ideals2)
        
        # Take average of best matches
        best_matches = np.max(similarities, axis=1)
        return np.mean(best_matches)

    def _create_cell_visualization(self, cell: Cell) -> Dict[str, Any]:
        """Create visualization properties for a cell.
        
        Args:
            cell: The cell to create visualization properties for
            
        Returns:
            Dictionary containing visualization properties:
            - color: Base color for the cell
            - texture: Texture pattern based on cell type
            - glow: Glow effect parameters
            - flow_lines: Flow line coordinates
        """
        # Generate base color from cell properties
        hue = cell.numerological_value / 22.0  # Normalize to [0,1]
        saturation = 0.7 + (len(cell.terms) / 20.0) * 0.3  # More terms = more saturated
        value = 0.5 + (cell.type.value / len(CellType)) * 0.5  # Cell type influences brightness
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Create texture pattern based on cell type
        texture = self._create_cell_texture(cell)
        
        # Generate glow effect parameters
        glow = {
            'inner_color': rgb_color,
            'outer_color': tuple(c * 0.7 for c in rgb_color),
            'intensity': cell.numerological_value / 11.0  # Master numbers glow more
        }
        
        # Generate flow lines showing semantic relationships
        flow_lines = self._create_cell_flow_lines(cell)
        
        return {
            'color': rgb_color,
            'texture': texture,
            'glow': glow,
            'flow_lines': flow_lines
        }
    
    def _create_cell_texture(self, cell: Cell) -> np.ndarray:
        """Create a texture pattern for a cell based on its type and properties."""
        size = 64  # Texture size
        texture = np.zeros((size, size))
        
        if cell.type == CellType.ELEMENTAL:
            # Create elemental patterns (waves, flames, etc.)
            x = np.linspace(0, 4*np.pi, size)
            y = np.linspace(0, 4*np.pi, size)
            X, Y = np.meshgrid(x, y)
            texture = np.sin(X) * np.cos(Y)
            
        elif cell.type == CellType.PLANETARY:
            # Create circular/orbital patterns
            center = size // 2
            for i in range(size):
                for j in range(size):
                    r = np.sqrt((i - center)**2 + (j - center)**2)
                    texture[i,j] = np.sin(r / 5)
                    
        elif cell.type == CellType.ZODIACAL:
            # Create zodiacal symbols pattern
            theta = np.linspace(0, 2*np.pi, size)
            r = np.linspace(0, 1, size)
            R, T = np.meshgrid(r, theta)
            texture = np.sin(12*T) * R  # 12 divisions for zodiac
            
        elif cell.type == CellType.TAROT:
            # Create mystical geometry pattern
            texture = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    x = 2 * (i/size - 0.5)
                    y = 2 * (j/size - 0.5)
                    texture[i,j] = np.sin(x*y*np.pi)
                    
        elif cell.type == CellType.SEPHIROTIC:
            # Create tree-like pattern
            x = np.linspace(-2, 2, size)
            y = np.linspace(-2, 2, size)
            X, Y = np.meshgrid(x, y)
            texture = np.sin(X*Y) * np.exp(-(X**2 + Y**2)/8)
            
        # Normalize to [0,1] range
        texture = (texture - texture.min()) / (texture.max() - texture.min())
        return texture
    
    def _create_cell_flow_lines(self, cell: Cell) -> np.ndarray:
        """Create flow lines showing semantic relationships between terms in the cell."""
        # Get embeddings for terms in the cell
        term_vectors = np.array([self.embeddings.get_embedding(term) for term in cell.terms])
        
        # Calculate principal directions of flow
        if len(term_vectors) > 1:
            pca = PCA(n_components=2)
            flow_directions = pca.fit_transform(term_vectors)
        else:
            # For single-term cells, use random orthogonal directions
            flow_directions = np.random.randn(1, 2)
            
        # Generate flow line points
        n_lines = min(10, len(cell.terms))
        points_per_line = 20
        
        flow_lines = []
        for i in range(n_lines):
            # Start from cell boundary
            if cell.boundary_points is not None:
                start = cell.boundary_points[i % len(cell.boundary_points)]
            else:
                angle = 2 * np.pi * i / n_lines
                start = cell.centroid + np.array([np.cos(angle), np.sin(angle)])
            
            # Create flow line
            line = [start]
            point = start.copy()
            
            # Follow flow field
            for _ in range(points_per_line):
                # Get flow direction at current point
                if len(flow_directions) > 1:
                    direction = flow_directions[i % len(flow_directions)]
                else:
                    direction = flow_directions[0]
                    
                # Move point along flow
                point = point + 0.1 * direction
                line.append(point.copy())
                
            flow_lines.append(line)
            
        return np.array(flow_lines)
    
    def visualize_manifold(
        self,
        ax: Optional[plt.Axes] = None,
        show_flow: bool = True,
        show_boundaries: bool = True,
        show_labels: bool = True
    ) -> plt.Axes:
        """
        Create a visualization of the manifold structure.
        
        Args:
            ax: Optional matplotlib axes to plot on
            show_flow: Whether to show semantic flow lines
            show_boundaries: Whether to show cell boundaries
            show_labels: Whether to show cell labels
            
        Returns:
            The matplotlib axes object with the visualization
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            
        # Set background
        ax.set_facecolor('black')
        
        # Plot cells
        for cell in self.cells.values():
            # Get visualization properties
            vis_props = self._create_cell_visualization(cell)
            
            # Plot cell region
            if cell.boundary_points is not None and show_boundaries:
                # Create polygon patch
                path = Path(cell.boundary_points, closed=True)
                patch = PathPatch(
                    path, 
                    facecolor=vis_props['color'],
                    alpha=0.3,
                    edgecolor='white',
                    linewidth=0.5
                )
                ax.add_patch(patch)
                
                # Add glow effect
                for i in range(3):
                    glow_patch = PathPatch(
                        path,
                        facecolor=vis_props['glow']['outer_color'],
                        alpha=0.1 * (3-i),
                        edgecolor='none',
                        linewidth=0
                    )
                    ax.add_patch(glow_patch)
            
            # Plot flow lines
            if show_flow:
                flow_lines = vis_props['flow_lines']
                for line in flow_lines:
                    ax.plot(
                        line[:, 0], line[:, 1],
                        color=vis_props['color'],
                        alpha=0.3,
                        linewidth=0.5
                    )
            
            # Add labels
            if show_labels:
                # Show cell type and numerological value
                label = f"{cell.type.name}\n({cell.numerological_value})"
                ax.text(
                    cell.centroid[0], cell.centroid[1],
                    label,
                    color='white',
                    ha='center',
                    va='center',
                    fontsize=8,
                    bbox=dict(
                        facecolor='black',
                        alpha=0.7,
                        edgecolor='none'
                    )
                )
        
        # Set axis properties
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        return ax
