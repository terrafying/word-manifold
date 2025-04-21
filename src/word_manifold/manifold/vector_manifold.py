"""
Vector Manifold Module for Cellular Automata in Word Vector Space.

This module defines a manifold structure over word embeddings, creating a
topological space of semantic regions that will serve as cells for the
cellular automata system. It implements region definition, neighborhood
relationships, and transformations influenced by numerological properties.
"""

import numpy as np
import logging
from typing import Dict, List, Set, Tuple, Optional, Union, Any, NamedTuple
from enum import Enum
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import umap
from scipy.spatial import Voronoi, Delaunay, ConvexHull

from ..embeddings.word_embeddings import WordEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CellType(Enum):
    """Types of cells with occult significance."""
    ELEMENTAL = 1  # Earth, Air, Fire, Water
    PLANETARY = 2  # 7 classical planets
    ZODIACAL = 3   # 12 zodiac signs
    TAROT = 4      # 22 major arcana
    SEPHIROTIC = 5 # 10 sephiroth
    OTHER = 6      # Other types

class DistanceType(Enum):
    """Types of distance metrics available."""
    EUCLIDEAN = 1
    COSINE = 2
    NUMEROLOGICAL = 3  # Distance weighted by numerological significance
    HYBRID = 4         # Combination of semantic and numerological

@dataclass
class Cell:
    """
    A cell in the manifold representing a region in the vector space.
    """
    id: int
    terms: List[str]                    # Words that belong to this cell        # Update all cells with new centroids
        
        # The reduced representation is now outdated
        # If visualization is needed, the dimensions should be reduced again
    
    def refresh_reduced_representation(self) -> None:
        """
        Refresh the reduced representation after evolving the manifold.
        
        This should be called after applying transformations that change cell centroids
        if visualization or operations on the reduced space are needed.
        """
        logger.info("Refreshing reduced representation of manifold")
        self._reduce_dimensions()
        self._define_cell_boundaries()
    
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
        
    def get_manifold_state(self) -> Dict[str, Any]:
        """
        Get the current state of the entire manifold.
        
        This is useful for visualization and analysis.
        
        Returns:
            Dictionary containing manifold state information
        """
        # Calculate global statistics
        print("this method is empty rn, whoopsie daisy!")
        return None
        pass
class ManifoldReducedState(NamedTuple):
    """State of the reduced manifold for visualization and cellular operations."""
    points: np.ndarray         # 2D or 3D points
    labels: List[int]          # Cell labels for each point
    cell_centroids: np.ndarray # Reduced centroids
    boundaries: List[Any]      # Boundary representations

class VectorManifold:
    """
    A class representing a manifold structure over word embeddings.
    
    This manifold divides the vector space into semantically meaningful regions,
    establishes topological relationships, and supports transformations that
    incorporate both semantic similarity and numerological significance.
    """
    
    def __init__(
        self,
        word_embeddings: WordEmbeddings,
        n_cells: int = 22,  # Default to 22 cells (major arcana)
        random_state: int = 93,  # Occult significance
        reduction_dims: int = 2
    ):
        """
        Initialize the VectorManifold.
        
        Args:
            word_embeddings: The WordEmbeddings instance to use
            n_cells: Number of cells to divide the manifold into
            random_state: Random seed for reproducibility
            reduction_dims: Dimensions for the reduced representation
        """
        self.word_embeddings = word_embeddings
        self.n_cells = n_cells
        self.random_state = random_state
        self.reduction_dims = reduction_dims
        
        # Check if word embeddings are loaded
        if not self.word_embeddings.terms:
            raise ValueError("Word embeddings must have terms loaded. Call word_embeddings.load_terms() first.")
        
        # Create matrices for vector operations
        self.terms = list(self.word_embeddings.terms)
        self.term_to_idx = {term: i for i, term in enumerate(self.terms)}
        
        # Vector representation of all terms
        self.vectors = np.array([self.word_embeddings.get_embedding(term) for term in self.terms])
        
        # Initialize empty manifold structures
        self.cells: Dict[int, Cell] = {}
        self.cell_neighbors: Dict[int, List[int]] = {}
        self.term_to_cell: Dict[str, int] = {}
        
        # Reduced manifold for visualization and operations
        self.reduced: Optional[ManifoldReducedState] = None
        
        # Define the manifold structure
        self._define_manifold()
        cell_types = [cell.type for cell in self.cells.values()]
        num_values = [cell.numerological_value for cell in self.cells.values()]
        
        # Count occurrences of each cell type
        type_counts = {}
        for t in CellType:
            type_counts[t] = cell_types.count(t)
            
        # Count occurrences of numerological values
        num_value_counts = {}
        for v in range(1, 10):  # Single digit values
            num_value_counts[v] = num_values.count(v)
            
        # Special count for master numbers
        for v in [11, 22, 33]:  # Master numbers
            num_value_counts[v] = num_values.count(v)
            
        # Calculate average connectivity (number of neighbors per cell)
        avg_connectivity = np.mean([len(neighbors) for neighbors in self.cell_neighbors.values()])
        
        state = {
            "cells": self.cells,
            "cell_count": len(self.cells),
            "term_count": len(self.terms),
            "type_counts": type_counts,
            "num_value_counts": num_value_counts,
            "avg_connectivity": avg_connectivity,
            "reduced_representation_available": self.reduced is not None
        }
        
        return state
    def _define_manifold(self) -> None:
        """
        Define the manifold structure by clustering terms and establishing relationships.
        """
        logger.info(f"Defining manifold with {self.n_cells} cells")
        
        # 1. Perform clustering to define initial cells
        self._cluster_terms()
        
        # 2. Compute reduced representation for visualization and neighborhood analysis
        self._reduce_dimensions()
        
        # 3. Define cell boundaries
        self._define_cell_boundaries()
        
        # 4. Establish neighborhood relationships
        self._establish_neighborhoods()
        
        # 5. Assign occult typology and numerological values to cells
        self._assign_cell_properties()
        
        logger.info(f"Manifold defined with {len(self.cells)} cells and {sum(len(n) for n in self.cell_neighbors.values())} neighborhood relationships")
    
    def _cluster_terms(self) -> None:
        """
        Cluster terms to define cells in the manifold.
        """
        # Use KMeans clustering to partition the space
        kmeans = KMeans(n_clusters=self.n_cells, random_state=self.random_state, n_init="auto")
        cluster_labels = kmeans.fit_predict(self.vectors)
        
        # Create cell structures
        for i in range(self.n_cells):
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
                terms=cluster_terms,
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
        reducer = umap.UMAP(n_components=self.reduction_dims, 
                          random_state=self.random_state,
                          n_neighbors=15,
                          min_dist=0.1)
        
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
            boundaries=[]  # Will be filled later
        )
    
    def _define_cell_boundaries(self) -> None:
        """
        Define boundaries between cells in the reduced space.
        """
        if self.reduced is None:
            raise ValueError("Reduced representation not available. Call _reduce_dimensions first.")
            
        boundaries = []
        
        # For 2D representation, use Voronoi diagrams
        if self.reduction_dims == 2:
            try:
                # Compute Voronoi diagram of cell centroids
                vor = Voronoi(self.reduced.cell_centroids)
                boundaries = vor
                
                # Update cell objects with boundary points if applicable
                for i, cell_id in enumerate(self.cells.keys()):
                    # Get the region for this cell
                    region_idx = vor.point_region[i]
                    region = vor.regions[region_idx]
                    
                    # Skip if region is empty or contains -1 (unbounded)
                    if -1 in region or not region:
                        continue
                        
                    # Get boundary vertices
                    boundary_points = vor.vertices[region]
                    self.cells[cell_id].boundary_points = boundary_points
            except Exception as e:
                logger.warning(f"Could not compute Voronoi diagram: {e}")
        
        # For 3D representation, use Convex Hull or Delaunay triangulation
        elif self.reduction_dims == 3:
            try:
                # We'll use Delaunay triangulation
                tri = Delaunay(self.reduced.cell_centroids)
                boundaries = tri
                
                # The boundaries in 3D are more complex, so we don't assign
                # boundary_points to cells directly
            except Exception as e:
                logger.warning(f"Could not compute 3D boundaries: {e}")
        
        # Store boundaries in reduced state
        if self.reduced:
            self.reduced = self.reduced._replace(boundaries=boundaries)
    
    def _establish_neighborhoods(self) -> None:
        """
        Establish neighborhood relationships between cells.
        """
        # Calculate distances between cell centroids
        cell_ids = list(self.cells.keys())
        cell_centroids = np.array([self.cells[cell_id].centroid for cell_id in cell_ids])
        
        # Use cosine distance for semantic relationships
        distances = cosine_distances(cell_centroids)
        
        # For each cell, find k nearest neighbors
        k = min(5, len(cell_ids) - 1)
        for i, cell_id in enumerate(cell_ids):
            # Sort distances (excluding self)
            neighbor_indices = np.argsort(distances[i])[1:k+1]
            neighbor_ids = [cell_ids[idx] for idx in neighbor_indices]
            
            # Store neighbors
            self.cell_neighbors[cell_id] = neighbor_ids
    
    def _assign_cell_properties(self) -> None:
        """
        Assign occult typology and numerological values to cells.
        """
        # We'll assign cell types based on dominant terms
        # and numerological values based on gematria
        
        cell_ids = list(self.cells.keys())
        
        # Assign types based on the semantic meaning of the terms in the cell
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
        
        for cell_id in cell_ids:
            cell = self.cells[cell_id]
            
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
            
            # If no specific type has matches, keep as OTHER
            if max_count > 0:
                cell.type = cell_type
            
            # Calculate numerological value based on terms in the cell
            # We use the average of the first 5 terms (or all if less than 5)
            terms_to_use = cell.terms[:min(5, len(cell.terms))]
            num_values = [self.word_embeddings.find_numerological_significance(term) 
                         for term in terms_to_use]
            
            cell.numerological_value = int(np.mean(num_values))
            
            # For terms with master numbers, give them more weight
            master_numbers = [11, 22, 33]
            for term in terms_to_use:
                value = self.word_embeddings.find_numerological_significance(term)
                if value in master_numbers:
                    # Increase the cell's numerological value to this master number
                    cell.numerological_value = value
                    break
    
    def get_cell_neighbors(self, cell_id: int) -> List[Cell]:
        """
        Get the neighboring cells for a given cell.
        
        Args:
            cell_id: The ID of the cell to get neighbors for
            
        Returns:
            List of neighboring Cell objects
        """
        if cell_id not in self.cell_neighbors:
            logger.warning(f"Cell ID {cell_id} not found in manifold")
            return []
            
        neighbor_ids = self.cell_neighbors[cell_id]
        return [self.cells[nid] for nid in neighbor_ids if nid in self.cells]
    
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
            # Reshape for cosine_distances function
            c1 = centroid1.reshape(1, -1)
            c2 = centroid2.reshape(1, -1)
            return float(cosine_distances(c1, c2)[0][0])
            
        elif distance_type == DistanceType.NUMEROLOGICAL:
            # Use numerological values to weight the distance
            num_diff = abs(cell1.numerological_value - cell2.numerological_value)
            
            # Master numbers have special significance in numerology
            master_numbers = [11, 22, 33]
            if cell1.numerological_value in master_numbers or cell2.numerological_value in master_numbers:
                # Reduce distance if either cell has a master number
                factor = 0.7
            else:
                factor = 1.0
                
            # Base distance is Euclidean, modified by numerological significance
            base_distance = float(np.linalg.norm(centroid1 - centroid2))
            return base_distance * (1.0 + (num_diff / 9.0) * factor)
            
        elif distance_type == DistanceType.HYBRID:
            # Combine semantic and numerological distances
            # Semantic component (cosine distance)
            c1 = centroid1.reshape(1, -1)
            c2 = centroid2.reshape(1, -1)
            semantic_dist = float(cosine_distances(c1, c2)[0][0])
            
            # Numerological component
            num_diff = abs(cell1.numerological_value - cell2.numerological_value)
            
            # Calculate numerological weight
            # Master numbers and matching numbers have special significance
            if cell1.numerological_value == cell2.numerological_value:
                num_weight = 0.7  # Cells with same numerological value are closer
            elif cell1.numerological_value in [11, 22, 33] or cell2.numerological_value in [11, 22, 33]:
                num_weight = 0.8  # Master numbers have special pull
            else:
                num_weight = 1.0 + (num_diff / 9.0)  # Normal scaling by difference
                
            # Return weighted combination
            return semantic_dist * num_weight
        
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")
    
    def get_cell_state(self, cell_id: int) -> Dict[str, Any]:
        """
        Get the current state of a cell for use in cellular automata rules.
        
        Args:
            cell_id: ID of the cell to get state for
            
        Returns:
            Dictionary containing cell state information
        """
        if cell_id not in self.cells:
            logger.warning(f"Cell ID {cell_id} not found in manifold")
            return {}
            
        cell = self.cells[cell_id]
        
        # Get neighboring cells
        neighbors = self.get_cell_neighbors(cell_id)
        
        # Calculate average distance to neighbors
        distances = [self.compute_cell_distance(cell_id, n.id) for n in neighbors]
        avg_distance = np.mean(distances) if distances else 0.0
        
        # Calculate semantic and numerological features
        neighbor_types = [n.type for n in neighbors]
        neighbor_num_values = [n.numerological_value for n in neighbors]
        
        # Count occurrences of each cell type in neighborhood
        type_counts = {}
        for t in CellType:
            type_counts[t] = neighbor_types.count(t)
            
        # Check for numerological relationships
        master_numbers = [11, 22, 33]
        has_master_neighbor = any(v in master_numbers for v in neighbor_num_values)
        
        # Create state dictionary
        state = {
            "cell": cell,
            "neighbors": neighbors,
            "neighbor_count": len(neighbors),
            "avg_distance": avg_distance,
            "type_counts": type_counts,
            "numerological_values": neighbor_num_values,
            "has_master_neighbor": has_master_neighbor,
            "mean_num_value": np.mean(neighbor_num_values) if neighbor_num_values else 0
        }
        
        return state
    
    def transform_cell(
        self,
        cell_id: int,
        transformation: str,
        magnitude: float = 1.0
    ) -> np.ndarray:
        """
        Apply a transformation to a cell's centroid based on its neighbors.
        
        This is key for implementing contrast-based evolution in the cellular automata.
        
        Args:
            cell_id: ID of the cell to transform
            transformation: Type of transformation to apply
                ('contrast', 'align', 'repel', 'numerological')
            magnitude: Strength of the transformation
            
        Returns:
            The new centroid vector after transformation
        """
        if cell_id not in self.cells:
            logger.warning(f"Cell ID {cell_id} not found in manifold")
            return np.array([])
            
        cell = self.cells[cell_id]
        neighbors = self.get_cell_neighbors(cell_id)
        
        if not neighbors:
            logger.warning(f"Cell {cell_id} has no neighbors for transformation")
            return cell.centroid
            
        # Get the original centroid
        centroid = cell.centroid
        
        if transformation == 'contrast':
            # Move away from the average of neighbors (increase contrast)
            neighbor_centroids = np.array([n.centroid for n in neighbors])
            neighbor_avg = np.mean(neighbor_centroids, axis=0)
            
            # Direction vector from neighbor average to current cell
            direction = centroid - neighbor_avg
            
            # Normalize the direction vector
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
                
            # Apply transformation (move away from neighbors)
            return centroid + direction * magnitude
            
        elif transformation == 'align':
            # Move toward neighbors with similar numerological values
            similar_neighbors = [n for n in neighbors 
                              if abs(n.numerological_value - cell.numerological_value) <= 2]
            
            if not similar_neighbors:
                return centroid
                
            # Calculate target as average of similar neighbors
            similar_centroids = np.array([n.centroid for n in similar_neighbors])
            target = np.mean(similar_centroids, axis=0)
            
            # Move toward the target
            direction = target - centroid
            return centroid + direction * magnitude
            
        elif transformation == 'repel':
            # Move away from dissimilar neighbors
            dissimilar_neighbors = [n for n in neighbors 
                               if abs(n.numerological_value - cell.numerological_value) > 2]
            
            if not dissimilar_neighbors:
                return centroid
                
            # Calculate point to move away from
            dissimilar_centroids = np.array([n.centroid for n in dissimilar_neighbors])
            repel_point = np.mean(dissimilar_centroids, axis=0)
            
            # Direction vector away from dissimilar neighbors
            direction = centroid - repel_point
            
            # Normalize the direction vector
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
                
            # Apply transformation (move away from dissimilar neighbors)
            return centroid + direction * magnitude
            
        elif transformation == 'numerological':
            # Transform based on numerological significance
            num_value = cell.numerological_value
            
            # Master numbers have special transformations
            if num_value in [11, 22, 33]:
                # For master numbers, amplify the vector (move outward)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    normalized = centroid / norm
                    return normalized * (norm * (1 + magnitude * 0.2))
                return centroid
                
            # For other numbers, apply transformations based on numerology
            if num_value == 1:  # Independence, new beginnings
                # Move away from all neighbors (emphasize uniqueness)
                neighbor_centroids = np.array([n.centroid for n in neighbors])
                neighbor_avg = np.mean(neighbor_centroids, axis=0)
                direction = centroid - neighbor_avg
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                return centroid + direction * magnitude
                
            elif num_value in [2, 6]:  # Partnership, harmony
                # Move toward the average of all neighbors
                neighbor_centroids = np.array([n.centroid for n in neighbors])
                neighbor_avg = np.mean(neighbor_centroids, axis=0)
                direction = neighbor_avg - centroid
                return centroid + direction * magnitude * 0.5
                
            elif num_value in [3, 9]:  # Creative expression, completion
                # Apply a rotation-like transformation in embedding space
                # This is simplified for high-dimensional space
                # We'll perturb the vector slightly in a way that
                # maintains its magnitude but changes its direction
                noise = np.random.normal(0, 0.1, size=centroid.shape)
                noise = noise - np.mean(noise)  # Zero-center the noise
                return centroid + noise * magnitude
                
            elif num_value in [4, 8]:  # Stability, power
                # Minimal movement, emphasize stability
                return centroid
                
            elif num_value in [5, 7]:  # Change, spirituality
                # Move in direction of most different neighbor
                diffs = [(n, abs(n.numerological_value - num_value)) for n in neighbors]
                if not diffs:
                    return centroid
                    
                most_different = max(diffs, key=lambda x: x[1])[0]
                direction = most_different.centroid - centroid
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                return centroid + direction * magnitude * 0.3
                
            else:
                # Default behavior for other values
                return centroid
        else:
            logger.warning(f"Unknown transformation: {transformation}")
            return centroid
    
    def apply_transformation_to_cell(
        self,
        cell_id: int,
        transformation: str,
        magnitude: float = 1.0
    ) -> None:
        """
        Apply and store a transformation to a cell's centroid.
        
        Args:
            cell_id: ID of the cell to transform
            transformation: Type of transformation to apply
            magnitude: Strength of the transformation
        """
        if cell_id not in self.cells:
            logger.warning(f"Cell ID {cell_id} not found in manifold")
            return
            
        # Calculate new centroid
        new_centroid = self.transform_cell(cell_id, transformation, magnitude)
        
        # Update the cell
        self.cells[cell_id].centroid = new_centroid
        
        # Note: After updating cells, the reduced representation and
        # boundaries will be outdated. They should be recomputed if needed.
    
    def evolve_manifold(self, evolution_rules: Dict[str, Any]) -> None:
        """
        Evolve the entire manifold according to specified rules.
        
        This is a high-level method for applying transformations to all cells,
        designed to be called by the cellular automata system.
        
        Args:
            evolution_rules: Dictionary of rules for evolution
                Must include:
                - 'transformation': Type of transformation to apply
                - 'magnitude': Base strength of transformation
                May include:
                - 'cell_type_weights': Dict mapping CellType to weight multiplier
                - 'numerological_weights': Dict mapping numerological values to weight multiplier
        """
        if 'transformation' not in evolution_rules or 'magnitude' not in evolution_rules:
            raise ValueError("Evolution rules must include 'transformation' and 'magnitude'")
            
        transformation = evolution_rules['transformation']
        base_magnitude = evolution_rules['magnitude']
        
        # Optional weights
        cell_type_weights = evolution_rules.get('cell_type_weights', {})
        numerological_weights = evolution_rules.get('numerological_weights', {})
        
        # Store original centroids to avoid interference during updates
        original_centroids = {cell_id: cell.centroid.copy() for cell_id, cell in self.cells.items()}
        
        # Calculate new centroids for all cells
        new_centroids = {}
        for cell_id, cell in self.cells.items():
            # Adjust magnitude based on cell properties
            magnitude = base_magnitude
            
            # Apply cell type weighting if specified
            if cell.type in cell_type_weights:
                magnitude *= cell_type_weights[cell.type]
                
            # Apply numerological weighting if specified
            if cell.numerological_value in numerological_weights:
                magnitude *= numerological_weights[cell.numerological_value]
                
            # Calculate new centroid
            new_centroids[cell_id] = self.transform_cell(cell_id, transformation, magnitude)
        
        # Update all cells with new centroids
        for cell_id, new_centroid in new_centroids.items():
            self.cells[cell_id].centroid = new_centroid
            
        logger.info(f"Evolved manifold using")
        
        if term not in self.term_to_cell:
            logger.warning(f"Term '{term}' not found in manifold")
            return None
            
        cell_id = self.term_to_cell[term]
        return self.cells.get(cell_id)
    
    # def get_cell_neighbors(self, cell_id: int) -> List[

