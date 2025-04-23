"""
Semantic Tree Visualizer for creating and visualizing hierarchical semantic relationships.

This module provides tools for building and visualizing semantic trees based on word embeddings
and similarity relationships. It includes support for hierarchical visualization with customizable
appearance and interactive exploration of semantic relationships.
"""

import os
import logging
from typing import List, Optional, Dict, Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticNode:
    """Represents a node in the semantic tree with hierarchical relationships."""
    
    def __init__(self, text: str, embedding: np.ndarray):
        """
        Initialize a semantic node.
        
        Args:
            text: The text content of the node
            embedding: The vector embedding of the text
        """
        self.text = text
        self.embedding = embedding
        self.children: List[SemanticNode] = []
        self.parent: Optional[SemanticNode] = None
        self.level = 0
        self.similarity_to_parent = 1.0
        self.semantic_weight = 1.0
    
    def add_child(self, child: 'SemanticNode', similarity: float) -> None:
        """
        Add a child node with its similarity score.
        
        Args:
            child: The child node to add
            similarity: Similarity score between parent and child
        """
        child.parent = self
        child.level = self.level + 1
        child.similarity_to_parent = similarity
        self.children.append(child)
        
    def calculate_semantic_weight(self) -> float:
        """
        Calculate the semantic weight of the node based on similarities.
        
        Returns:
            The calculated semantic weight
        """
        if self.parent is None:
            return 1.0
        
        parent_weight = self.parent.calculate_semantic_weight()
        self.semantic_weight = parent_weight * self.similarity_to_parent * self.similarity_to_parent
        return self.semantic_weight

class SemanticTreeVisualizer:
    """Visualizer for semantic trees with customizable appearance and layout."""
    
    def __init__(
        self,
        output_dir: str = "visualizations/semantic_trees",
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        color_scheme: str = "viridis",
        node_size_base: int = 1000,
        min_similarity: float = 0.3
    ):
        """
        Initialize the semantic tree visualizer.
        
        Args:
            output_dir: Directory for saving visualizations
            model_name: Name of the sentence transformer model
            color_scheme: Color scheme for visualization
            node_size_base: Base size for nodes
            min_similarity: Minimum similarity threshold for relationships
        """
        self.output_dir = output_dir
        self.color_scheme = color_scheme
        self.node_size_base = node_size_base
        self.min_similarity = min_similarity
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the embedding model
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            logger.info("Falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Create custom colormap for similarity scores
        self.similarity_cmap = LinearSegmentedColormap.from_list(
            'similarity',
            ['#ff9999', '#99ff99']  # Red to Green
        )
    
    def build_semantic_tree(
        self,
        root_text: str,
        related_terms: List[str],
        max_depth: int = 3,
        branching_factor: int = 4
    ) -> SemanticNode:
        """
        Build a semantic tree from root text and related terms.
        
        Args:
            root_text: The root term of the tree
            related_terms: List of terms to organize in the tree
            max_depth: Maximum depth of the tree
            branching_factor: Maximum children per node
            
        Returns:
            The root node of the built tree
        """
        # Filter invalid terms
        related_terms = [term for term in related_terms if isinstance(term, str) and term.strip()]
        
        if not related_terms:
            logger.warning("No valid related terms provided")
            root_embedding = self.model.encode([root_text])[0]
            return SemanticNode(root_text, root_embedding)
        
        # Get embeddings for all terms
        all_terms = [root_text] + related_terms
        try:
            embeddings = self.model.encode(all_terms)
        except Exception as e:
            logger.error("Error encoding terms", exc_info=e, stack_info=True, stacklevel=2)
            return SemanticNode(root_text, np.zeros(768))
        
        # Create root node
        root = SemanticNode(root_text, embeddings[0])
        
        # Build tree recursively
        term_embeddings = list(zip(related_terms, embeddings[1:]))
        self._build_tree_recursive(root, term_embeddings, max_depth, branching_factor)
        
        return root
    
    def _build_tree_recursive(
        self,
        parent: SemanticNode,
        available_terms: List[Tuple[str, np.ndarray]],
        depth: int,
        branching_factor: int
    ) -> None:
        """
        Recursively build the tree by adding children based on similarity.
        
        Args:
            parent: The parent node
            available_terms: List of (term, embedding) tuples
            depth: Current depth in the tree
            branching_factor: Maximum children per node
        """
        if depth <= 0 or not available_terms:
            return
        
        # Calculate similarities to parent
        similarities = [
            (term, emb, self._calculate_similarity(parent.embedding, emb))
            for term, emb in available_terms
        ]
        
        # Filter by minimum similarity and sort
        valid_terms = [
            (term, emb, sim) for term, emb, sim in similarities
            if sim >= self.min_similarity
        ]
        valid_terms.sort(key=lambda x: x[2], reverse=True)
        
        # Select top terms as children
        selected_terms = valid_terms[:branching_factor]
        remaining_terms = [
            (term, emb) for term, emb, _ in valid_terms[branching_factor:]
        ]
        
        # Add children
        for term, embedding, similarity in selected_terms:
            child = SemanticNode(term, embedding)
            parent.add_child(child, similarity)
            
            # Recursively build subtrees
            self._build_tree_recursive(
                child,
                remaining_terms,
                depth - 1,
                branching_factor
            )
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def visualize_tree(
        self,
        root: SemanticNode,
        title: str = "Semantic Tree",
        show_weights: bool = True,
        show_similarities: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ) -> str:
        """
        Create and save a visualization of the semantic tree.
        
        Args:
            root: Root node of the tree
            title: Title for the visualization
            show_weights: Whether to show semantic weights
            show_similarities: Whether to show similarity scores
            figsize: Figure size (width, height)
            
        Returns:
            Path to the saved visualization
        """
        # Create directed graph
        G = nx.DiGraph()
        self._add_node_to_graph(G, root)
        
        # Create figure
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        node_colors = [
            plt.cm.get_cmap(self.color_scheme)(n.level / 3)
            for n in G.nodes()
        ]
        node_sizes = [
            self.node_size_base * n.semantic_weight
            for n in G.nodes()
        ]
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes
        )
        
        # Draw edges with similarity colors if requested
        if show_similarities:
            edge_colors = [
                self.similarity_cmap(G.edges[e]['similarity'])
                for e in G.edges()
            ]
        else:
            edge_colors = ['gray' for _ in G.edges()]
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=20
        )
        
        # Add labels
        labels = {}
        for node in G.nodes():
            label = node.text
            if show_weights:
                label += f"\n(w={node.semantic_weight:.2f})"
            labels[node] = label
        
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8
        )
        
        plt.title(title)
        
        # Save visualization
        output_path = os.path.join(
            self.output_dir,
            f"{title.lower().replace(' ', '_')}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_path}")
        return output_path
    
    def _add_node_to_graph(self, G: nx.DiGraph, node: SemanticNode) -> None:
        """
        Recursively add nodes and edges to the graph.
        
        Args:
            G: NetworkX directed graph
            node: Current node to add
        """
        G.add_node(node)
        
        for child in node.children:
            G.add_node(child)
            G.add_edge(
                node,
                child,
                similarity=child.similarity_to_parent
            )
            self._add_node_to_graph(G, child) 