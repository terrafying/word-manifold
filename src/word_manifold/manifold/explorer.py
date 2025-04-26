from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class ManifoldExplorer:
    """A class for exploring and analyzing semantic manifolds."""
    
    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        dimension: int = 768,
        metric: str = "cosine",
        neighborhood_size: int = 5
    ):
        """
        Initialize the ManifoldExplorer.
        
        Args:
            embeddings: Dictionary mapping words to their embedding vectors
            dimension: Dimension of the embedding space
            metric: Distance metric to use ("cosine" or "euclidean")
            neighborhood_size: Number of nearest neighbors to consider
        """
        self.embeddings = embeddings
        self.dimension = dimension
        self.metric = metric
        self.neighborhood_size = neighborhood_size
        self.graph = self._build_graph()
        
    def _build_graph(self) -> nx.Graph:
        """Build a graph representation of the semantic manifold."""
        G = nx.Graph()
        
        # Add nodes
        for word in self.embeddings:
            G.add_node(word, embedding=self.embeddings[word])
            
        # Add edges based on similarity
        for word1 in self.embeddings:
            similarities = []
            for word2 in self.embeddings:
                if word1 != word2:
                    sim = self._compute_similarity(
                        self.embeddings[word1],
                        self.embeddings[word2]
                    )
                    similarities.append((word2, sim))
            
            # Add edges to top-k similar words
            similarities.sort(key=lambda x: x[1], reverse=True)
            for word2, sim in similarities[:self.neighborhood_size]:
                G.add_edge(word1, word2, weight=sim)
                
        return G
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity between two vectors."""
        if self.metric == "cosine":
            return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        else:  # euclidean
            return -np.linalg.norm(vec1 - vec2)  # Negative for consistency with cosine
            
    def analyze_structure(self) -> Dict:
        """
        Analyze the geometric structure of the manifold.
        
        Returns:
            Dictionary containing structural analysis results
        """
        # Compute basic graph metrics
        metrics = {
            "density": nx.density(self.graph),
            "average_clustering": nx.average_clustering(self.graph),
            "average_shortest_path": nx.average_shortest_path_length(self.graph),
            "diameter": nx.diameter(self.graph),
        }
        
        # Find communities
        communities = nx.community.greedy_modularity_communities(self.graph)
        metrics["num_communities"] = len(communities)
        metrics["community_sizes"] = [len(c) for c in communities]
        
        return metrics
    
    def get_geodesic(self, word1: str, word2: str) -> List[str]:
        """
        Find the geodesic (shortest path) between two words.
        
        Args:
            word1: Starting word
            word2: Target word
            
        Returns:
            List of words forming the path
        """
        try:
            path = nx.shortest_path(
                self.graph,
                source=word1,
                target=word2,
                weight="weight"
            )
            return path
        except nx.NetworkXNoPath:
            return []
            
    def visualize(self, output_file: Optional[str] = None):
        """
        Visualize the manifold using t-SNE.
        
        Args:
            output_file: Optional file to save the visualization
        """
        import matplotlib.pyplot as plt
        
        # Prepare data for t-SNE
        words = list(self.embeddings.keys())
        vectors = np.array([self.embeddings[w] for w in words])
        
        # Reduce dimensionality
        tsne = TSNE(n_components=2, random_state=42)
        coords = tsne.fit_transform(vectors)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.5)
        
        # Add labels
        for i, word in enumerate(words):
            plt.annotate(word, (coords[i, 0], coords[i, 1]))
            
        plt.title("Semantic Manifold Visualization")
        plt.axis("off")
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show() 