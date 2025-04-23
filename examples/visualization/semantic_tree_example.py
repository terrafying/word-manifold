"""
Example demonstrating the semantic tree visualization capabilities.
"""

from word_manifold.visualization.semantic_tree_visualizer import SemanticTreeVisualizer

def main():
    # Initialize visualizer
    visualizer = SemanticTreeVisualizer(
        output_dir="visualizations/semantic_trees",
        node_size_base=800,
        min_similarity=0.3
    )
    
    # Example 1: Simple concept hierarchy
    root = "consciousness"
    related_terms = set([
        "awareness", "perception", "mindfulness",
        "thought", "cognition", "attention",
        "meditation", "self-awareness", "alertness",
        "unconsciousness", "subconsciousness", "unconscious", "dreams",
        "sleep", "consciousness", "awareness", "perception", "mindfulness",
        "thought", "cognition", "attention",
        "meditation", "self-awareness", "alertness",
        "unconsciousness", "subconsciousness", "unconscious", "dreams",
        "sleep", "consciousness", "awareness", "perception", "mindfulness",
        "thought", "cognition", "attention",
    ])
    
    # Build and visualize tree
    tree = visualizer.build_semantic_tree(
        root,
        related_terms,
        max_depth=3,
        branching_factor=3
    )
    
    path = visualizer.visualize_tree(
        tree,
        title="Consciousness Concept Tree",
        show_weights=True,
        show_similarities=True
    )
    print(f"Visualization saved to: {path}")
    
    # Example 2: Technical concept mapping
    root = "machine learning"
    related_terms = [
        "neural networks", "deep learning", "supervised learning",
        "reinforcement learning", "clustering", "classification",
        "regression", "decision trees", "support vector machines",
        "natural language processing", "computer vision"
    ]
    
    tree = visualizer.build_semantic_tree(
        root,
        related_terms,
        max_depth=4,
        branching_factor=2
    )
    
    path = visualizer.visualize_tree(
        tree,
        title="Machine Learning Concepts",
        show_weights=True,
        show_similarities=True
    )
    print(f"Visualization saved to: {path}")

if __name__ == "__main__":
    main() 