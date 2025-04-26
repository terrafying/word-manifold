import pytest
import numpy as np
import matplotlib.pyplot as plt
from word_manifold.manifold.pattern_selection import PatternSelector, PatternSource

def test_pattern_loading():
    """Test pattern loading from different sources."""
    for source in PatternSource:
        selector = PatternSelector(source)
        patterns = selector.patterns
        assert len(patterns) > 0
        for pattern_name, pattern in patterns.items():
            assert isinstance(pattern, list)
            assert all(isinstance(x, float) for x in pattern)
            assert len(pattern) > 0

def test_pattern_selection():
    """Test pattern selection with multiple valid solutions."""
    selector = PatternSelector(PatternSource.HERMETIC)
    
    # Create some valid solutions
    solutions = [
        np.array([1.0, 0.0, 0.0]),  # Mentalism-like
        np.array([0.0, 1.0, 0.0]),  # Correspondence-like
        np.array([0.0, 0.0, 1.0])   # Vibration-like
    ]
    
    # Select pattern with different contexts
    context1 = {"mentalism": 2.0}  # Favor mentalism
    selected1 = selector.select_pattern(solutions, context1)
    assert np.allclose(selected1, solutions[0])
    
    context2 = {"correspondence": 2.0}  # Favor correspondence
    selected2 = selector.select_pattern(solutions, context2)
    assert np.allclose(selected2, solutions[1])

def test_pattern_similarity():
    """Test pattern similarity calculation."""
    selector = PatternSelector(PatternSource.HERMETIC)
    
    # Test similar patterns
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.9, 0.1, 0.0])
    similarity = selector._calculate_similarity(vec1, vec2)
    assert similarity > 0.9
    
    # Test different patterns
    vec3 = np.array([0.0, 0.0, 1.0])
    similarity = selector._calculate_similarity(vec1, vec3)
    assert similarity < 0.1

def test_pattern_descriptions():
    """Test pattern description retrieval."""
    selector = PatternSelector(PatternSource.HERMETIC)
    
    # Test known pattern
    desc = selector.get_pattern_description("mentalism")
    assert "mind" in desc.lower()
    
    # Test unknown pattern
    desc = selector.get_pattern_description("unknown")
    assert desc == "Unknown pattern"

def visualize_patterns():
    """Visualize patterns from different sources."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, source in enumerate(PatternSource):
        selector = PatternSelector(source)
        patterns = selector.patterns
        
        # Create 2D projection of patterns
        for pattern_name, pattern in patterns.items():
            # Project to 2D if needed
            if len(pattern) > 2:
                pattern_2d = pattern[:2]
            else:
                pattern_2d = pattern
                
            # Plot pattern
            axes[i].scatter(pattern_2d[0], pattern_2d[1], label=pattern_name)
            axes[i].text(pattern_2d[0], pattern_2d[1], pattern_name)
            
        axes[i].set_title(f"{source.value.title()} Patterns")
        axes[i].grid(True)
        axes[i].legend()
        
    plt.tight_layout()
    plt.savefig("outputs/visualizations/pattern_visualization.png")
    plt.close()

if __name__ == "__main__":
    # Run tests
    test_pattern_loading()
    test_pattern_selection()
    test_pattern_similarity()
    test_pattern_descriptions()
    
    # Generate visualization
    visualize_patterns() 