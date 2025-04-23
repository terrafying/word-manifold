"""
Test suite for the RitualVisualizer class.

Tests cover initialization, core functionality, transformation processing,
visualization generation, error handling, and state tracking.
"""

import os
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Set

from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.visualization.ritual_visualizer import RitualVisualizer, RitualPhase
from word_manifold.automata.hermetic_principles import HermeticPrinciple

@pytest.fixture
def word_embeddings():
    """Create a WordEmbeddings instance for testing."""
    embeddings = WordEmbeddings(model_name="all-MiniLM-L6-v2")
    test_terms = ["light", "wisdom", "truth", "love", "power"]
    embeddings.load_terms(test_terms)
    return embeddings

@pytest.fixture
def test_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "test_ritual_viz"
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)

@pytest.fixture
def ritual_visualizer(word_embeddings, test_output_dir):
    """Create a RitualVisualizer instance for testing."""
    return RitualVisualizer(
        word_embeddings=word_embeddings,
        output_dir=test_output_dir,
        n_dims=3,
        frame_duration=100,  # Faster for testing
        energy_threshold=0.7,
        resonance_threshold=0.8
    )

def test_initialization(ritual_visualizer, test_output_dir):
    """Test proper initialization of RitualVisualizer."""
    assert ritual_visualizer.output_dir == test_output_dir
    assert ritual_visualizer.n_dims == 3
    assert ritual_visualizer.frame_duration == 100
    assert ritual_visualizer.energy_threshold == 0.7
    assert ritual_visualizer.resonance_threshold == 0.8
    assert os.path.exists(test_output_dir)
    assert isinstance(ritual_visualizer.states, list)
    assert isinstance(ritual_visualizer.term_evolution, dict)

def test_process_transformation(ritual_visualizer):
    """Test processing of term transformations."""
    initial_terms = {"light", "wisdom"}
    transformed_terms = {"illumination", "understanding"}
    
    ritual_visualizer.process_transformation(
        initial_terms,
        transformed_terms,
        RitualPhase.TRANSFORMATION
    )
    
    assert len(ritual_visualizer.states) == 1
    state = ritual_visualizer.states[0]
    assert state.phase == RitualPhase.TRANSFORMATION
    assert state.active_terms == transformed_terms
    assert isinstance(state.energy_level, float)
    assert isinstance(state.resonance_pattern, dict)
    assert isinstance(state.dominant_principle, HermeticPrinciple)

def test_calculate_energy_levels(ritual_visualizer):
    """Test energy level calculations."""
    terms = {"light", "wisdom", "truth"}
    energy = ritual_visualizer._calculate_energy_level(terms)
    
    assert isinstance(energy, float)
    assert 0 <= energy <= 1

def test_determine_resonance(ritual_visualizer):
    """Test resonance pattern determination."""
    terms = {"light", "wisdom", "truth"}
    resonance = ritual_visualizer._determine_resonance_pattern(terms)
    
    assert isinstance(resonance, dict)
    assert all(isinstance(k, HermeticPrinciple) for k in resonance.keys())
    assert all(isinstance(v, float) for v in resonance.values())
    assert all(0 <= v <= 1 for v in resonance.values())

def test_identify_dominant_principle(ritual_visualizer):
    """Test dominant principle identification."""
    terms = {"light", "wisdom", "truth"}
    principle = ritual_visualizer._identify_dominant_principle(terms)
    
    assert isinstance(principle, HermeticPrinciple)

def test_generate_visualization(ritual_visualizer, test_output_dir):
    """Test visualization generation."""
    # Process a few transformations
    initial = {"light", "wisdom"}
    transformed = {"illumination", "understanding"}
    final = {"enlightenment", "knowledge"}
    
    ritual_visualizer.process_transformation(initial, transformed, RitualPhase.PREPARATION)
    ritual_visualizer.process_transformation(transformed, final, RitualPhase.TRANSFORMATION)
    
    # Generate visualization
    output_path = ritual_visualizer.generate_visualization()
    
    assert output_path is not None
    assert os.path.exists(output_path)
    assert output_path.startswith(test_output_dir)

def test_error_handling(ritual_visualizer):
    """Test error handling for invalid inputs."""
    # Empty term sets
    with pytest.raises(ValueError):
        ritual_visualizer.process_transformation(set(), {"light"}, RitualPhase.PREPARATION)
    
    with pytest.raises(ValueError):
        ritual_visualizer.process_transformation({"light"}, set(), RitualPhase.PREPARATION)
    
    # Invalid terms
    with pytest.raises(ValueError):
        ritual_visualizer.process_transformation(
            {"nonexistent_term"},
            {"light"},
            RitualPhase.PREPARATION
        )

def test_state_tracking(ritual_visualizer):
    """Test proper tracking of ritual state history."""
    transformations = [
        ({"light"}, {"illumination"}, RitualPhase.PREPARATION),
        ({"illumination"}, {"enlightenment"}, RitualPhase.TRANSFORMATION),
        ({"enlightenment"}, {"wisdom"}, RitualPhase.INTEGRATION)
    ]
    
    for initial, transformed, phase in transformations:
        ritual_visualizer.process_transformation(initial, transformed, phase)
    
    assert len(ritual_visualizer.states) == len(transformations)
    assert all(state.phase == phase for state, (_, _, phase) in 
              zip(ritual_visualizer.states, transformations))
    
    # Check term evolution tracking
    assert "light" in ritual_visualizer.term_evolution
    assert len(ritual_visualizer.term_evolution["light"]) > 0 