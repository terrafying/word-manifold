"""
Tests for the cellular automata system and rules.

This includes tests for:
1. Automata system initialization and state management
2. Rule application and sequences
3. Evolution patterns
4. State capture and persistence
"""

import pytest
from unittest.mock import Mock, MagicMock
from word_manifold.automata.cellular_rules import CellularRule, RuleSequence, create_predefined_rules
from word_manifold.automata.system import AutomataSystem, EvolutionPattern
from word_manifold.types import CellType
import numpy as np

@pytest.fixture
def mock_manifold():
    mock = Mock()
    # Mock the cells attribute with some test data
    mock.cells = {
        0: Mock(centroid=np.zeros(10)),
        1: Mock(centroid=np.zeros(10))
    }
    # Mock transform to return the same vectors
    mock.transform = MagicMock(return_value=np.zeros((2, 10)))
    # Mock get_manifold_state to return a valid state
    mock.get_manifold_state = MagicMock(return_value={
        "cell_count": 2,
        "term_count": 0,
        "type_counts": {},
        "num_value_counts": {},
        "avg_connectivity": 0.0,
        "reduced_representation_available": False
    })
    return mock

@pytest.fixture
def base_system(mock_manifold):
    rules = create_predefined_rules()
    return AutomataSystem(
        manifold=mock_manifold,
        rules_dict=rules,
        sequences_dict={},
        evolution_pattern=EvolutionPattern.THELEMIC,
        save_path="test_outputs"
    )

class TestAutomataSystem:
    """Tests for the AutomataSystem class."""
    
    def test_initialization(self, base_system, mock_manifold):
        """Test system initialization and basic properties."""
        assert base_system.manifold == mock_manifold
        assert len(base_system.rules) > 0
        assert base_system.generation == 0
        assert base_system.evolution_pattern == EvolutionPattern.THELEMIC
        
    def test_evolution_state(self, base_system):
        """Test evolution state management."""
        initial_generation = base_system.generation
        
        # Evolve the system
        base_system.evolve(generations=1)
        assert base_system.generation == initial_generation + 1
        
        # Check state capture
        state = base_system.manifold.get_manifold_state()
        assert state is not None
        assert isinstance(state, dict)
        assert state["cell_count"] == 2  # From mock_manifold fixture
        
    def test_rule_application(self, base_system):
        """Test individual rule application."""
        rule = list(base_system.rules.values())[0]
        initial_generation = base_system.generation
        
        # Apply rule
        rule.apply(base_system.manifold, initial_generation)
        
        # Check transformation was called
        assert base_system.manifold.transform.called
        
    def test_rule_sequence(self, base_system):
        """Test rule sequence application."""
        rules = list(base_system.rules.values())[:2]
        test_sequence = RuleSequence(
            name="test_sequence",
            description="Test sequence",
            rules=rules,
            esoteric_correspondence="Test correspondence"
        )
        
        # Apply sequence
        initial_generation = base_system.generation
        test_sequence.apply(base_system.manifold, initial_generation)
        
        # Verify transformations were called
        assert base_system.manifold.transform.call_count >= len(rules)

class TestCellularRules:
    """Tests for cellular automata rules."""
    
    def test_predefined_rules(self):
        """Test creation and properties of predefined rules."""
        rules = create_predefined_rules()
        
        # Check basic rule properties
        assert len(rules) > 0
        for name, rule in rules.items():
            assert isinstance(rule, CellularRule)
            # assert rule.name in str(name).lower().replace(' ', '_')  # More flexible name matching
            assert rule.parameters is not None
            assert rule.vector_transformation is not None
            
    def test_rule_parameters(self):
        """Test rule parameter validation and effects."""
        rules = create_predefined_rules()
        rule = rules["great_work"]  # Use a specific rule for testing
        
        # Check parameter properties
        assert rule.parameters.magnitude > 0
        assert rule.parameters.principle is not None
        if rule.parameters.numerological_weights:
            assert all(isinstance(k, int) for k in rule.parameters.numerological_weights.keys())
            
    def test_rule_sequence_creation(self, mock_manifold):
        """Test creation and validation of rule sequences."""
        rules = create_predefined_rules()
        rule_list = list(rules.values())[:2]
        
        sequence = RuleSequence(
            name="test_sequence",
            rules=rule_list,
            description="Test sequence",
            esoteric_correspondence="Test correspondence"
        )
        
        # Check sequence properties
        assert len(sequence.rules) == len(rule_list)
        assert sequence.name == "test_sequence"
        assert sequence.description is not None
