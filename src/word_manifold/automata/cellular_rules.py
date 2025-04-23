"""
Cellular Automata Rules Module for Word Vector Space.

This module implements transformation rules for cellular automata in word vector space
based on hermetic principles and occult correspondences. It structures the evolution
of semantic regions according to the Law of Correspondence: "As above, so below".

The module operates on three levels of reality:
1. Physical/Vector Space - The mathematical operations in embedding space
2. Astral/Semantic Space - The meaningful relationships between concepts 
3. Divine/Numerological Space - The numerological significance and symbolic patterns
"""

import logging
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable, NamedTuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter

from word_manifold.types import CellType, DistanceType
from word_manifold.manifold.vector_manifold import VectorManifold
from ..embeddings.word_embeddings import WordEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HermeticPrinciple(Enum):
    """The seven Hermetic principles that govern transformation rules."""
    MENTALISM = auto()         # "THE ALL is MIND; The Universe is Mental."
    CORRESPONDENCE = auto()    # "As above, so below; as below, so above."
    VIBRATION = auto()         # "Nothing rests; everything moves; everything vibrates."
    POLARITY = auto()          # "Everything is Dual; everything has poles."
    RHYTHM = auto()            # "Everything flows, out and in; everything has its tides."
    CAUSE_EFFECT = auto()      # "Every Cause has its Effect; Every Effect has its Cause."
    GENDER = auto()            # "Gender is in everything; everything has its Masculine and Feminine."

class ElementalForce(Enum):
    """The four elemental forces that influence transformation."""
    EARTH = auto()  # Stability, materiality, resistance to change
    AIR = auto()    # Intellect, communication, adaptability
    FIRE = auto()   # Energy, transformation, creation/destruction
    WATER = auto()  # Emotion, intuition, connection

class VibrationDirection(Enum):
    """Possible directions of vibrational change in the vector space."""
    ASCENDING = auto()  # Moving towards higher vibration (complexity, abstraction)
    DESCENDING = auto() # Moving towards lower vibration (simplicity, concreteness)
    EXPANDING = auto()  # Increasing in scope or influence
    CONTRACTING = auto() # Decreasing in scope or influence
    HARMONIZING = auto() # Moving towards balance with neighbors
    POLARIZING = auto()  # Moving away from neighbors, increasing distinction

@dataclass
class RuleParameterSet:
    """Parameters that define how a transformation rule behaves."""
    magnitude: float = 1.0                  # Base strength of transformation
    principle: HermeticPrinciple = HermeticPrinciple.CORRESPONDENCE
    elemental_influence: Dict[ElementalForce, float] = None  # Influence of each element
    numerological_weights: Dict[int, float] = None  # Weights by numerological value
    cell_type_weights: Dict[CellType, float] = None  # Weights by cell type
    vibration_direction: VibrationDirection = VibrationDirection.HARMONIZING
    
    def __init__(self, magnitude=1.0, principle=None, vibration_direction=None,
                 numerological_weights=None, elemental_influence=None, cell_type_weights=None):
        self.magnitude = magnitude
        self.principle = principle
        self.vibration_direction = vibration_direction
        self.numerological_weights = numerological_weights if numerological_weights is not None else {}
        self.elemental_influence = elemental_influence if elemental_influence is not None else {}
        self.cell_type_weights = cell_type_weights if cell_type_weights is not None else {}


class CellularRule:
    """
    A rule that defines how cells transform in the vector space.
    
    Each rule embodies one or more hermetic principles and governs 
    the evolution of the cellular automata system.
    """
    
    def __init__(
        self, 
        name: str, 
        description: str,
        parameters: RuleParameterSet,
        vector_transformation: str,
        esoteric_correspondence: str
    ):
        """
        Initialize a cellular automata rule.
        
        Args:
            name: Name of the rule
            description: Technical description of the rule
            parameters: Parameters that control rule behavior
            vector_transformation: Type of vector transformation to apply
            esoteric_correspondence: Description of occult/hermetic correspondence
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.vector_transformation = vector_transformation
        self.esoteric_correspondence = esoteric_correspondence
    
    def apply(self, manifold: VectorManifold, generation: int = 0) -> None:
        """
        Apply the rule to transform the entire manifold.
        
        Args:
            manifold: The vector manifold to transform
            generation: Current generation number (used for cyclical patterns)
        """
        logger.info(f"Applying rule '{self.name}' to manifold (generation {generation})")
        
        # Construct the evolution rules dictionary from parameters
        evolution_rules = {
            'transformation': self.vector_transformation,
            'magnitude': self.parameters.magnitude,
            'cell_type_weights': self.parameters.cell_type_weights,
            'numerological_weights': self.parameters.numerological_weights
        }
        
        # Apply modifiers based on hermetic principles
        self._apply_hermetic_modifiers(evolution_rules, generation)
        
        # Get all cell vectors that need to be transformed
        cell_vectors = np.array([cell.centroid for cell in manifold.cells.values()])
        
        # Apply the transformation
        transformed_vectors = manifold.transform(cell_vectors, evolution_rules)
        
        # Update cell centroids with transformed vectors
        for i, (cell_id, cell) in enumerate(manifold.cells.items()):
            cell.centroid = transformed_vectors[i]
        
        logger.info(f"Rule '{self.name}' applied successfully")
    
    def _apply_hermetic_modifiers(self, evolution_rules: Dict[str, Any], generation: int) -> None:
        """
        Apply modifiers to the evolution rules based on hermetic principles.
        
        Args:
            evolution_rules: The rules dictionary to modify
            generation: Current generation number
        """
        principle = self.parameters.principle
        
        if principle == HermeticPrinciple.MENTALISM:
            # The Universe is Mental - transformation is influenced by the "mental" 
            # state of the system (the global semantic context)
            pass  # Implemented in the manifold evolution method
            
        elif principle == HermeticPrinciple.CORRESPONDENCE:
            # As above, so below - higher-level patterns should influence lower ones
            # and vice versa (implemented through numerological weights)
            pass  # Already in the numerological_weights
            
        elif principle == HermeticPrinciple.VIBRATION:
            # Everything vibrates - add cyclical variations based on generation
            cycle_amplitude = 0.2  # How much the cycle affects magnitude
            cycle_period = 7  # Complete cycle every 7 generations
            cycle_phase = (generation % cycle_period) / cycle_period
            cycle_factor = 1.0 + cycle_amplitude * np.sin(2 * np.pi * cycle_phase)
            evolution_rules['magnitude'] *= cycle_factor
            
        elif principle == HermeticPrinciple.POLARITY:
            # Everything is dual - strengthen movements that create clear distinctions
            # by amplifying the transformation magnitude for cells with strong neighbors
            vib_dir = self.parameters.vibration_direction
            if vib_dir == VibrationDirection.POLARIZING:
                evolution_rules['magnitude'] *= 1.2
            elif vib_dir == VibrationDirection.HARMONIZING:
                evolution_rules['magnitude'] *= 0.8
            
        elif principle == HermeticPrinciple.RHYTHM:
            # Everything flows, has tides - similar to vibration but affects
            # which cells are transformed more (implemented in application)
            pass
            
        elif principle == HermeticPrinciple.CAUSE_EFFECT:
            # Every cause has its effect - transformations should cascade
            # through the system (implemented in the manifold evolution)
            pass
            
        elif principle == HermeticPrinciple.GENDER:
            # Gender is in everything - balance of active/passive forces
            # Implemented through how "masculine" (outward) and "feminine" (inward)
            # transformations are applied
            vib_dir = self.parameters.vibration_direction
            if vib_dir in [VibrationDirection.EXPANDING, VibrationDirection.ASCENDING]:
                # More "masculine"/active energy 
                evolution_rules['transformation'] = 'contrast'
            elif vib_dir in [VibrationDirection.CONTRACTING, VibrationDirection.DESCENDING]:
                # More "feminine"/receptive energy
                evolution_rules['transformation'] = 'align'


# Define a set of predefined rules based on hermetic and occult principles

def create_predefined_rules() -> Dict[str, CellularRule]:
    """
    Create a set of predefined cellular automata rules based on
    hermetic principles and occult correspondences.
    
    Returns:
        Dictionary mapping rule names to CellularRule objects
    """
    rules = {}
    
    # The Great Work Rule - Based on alchemical transformation
    great_work_params = RuleParameterSet(
        magnitude=1.3,
        principle=HermeticPrinciple.VIBRATION,
        vibration_direction=VibrationDirection.ASCENDING,
        numerological_weights={
            1: 1.3,  # Unity
            3: 1.4,  # Creation
            7: 1.5,  # Mysticism
            9: 1.6   # Completion
        }
    )
    rules["great_work"] = CellularRule(
        name="The Great Work",
        description="Transforms cells toward higher spiritual/conceptual states, emphasizing numerological significance",
        parameters=great_work_params,
        vector_transformation="numerological",
        esoteric_correspondence="The alchemical transformation of the soul from base matter (nigredo) through purification (albedo) to enlightenment (rubedo)"
    )
    
    # The Equilibrium Rule - Based on balance of polarities
    equilibrium_params = RuleParameterSet(
        magnitude=0.5,
        principle=HermeticPrinciple.POLARITY,
        vibration_direction=VibrationDirection.HARMONIZING,
        elemental_influence={
            ElementalForce.EARTH: 1.2,  # Strengthen stability
            ElementalForce.AIR: 0.8,    # Reduce volatility
            ElementalForce.FIRE: 0.7,   # Reduce transformation
            ElementalForce.WATER: 1.3   # Enhance connection
        }
    )
    rules["equilibrium"] = CellularRule(
        name="The Equilibrium",
        description="Seeks balance by moving cells closer to semantic consensus with their neighbors",
        parameters=equilibrium_params,
        vector_transformation="align",
        esoteric_correspondence="The balance of opposing forces (solve et coagula) and the reconciliation of opposites into harmonious unity"
    )
    
    # The Tower Rule - Based on sudden, disruptive change
    tower_params = RuleParameterSet(
        magnitude=1.5,
        principle=HermeticPrinciple.VIBRATION,
        vibration_direction=VibrationDirection.POLARIZING,
        elemental_influence={
            ElementalForce.EARTH: 0.5,  # Decrease stability
            ElementalForce.AIR: 1.0,    # Normal volatility
            ElementalForce.FIRE: 2.0,   # Strong transformation
            ElementalForce.WATER: 0.7   # Decrease connection
        }
    )
    rules["tower"] = CellularRule(
        name="The Tower",
        description="Creates dramatic separations and distinctions between neighboring cells",
        parameters=tower_params,
        vector_transformation="contrast",
        esoteric_correspondence="The Tower (XVI) tarot archetype - sudden revelation, breakdown of structures, and liberation through destruction"
    )
    
    # The Hermit Rule - Based on inward, solitary exploration
    hermit_params = RuleParameterSet(
        magnitude=1.2,
        principle=HermeticPrinciple.MENTALISM,
        vibration_direction=VibrationDirection.CONTRACTING,
        cell_type_weights={
            CellType.ELEMENTAL: 0.8,
            CellType.PLANETARY: 1.0,
            CellType.ZODIACAL: 0.9,
            CellType.TAROT: 1.3,
            CellType.SEPHIROTIC: 1.5,
            CellType.OTHER: 1.0
        }
    )
    rules["hermit"] = CellularRule(
        name="The Hermit",
        description="Intensifies the unique properties of each cell, increasing distinctiveness",
        parameters=hermit_params,
        vector_transformation="contrast",
        esoteric_correspondence="The Hermit (IX) tarot archetype - introspection, solitude, inner guidance, and the quest for deeper wisdom"
    )
    
    # The Wheel Rule - Based on cyclical change and fate
    wheel_params = RuleParameterSet(
        magnitude=1.0,
        principle=HermeticPrinciple.RHYTHM,
        vibration_direction=VibrationDirection.EXPANDING,
    )
    rules["wheel"] = CellularRule(
        name="The Wheel",
        description="Creates cyclic patterns of expansion and contraction over generations",
        parameters=wheel_params,
        vector_transformation="repel",
        esoteric_correspondence="The Wheel of Fortune (X) tarot archetype - cycles, karma, and the eternal recurrence of patterns"
    )
    
    # The Star Rule - Based on hope, inspiration, and connection
    star_params = RuleParameterSet(
        magnitude=0.7,
        principle=HermeticPrinciple.CORRESPONDENCE,
        vibration_direction=VibrationDirection.HARMONIZING,
        numerological_weights={
            7: 1.5,  # Number of mysticism
            9: 1.3,  # Completion
            11: 1.7, # Vision
            17: 1.5, # The Star card number
            22: 1.6  # Master builder
        }
    )
    rules["star"] = CellularRule(
        name="The Star",
        description="Creates harmonious connections between cells based on higher numerological patterns",
        parameters=star_params,
        vector_transformation="align",
        esoteric_correspondence="The Star (XVII) tarot archetype - hope, inspiration, spiritual connection, and the bridge between worlds"
    )
    
    # The Lovers Rule - Based on union and harmony of opposites
    lovers_params = RuleParameterSet(
        magnitude=0.9,
        principle=HermeticPrinciple.GENDER,
        vibration_direction=VibrationDirection.HARMONIZING,
        cell_type_weights={
            CellType.ELEMENTAL: 1.2,
            CellType.PLANETARY: 1.1,
            CellType.ZODIACAL: 0.9,
            CellType.TAROT: 1.0,
            CellType.SEPHIROTIC: 0.8,
            CellType.OTHER: 1.0
        }
    )
    rules["lovers"] = CellularRule(
        name="The Lovers",
        description="Unites opposing semantic concepts by reducing contrast between polar opposites",
        parameters=lovers_params,
        vector_transformation="align",
        esoteric_correspondence="The Lovers (VI) tarot archetype - union, relationship, harmony of opposites, and the alchemical wedding"
    )
    
    # The Magician Rule - Based on creative transformation
    magician_params = RuleParameterSet(
        magnitude=1.1,
        principle=HermeticPrinciple.MENTALISM,
        vibration_direction=VibrationDirection.EXPANDING,
        elemental_influence={
            ElementalForce.EARTH: 1.0,
            ElementalForce.AIR: 1.2,
            ElementalForce.FIRE: 1.3,
            ElementalForce.WATER: 1.1
        }
    )
    rules["magician"] = CellularRule(
        name="The Magician",
        description="Transforms cells through the power of active will and directed intention",
        parameters=magician_params,
        vector_transformation="numerological",
        esoteric_correspondence="The Magician (I) tarot archetype - will, skill, manifestation, and the principle of 'as above, so below'"
    )
    
    sun_params = RuleParameterSet(
        principle=HermeticPrinciple.POLARITY,
        vibration_direction=VibrationDirection.ASCENDING,
        elemental_influence={
            ElementalForce.EARTH: 0.0,
            ElementalForce.AIR: 1.2,
            ElementalForce.FIRE: 1.5,
            ElementalForce.WATER: 0.2,
        }
    )
    
    rules["sun"] = CellularRule(
        name="The Sun",
        description="The sun",
        parameters=sun_params,
        vector_transformation="numerological",
        esoteric_correspondence="praise the sun"
    )
    
    return rules


class RuleSequence:
    """
    A sequence of cellular automata rules to be applied in a specific order.
    
    The sequence can be applied in different ways:
    - Sequentially (default): Rules are applied in order
    - Conditionally: Rules are applied based on conditions
    - With branching: Different paths can be taken based on state
    
    The sequence also supports:
    - Dependencies between rules
    - Conditions for rule application
    - Branching paths based on manifold state
    - Platonic ideal inference during transformation
    """
    
    def __init__(
        self,
        name: str,
        rules: List[CellularRule],
        application_order: Optional[str] = None,
        description: Optional[str] = None,
        esoteric_correspondence: Optional[str] = None,
        dependencies: Optional[Dict[str, List[str]]] = None,
        conditions: Optional[Dict[str, Dict[str, Any]]] = None,
        branches: Optional[Dict[str, List[str]]] = None,
        platonic_inference: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a rule sequence.
        
        Args:
            name: Name of the sequence
            rules: List of CellularRule objects to apply
            application_order: How to apply rules - "sequential" (default), "conditional", or "branching"
            description: Description of the sequence's purpose
            esoteric_correspondence: Description of occult/hermetic correspondence
            dependencies: Dict mapping rule names to lists of prerequisite rule names
            conditions: Dict mapping rule names to condition dictionaries
            branches: Dict defining possible branching paths
            platonic_inference: Settings for inferring platonic ideals during transformation
        """
        self.name = name
        self.rules = {rule.name: rule for rule in rules}  # Map names to rules
        self.application_order = application_order or "sequential"
        self.description = description
        self.esoteric_correspondence = esoteric_correspondence
        self.dependencies = dependencies or {}
        self.conditions = conditions or {}
        self.branches = branches or {}
        self.platonic_inference = platonic_inference or {}
        
        # Initialize tracking attributes
        self.manifold = None  # Will be set when apply() is called
        self.applied_rules = []  # Track which rules have been applied
        self.transformation_history = []  # Track transformation effects
        
        # Validate the sequence
        self._validate_sequence()
        
    def apply(self, manifold: VectorManifold, start_generation: int = 0) -> None:
        """
        Apply the sequence of rules to transform the manifold.
        
        Args:
            manifold: The vector manifold to transform
            start_generation: Starting generation number for the sequence
            
        The method applies rules according to the specified application_order:
        - 'sequential': Apply rules in order, checking dependencies
        - 'conditional': Apply rules based on conditions
        - 'branching': Apply rules following branching paths
        """
        logger.info(f"Applying rule sequence '{self.name}' to manifold")
        
        # Validate the sequence before applying
        self._validate_sequence()
        
        # Clear transformation history
        self.transformation_history = []
        self.applied_rules = []
        
        # Apply rules according to specified order
        if self.application_order == 'conditional':
            self._apply_conditional(manifold, start_generation)
        elif self.application_order == 'branching':
            self._apply_branching(manifold, start_generation)
        else:  # Default to sequential
            self._apply_sequential(manifold, start_generation)
            
        logger.info(f"Rule sequence '{self.name}' applied successfully")
        logger.info(f"Applied rules: {', '.join(self.applied_rules)}")
    
    def _validate_sequence(self) -> None:
        """Validate the sequence configuration."""
        # Check that all rules in dependencies exist
        for rule_name, deps in self.dependencies.items():
            if rule_name not in self.rules:
                raise ValueError(f"Rule '{rule_name}' in dependencies not found in rules")
            for dep in deps:
                if dep not in self.rules:
                    raise ValueError(f"Dependency '{dep}' for rule '{rule_name}' not found in rules")
                    
        # Check that all rules in conditions exist
        for rule_name in self.conditions:
            if rule_name not in self.rules:
                raise ValueError(f"Rule '{rule_name}' in conditions not found in rules")
                
        # Check that all rules in branches exist
        for rule_name, next_rules in self.branches.items():
            if rule_name not in self.rules:
                raise ValueError(f"Rule '{rule_name}' in branches not found in rules")
            for next_rule in next_rules:
                if next_rule not in self.rules:
                    raise ValueError(f"Next rule '{next_rule}' for rule '{rule_name}' not found in rules")
                    
        # Check for circular dependencies
        self._check_circular_dependencies()
        
    def _check_circular_dependencies(self) -> None:
        """Check for circular dependencies in the rule sequence."""
        def has_cycle(node: str, visited: Set[str], path: Set[str]) -> bool:
            visited.add(node)
            path.add(node)
            
            for neighbor in self.dependencies.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, path):
                        return True
                elif neighbor in path:
                    return True
                    
            path.remove(node)
            return False
            
        visited = set()
        path = set()
        
        for rule_name in self.rules:
            if rule_name not in visited:
                if has_cycle(rule_name, visited, path):
                    raise ValueError("Circular dependency detected in rule sequence")
                    
    def get_transformation_history(self) -> List[Dict[str, Any]]:
        """Get the history of transformations applied by this sequence."""
        return self.transformation_history.copy()
        
    def get_applied_rules(self) -> List[str]:
        """Get the list of rules that have been applied."""
        return self.applied_rules.copy()

    def _apply_sequential(self, manifold: VectorManifold, start_generation: int) -> None:
        """
        Apply rules sequentially while checking dependencies and inferring platonic ideals.
        
        Args:
            manifold: The vector manifold to transform
            start_generation: Starting generation number
        """
        current_generation = start_generation
        manifold.transform(self.rules, current_generation) 
        for rule in self.rules:
            # Check dependencies
            if not self._check_dependencies(rule, manifold):
                logger.warning(f"Skipping rule {rule.name} due to unmet dependencies")
                continue
                
            # Infer and apply platonic ideals if needed
            # if rule.requires_ideals:
            #     self._infer_and_apply_ideals(manifold)
                
            # Apply the rule
            logger.info(f"Applying rule {rule.name} at generation {current_generation}")
            logger.info(f"Manifold state: {manifold.get_state_snapshot()}")
            logger.info(f"Rule parameters: {rule.parameters}")
            
            # manifold.transform(rules, current_generation)
            self.applied_rules.append(rule.name)
            
            # Record transformation
            self.transformation_history.append({
                'generation': current_generation,
                'rule': rule.name,
                'state': manifold.get_state_snapshot()
            })
            
            current_generation += 1
            
    def _apply_conditional(self, manifold: VectorManifold, start_generation: int) -> None:
        """
        Apply rules based on conditions and dependencies.
        
        Args:
            manifold: The vector manifold to transform
            start_generation: Starting generation number
        """
        current_generation = start_generation
        rules_to_apply = self.rules.copy()
        
        while rules_to_apply:
            # Evaluate conditions for each remaining rule
            applicable_rules = [
                rule for rule in rules_to_apply
                if self._evaluate_conditions(rule, manifold)
            ]
            
            if not applicable_rules:
                break
                
            # Apply the first applicable rule
            rule = applicable_rules[0]
            rules_to_apply.remove(rule)
            
            # Infer and apply platonic ideals if needed
            if rule.requires_ideals:
                self._infer_and_apply_ideals(manifold)
                
            # Apply the rule
            rule.apply(manifold, current_generation)
            self.applied_rules.append(rule.name)
            
            # Record transformation
            self.transformation_history.append({
                'generation': current_generation,
                'rule': rule.name,
                'state': manifold.get_state_snapshot()
            })
            
            current_generation += 1
            
    def _apply_branching(self, manifold: VectorManifold, start_generation: int) -> None:
        """
        Apply rules following branching paths based on manifold state.
        
        Args:
            manifold: The vector manifold to transform
            start_generation: Starting generation number
        """
        current_generation = start_generation
        branches = self._evaluate_branches(manifold)
        
        for branch in branches:
            branch_rules = self.rules[branch['start']:branch['end']]
            
            for rule in branch_rules:
                # Check if branch conditions are still valid
                if not self._evaluate_branch_conditions(rule, manifold, branch):
                    break
                    
                # Infer and apply platonic ideals if needed
                if rule.requires_ideals:
                    self._infer_and_apply_ideals(manifold)
                    
                # Apply the rule
                rule.apply(manifold, current_generation)
                self.applied_rules.append(rule.name)
                
                # Record transformation
                self.transformation_history.append({
                    'generation': current_generation,
                    'rule': rule.name,
                    'branch': branch['name'],
                    'state': manifold.get_state_snapshot()
                })
                
                current_generation += 1
                
    def _evaluate_conditions(self, rule: CellularRule, manifold: VectorManifold) -> bool:
        """
        Evaluate conditions for rule application.
        
        Args:
            rule: The rule to evaluate
            manifold: The current manifold state
            
        Returns:
            bool: True if conditions are met, False otherwise
        """
        # Check dependencies
        if not self._check_dependencies(rule, manifold):
            return False
            
        # Check rule-specific conditions
        if hasattr(rule, 'conditions'):
            for condition in rule.conditions:
                if not condition(manifold):
                    return False
                    
        return True
        
    def _evaluate_branches(self, manifold: VectorManifold) -> List[Dict]:
        """
        Determine branches for rule application based on manifold state.
        
        Args:
            manifold: The current manifold state
            
        Returns:
            List[Dict]: List of branch definitions with start/end indices
        """
        branches = []
        current_branch = {'start': 0, 'name': 'main'}
        
        for i, rule in enumerate(self.rules):
            # Check for branch points
            if hasattr(rule, 'branch_point') and rule.branch_point:
                # End current branch
                current_branch['end'] = i
                branches.append(current_branch)
                
                # Start new branches
                for branch_name in rule.branch_options:
                    if self._evaluate_branch_conditions(rule, manifold, {'name': branch_name}):
                        branches.append({
                            'start': i + 1,
                            'name': branch_name,
                            'end': len(self.rules)
                        })
                        
        # Add final branch if not already added
        if not branches or branches[-1]['end'] < len(self.rules):
            current_branch['end'] = len(self.rules)
            branches.append(current_branch)
            
        return branches
        
    def _evaluate_branch_conditions(self, rule: CellularRule, 
                                  manifold: VectorManifold,
                                  branch: Dict) -> bool:
        """
        Evaluate conditions for a specific branch.
        
        Args:
            rule: The rule to evaluate
            manifold: The current manifold state
            branch: Branch information
            
        Returns:
            bool: True if branch conditions are met, False otherwise
        """
        if not hasattr(rule, 'branch_conditions'):
            return True
            
        branch_conditions = rule.branch_conditions.get(branch['name'], [])
        return all(condition(manifold) for condition in branch_conditions)
        
    def _infer_and_apply_ideals(self, manifold: VectorManifold) -> None:
        """
        Infer and apply platonic ideals during transformation.
        
        Args:
            manifold: The manifold to transform
        """
        # Get current manifold state
        vectors = manifold.get_vectors()
        
        # Infer platonic ideals
        ideals = manifold._infer_platonic_ideals(
            method='geometric',  # Use geometric analysis by default
            n_ideals=min(5, len(vectors))  # Infer up to 5 ideals
        )
        
        # Apply ideals to influence transformation
        for ideal in ideals:
            manifold.add_attractor(ideal)
            
    def _check_dependencies(self, rule: CellularRule, manifold: VectorManifold) -> bool:
        """
        Check if rule dependencies are satisfied.
        
        Args:
            rule: The rule to check
            manifold: The current manifold state
            
        Returns:
            bool: True if dependencies are met, False otherwise
        """
        if not hasattr(rule, 'dependencies'):
            return True
            
        return all(
            dep_rule.name in self.applied_rules 
            for dep_rule in rule.dependencies
        )


def create_predefined_sequences() -> Dict[str, RuleSequence]:
    """
    Create a set of predefined rule sequences based on
    magical rituals and occult correspondences.
    
    Returns:
        Dictionary mapping sequence names to RuleSequence objects
    """
    rules = create_predefined_rules()
    sequences = {}
    
    # The Great Work Sequence - Alchemical transformation from base to divine
    great_work_seq = RuleSequence(
        name="The Great Work",
        description="A sequence of transformations corresponding to alchemical stages",
        rules=[
            rules["tower"],       # Nigredo (blackening) - breakdown
            rules["hermit"],      # Separation and introspection
            rules["equilibrium"], # Albedo (whitening) - purification
            rules["star"],        # Citrinitas (yellowing) - awakening
            rules["great_work"]   # Rubedo (reddening) - completion
        ],
        esoteric_correspondence="The alchemical Great Work (Magnum Opus) - the complete transformation of the prima materia into the philosopher's stone"
    )
    sequences["great_work"] = great_work_seq
    
    # The Thelemic Sequence - Based on Crowley's principle
    thelemic_seq = RuleSequence(
        name="Thelemic Transformation",
        description="A sequence embodying the principle of 'Do what thou wilt shall be the whole of the Law'",
        rules=[
            rules["hermit"],    # Know thyself
            rules["magician"],  # Exercise will
            rules["lovers"],    # Union with True Will
            rules["wheel"],     # Accept cycles and karma
            rules["tower"],     # Break false structures
            rules["star"]       # Attain spiritual insight
        ],
        esoteric_correspondence="The Thelemic principle of discovering and manifesting one's True Will in harmony with the universe"
    )
    sequences["thelemic"] = thelemic_seq
    
    # The Kabbalistic Sequence - Based on ascent through Sephiroth
    kabbalistic_seq = RuleSequence(
        name="Path of the Sephiroth",
        description="A sequence corresponding to mystical ascent through the Tree of Life",
        rules=[
            rules["equilibrium"],  # Malkuth (stability)
            rules["lovers"],       # Yesod (foundation) -> Hod/Netzach (balance)
            rules["tower"],        # Breaking through to Tiphareth
            rules["star"],         # Tiphareth (beauty) -> higher realms
            rules["great_work"]    # Approaching the supernals
        ],
        esoteric_correspondence="The mystical ascent through the Kabbalistic Tree of Life, from the material realm to divine unity"
    )
    sequences["kabbalistic"] = kabbalistic_seq
    
    return sequences

