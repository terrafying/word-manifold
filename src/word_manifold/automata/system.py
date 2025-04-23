"""
Automata System Module for Word Vector Space Cellular Automata.

This module implements the AutomataSystem class that orchestrates the application
of cellular automata rules to a word vector manifold. It provides functionality
for managing rule sequences, tracking system state across generations, and
implementing different patterns of evolution, all while integrating hermetic
principles at the system level.

"Do what thou wilt shall be the whole of the Law."
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from word_manifold.manifold.vector_manifold import VectorManifold, Cell, CellType
from word_manifold.embeddings.word_embeddings import WordEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvolutionPattern(Enum):
    """Patterns of evolution that the automata system can follow."""
    LINEAR = auto()      # Sequential application of rules
    CYCLIC = auto()      # Repeated application of rules in a cycle
    SPIRAL = auto()      # Cyclic with increasing intensity
    CHAOTIC = auto()     # Random selection of rules
    THELEMIC = auto()    # Rules selected based on True Will principle
    KABBALISTIC = auto() # Rules follow Tree of Life pattern

@dataclass
class SystemState:
    """State of the automata system at a point in time."""
    generation: int                # Current generation number
    active_rules: List[str]        # Names of rules currently active
    manifold_state: Dict[str, Any] # State snapshot of the manifold
    timestamp: float               # Unix timestamp when state was captured
    metrics: Dict[str, float]      # Metrics about the system's state

class AutomataSystem:
    """
    A system that orchestrates the application of cellular automata rules
    to a word vector manifold according to hermetic principles.
    
    This class manages the evolution of the manifold through generations,
    applying rules according to specified patterns and tracking the system's
    state over time.
    """
    
    def __init__(
        self,
        manifold: VectorManifold,
        rules_dict: Dict[str, Any],
        sequences_dict: Dict[str, Any],
        evolution_pattern: EvolutionPattern = EvolutionPattern.LINEAR,
        save_path: Optional[str] = None
    ):
        """
        Initialize the automata system.

        Args:
            manifold: The manifold to evolve
            rules_dict: Dictionary mapping rule names to rule objects
            sequences_dict: Dictionary mapping sequence names to sequence objects
            evolution_pattern: Pattern to use for rule selection
            save_path: Optional path to save system states
        """
        self.manifold = manifold
        self.rules = rules_dict
        self.sequences = sequences_dict
        self.evolution_pattern = evolution_pattern
        self.save_path = Path(save_path) if save_path else None
        
        # Initialize system state
        self.generation = 0
        self.history = []
        
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized automata system with {len(self.rules)} rules and {len(self.sequences)} sequences")
    
    def evolve(self, generations: int = 1) -> None:
        """
        Evolve the system for a specified number of generations.

        Args:
            generations: Number of generations to evolve
        """
        for _ in range(generations):
            # Select next rule based on evolution pattern
            rule_name = self._select_next_rule()
            if rule_name and rule_name in self.rules:
                # Apply the selected rule
                rule = self.rules[rule_name]
                # Get all cell vectors that need to be transformed
                cell_vectors = np.array([cell.centroid for cell in self.manifold.cells.values()])
                # Construct evolution rules
                evolution_rules = {
                    'transformation': rule.vector_transformation,
                    'magnitude': rule.parameters.magnitude,
                    'cell_type_weights': rule.parameters.cell_type_weights,
                    'numerological_weights': rule.parameters.numerological_weights
                }
                # Transform the vectors
                transformed_vectors = self.manifold.transform(cell_vectors, evolution_rules)
                # Update cell centroids
                for i, (cell_id, cell) in enumerate(self.manifold.cells.items()):
                    cell.centroid = transformed_vectors[i]
            
            # Increment generation counter
            self.generation += 1
            
            # Capture system state
            self._capture_state([rule_name] if rule_name else [])
    
    def apply_sequence(self, sequence_name: str) -> None:
        """
        Apply a named sequence of rules to the system.

        Args:
            sequence_name: Name of the sequence to apply
        """
        if sequence_name not in self.sequences:
            logger.warning(f"Sequence '{sequence_name}' not found")
            return
            
        sequence = self.sequences[sequence_name]
        sequence.apply(self.manifold, self.generation)
        self.generation += len(sequence.rules)
        
        # Capture final state after sequence
        self._capture_state([r.name for r in sequence.rules])
    
    def _capture_state(self, active_rules: List[str]) -> None:
        """
        Capture the current state of the system.

        Args:
            active_rules: List of currently active rule names
        """
        state = SystemState(
            generation=self.generation,
            active_rules=active_rules,
            manifold_state=self.manifold.get_manifold_state(),
            timestamp=time.time(),
            metrics=self._calculate_system_metrics()
        )
        
        # Add to history
        self.history.append(state)
        
        # Save state if path is configured
        if self.save_path:
            self._save_state(state)
    
    def _calculate_system_metrics(self) -> Dict[str, float]:
        """Calculate metrics about the current system state."""
        metrics = {}
        
        # Calculate basic metrics about the manifold
        manifold_state = self.manifold.get_manifold_state()
        
        # Average cell connectivity
        if "avg_connectivity" in manifold_state:
            metrics["avg_connectivity"] = manifold_state["avg_connectivity"]
            
        # Distribution of cell types
        if "type_counts" in manifold_state:
            total_cells = sum(manifold_state["type_counts"].values())
            for cell_type, count in manifold_state["type_counts"].items():
                metrics[f"cell_type_{cell_type}_ratio"] = count / total_cells
                
        # Distribution of numerological values
        if "num_value_counts" in manifold_state:
            total_values = sum(manifold_state["num_value_counts"].values())
            for value, count in manifold_state["num_value_counts"].items():
                metrics[f"num_value_{value}_ratio"] = count / total_values
                
        return metrics
    
    def _save_state(self, state: SystemState) -> None:
        """
        Save a system state to disk.

        Args:
            state: The state to save
        """
        if not self.save_path:
            return
            
        # Create state directory for this generation
        state_dir = self.save_path / f"generation_{state.generation}"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Save state metadata
        metadata = {
            "generation": state.generation,
            "active_rules": state.active_rules,
            "timestamp": state.timestamp,
            "metrics": state.metrics
        }
        
        with open(state_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Save manifold state (if supported)
        if hasattr(self.manifold, "save_state"):
            self.manifold.save_state(state_dir / "manifold_state.npz")
    
    def get_history_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of the system's evolution history."""
        summary = []
        for state in self.history:
            summary.append({
                "generation": state.generation,
                "active_rules": state.active_rules,
                "timestamp": state.timestamp,
                "metrics": state.metrics
            })
        return summary
    
    def reset(self) -> None:
        """Reset the system to its initial state."""
        self.generation = 0
        self.history = []
        # Reset manifold if supported
        if hasattr(self.manifold, "reset"):
            self.manifold.reset()
            
    def _calculate_thelemic_resonance(self, rule: Any) -> float:
        """
        Calculate how well a rule resonates with the current system state
        according to Thelemic principles.
        
        Args:
            rule: The rule to evaluate
            
        Returns:
            Resonance score between 0.0 and 1.0
        """
        # Get current manifold state
        manifold_state = self.manifold.get_manifold_state()
        
        # Base resonance on several factors:
        
        # 1. Numerological correspondence
        num_resonance = 0.0
        if "num_value_counts" in manifold_state:
            # Check if rule's numerological values align with manifold state
            rule_values = set()
            if hasattr(rule, "parameters") and hasattr(rule.parameters, "numerological_weights"):
                rule_values = set(rule.parameters.numerological_weights.keys())
            
            manifold_values = set(manifold_state["num_value_counts"].keys())
            # Calculate overlap
            if rule_values and manifold_values:
                num_resonance = len(rule_values & manifold_values) / len(rule_values | manifold_values)
        
        # 2. Cell type correspondence
        type_resonance = 0.0
        if "type_counts" in manifold_state:
            # Check if rule's cell type weights align with manifold state
            rule_types = set()
            if hasattr(rule, "parameters") and hasattr(rule.parameters, "cell_type_weights"):
                rule_types = set(rule.parameters.cell_type_weights.keys())
            
            manifold_types = set(manifold_state["type_counts"].keys())
            # Calculate overlap
            if rule_types and manifold_types:
                type_resonance = len(rule_types & manifold_types) / len(rule_types | manifold_types)
        
        # 3. Vibrational correspondence
        vib_resonance = 0.0
        if hasattr(rule, "parameters") and hasattr(rule.parameters, "vibration_direction"):
            # Different vibration directions are more appropriate at different stages
            # This is a simplified model - could be made more sophisticated
            cycle_position = self.generation % 7  # 7 is significant in Thelema
            
            # Map cycle positions to preferred vibration directions
            preferred_directions = {
                0: "ASCENDING",    # Beginning of cycle - rise
                1: "EXPANDING",    # Early cycle - grow
                2: "HARMONIZING", # Mid cycle - stabilize
                3: "POLARIZING",  # Mid cycle - differentiate
                4: "CONTRACTING", # Late cycle - consolidate
                5: "DESCENDING",  # End cycle - ground
                6: "HARMONIZING"  # Transition - rebalance
            }
            
            if rule.parameters.vibration_direction.name == preferred_directions[cycle_position]:
                vib_resonance = 1.0
            else:
                vib_resonance = 0.3  # Base resonance for any vibration
        
        # Combine factors (weights could be adjusted)
        resonance = (num_resonance * 0.4 + 
                    type_resonance * 0.3 + 
                    vib_resonance * 0.3)
                    
        return resonance
    
    def _select_next_rule(self) -> Optional[str]:
        """
        Select the next rule to apply based on the evolution pattern.
        
        Returns:
            Name of the selected rule, or None if no rule is selected
        """
        rule_names = list(self.rules.keys())
        if not rule_names:
            return None
            
        if self.evolution_pattern == EvolutionPattern.LINEAR:
            # Select rules in order
            return rule_names[self.generation % len(rule_names)]
            
        elif self.evolution_pattern == EvolutionPattern.CYCLIC:
            # Select rules in a repeating cycle
            return rule_names[self.generation % len(rule_names)]
            
        elif self.evolution_pattern == EvolutionPattern.SPIRAL:
            # Select rules in a cycle with increasing intensity
            rule_idx = self.generation % len(rule_names)
            rule = self.rules[rule_names[rule_idx]]
            
            # Increase magnitude based on cycle number
            cycle_num = self.generation // len(rule_names)
            intensity_factor = 1.0 + (cycle_num * 0.1)  # 10% increase per cycle
            
            # Modify rule parameters
            rule.parameters.magnitude *= intensity_factor
            
            return rule_names[rule_idx]
            
        elif self.evolution_pattern == EvolutionPattern.CHAOTIC:
            # Select rules randomly
            return np.random.choice(rule_names)
            
        elif self.evolution_pattern == EvolutionPattern.THELEMIC:
            # Select rules based on True Will principle
            # This is a more complex selection that considers the current state
            # and tries to select the rule that best advances the system's "will"
            
            # For now, implement a simple heuristic
            # Choose rule that has the highest "resonance" with current state
            best_rule = None
            best_score = -1
            
            for rule_name, rule in self.rules.items():
                # Calculate resonance based on numerological correspondences
                score = self._calculate_thelemic_resonance(rule)
                if score > best_score:
                    best_score = score
                    best_rule = rule_name
                    
            return best_rule
            
        elif self.evolution_pattern == EvolutionPattern.KABBALISTIC:
            # Select rules following Tree of Life pattern
            # This follows a specific order corresponding to the sephiroth
            kabbalistic_order = [
                "malkuth", "yesod", "hod", "netzach", "tiphareth",
                "geburah", "chesed", "binah", "chokmah", "kether"
            ]
            
            # Find rules that correspond to these sephiroth
            for sephirah in kabbalistic_order:
                for rule_name in rule_names:
                    if sephirah in rule_name.lower():
                        return rule_name
                        
            # If no matching rule, fall back to linear selection
            return rule_names[self.generation % len(rule_names)]
            
        return None
    
    def _calculate_system_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics about the current system state.
        
        Returns:
            Dictionary of metric names to values
        """
        # This is a placeholder for more sophisticated metrics
        # In a full implementation, this would calculate various metrics
        # about the system state, such as entropy, diversity, etc.
        
        metrics = {
            "entropy": 0.0,
            "diversity": 0.0,
            "coherence": 0.0,
            "stability": 0.0
        }
        
        # Get cell data from manifold
        cells = list(self.manifold.cells.values())
        
        if cells:
            # Calculate entropy based on cell type distribution
            type_counts = {}
            for cell in cells:
                cell_type = cell.type
                type_counts[cell_type] = type_counts.get(cell_type, 0) + 1
                
            # Shannon entropy
            total = len(cells)
            entropy = 0.0
            for count in type_counts.values():
                p = count / total
                entropy -= p * np.log2(p)
                
            # Normalize to [0, 1]
            max_entropy = np.log2(len(type_counts)) if type_counts else 0
            if max_entropy > 0:
                metrics["entropy"] = entropy / max_entropy
                
            # Calculate diversity as number of distinct cell types / max possible
            metrics["diversity"] = len(type_counts) / len(CellType)
            
            # Other metrics would require more complex calculations
            # with the actual manifold state
        
        return metrics
    
    def _save_state(self, state: SystemState) -> None:
        """
        Save the system state to disk.
        
        Args:
            state: The system state to save
        """
        if not self.save_path:
            return
            
        # Create a directory for this generation
        gen_dir = Path(self.save_path) / f"generation_{state.generation:04d}"
        gen_dir.mkdir(exist_ok=True)
        
        # Save state metadata as JSON
        metadata = {
            "generation": state.generation,
            "active_rules": state.active_rules,
            "timestamp": state.timestamp,
            "metrics": state.metrics
        }
        
        with open(gen_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Note: We're not saving the full manifold state here as it might be large
        # and complex. In a real implementation, you might want to save specific
        # aspects of it or use a more efficient serialization format.
        
        logger.info(f"Saved system state for generation {state.generation}")
    
    def get_history_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of the system's evolution history.
        
        Returns:
            List of dictionaries with summary information for each generation
        """
        summary = []
        
        for state in self.history:
            summary.append({
                "generation": state.generation,
                "active_rules": state.active_rules,
                "timestamp": state.timestamp,
                "metrics": state.metrics
            })
            
        return summary
    
    def _reset(self) -> None:
        """Reset the system to its initial state."""
        self.generation = 0
        self.history = []
        self.active_sequence = None
        self.sequence_position = 0
        
        logger.info("Reset automata system to initial state")

