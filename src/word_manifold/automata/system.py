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
        sequences_dict: Dict[str, Any] = None,
        evolution_pattern: EvolutionPattern = EvolutionPattern.LINEAR,
        save_path: Optional[str] = None
    ):
        """
        Initialize the automata system.
        
        Args:
            manifold: The vector manifold to evolve
            rules_dict: Dictionary of available rules
            sequences_dict: Dictionary of available rule sequences
            evolution_pattern: Pattern to follow when applying rules
            save_path: Directory to save system states and visualizations
        """
        self.manifold = manifold
        self.rules = rules_dict
        self.sequences = sequences_dict or {}
        self.evolution_pattern = evolution_pattern
        self.save_path = save_path
        
        # Initialize system state
        self.generation = 0
        self.history = []  # List of SystemState objects
        self.active_sequence = None
        self.sequence_position = 0
        
        logger.info(f"Initialized AutomataSystem with {len(self.rules)} rules and {len(self.sequences)} sequences")
        
        # Create save directory if specified
        if self.save_path:
            Path(self.save_path).mkdir(parents=True, exist_ok=True)
    
    def evolve(self, generations: int = 1) -> None:
        """
        Evolve the system for a specified number of generations.
        
        Args:
            generations: Number of generations to evolve
        """
        logger.info(f"Evolving system for {generations} generations following {self.evolution_pattern.name} pattern")
        
        for _ in range(generations):
            # Select and apply rule based on evolution pattern
            rule_name = self._select_next_rule()
            if rule_name and rule_name in self.rules:
                rule = self.rules[rule_name]
                rule.apply(self.manifold, self.generation)
                
                # Capture system state
                self._capture_state([rule_name])
                
                # Increment generation
                self.generation += 1
                
                logger.info(f"Generation {self.generation}: Applied rule '{rule_name}'")
            else:
                logger.warning(f"No valid rule selected for generation {self.generation}")
    
    def apply_sequence(self, sequence_name: str) -> None:
        """
        Apply a named sequence of rules.
        
        Args:
            sequence_name: Name of the sequence to apply
        """
        if sequence_name not in self.sequences:
            logger.error(f"Sequence '{sequence_name}' not found")
            return
            
        sequence = self.sequences[sequence_name]
        logger.info(f"Applying sequence '{sequence_name}' to manifold")
        
        sequence.apply(self.manifold, self.generation)
        
        # Capture system state with all rules in the sequence
        rule_names = [rule.name for rule in sequence.rules]
        self._capture_state(rule_names)
        
        # Increment generation by the number of rules in the sequence
        self.generation += len(sequence.rules)
        
        logger.info(f"Sequence '{sequence_name}' applied successfully")
    
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
    
    def _calculate_thelemic_resonance(self, rule: Any) -> float:
        """
        Calculate how well a rule resonates with the current system state
        according to Thelemic principles.
        
        Args:
            rule: The rule to evaluate
            
        Returns:
            Resonance score between 0.0 and 1.0
        """
        # This is a placeholder for a more sophisticated calculation
        # In a full implementation, this would analyze the current manifold state
        # and determine which rule best aligns with the system's "True Will"
        
        # For now, use a simple calculation based on the rule's parameters
        base_score = 0.5
        
        # Adjust based on generation number - different rules resonate at different times
        phase = (self.generation % 22) / 22.0  # 22 = number of major arcana
        
        # Each rule has peak resonance at a different phase of the cycle
        rule_name = rule.name.lower()
        if "star" in rule_name:
            peak_phase = 17/22  # Star is arcana 17
        elif "tower" in rule_name:
            peak_phase = 16/22
        elif "lovers" in rule_name:
            peak_phase = 6/22
        else:
            # Assign a random but consistent phase for other rules
            peak_phase = sum(ord(c) for c in rule_name) % 22 / 22
            
        # Calculate distance from peak (in phase space)
        phase_distance = min(abs(phase - peak_phase), 1 - abs(phase - peak_phase))
        
        # Higher score when closer to peak
        resonance = base_score + (1 - phase_distance)
        
        return resonance
    
    def _capture_state(self, active_rules: List[str]) -> None:
        """
        Capture the current state of the system.
        
        Args:
            active_rules: Names of rules that were applied in this generation
        """
        # Get manifold state
        manifold_state = self.manifold.get_manifold_state()
        
        # Calculate some metrics about the system state
        metrics = self._calculate_system_metrics()
        
        # Create system state object
        state = SystemState(
            generation=self.generation,
            active_rules=active_rules,
            manifold_state=manifold_state,
            timestamp=time.time(),
            metrics=metrics
        )
        
        # Add to history
        self.history.append(state)
        
        # Save state if save path is specified
        if self.save_path:
            self._save_state(state)
    
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
    
    def reset(self) -> None:
        """Reset the system to its initial state."""
        self.generation = 0
        self.history = []
        self.active_sequence = None
        self.sequence_position = 0
        
        logger.info("Reset automata system to initial state")

