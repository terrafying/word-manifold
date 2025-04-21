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

from ..manifold.vector_manifold import VectorManifold, Cell, CellType, DistanceType
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
    
    def __post_init__(self):
        """Initialize default dictionaries if None."""
        if self.elemental_influence is None:
            self.elemental_influence = {e: 1.0 for e in ElementalForce}
        if self.numerological_weights is None:
            self.numerological_weights = {}
            # Master numbers have special weight
            for i in range(1, 10):
                self.numerological_weights[i] = 1.0
            # Master numbers have higher weight
            for num in [11, 22, 33]:
                self.numerological_weights[num] = 1.5
        if self.cell_type_weights is None:
            self.cell_type_weights = {ct: 1.0 for ct in CellType}


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
        
        # Evolve the manifold using the modified rules
        manifold.evolve_manifold(evolution_rules)
        
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
        magnitude=0.8,
        principle=HermeticPrinciple.CORRESPONDENCE,
        vibration_direction=VibrationDirection.ASCENDING,
        numerological_weights={
            1: 1.0,   # Unity
            2: 0.8,   # Duality
            3: 1.2,   # Synthesis
            4: 0.9,   # Stability
            5: 1.1,   # Change
            6: 0.7,   # Harmony
            7: 1.3,   # Spirituality
            8: 1.0,   # Power
            9: 1.2,   # Completion
            11: 1.5,  # Master vibration
            22: 1.7,  # Master builder
            33: 2.0   # Master teacher
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
    
    return rules


class RuleSequence:
    """
    A sequence of cellular rules to be applied in a specific order,
    representing a ritual or magical working in the vector space.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        rules: List[CellularRule],
        esoteric_correspondence: str
    ):
        """
        Initialize a rule sequence.
        
        Args:
            name: Name of the sequence
            description: Technical description of the sequence
            rules: List of rules to apply in order
            esoteric_correspondence: Description of occult/hermetic correspondence
        """
        self.name = name
        self.description = description
        self.rules = rules
        self.esoteric_correspondence = esoteric_correspondence
    
    def apply(self, manifold: VectorManifold, start_generation: int = 0) -> None:
        """
        Apply the sequence of rules to transform the manifold.
        
        Args:
            manifold: The vector manifold to transform
            start_generation: Starting generation number
        """
        logger.info(f"Applying rule sequence '{self.name}' to manifold (starting at generation {start_generation})")
        
        for i, rule in enumerate(self.rules):
            generation = start_generation + i
            rule.apply(manifold, generation)
            
        logger.info(f"Rule sequence '{self.name}' applied successfully")


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
    rules["star"] =

