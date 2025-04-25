"""
I Ching Hexagram Rules Module.

This module implements cellular automata rules based on I Ching hexagrams,
integrating the 64 hexagrams with vector space transformations.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
import random
from datetime import datetime

from .cellular_rules import CellularRule, RuleParameterSet, HermeticPrinciple, VibrationDirection
from ..manifold.vector_manifold import VectorManifold

# Default sacred texts for when no input is provided
DEFAULT_TEXTS = {
    'hermetic': [
        "As above, so below; as below, so above.",
        "All is mind; the universe is mental.",
        "Nothing rests; everything moves; everything vibrates.",
        "Everything is dual; everything has poles.",
        "Everything flows, out and in; everything has its tides.",
    ],
    'alchemical': [
        "Solve et Coagula",
        "The philosopher's stone transmutes all it touches",
        "In perfect unity, spirit and matter become one",
        "Through dissolution, the essence is revealed",
    ],
    'mystical': [
        "The light shines in the darkness",
        "In silence, wisdom speaks",
        "Unity manifests through diversity",
        "The eternal now contains all moments",
    ]
}

class Line(Enum):
    """I Ching line types."""
    YIN = 0   # Broken line ⚋
    YANG = 1  # Solid line ⚊

class CastingMethod(Enum):
    """Methods for casting hexagrams."""
    YARROW_STALKS = "yarrow"  # Traditional 50 yarrow stalks method
    COINS = "coins"           # Three coins method
    ORACLE_BONES = "bones"    # Ancient oracle bone method

@dataclass
class Hexagram:
    """Represents an I Ching hexagram."""
    number: int  # 1-64
    lines: List[Line]  # Bottom to top
    name: str
    attribute: str
    image: str
    nuclear_hexagram: Optional['Hexagram'] = None
    opposite_hexagram: Optional['Hexagram'] = None
    
    @property
    def binary_value(self) -> int:
        """Convert lines to binary number."""
        return sum(line.value << i for i, line in enumerate(self.lines))
    
    def get_trigrams(self) -> Tuple[List[Line], List[Line]]:
        """Get lower and upper trigrams."""
        return self.lines[:3], self.lines[3:]
    
    def get_nuclear_hexagram(self) -> 'Hexagram':
        """Get the nuclear hexagram (inner meaning)."""
        if not self.nuclear_hexagram:
            nuclear_lines = self.lines[1:4] + self.lines[2:5]
            self.nuclear_hexagram = hexagram_lookup[nuclear_lines_to_number(nuclear_lines)]
        return self.nuclear_hexagram

def cast_hexagram(method: CastingMethod = CastingMethod.ORACLE_BONES) -> Tuple[Hexagram, List[int]]:
    """
    Cast a hexagram using the specified method.
    
    Args:
        method: The divination method to use
        
    Returns:
        Tuple of (resulting hexagram, list of changing lines)
    """
    lines = []
    changing_lines = []
    
    if method == CastingMethod.YARROW_STALKS:
        # Simplified yarrow stalk probabilities
        for i in range(6):
            # Yarrow stalk method has uneven probabilities
            value = random.choices(
                [Line.YIN, Line.YANG],
                weights=[3, 5],  # Historical probabilities
                k=1
            )[0]
            changing = random.random() < 0.2  # 1/5 chance of changing
            lines.append(value)
            if changing:
                changing_lines.append(i)
                
    elif method == CastingMethod.COINS:
        # Three coins method
        for i in range(6):
            coins = [random.choice([0, 1]) for _ in range(3)]
            total = sum(coins)
            value = Line.YANG if total > 1 else Line.YIN
            changing = total in [0, 3]  # All heads or all tails
            lines.append(value)
            if changing:
                changing_lines.append(i)
                
    else:  # ORACLE_BONES
        # Use current time for divination
        timestamp = datetime.now().timestamp()
        seed = int(timestamp * 1000)
        random.seed(seed)
        
        # Create patterns based on natural cycles
        hour = datetime.now().hour
        day = datetime.now().day
        month = datetime.now().month
        
        for i in range(6):
            # Complex pattern based on time cycles
            cycle_value = (hour + day + month + i) % 6
            natural_tendency = cycle_value / 5.0  # 0.0 to 1.0
            
            # Incorporate randomness with natural tendency
            value = Line.YANG if random.random() < natural_tendency else Line.YIN
            changing = abs(natural_tendency - 0.5) < 0.1  # Change near balance points
            
            lines.append(value)
            if changing:
                changing_lines.append(i)
    
    # Get the hexagram from the lookup table
    hexagram_number = sum(line.value << i for i, line in enumerate(lines)) + 1
    hexagram = hexagram_lookup[hexagram_number]
    
    return hexagram, changing_lines

def get_default_text() -> str:
    """Get default text for oracle readings."""
    return random.choice(DEFAULT_TEXTS['mystical'])

class HexagramRule(CellularRule):
    """
    A cellular automata rule based on I Ching hexagram transformations.
    
    Each rule embodies the transformative principles of a specific hexagram,
    applying its wisdom to vector space evolution.
    """
    
    def __init__(
        self,
        hexagram: Hexagram,
        parameters: Optional[RuleParameterSet] = None,
        description: Optional[str] = None
    ):
        """Initialize hexagram-based rule."""
        if parameters is None:
            # Create default parameters based on hexagram properties
            parameters = self._derive_parameters(hexagram)
            
        super().__init__(
            name=f"Hexagram_{hexagram.number}_{hexagram.name}",
            description=description or f"Transformation rule based on {hexagram.name} hexagram",
            parameters=parameters,
            vector_transformation=self._get_transformation_type(hexagram),
            esoteric_correspondence=f"Based on I Ching hexagram {hexagram.number}: {hexagram.name}"
        )
        self.hexagram = hexagram
    
    def _derive_parameters(self, hexagram: Hexagram) -> RuleParameterSet:
        """Derive rule parameters from hexagram properties."""
        # Count yin and yang lines
        yin_count = sum(1 for line in hexagram.lines if line == Line.YIN)
        yang_count = 6 - yin_count
        
        # Calculate base magnitude based on balance
        magnitude = 0.5 + abs(yang_count - yin_count) / 6
        
        # Determine vibration direction based on line arrangement
        lower_trigram, upper_trigram = hexagram.get_trigrams()
        if sum(l.value for l in upper_trigram) > sum(l.value for l in lower_trigram):
            vib_dir = VibrationDirection.ASCENDING
        else:
            vib_dir = VibrationDirection.DESCENDING
            
        # Create numerological weights based on hexagram number
        num_weights = {hexagram.number: 1.2, hexagram.nuclear_hexagram.number: 0.8}
        
        return RuleParameterSet(
            magnitude=magnitude,
            principle=HermeticPrinciple.CORRESPONDENCE,
            vibration_direction=vib_dir,
            numerological_weights=num_weights
        )
    
    def _get_transformation_type(self, hexagram: Hexagram) -> str:
        """Determine transformation type based on hexagram properties."""
        # Use binary value to select transformation
        binary = hexagram.binary_value
        if binary & 0b111000:  # Upper trigram has more yang lines
            return 'expand'
        elif binary & 0b000111:  # Lower trigram has more yang lines
            return 'contract'
        else:
            return 'harmonize'
    
    def apply(self, manifold: VectorManifold, generation: int = 0) -> VectorManifold:
        """Apply hexagram-based transformation to manifold."""
        # Get nuclear hexagram for inner transformation
        nuclear = self.hexagram.get_nuclear_hexagram()
        
        # First apply outer transformation based on main hexagram
        transformed = super().apply(manifold, generation)
        
        # Then apply subtle inner transformation based on nuclear hexagram
        if nuclear != self.hexagram:
            nuclear_rule = HexagramRule(nuclear)
            # Reduce magnitude for nuclear transformation
            nuclear_rule.parameters.magnitude *= 0.5
            transformed = nuclear_rule.apply(transformed, generation)
            
        return transformed

def create_hexagram_rules() -> Dict[str, HexagramRule]:
    """Create the complete set of 64 hexagram-based rules."""
    rules = {}
    
    # Create rules for each hexagram
    for hexagram in hexagram_lookup.values():
        rule = HexagramRule(hexagram)
        rules[rule.name] = rule
        
    return rules

# Initialize hexagram lookup table with complete set
hexagram_lookup: Dict[int, Hexagram] = {
    1: Hexagram(1, [Line.YANG] * 6, "The Creative", "Creation", "Heaven"),
    2: Hexagram(2, [Line.YIN] * 6, "The Receptive", "Reception", "Earth"),
    3: Hexagram(3, [Line.YANG, Line.YIN, Line.YIN, Line.YIN, Line.YANG, Line.YIN], "Difficulty at the Beginning", "Growth", "Thunder over Water"),
    4: Hexagram(4, [Line.YIN, Line.YANG, Line.YIN, Line.YIN, Line.YIN, Line.YANG], "Youthful Folly", "Learning", "Mountain over Water"),
    5: Hexagram(5, [Line.YANG, Line.YANG, Line.YIN, Line.YIN, Line.YIN, Line.YANG], "Waiting", "Patience", "Water over Heaven"),
    6: Hexagram(6, [Line.YANG, Line.YIN, Line.YIN, Line.YIN, Line.YANG, Line.YANG], "Conflict", "Dispute", "Heaven over Water"),
    7: Hexagram(7, [Line.YIN, Line.YIN, Line.YIN, Line.YIN, Line.YANG, Line.YIN], "The Army", "Leadership", "Earth over Water"),
    8: Hexagram(8, [Line.YIN, Line.YANG, Line.YIN, Line.YIN, Line.YIN, Line.YIN], "Holding Together", "Union", "Water over Earth"),
    9: Hexagram(9, [Line.YANG, Line.YANG, Line.YIN, Line.YIN, Line.YANG, Line.YANG], "Small Taming", "Restraint", "Wind over Heaven"),
    10: Hexagram(10, [Line.YANG, Line.YANG, Line.YANG, Line.YIN, Line.YANG, Line.YANG], "Treading", "Conduct", "Heaven over Lake"),
    11: Hexagram(11, [Line.YANG, Line.YANG, Line.YANG, Line.YIN, Line.YIN, Line.YIN], "Peace", "Harmony", "Earth over Heaven"),
    12: Hexagram(12, [Line.YIN, Line.YIN, Line.YIN, Line.YANG, Line.YANG, Line.YANG], "Standstill", "Stagnation", "Heaven over Earth"),
    13: Hexagram(13, [Line.YANG, Line.YANG, Line.YANG, Line.YANG, Line.YIN, Line.YANG], "Fellowship", "Community", "Heaven over Fire"),
    14: Hexagram(14, [Line.YANG, Line.YIN, Line.YANG, Line.YANG, Line.YANG, Line.YANG], "Great Possession", "Wealth", "Fire over Heaven"),
    15: Hexagram(15, [Line.YIN, Line.YIN, Line.YANG, Line.YIN, Line.YIN, Line.YIN], "Modesty", "Humility", "Earth over Mountain"),
    16: Hexagram(16, [Line.YIN, Line.YIN, Line.YIN, Line.YANG, Line.YIN, Line.YIN], "Enthusiasm", "Joy", "Thunder over Earth"),
    17: Hexagram(17, [Line.YANG, Line.YIN, Line.YANG, Line.YIN, Line.YIN, Line.YANG], "Following", "Adaptation", "Lake over Thunder"),
    18: Hexagram(18, [Line.YANG, Line.YIN, Line.YIN, Line.YANG, Line.YIN, Line.YANG], "Work on the Decayed", "Repair", "Mountain over Wind"),
    19: Hexagram(19, [Line.YANG, Line.YANG, Line.YIN, Line.YIN, Line.YIN, Line.YIN], "Approach", "Progress", "Earth over Lake"),
    20: Hexagram(20, [Line.YIN, Line.YIN, Line.YIN, Line.YIN, Line.YANG, Line.YANG], "Contemplation", "View", "Wind over Earth"),
    21: Hexagram(21, [Line.YANG, Line.YIN, Line.YANG, Line.YANG, Line.YIN, Line.YANG], "Biting Through", "Decision", "Fire over Thunder"),
    22: Hexagram(22, [Line.YANG, Line.YIN, Line.YANG, Line.YANG, Line.YIN, Line.YIN], "Grace", "Beauty", "Mountain over Fire"),
    23: Hexagram(23, [Line.YIN, Line.YIN, Line.YIN, Line.YIN, Line.YIN, Line.YANG], "Splitting Apart", "Deterioration", "Mountain over Earth"),
    24: Hexagram(24, [Line.YANG, Line.YIN, Line.YIN, Line.YIN, Line.YIN, Line.YIN], "Return", "Turning Point", "Earth over Thunder"),
    25: Hexagram(25, [Line.YANG, Line.YIN, Line.YIN, Line.YANG, Line.YANG, Line.YANG], "Innocence", "Spontaneity", "Heaven over Thunder"),
    26: Hexagram(26, [Line.YANG, Line.YANG, Line.YANG, Line.YIN, Line.YIN, Line.YANG], "Great Taming", "Power", "Mountain over Heaven"),
    27: Hexagram(27, [Line.YANG, Line.YIN, Line.YIN, Line.YIN, Line.YIN, Line.YANG], "Nourishment", "Care", "Mountain over Thunder"),
    28: Hexagram(28, [Line.YANG, Line.YANG, Line.YANG, Line.YANG, Line.YIN, Line.YIN], "Great Preponderance", "Critical Mass", "Lake over Wind"),
    29: Hexagram(29, [Line.YIN, Line.YANG, Line.YIN, Line.YIN, Line.YANG, Line.YIN], "The Abysmal", "Danger", "Water over Water"),
    30: Hexagram(30, [Line.YANG, Line.YIN, Line.YANG, Line.YANG, Line.YIN, Line.YANG], "The Clinging", "Clarity", "Fire over Fire"),
    31: Hexagram(31, [Line.YIN, Line.YIN, Line.YANG, Line.YANG, Line.YANG, Line.YIN], "Influence", "Attraction", "Lake over Mountain"),
    32: Hexagram(32, [Line.YIN, Line.YANG, Line.YANG, Line.YANG, Line.YIN, Line.YIN], "Duration", "Persistence", "Thunder over Wind"),
    33: Hexagram(33, [Line.YIN, Line.YIN, Line.YANG, Line.YANG, Line.YANG, Line.YANG], "Retreat", "Withdrawal", "Heaven over Mountain"),
    34: Hexagram(34, [Line.YANG, Line.YANG, Line.YANG, Line.YANG, Line.YIN, Line.YIN], "Great Power", "Strength", "Thunder over Heaven"),
    35: Hexagram(35, [Line.YIN, Line.YIN, Line.YIN, Line.YANG, Line.YANG, Line.YIN], "Progress", "Advancement", "Fire over Earth"),
    36: Hexagram(36, [Line.YIN, Line.YANG, Line.YANG, Line.YIN, Line.YIN, Line.YIN], "Darkening of the Light", "Adversity", "Earth over Fire"),
    37: Hexagram(37, [Line.YANG, Line.YIN, Line.YANG, Line.YIN, Line.YIN, Line.YIN], "The Family", "Clan", "Wind over Fire"),
    38: Hexagram(38, [Line.YIN, Line.YIN, Line.YIN, Line.YANG, Line.YIN, Line.YANG], "Opposition", "Contrast", "Fire over Lake"),
    39: Hexagram(39, [Line.YIN, Line.YIN, Line.YANG, Line.YIN, Line.YANG, Line.YIN], "Obstruction", "Difficulty", "Water over Mountain"),
    40: Hexagram(40, [Line.YIN, Line.YANG, Line.YIN, Line.YANG, Line.YIN, Line.YIN], "Deliverance", "Relief", "Thunder over Water"),
    41: Hexagram(41, [Line.YANG, Line.YIN, Line.YIN, Line.YIN, Line.YANG, Line.YANG], "Decrease", "Reduction", "Mountain over Lake"),
    42: Hexagram(42, [Line.YANG, Line.YANG, Line.YIN, Line.YIN, Line.YIN, Line.YANG], "Increase", "Expansion", "Wind over Thunder"),
    43: Hexagram(43, [Line.YANG, Line.YANG, Line.YANG, Line.YANG, Line.YANG, Line.YIN], "Break-through", "Resolution", "Lake over Heaven"),
    44: Hexagram(44, [Line.YIN, Line.YANG, Line.YANG, Line.YANG, Line.YANG, Line.YANG], "Coming to Meet", "Encounter", "Heaven over Wind"),
    45: Hexagram(45, [Line.YIN, Line.YIN, Line.YIN, Line.YANG, Line.YANG, Line.YIN], "Gathering Together", "Collection", "Lake over Earth"),
    46: Hexagram(46, [Line.YIN, Line.YANG, Line.YANG, Line.YIN, Line.YIN, Line.YIN], "Pushing Upward", "Advancement", "Earth over Wind"),
    47: Hexagram(47, [Line.YIN, Line.YANG, Line.YIN, Line.YANG, Line.YANG, Line.YIN], "Oppression", "Exhaustion", "Lake over Water"),
    48: Hexagram(48, [Line.YIN, Line.YANG, Line.YANG, Line.YIN, Line.YANG, Line.YIN], "The Well", "Source", "Water over Wind"),
    49: Hexagram(49, [Line.YANG, Line.YIN, Line.YANG, Line.YANG, Line.YANG, Line.YIN], "Revolution", "Change", "Lake over Fire"),
    50: Hexagram(50, [Line.YIN, Line.YANG, Line.YANG, Line.YANG, Line.YIN, Line.YANG], "The Cauldron", "Transformation", "Fire over Wind"),
    51: Hexagram(51, [Line.YANG, Line.YIN, Line.YIN, Line.YANG, Line.YIN, Line.YANG], "The Arousing", "Shock", "Thunder over Thunder"),
    52: Hexagram(52, [Line.YANG, Line.YIN, Line.YANG, Line.YIN, Line.YIN, Line.YANG], "Keeping Still", "Stillness", "Mountain over Mountain"),
    53: Hexagram(53, [Line.YANG, Line.YIN, Line.YIN, Line.YANG, Line.YIN, Line.YIN], "Development", "Gradual Progress", "Wind over Mountain"),
    54: Hexagram(54, [Line.YIN, Line.YANG, Line.YANG, Line.YIN, Line.YANG, Line.YANG], "The Marrying Maiden", "Union", "Thunder over Lake"),
    55: Hexagram(55, [Line.YANG, Line.YIN, Line.YANG, Line.YANG, Line.YANG, Line.YIN], "Abundance", "Fullness", "Thunder over Fire"),
    56: Hexagram(56, [Line.YIN, Line.YANG, Line.YANG, Line.YANG, Line.YIN, Line.YANG], "The Wanderer", "Travel", "Fire over Mountain"),
    57: Hexagram(57, [Line.YANG, Line.YIN, Line.YANG, Line.YIN, Line.YANG, Line.YANG], "The Gentle", "Penetration", "Wind over Wind"),
    58: Hexagram(58, [Line.YANG, Line.YANG, Line.YIN, Line.YANG, Line.YIN, Line.YANG], "The Joyous", "Joy", "Lake over Lake"),
    59: Hexagram(59, [Line.YANG, Line.YIN, Line.YANG, Line.YIN, Line.YANG, Line.YIN], "Dispersion", "Dissolution", "Wind over Water"),
    60: Hexagram(60, [Line.YIN, Line.YANG, Line.YIN, Line.YANG, Line.YIN, Line.YANG], "Limitation", "Restriction", "Water over Lake"),
    61: Hexagram(61, [Line.YANG, Line.YIN, Line.YANG, Line.YIN, Line.YANG, Line.YANG], "Inner Truth", "Sincerity", "Wind over Lake"),
    62: Hexagram(62, [Line.YANG, Line.YANG, Line.YIN, Line.YANG, Line.YANG, Line.YIN], "Small Preponderance", "Detail", "Thunder over Mountain"),
    63: Hexagram(63, [Line.YANG, Line.YIN, Line.YANG, Line.YIN, Line.YANG, Line.YIN], "After Completion", "Fulfillment", "Water over Fire"),
    64: Hexagram(64, [Line.YIN, Line.YANG, Line.YIN, Line.YIN, Line.YANG, Line.YIN], "Before Completion", "Transition", "Fire over Water")
}

def nuclear_lines_to_number(lines: List[Line]) -> int:
    """Convert nuclear hexagram lines to hexagram number."""
    # Implementation to convert lines to hexagram number
    return sum(line.value << i for i, line in enumerate(lines)) + 1 