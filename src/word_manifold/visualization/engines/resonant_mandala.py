"""
Resonant Mandala Engine

Explores morphic resonance fields through mandala evolution.
Each mandala can tap into archetypal patterns and evolve based on
resonant frequencies and narrative threads.
"""

import numpy as np
import ray
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import math
import random
from datetime import datetime
import logging
from ..engines.mandala import MandalaConfig, MandalaStyle

logger = logging.getLogger(__name__)

class ResonanceField(Enum):
    """Core resonance fields that mandalas can tap into."""
    COSMIC_MEMORY = "cosmic_memory"      # Universal patterns and cosmic archetypes
    NATURAL_WISDOM = "natural_wisdom"    # Earth's living patterns and rhythms
    ANCESTRAL_ECHO = "ancestral_echo"    # Cultural and historical patterns
    QUANTUM_DREAM = "quantum_dream"      # Quantum-level consciousness patterns
    MYTHIC_CURRENT = "mythic_current"    # Mythological and archetypal streams
    DREAM_LOGIC = "dream_logic"          # Non-linear dream-state patterns
    EMOTIONAL_FIELD = "emotional_field"  # Emotional resonance patterns
    TIME_SPIRAL = "time_spiral"          # Temporal pattern dynamics
    VOID_WHISPERS = "void_whispers"      # Patterns from the primordial void
    SACRED_GEOMETRY = "sacred_geometry"  # Universal geometric principles
    AKASHIC_STREAM = "akashic_stream"    # Universal knowledge patterns
    GAIA_PULSE = "gaia_pulse"            # Earth's electromagnetic patterns

@dataclass
class ResonanceConfig:
    """Configuration for resonance exploration."""
    primary_field: ResonanceField
    secondary_fields: Set[ResonanceField]
    intensity: float = 0.7  # How strongly to tap into the field
    evolution_rate: float = 0.3  # How quickly patterns evolve
    narrative_coherence: float = 0.5  # How strongly to maintain story threads
    meditation_depth: int = 3  # Levels of recursive exploration
    collective_resonance: bool = False  # Whether to participate in network resonance

@dataclass
class CollectiveState:
    """State information for collective resonance."""
    participating_nodes: Set[str]
    field_strengths: Dict[ResonanceField, float]
    shared_patterns: Dict[str, np.ndarray]
    collective_coherence: float
    last_update: datetime

@ray.remote
class ResonanceWorker:
    """Distributed worker for exploring resonance patterns."""
    
    def __init__(self, field: ResonanceField):
        self.field = field
        self.patterns = {}
        self.current_evolution = 0
        self.collective_state: Optional[CollectiveState] = None
    
    def explore_pattern(
        self,
        seed_pattern: np.ndarray,
        depth: int,
        collective_state: Optional[CollectiveState] = None
    ) -> Dict[str, np.ndarray]:
        """Explore resonant patterns from a seed."""
        self.collective_state = collective_state
        patterns = {}
        current = seed_pattern.copy()
        
        for i in range(depth):
            # Apply field-specific transformations
            if self.field == ResonanceField.COSMIC_MEMORY:
                current = self._apply_cosmic_transformation(current)
            elif self.field == ResonanceField.NATURAL_WISDOM:
                current = self._apply_natural_transformation(current)
            elif self.field == ResonanceField.ANCESTRAL_ECHO:
                current = self._apply_ancestral_transformation(current)
            elif self.field == ResonanceField.QUANTUM_DREAM:
                current = self._apply_quantum_transformation(current)
            elif self.field == ResonanceField.MYTHIC_CURRENT:
                current = self._apply_mythic_transformation(current)
            elif self.field == ResonanceField.DREAM_LOGIC:
                current = self._apply_dream_transformation(current)
            elif self.field == ResonanceField.EMOTIONAL_FIELD:
                current = self._apply_emotional_transformation(current)
            elif self.field == ResonanceField.TIME_SPIRAL:
                current = self._apply_temporal_transformation(current)
            elif self.field == ResonanceField.VOID_WHISPERS:
                current = self._apply_void_transformation(current)
            elif self.field == ResonanceField.SACRED_GEOMETRY:
                current = self._apply_geometric_transformation(current)
            elif self.field == ResonanceField.AKASHIC_STREAM:
                current = self._apply_akashic_transformation(current)
            elif self.field == ResonanceField.GAIA_PULSE:
                current = self._apply_gaia_transformation(current)
            
            # Apply collective resonance if available
            if self.collective_state:
                current = self._apply_collective_resonance(current)
            
            patterns[f"depth_{i}"] = current
        
        return patterns
    
    def _apply_collective_resonance(self, pattern: np.ndarray) -> np.ndarray:
        """Apply collective resonance from other nodes."""
        if not self.collective_state:
            return pattern
            
        # Blend with shared patterns based on field strengths
        result = pattern.copy()
        total_strength = sum(self.collective_state.field_strengths.values())
        
        if total_strength > 0:
            for node_pattern in self.collective_state.shared_patterns.values():
                # Scale influence by field strength and coherence
                influence = (self.collective_state.collective_coherence * 
                           self.collective_state.field_strengths.get(self.field, 0) /
                           total_strength)
                result += node_pattern * influence
            
            result = np.clip(result, 0, 1)
        
        return result
    
    def _apply_dream_transformation(self, pattern: np.ndarray) -> np.ndarray:
        """Apply dream-logic transformations."""
        # Create non-linear dream-like distortions
        y, x = np.ogrid[-pattern.shape[0]//2:pattern.shape[0]//2, 
                       -pattern.shape[1]//2:pattern.shape[1]//2]
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        
        # Dream waves with varying frequencies
        dream_wave = np.sin(r/7 + theta*2) * np.cos(r/13 + theta*3)
        
        # Add some quantum uncertainty
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(pattern, sigma=random.uniform(0.5, 2.0))
        
        return np.clip(blurred + dream_wave * 0.3, 0, 1)
    
    def _apply_emotional_transformation(self, pattern: np.ndarray) -> np.ndarray:
        """Apply emotional field transformations."""
        # Create flowing emotional currents
        y, x = np.ogrid[-pattern.shape[0]//2:pattern.shape[0]//2, 
                       -pattern.shape[1]//2:pattern.shape[1]//2]
        r = np.sqrt(x*x + y*y)
        
        # Emotional resonance waves
        intensity = np.exp(-r/(pattern.shape[0]/3))
        flow = np.sin(r/15) * np.cos(r/10) * intensity
        
        # Add organic variation
        variation = np.random.rand(*pattern.shape) * 0.1
        
        return np.clip(pattern + flow * 0.4 + variation, 0, 1)
    
    def _apply_temporal_transformation(self, pattern: np.ndarray) -> np.ndarray:
        """Apply temporal pattern transformations."""
        # Create spiral time patterns
        y, x = np.ogrid[-pattern.shape[0]//2:pattern.shape[0]//2, 
                       -pattern.shape[1]//2:pattern.shape[1]//2]
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        
        # Time spiral with multiple frequencies
        time_spiral = np.sin(r/8 + theta*5) * np.exp(-r/(pattern.shape[0]/2))
        
        # Add temporal echoes
        echoes = np.zeros_like(pattern)
        for i in range(3):  # Three temporal echoes
            phase = 2 * np.pi * i / 3
            echo = np.sin(r/12 + theta + phase) * np.exp(-r/(pattern.shape[0]/3))
            echoes += echo
        
        return np.clip(pattern + time_spiral * 0.3 + echoes * 0.2, 0, 1)
    
    def _apply_void_transformation(self, pattern: np.ndarray) -> np.ndarray:
        """Apply void pattern transformations."""
        # Create void-like patterns
        y, x = np.ogrid[-pattern.shape[0]//2:pattern.shape[0]//2, 
                       -pattern.shape[1]//2:pattern.shape[1]//2]
        r = np.sqrt(x*x + y*y)
        
        # Void ripples
        void = 1 - np.exp(-r/(pattern.shape[0]/4))
        
        # Add quantum fluctuations
        fluctuations = np.random.rand(*pattern.shape) * 0.1
        
        return np.clip(pattern * void + fluctuations, 0, 1)
    
    def _apply_geometric_transformation(self, pattern: np.ndarray) -> np.ndarray:
        """Apply sacred geometric transformations."""
        # Create geometric patterns
        y, x = np.ogrid[-pattern.shape[0]//2:pattern.shape[0]//2, 
                       -pattern.shape[1]//2:pattern.shape[1]//2]
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        
        # Vesica piscis
        vesica = np.sin(theta * 2) * (r < pattern.shape[0]/3)
        
        # Flower of life pattern
        phi = (1 + np.sqrt(5)) / 2
        flower = np.sin(r/phi + theta*6) * np.exp(-r/(pattern.shape[0]/3))
        
        return np.clip(pattern + vesica * 0.3 + flower * 0.3, 0, 1)
    
    def _apply_akashic_transformation(self, pattern: np.ndarray) -> np.ndarray:
        """Apply akashic record transformations."""
        # Create knowledge pattern streams
        y, x = np.ogrid[-pattern.shape[0]//2:pattern.shape[0]//2, 
                       -pattern.shape[1]//2:pattern.shape[1]//2]
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        
        # Knowledge streams
        streams = np.zeros_like(pattern)
        for i in range(9):  # Nine streams of knowledge
            phase = 2 * np.pi * i / 9
            stream = np.sin(r/9 + theta*3 + phase) * np.exp(-r/(pattern.shape[0]/4))
            streams += stream
        
        return np.clip(pattern + streams * 0.25, 0, 1)
    
    def _apply_gaia_transformation(self, pattern: np.ndarray) -> np.ndarray:
        """Apply Earth's electromagnetic pattern transformations."""
        # Create Schumann resonance patterns
        y, x = np.ogrid[-pattern.shape[0]//2:pattern.shape[0]//2, 
                       -pattern.shape[1]//2:pattern.shape[1]//2]
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        
        # Schumann resonance (7.83 Hz base frequency)
        schumann = np.sin(r/7.83 + theta) * np.exp(-r/(pattern.shape[0]/3))
        
        # Add geomagnetic patterns
        field_lines = np.sin(theta * 2) * np.exp(-r/(pattern.shape[0]/4))
        
        return np.clip(pattern + schumann * 0.3 + field_lines * 0.2, 0, 1)

class ResonantMandalaEngine:
    """Engine for generating resonant mandalas that tell stories."""
    
    def __init__(self):
        self.workers = {
            field: ResonanceWorker.remote(field)
            for field in ResonanceField
        }
        self.evolution_history = []
    
    def generate_resonant_mandala(
        self,
        base_config: MandalaConfig,
        resonance_config: ResonanceConfig
    ) -> Dict[str, Any]:
        """Generate a mandala that explores resonant patterns."""
        
        # Generate base mandala
        from .mandala import MandalaEngine
        base_engine = MandalaEngine()
        base_pattern = base_engine.generate_mandala(base_config)
        
        # Explore primary resonance field
        primary_patterns = ray.get(
            self.workers[resonance_config.primary_field].explore_pattern.remote(
                base_pattern,
                resonance_config.meditation_depth
            )
        )
        
        # Explore secondary fields in parallel
        secondary_futures = [
            self.workers[field].explore_pattern.remote(
                base_pattern,
                max(1, resonance_config.meditation_depth - 1)
            )
            for field in resonance_config.secondary_fields
        ]
        secondary_patterns = ray.get(secondary_futures)
        
        # Combine patterns based on resonance
        final_pattern = base_pattern.copy()
        
        # Add primary field patterns with full intensity
        for pattern in primary_patterns.values():
            final_pattern += pattern * resonance_config.intensity
        
        # Add secondary field patterns with reduced intensity
        secondary_intensity = resonance_config.intensity * 0.5
        for field_patterns in secondary_patterns:
            for pattern in field_patterns.values():
                final_pattern += pattern * secondary_intensity
        
        final_pattern = np.clip(final_pattern, 0, 1)
        
        # Generate narrative interpretation
        narrative = self._interpret_resonance(
            resonance_config,
            len(primary_patterns),
            len(secondary_patterns)
        )
        
        return {
            "pattern": final_pattern,
            "narrative": narrative,
            "resonance_metrics": {
                "primary_depth": len(primary_patterns),
                "secondary_depths": [len(p) for p in secondary_patterns],
                "coherence": self._calculate_coherence(final_pattern),
                "evolution_stage": len(self.evolution_history)
            }
        }
    
    def evolve_mandala(
        self,
        current_state: Dict[str, Any],
        resonance_config: ResonanceConfig,
        evolution_steps: int = 1
    ) -> List[Dict[str, Any]]:
        """Evolve a mandala through its resonance fields."""
        evolution = []
        current_pattern = current_state["pattern"]
        
        for step in range(evolution_steps):
            # Adjust resonance based on evolution
            evolved_config = self._evolve_resonance(resonance_config, step)
            
            # Generate new state
            new_state = ray.get(
                self.workers[evolved_config.primary_field].explore_pattern.remote(
                    current_pattern,
                    evolved_config.meditation_depth
                )
            )
            
            # Select most resonant pattern
            evolved_pattern = max(
                new_state.values(),
                key=lambda p: self._calculate_coherence(p)
            )
            
            # Generate narrative for this evolution
            narrative = self._interpret_evolution(
                current_state["narrative"],
                evolved_config,
                step
            )
            
            state = {
                "pattern": evolved_pattern,
                "narrative": narrative,
                "evolution_metrics": {
                    "step": step,
                    "coherence_delta": self._calculate_coherence(evolved_pattern) -
                                     self._calculate_coherence(current_pattern),
                    "field_strength": self._calculate_field_strength(evolved_pattern)
                }
            }
            
            evolution.append(state)
            current_pattern = evolved_pattern
            self.evolution_history.append(state)
        
        return evolution
    
    def _calculate_coherence(self, pattern: np.ndarray) -> float:
        """Calculate the coherence of a pattern."""
        # Use gradient magnitude as a measure of pattern coherence
        from scipy.ndimage import sobel
        gx = sobel(pattern, axis=0)
        gy = sobel(pattern, axis=1)
        gradient_mag = np.sqrt(gx*gx + gy*gy)
        return 1.0 - np.mean(gradient_mag)
    
    def _calculate_field_strength(self, pattern: np.ndarray) -> float:
        """Calculate the resonant field strength."""
        # Use pattern energy and organization as field strength
        energy = np.mean(pattern)
        organization = 1.0 - np.std(pattern)
        return (energy + organization) / 2
    
    def _evolve_resonance(
        self,
        config: ResonanceConfig,
        step: int
    ) -> ResonanceConfig:
        """Evolve resonance configuration."""
        # Deepen meditation as evolution progresses
        evolved_depth = config.meditation_depth + step // 3
        
        # Adjust field intensity based on coherence
        evolved_intensity = config.intensity * (1.0 + 0.1 * math.sin(step/3))
        
        # Potentially add new secondary fields
        available_fields = set(ResonanceField) - {config.primary_field}
        n_secondary = len(config.secondary_fields)
        if random.random() < 0.3 and n_secondary < len(available_fields):
            new_field = random.choice(list(available_fields - config.secondary_fields))
            secondary_fields = config.secondary_fields | {new_field}
        else:
            secondary_fields = config.secondary_fields
        
        return ResonanceConfig(
            primary_field=config.primary_field,
            secondary_fields=secondary_fields,
            intensity=evolved_intensity,
            evolution_rate=config.evolution_rate,
            narrative_coherence=config.narrative_coherence,
            meditation_depth=evolved_depth,
            collective_resonance=config.collective_resonance
        )
    
    def _interpret_resonance(
        self,
        config: ResonanceConfig,
        primary_depth: int,
        secondary_depths: List[int]
    ) -> Dict[str, Any]:
        """Generate narrative interpretation of resonance patterns."""
        
        field_narratives = {
            ResonanceField.COSMIC_MEMORY: [
                "echoes of stellar birth",
                "whispers of galactic winds",
                "cosmic dance patterns",
                "universal breathing"
            ],
            ResonanceField.NATURAL_WISDOM: [
                "earth's living rhythms",
                "forest consciousness",
                "oceanic memories",
                "mountain dreams"
            ],
            ResonanceField.ANCESTRAL_ECHO: [
                "ancient knowledge streams",
                "tribal memories",
                "wisdom of elders",
                "cultural heartbeats"
            ],
            ResonanceField.QUANTUM_DREAM: [
                "quantum possibility waves",
                "probability dances",
                "entangled consciousness",
                "quantum foam patterns"
            ],
            ResonanceField.MYTHIC_CURRENT: [
                "archetypal rivers",
                "mythic story threads",
                "hero's journey spirals",
                "divine play patterns"
            ]
        }
        
        # Select narrative elements based on fields and depths
        primary_narrative = random.choice(field_narratives[config.primary_field])
        secondary_narratives = [
            random.choice(field_narratives[field])
            for field in config.secondary_fields
        ]
        
        # Combine into coherent narrative
        return {
            "primary_theme": primary_narrative,
            "secondary_themes": secondary_narratives,
            "depth_interpretation": f"Exploring to depth {primary_depth}",
            "resonance_quality": f"Resonating at {config.intensity:.1f} intensity",
            "field_interaction": len(config.secondary_fields) > 0
        }
    
    def _interpret_evolution(
        self,
        previous_narrative: Dict[str, Any],
        config: ResonanceConfig,
        step: int
    ) -> Dict[str, Any]:
        """Interpret the evolution of the narrative."""
        
        evolution_patterns = [
            "deepening into {theme}",
            "spiraling through {theme}",
            "dancing with {theme}",
            "weaving through {theme}"
        ]
        
        pattern = random.choice(evolution_patterns)
        primary_evolution = pattern.format(
            theme=previous_narrative["primary_theme"]
        )
        
        secondary_evolutions = [
            pattern.format(theme=theme)
            for theme in previous_narrative["secondary_themes"]
        ]
        
        return {
            "evolution_stage": step,
            "primary_movement": primary_evolution,
            "secondary_movements": secondary_evolutions,
            "depth_reached": config.meditation_depth,
            "field_strength": config.intensity
        } 