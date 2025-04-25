"""
Mandala Visualization Engine

Generates mandala patterns with configurable resolution and frame limits.
Supports both static and animated mandalas with various styles.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class MandalaStyle(Enum):
    """Available mandala styles."""
    MYSTICAL = "mystical"
    GEOMETRIC = "geometric"
    NATURAL = "natural"
    COSMIC = "cosmic"
    FRACTAL = "fractal"

@dataclass
class MandalaConfig:
    """Configuration for mandala generation."""
    radius: int
    resolution: Tuple[int, int]  # (width, height)
    style: MandalaStyle
    symmetry: int = 8
    density: float = 0.7
    complexity: float = 1.0
    frame_limit: Optional[int] = None
    frame_interval: float = 0.1  # seconds between frames

class MandalaEngine:
    """Engine for generating mandala patterns."""
    
    def __init__(self):
        self.current_frame = 0
        self.patterns: Dict[str, np.ndarray] = {}
    
    def generate_mandala(
        self,
        config: MandalaConfig,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate a static mandala pattern."""
        if seed is not None:
            np.random.seed(seed)
        
        width, height = config.resolution
        pattern = np.zeros((height, width), dtype=float)
        
        # Generate base pattern based on style
        if config.style == MandalaStyle.MYSTICAL:
            pattern = self._generate_mystical(config)
        elif config.style == MandalaStyle.GEOMETRIC:
            pattern = self._generate_geometric(config)
        elif config.style == MandalaStyle.NATURAL:
            pattern = self._generate_natural(config)
        elif config.style == MandalaStyle.COSMIC:
            pattern = self._generate_cosmic(config)
        elif config.style == MandalaStyle.FRACTAL:
            pattern = self._generate_fractal(config)
        
        return pattern
    
    def create_animation_frames(
        self,
        config: MandalaConfig,
        n_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """Create animation frames for mandala."""
        if n_frames is None:
            n_frames = config.frame_limit or 30
            
        frames = []
        for i in range(n_frames):
            # Adjust parameters for each frame
            frame_config = MandalaConfig(
                radius=config.radius,
                resolution=config.resolution,
                style=config.style,
                symmetry=config.symmetry,
                density=config.density,
                complexity=config.complexity * (1 + 0.1 * math.sin(2 * math.pi * i / n_frames))
            )
            
            frame = self.generate_mandala(frame_config, seed=i)
            frames.append(frame)
            
            # Check frame limit
            if config.frame_limit and len(frames) >= config.frame_limit:
                break
        
        return frames
    
    def _generate_mystical(self, config: MandalaConfig) -> np.ndarray:
        """Generate mystical style mandala."""
        width, height = config.resolution
        pattern = np.zeros((height, width))
        
        # Create circular base
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
        dist = np.sqrt(x*x + y*y)
        
        # Add mystical elements
        for i in range(config.symmetry):
            angle = 2 * np.pi * i / config.symmetry
            # Add spiral arms
            spiral = np.sin(dist/10 + angle) * np.exp(-dist/(config.radius*2))
            pattern += spiral * config.complexity
            
            # Add circular patterns
            rings = np.sin(dist/5) * np.exp(-dist/config.radius)
            pattern += rings * 0.5
        
        return np.clip(pattern, 0, 1)
    
    def _generate_geometric(self, config: MandalaConfig) -> np.ndarray:
        """Generate geometric style mandala."""
        width, height = config.resolution
        pattern = np.zeros((height, width))
        
        # Create geometric base
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
        dist = np.sqrt(x*x + y*y)
        angle = np.arctan2(y, x)
        
        # Add geometric elements
        for i in range(config.symmetry):
            # Add polygonal shapes
            poly = np.cos(angle * config.symmetry) * (dist < config.radius)
            pattern += poly * config.complexity
            
            # Add concentric patterns
            circles = ((dist/10).astype(int) % 2) * (dist < config.radius)
            pattern += circles * 0.5
        
        return np.clip(pattern, 0, 1)
    
    def _generate_natural(self, config: MandalaConfig) -> np.ndarray:
        """Generate natural style mandala."""
        width, height = config.resolution
        pattern = np.zeros((height, width))
        
        # Create organic base
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
        dist = np.sqrt(x*x + y*y)
        
        # Add natural elements
        for i in range(config.symmetry):
            angle = 2 * np.pi * i / config.symmetry
            # Add leaf-like patterns
            leaves = np.exp(-((dist-config.radius/2)**2)/(config.radius*10))
            pattern += leaves * np.cos(angle + np.arctan2(y, x) * 4) * config.complexity
            
            # Add organic texture
            texture = np.random.rand(height, width) * 0.1
            pattern += texture * np.exp(-dist/config.radius)
        
        return np.clip(pattern, 0, 1)
    
    def _generate_cosmic(self, config: MandalaConfig) -> np.ndarray:
        """Generate cosmic style mandala."""
        width, height = config.resolution
        pattern = np.zeros((height, width))
        
        # Create cosmic base
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
        dist = np.sqrt(x*x + y*y)
        
        # Add cosmic elements
        for i in range(config.symmetry):
            # Add galaxy-like spirals
            spiral = np.exp(-dist/(config.radius*2)) * np.sin(dist/10 + np.arctan2(y, x) * 4)
            pattern += spiral * config.complexity
            
            # Add star-like points
            stars = np.random.rand(height, width) * (dist < config.radius)
            pattern += stars * 0.2
        
        return np.clip(pattern, 0, 1)
    
    def _generate_fractal(self, config: MandalaConfig) -> np.ndarray:
        """Generate fractal style mandala."""
        width, height = config.resolution
        pattern = np.zeros((height, width))
        
        # Create fractal base
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
        dist = np.sqrt(x*x + y*y)
        
        # Add fractal elements using multiple scales
        scales = [1, 2, 4, 8]
        for scale in scales:
            # Add self-similar patterns at different scales
            pattern_scale = np.sin(dist * scale/10) * np.exp(-dist/(config.radius/scale))
            pattern += pattern_scale * (config.complexity / scale)
            
            # Add rotational symmetry
            for i in range(config.symmetry):
                angle = 2 * np.pi * i / config.symmetry
                rotated = np.sin(dist * scale/10 + angle) * np.exp(-dist/(config.radius/scale))
                pattern += rotated * (config.complexity / (scale * 2))
        
        return np.clip(pattern, 0, 1)
    
    def optimize_resolution(self, target_resolution: Tuple[int, int], max_size: int = 2048) -> Tuple[int, int]:
        """Optimize resolution while maintaining aspect ratio."""
        width, height = target_resolution
        aspect_ratio = width / height
        
        if width > max_size:
            width = max_size
            height = int(width / aspect_ratio)
        if height > max_size:
            height = max_size
            width = int(height * aspect_ratio)
            
        return (width, height)
    
    def estimate_memory_usage(self, config: MandalaConfig, n_frames: Optional[int] = None) -> int:
        """Estimate memory usage in bytes."""
        width, height = config.resolution
        frame_size = width * height * 8  # 8 bytes per float64
        n_frames = n_frames or config.frame_limit or 1
        
        return frame_size * n_frames 