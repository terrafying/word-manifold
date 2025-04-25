"""
ASCII Engine for Symbolic Visualizations

This module provides the core engine for generating ASCII art visualizations,
including mandalas, patterns, and symbolic fields.
"""

import numpy as np
from typing import List, Tuple, Optional
import math

class ASCIIEngine:
    """Engine for generating ASCII art visualizations."""
    
    def __init__(self):
        # ASCII characters for different densities
        self.density_chars = ' .:-=+*#%@'
        self.special_chars = '♠♡♢♣★☆⚝✧✦❈❉❊❋✺✹✸✷✶✵✴✳'
        
    def generate_mandala_pattern(
        self,
        term: str,
        radius: int,
        rotation: float = 0.0
    ) -> str:
        """Generate a mandala pattern centered around a term."""
        # Create empty canvas
        size = 2 * radius + 1
        pattern = [[' ' for _ in range(size)] for _ in range(size)]
        center = radius
        
        # Generate base pattern using term length as complexity factor
        complexity = len(term) / 10  # Scale complexity with term length
        
        for y in range(size):
            for x in range(size):
                # Calculate polar coordinates
                dx = x - center
                dy = y - center
                r = math.sqrt(dx*dx + dy*dy)
                theta = math.atan2(dy, dx) + rotation
                
                if r <= radius:
                    # Create mandala pattern using mathematical functions
                    value = (
                        math.sin(theta * complexity) * 
                        math.cos(r / radius * math.pi * 2) +
                        math.cos(theta * complexity/2) * 
                        math.sin(r / radius * math.pi)
                    )
                    
                    # Map value to ASCII character
                    char_idx = int((value + 1) * (len(self.density_chars) - 1) / 2)
                    pattern[y][x] = self.density_chars[char_idx]
        
        return '\n'.join(''.join(row) for row in pattern)
    
    def generate_term_influence(
        self,
        term: str,
        phase: float = 0.0
    ) -> str:
        """Generate a pattern representing a term's influence."""
        # Create pattern based on term characteristics
        width = len(term) * 2
        height = len(term)
        pattern = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Create wave-like pattern
        for y in range(height):
            for x in range(width):
                # Generate value using term-based parameters
                value = math.sin(
                    x / width * math.pi * 2 * len(term) +
                    y / height * math.pi * len(term) +
                    phase
                )
                
                # Map to ASCII character
                char_idx = int((value + 1) * (len(self.special_chars) - 1) / 2)
                pattern[y][x] = self.special_chars[char_idx]
        
        return '\n'.join(''.join(row) for row in pattern)
    
    def combine_patterns(self, pattern1: str, pattern2: str) -> str:
        """Combine two ASCII patterns with blending."""
        lines1 = pattern1.split('\n')
        lines2 = pattern2.split('\n')
        
        # Get dimensions
        height1, width1 = len(lines1), len(lines1[0])
        height2, width2 = len(lines2), len(lines2[0])
        
        # Create output pattern with maximum dimensions
        height = max(height1, height2)
        width = max(width1, width2)
        
        result = []
        for y in range(height):
            row = ''
            for x in range(width):
                # Get characters from both patterns
                char1 = lines1[y % height1][x % width1] if y < height1 and x < width1 else ' '
                char2 = lines2[y % height2][x % width2] if y < height2 and x < width2 else ' '
                
                # Blend characters (use non-space character if available)
                row += char2 if char1 == ' ' else char1
            
            result.append(row)
        
        return '\n'.join(result)
    
    def generate_symbolic_field(
        self,
        text: str,
        width: int,
        height: int,
        density: float = 0.7
    ) -> str:
        """Generate a symbolic field based on text."""
        # Create empty field
        field = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Generate field pattern based on text characteristics
        words = text.split()
        for y in range(height):
            for x in range(width):
                # Calculate pattern value based on text properties
                value = 0
                for i, word in enumerate(words):
                    # Create interference pattern for each word
                    freq = (len(word) + 1) / 10
                    phase = i * math.pi / len(words)
                    value += math.sin(x * freq + phase) * math.cos(y * freq + phase)
                
                # Normalize value and apply density threshold
                value = (value / len(words) + 1) / 2
                if value > (1 - density):
                    char_idx = int(value * (len(self.special_chars) - 1))
                    field[y][x] = self.special_chars[char_idx]
        
        return '\n'.join(''.join(row) for row in field)
    
    def generate_pattern(
        self,
        text: str,
        width: int,
        height: int
    ) -> str:
        """Generate a pattern based on text."""
        # Create pattern using text properties
        pattern = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Use text characteristics to generate pattern
        text_len = len(text)
        word_count = len(text.split())
        
        for y in range(height):
            for x in range(width):
                # Create pattern using text properties
                value = math.sin(x * text_len/50 + y * word_count/20)
                value *= math.cos(y * text_len/40 + x * word_count/30)
                
                # Map to ASCII character
                if abs(value) > 0.3:  # Threshold for pattern density
                    char_idx = int((value + 1) * (len(self.density_chars) - 1) / 2)
                    pattern[y][x] = self.density_chars[char_idx]
        
        return '\n'.join(''.join(row) for row in pattern)
    
    def create_transformation_sequence(
        self,
        source_pattern: str,
        target_pattern: str,
        n_steps: int
    ) -> List[str]:
        """Create a sequence of patterns transforming from source to target."""
        source_lines = source_pattern.split('\n')
        target_lines = target_pattern.split('\n')
        
        # Get dimensions
        height = max(len(source_lines), len(target_lines))
        width = max(len(source_lines[0]), len(target_lines[0]))
        
        # Pad patterns to same size
        source_lines = [line.ljust(width) for line in source_lines] + [' ' * width] * (height - len(source_lines))
        target_lines = [line.ljust(width) for line in target_lines] + [' ' * width] * (height - len(target_lines))
        
        # Create transformation sequence
        sequence = []
        for step in range(n_steps):
            # Calculate interpolation factor
            t = step / (n_steps - 1)
            
            # Create interpolated pattern
            pattern = []
            for y in range(height):
                row = ''
                for x in range(width):
                    # Get source and target characters
                    source_char = source_lines[y][x]
                    target_char = target_lines[y][x]
                    
                    # Simple character interpolation
                    if t < 0.5:
                        row += source_char if source_char != ' ' else target_char
                    else:
                        row += target_char if target_char != ' ' else source_char
                
                pattern.append(row)
            
            sequence.append('\n'.join(pattern))
        
        return sequence 