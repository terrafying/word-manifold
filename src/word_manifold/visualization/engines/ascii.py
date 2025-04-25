"""
ASCII Engine

Core engine for generating ASCII art patterns and visualizations.
Features rich UTF-8 character sets and advanced pattern generation.
"""

import numpy as np
import math
from typing import List, Optional, Dict, Tuple
from ...types.patterns import Pattern, Mandala, Field

class ASCIIEngine:
    """Engine for generating ASCII art visualizations."""
    
    # Extended character sets for different densities and styles
    DENSITY_CHARS = ' ░▒▓█'  # Basic density progression
    
    # Rich character sets for different pattern types
    DOTS = '·•●⬤○⚪⚫⊙⊚⊛'
    BLOCKS = '▀▁▂▃▄▅▆▇█▌▐░▒▓'
    SHADES = ' ░▒▓█▉▊▋▌▍▎▏'
    LINES = '─│┌┐└┘├┤┬┴┼╌╍═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬'
    CURVES = '╭╮╯╰╴╵╶╷╸╹╺╻'
    DOUBLES = '═║╔╗╚╝╠╣╦╩╬'
    SPECIAL = '★☆✦✧✶✷✸✹✺✻✼✽✾✿❀❁❂❃❄❅❆❇❈❉❊❋'
    CIRCLES = '◌○◎●◐◑◒◓◔◕◖◗◴◵◶◷⚪⚫⚬'
    SQUARES = '◰◱◲◳▀▄▌▐░▒▓█▂▃▄▅▆▇'
    TRIANGLES = '◄►▲▼△▽◢◣◤◥'
    GEOMETRIC = '■□▢▣▤▥▦▧▨▩▪▫◊'
    ORNATE = '❦❧☙❈❉❊❋✺✻✼✽✾✿❀❁❂❃❄❅'
    ARROWS = '←↑→↓↔↕↖↗↘↙↚↛↜↝↞↟↠↡↢↣↤↥↦↧↨↩↪↫↬↭↮↯↰↱↲↳↴↵'
    MATH = '∀∁∂∃∄∅∆∇∈∉∊∋∌∍∎∏∐∑−∓∔∕∖∗∘∙√∛∜∝∞∟∠∡∢∣∤∥'
    BRAILLE = '⠀⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟⠠⠡⠢⠣⠤⠥⠦⠧⠨⠩⠪⠫⠬⠭⠮⠯'
    RUNES = 'ᚠᚡᚢᚣᚤᚥᚦᚧᚨᚩᚪᚫᚬᚭᚮᚯᚰᚱᚲᚳᚴᚵᚶᚷᚸᚹᚺᚻᚼᚽᚾᚿ'
    
    # Color gradients for different themes
    THEMES = {
        'fire': ['red', 'yellow', 'white'],
        'water': ['blue', 'cyan', 'white'],
        'earth': ['green', 'yellow', 'brown'],
        'air': ['cyan', 'white', 'blue'],
        'cosmic': ['magenta', 'cyan', 'white'],
        'mystic': ['magenta', 'blue', 'purple'],
        'rainbow': ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta'],
        'sunset': ['red', 'orange', 'purple'],
        'forest': ['dark_green', 'green', 'light_green'],
        'ocean': ['dark_blue', 'blue', 'light_blue'],
        'ethereal': ['white', 'cyan', 'magenta', 'white'],
        'void': ['black', 'dark_gray', 'gray']
    }
    
    # Pattern styles combining different character sets
    STYLES = {
        'mystical': SPECIAL + ORNATE + CIRCLES,
        'geometric': BLOCKS + GEOMETRIC + SQUARES + TRIANGLES,
        'natural': CURVES + CIRCLES + DOTS,
        'runic': RUNES + SPECIAL + GEOMETRIC,
        'mathematical': MATH + GEOMETRIC + CIRCLES,
        'cyberpunk': BLOCKS + LINES + ARROWS,
        'ethereal': SPECIAL + CIRCLES + DOTS,
        'ancient': RUNES + ORNATE + GEOMETRIC,
        'digital': BLOCKS + SQUARES + LINES,
        'cosmic': SPECIAL + CIRCLES + MATH,
        'braille': BRAILLE + DOTS + GEOMETRIC
    }
    
    def __init__(self):
        """Initialize the ASCII engine."""
        # Create combined character sets for different purposes
        self.MYSTICAL = self.STYLES['mystical']
        self.GEOMETRIC = self.STYLES['geometric']
        self.NATURAL = self.STYLES['natural']
        
    def generate_mandala(
        self,
        radius: int,
        complexity: float = 1.0,
        rotation: float = 0.0,
        style: str = 'mystical',
        layers: int = 3,
        symmetry: int = 8
    ) -> Mandala:
        """Generate a mandala pattern with rich characters.
        
        Args:
            radius: Radius of the mandala
            complexity: Pattern complexity factor
            rotation: Rotation angle in radians
            style: Visual style from STYLES
            layers: Number of concentric layers
            symmetry: Number of symmetry axes
        """
        # Select character set based on style
        chars = self.STYLES.get(style, self.MYSTICAL)
            
        # Ensure we have at least one character
        if not chars:
            chars = self.DENSITY_CHARS
            
        mandala = Mandala.create(radius, chars, rotation)
        center = radius
        
        for y in range(mandala.height):
            for x in range(mandala.width):
                # Calculate polar coordinates
                dx = x - center
                dy = y - center
                r = math.sqrt(dx*dx + dy*dy)
                theta = math.atan2(dy, dx) + rotation
                
                if r <= radius:
                    # Create multi-layered mandala pattern
                    value = 0
                    
                    # Add layer patterns
                    for layer in range(layers):
                        layer_freq = (layer + 1) * complexity
                        layer_weight = 1.0 / (layer + 1)
                        
                        # Radial pattern
                        radial = math.sin(theta * symmetry * (layer + 1))
                        
                        # Concentric pattern
                        concentric = math.cos(r / radius * math.pi * 2 * (layer + 1))
                        
                        # Spiral pattern
                        spiral = math.sin(theta * layer_freq + r / radius * math.pi * 2)
                        
                        # Combine patterns with weights
                        value += layer_weight * (radial + concentric + spiral) / 3
                    
                    # Normalize value
                    value = (value + 1) / 2
                    
                    # Add some variation based on radius
                    value += 0.2 * math.sin(r / radius * math.pi * complexity)
                    value = max(0, min(1, value))  # Clamp to [0, 1]
                    
                    # Map value to character
                    char_idx = min(int(value * len(chars)), len(chars) - 1)
                    mandala.data[y, x] = chars[char_idx]
        
        return mandala
    
    def generate_field(
        self,
        width: int,
        height: int,
        density: float = 0.7,
        style: str = 'natural',
        pattern_type: str = 'organic'
    ) -> Field:
        """Generate a field pattern with specified style and type.
        
        Args:
            width: Field width
            height: Field height
            density: Pattern density
            style: Visual style from STYLES
            pattern_type: Type of pattern ('organic', 'crystalline', 'flowing', 'chaotic')
        """
        # Select character set based on style
        chars = self.STYLES.get(style, self.NATURAL)
        field = Field.create(width, height, chars, density)
        
        # Apply pattern-specific generation
        if pattern_type == 'crystalline':
            # Create crystalline growth patterns
            centers = [(width//2, height//2)]  # Start from center
            for _ in range(int(width * height * density * 0.1)):
                x, y = centers[np.random.randint(len(centers))]
                for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_x < width and 0 <= new_y < height and 
                        np.random.random() < density):
                        centers.append((new_x, new_y))
                        char_idx = np.random.randint(len(chars))
                        field.data[new_y, new_x] = chars[char_idx]
                        
        elif pattern_type == 'flowing':
            # Create flowing patterns using noise
            scale = 20.0
            octaves = 4
            for y in range(height):
                for x in range(width):
                    value = 0
                    for octave in range(octaves):
                        freq = 1.0 * (2 ** octave)
                        amp = 1.0 / (2 ** octave)
                        value += amp * math.sin(x/scale * freq) * math.cos(y/scale * freq)
                    value = (value + 1) / 2
                    if value > (1 - density):
                        char_idx = int(value * (len(chars) - 1))
                        field.data[y, x] = chars[char_idx % len(chars)]
                        
        elif pattern_type == 'chaotic':
            # Create chaotic patterns using cellular automata rules
            for y in range(height):
                for x in range(width):
                    if np.random.random() < density:
                        neighbors = sum(1 for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
                                     if 0 <= x+dx < width and 0 <= y+dy < height and field.data[y+dy, x+dx] != ' ')
                        if neighbors == 2 or neighbors == 3:
                            char_idx = np.random.randint(len(chars))
                            field.data[y, x] = chars[char_idx]
                            
        else:  # organic
            # Create organic patterns using noise and curves
            for y in range(height):
                for x in range(width):
                    value = (math.sin(x/10) + math.cos(y/10) + 
                            math.sin(math.sqrt(x*x + y*y)/10)) / 3
                    value = (value + 1) / 2
                    if value > (1 - density):
                        char_idx = int(value * (len(chars) - 1))
                        field.data[y, x] = chars[char_idx]
        
        return field
    
    def add_wave_pattern(
        self,
        field: Field,
        frequency: float = 0.1,
        phase: float = 0.0,
        amplitude: float = 1.0,
        wave_type: str = 'sine',
        interference: bool = False
    ) -> None:
        """Add wave interference pattern to a field.
        
        Args:
            field: Field to modify
            frequency: Pattern frequency
            phase: Pattern phase shift
            amplitude: Wave amplitude
            wave_type: Type of wave ('sine', 'square', 'triangle', 'sawtooth')
            interference: Whether to add interference patterns
        """
        for y in range(field.height):
            for x in range(field.width):
                # Calculate base wave
                if wave_type == 'square':
                    value = np.sign(math.sin(x * frequency + phase))
                elif wave_type == 'triangle':
                    value = 2 * abs(2 * (x * frequency + phase) / (2 * math.pi) - 
                                  math.floor((x * frequency + phase) / (2 * math.pi) + 0.5)) - 1
                elif wave_type == 'sawtooth':
                    value = 2 * ((x * frequency + phase) / (2 * math.pi) - 
                                math.floor(0.5 + (x * frequency + phase) / (2 * math.pi))) 
                else:  # sine
                    value = math.sin(x * frequency + phase)
                
                # Add vertical component
                if wave_type == 'square':
                    v_value = np.sign(math.cos(y * frequency + phase))
                elif wave_type == 'triangle':
                    v_value = 2 * abs(2 * (y * frequency + phase) / (2 * math.pi) - 
                                    math.floor((y * frequency + phase) / (2 * math.pi) + 0.5)) - 1
                elif wave_type == 'sawtooth':
                    v_value = 2 * ((y * frequency + phase) / (2 * math.pi) - 
                                  math.floor(0.5 + (y * frequency + phase) / (2 * math.pi)))
                else:  # sine
                    v_value = math.cos(y * frequency + phase)
                
                # Add interference if requested
                if interference:
                    i_value = math.sin((x+y) * frequency * 0.7 + phase)
                    value = (value + v_value + i_value) * amplitude / 3
                else:
                    value = (value + v_value) * amplitude / 2
                    
                value = (value + 1) / 2  # Normalize to [0, 1]
                
                if value > (1 - field.density):
                    char_idx = int(value * (len(field.symbols) - 1))
                    field.data[y, x] = field.symbols[char_idx]
    
    def resize_pattern(self, pattern: Pattern, width: int, height: int) -> Pattern:
        """Resize a pattern to new dimensions while preserving aspect ratio.
        
        Args:
            pattern: Pattern to resize
            width: Target width
            height: Target height
        """
        result = Pattern.create_empty(width, height, pattern.symbols)
        
        # Calculate scaling factors
        scale_x = width / pattern.width
        scale_y = height / pattern.height
        
        # Fill new pattern
        for y in range(height):
            for x in range(width):
                # Map new coordinates back to original pattern
                src_x = int(x / scale_x)
                src_y = int(y / scale_y)
                
                # Keep coordinates in bounds
                src_x = min(src_x, pattern.width - 1)
                src_y = min(src_y, pattern.height - 1)
                
                # Copy character
                result.data[y, x] = pattern.data[src_y, src_x]
        
        return result

    def blend_patterns(
        self,
        pattern1: Pattern,
        pattern2: Pattern,
        alpha: float = 0.5,
        blend_mode: str = 'overlay'
    ) -> Pattern:
        """Blend two patterns with various blend modes.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            alpha: Blend factor (0.0 to 1.0)
            blend_mode: Blending mode ('overlay', 'add', 'multiply', 'screen', 'difference')
        """
        # Handle different dimensions by resizing to the larger dimensions
        max_width = max(pattern1.width, pattern2.width)
        max_height = max(pattern1.height, pattern2.height)
        
        if pattern1.width != max_width or pattern1.height != max_height:
            pattern1 = self.resize_pattern(pattern1, max_width, max_height)
        if pattern2.width != max_width or pattern2.height != max_height:
            pattern2 = self.resize_pattern(pattern2, max_width, max_height)
        
        result = Pattern.create_empty(max_width, max_height)
        result.symbols = pattern1.symbols + pattern2.symbols  # Combine symbol sets
        
        for y in range(result.height):
            for x in range(result.width):
                # Get character densities
                char1 = pattern1.data[y, x]
                char2 = pattern2.data[y, x]
                density1 = pattern1.symbols.find(char1) / len(pattern1.symbols) if char1 in pattern1.symbols else 0
                density2 = pattern2.symbols.find(char2) / len(pattern2.symbols) if char2 in pattern2.symbols else 0
                
                # Apply blend mode
                if blend_mode == 'add':
                    value = min(density1 + density2, 1.0)
                elif blend_mode == 'multiply':
                    value = density1 * density2
                elif blend_mode == 'screen':
                    value = 1.0 - (1.0 - density1) * (1.0 - density2)
                elif blend_mode == 'difference':
                    value = abs(density1 - density2)
                else:  # overlay
                    if density1 < 0.5:
                        value = 2 * density1 * density2
                    else:
                        value = 1.0 - 2 * (1.0 - density1) * (1.0 - density2)
                
                # Apply alpha blending
                value = value * alpha + density1 * (1 - alpha)
                
                # Map to character
                char_idx = int(value * (len(result.symbols) - 1))
                result.data[y, x] = result.symbols[char_idx]
        
        return result
    
    def create_animation_frames(
        self,
        pattern: Pattern,
        n_frames: int,
        animation_type: str = 'rotate'
    ) -> List[Pattern]:
        """Create animation frames with various effects.
        
        Args:
            pattern: Base pattern
            n_frames: Number of frames
            animation_type: Type of animation ('rotate', 'pulse', 'wave')
        """
        frames = []
        
        for i in range(n_frames):
            phase = 2 * math.pi * i / n_frames
            
            if isinstance(pattern, Mandala):
                if animation_type == 'pulse':
                    # Pulsing effect
                    scale = 0.5 + 0.5 * math.sin(phase)
                    radius = max(1, int(pattern.radius * scale))
                    frame = self.generate_mandala(
                        radius,
                        complexity=1.0 + 0.5 * math.sin(phase),
                        rotation=pattern.rotation
                    )
                else:  # rotate
                    frame = self.generate_mandala(
                        pattern.radius,
                        complexity=1.0,
                        rotation=pattern.rotation + phase
                    )
            else:
                # Create phase-shifted copy
                frame = Pattern.create_empty(pattern.width, pattern.height, pattern.symbols)
                frame.data = pattern.data.copy()
                
                if animation_type == 'wave':
                    # Wave effect
                    for y in range(frame.height):
                        shift = int(math.sin(phase + y * 0.2) * 3)
                        frame.data[y] = np.roll(frame.data[y], shift)
                else:
                    # Character cycling
                    for y in range(frame.height):
                        for x in range(frame.width):
                            if frame.data[y, x] != ' ':
                                char_idx = (x + y + int(phase * 10)) % len(frame.symbols)
                                frame.data[y, x] = frame.symbols[char_idx]
            
            frames.append(frame)
        
        return frames 