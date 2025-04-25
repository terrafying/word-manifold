"""
ASCII Renderer

Handles rendering of ASCII patterns to terminal and files.
Features rich color support and advanced effects.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from ...types.patterns import Pattern, ASCIIPattern, Mandala, Field

class ASCIIRenderer:
    """Renderer for ASCII art visualizations."""
    
    # ANSI color codes
    COLORS = {
        # Basic colors
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        
        # Bright colors
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        
        # Background colors
        'bg_black': '\033[40m',
        'bg_red': '\033[41m',
        'bg_green': '\033[42m',
        'bg_yellow': '\033[43m',
        'bg_blue': '\033[44m',
        'bg_magenta': '\033[45m',
        'bg_cyan': '\033[46m',
        'bg_white': '\033[47m',
        
        'reset': '\033[0m'
    }
    
    # Special formatting
    FORMATS = {
        'bold': '\033[1m',
        'dim': '\033[2m',
        'italic': '\033[3m',
        'underline': '\033[4m',
        'blink': '\033[5m',
        'reverse': '\033[7m',
        'hidden': '\033[8m',
        'strike': '\033[9m'
    }
    
    # Color themes
    THEMES = {
        'fire': [('red', None), ('yellow', None), ('bright_white', None)],
        'water': [('blue', None), ('cyan', None), ('bright_white', None)],
        'earth': [('green', None), ('yellow', None), ('bright_black', None)],
        'air': [('cyan', None), ('bright_white', None), ('bright_blue', None)],
        'cosmic': [('magenta', None), ('cyan', None), ('bright_white', None)],
        'mystic': [('magenta', 'bg_black'), ('blue', 'bg_black'), ('bright_magenta', 'bg_black')]
    }
    
    def __init__(self):
        """Initialize the renderer."""
        # Check terminal capabilities
        self.supports_color = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        self.supports_unicode = True  # Most modern terminals support Unicode
        
        try:
            # Test if terminal supports RGB colors
            self.supports_rgb = sys.stdout.write('\033[38;2;255;0;0mTest\033[0m') > 0
        except:
            self.supports_rgb = False
    
    def apply_color(
        self,
        text: str,
        fg_color: Optional[str] = None,
        bg_color: Optional[str] = None,
        format_type: Optional[str] = None
    ) -> str:
        """Apply color and formatting to text."""
        if not self.supports_color:
            return text
            
        result = ''
        
        # Apply background color first
        if bg_color and bg_color.startswith('bg_') and bg_color in self.COLORS:
            result += self.COLORS[bg_color]
            
        # Apply foreground color
        if fg_color and fg_color in self.COLORS:
            result += self.COLORS[fg_color]
            
        # Apply format
        if format_type and format_type in self.FORMATS:
            result += self.FORMATS[format_type]
            
        result += text + self.COLORS['reset']
        return result
    
    def apply_theme(
        self,
        text: str,
        theme: str,
        position: float
    ) -> str:
        """Apply color theme based on position in pattern."""
        if not self.supports_color or theme not in self.THEMES:
            return text
            
        # Get theme colors
        colors = self.THEMES[theme]
        
        # Find colors to interpolate between
        color_idx = int(position * (len(colors) - 1))
        fg_color, bg_color = colors[color_idx]
        
        return self.apply_color(text, fg_color, bg_color)
    
    def render_pattern(
        self,
        pattern: Pattern,
        color: Optional[str] = None,
        theme: Optional[str] = None,
        format_type: Optional[str] = None,
        add_border: bool = False
    ) -> str:
        """Render a pattern with colors and effects."""
        if not pattern:
            return ""
            
        # Add border if requested
        width = pattern.width + (2 if add_border else 0)
        height = pattern.height + (2 if add_border else 0)
        
        lines = []
        if add_border:
            border = '─' * (width - 2)
            lines.append(f'╭{border}╮')
        
        for y in range(pattern.height):
            line = '│ ' if add_border else ''
            for x in range(pattern.width):
                char = pattern.data[y, x]
                
                # Apply color/theme
                if theme:
                    # Calculate position for theme interpolation
                    pos = (x + y) / (pattern.width + pattern.height)
                    line += self.apply_theme(char, theme, pos)
                elif color:
                    line += self.apply_color(char, color, format_type=format_type)
                else:
                    line += char
                    
            if add_border:
                line += ' │'
            lines.append(line)
            
        if add_border:
            border = '─' * (width - 2)
            lines.append(f'╰{border}╯')
            
        return '\n'.join(lines)
    
    def render_animation(
        self,
        frames: List[Pattern],
        frame_delay: float = 0.1,
        color: Optional[str] = None,
        theme: Optional[str] = None,
        loop: bool = True,
        add_border: bool = True
    ) -> None:
        """Render an animation with colors and effects."""
        if not frames:
            return
            
        try:
            frame_count = 0
            while True:
                for frame in frames:
                    # Clear screen and move cursor to top
                    sys.stdout.write('\033[2J\033[H')
                    
                    # Render frame with effects
                    output = self.render_pattern(
                        frame,
                        color=color,
                        theme=theme,
                        add_border=add_border
                    )
                    
                    # Add frame counter
                    if add_border:
                        frame_count += 1
                        counter = f" Frame {frame_count} "
                        output += f"\n{counter:^{frame.width + 4}}"
                    
                    sys.stdout.write(output)
                    sys.stdout.flush()
                    time.sleep(frame_delay)
                
                if not loop:
                    break
                    
        except KeyboardInterrupt:
            # Clear screen and reset cursor on exit
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.flush()
    
    def save_pattern(
        self,
        pattern: Pattern,
        filepath: Union[str, Path],
        include_metadata: bool = True,
        add_border: bool = True
    ) -> None:
        """Save a pattern to a file with optional border."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            if include_metadata:
                f.write(f"# Pattern Type: {pattern.__class__.__name__}\n")
                f.write(f"# Dimensions: {pattern.width}x{pattern.height}\n")
                f.write(f"# Character Set: {pattern.symbols}\n")
                f.write("#" + "=" * 40 + "\n\n")
            
            # Write pattern with optional border
            output = self.render_pattern(pattern, add_border=add_border)
            f.write(output)
    
    def save_animation(
        self,
        frames: List[Pattern],
        filepath: Union[str, Path],
        include_metadata: bool = True,
        add_border: bool = True
    ) -> None:
        """Save an animation to a file with optional border."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            if include_metadata:
                f.write(f"# Animation Frames: {len(frames)}\n")
                if frames:
                    f.write(f"# Frame Dimensions: {frames[0].width}x{frames[0].height}\n")
                f.write("#" + "=" * 40 + "\n\n")
            
            for i, frame in enumerate(frames, 1):
                f.write(f"\nFrame {i}/{len(frames)}\n")
                f.write("=" * 40 + "\n")
                output = self.render_pattern(frame, add_border=add_border)
                f.write(output + "\n") 