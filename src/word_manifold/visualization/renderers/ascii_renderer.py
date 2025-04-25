"""
ASCII Renderer Module

Provides rendering capabilities for ASCII art visualizations,
with support for external APIs and local fallback.
"""

import logging
import requests
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
import time
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoints
ASCII_ART_API = "https://artii.herokuapp.com/make"
TEXT_TO_ASCII_API = "https://asciified.thelicato.io/api/v2/ascii"

@dataclass
class RenderConfig:
    """Configuration for ASCII rendering."""
    use_external_api: bool = True
    api_timeout: int = 5
    max_retries: int = 3
    font: str = "standard"
    width: int = 80
    height: int = 40
    color_support: bool = True

class ASCIIRenderer:
    """Renderer for ASCII art visualizations with API support."""
    
    def __init__(self, config: Optional[RenderConfig] = None):
        """Initialize renderer with configuration."""
        self.config = config or RenderConfig()
        self.supports_color = self._check_color_support()
        self._cached_patterns: Dict[str, str] = {}
    
    def render_pattern(
        self,
        pattern: Union[str, Dict],
        color: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Render an ASCII pattern, optionally using external APIs.
        
        Args:
            pattern: Pattern to render (string or dict with metadata)
            color: Optional color for the pattern
            use_cache: Whether to use pattern caching
            
        Returns:
            Rendered ASCII pattern
        """
        # Check cache first
        cache_key = str(pattern)
        if use_cache and cache_key in self._cached_patterns:
            rendered = self._cached_patterns[cache_key]
        else:
            # Try external API first if enabled
            rendered = None
            if self.config.use_external_api:
                try:
                    rendered = self._render_with_api(pattern)
                except Exception as e:
                    logger.warning(f"API rendering failed, falling back to local: {e}")
            
            # Fall back to local rendering if needed
            if rendered is None:
                rendered = self._render_local(pattern)
            
            # Cache the result
            if use_cache:
                self._cached_patterns[cache_key] = rendered
        
        # Apply color if requested and supported
        if color and self.supports_color:
            rendered = self._apply_color(rendered, color)
        
        return rendered
    
    def render_animation(
        self,
        frames: List[Union[str, Dict]],
        frame_delay: float = 0.1,
        color: Optional[str] = None,
        loop: bool = False
    ) -> None:
        """Render an animation in the terminal."""
        try:
            while True:
                for frame in frames:
                    rendered = self.render_pattern(frame, color)
                    # Clear screen and render frame
                    os.system('cls' if os.name == 'nt' else 'clear')
                    sys.stdout.write(rendered)
                    sys.stdout.flush()
                    time.sleep(frame_delay)
                
                if not loop:
                    break
                    
        except KeyboardInterrupt:
            # Clean exit on Ctrl+C
            sys.stdout.write('\n')
    
    def save_pattern(
        self,
        pattern: Union[str, Dict],
        output_path: Union[str, Path],
        include_metadata: bool = False
    ) -> None:
        """Save pattern to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rendered = self.render_pattern(pattern)
        
        # Save with metadata if requested
        if include_metadata and isinstance(pattern, dict):
            metadata = {
                'pattern': rendered,
                'metadata': {
                    k: v for k, v in pattern.items()
                    if k != 'pattern' and not k.startswith('_')
                }
            }
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        else:
            with open(output_path, 'w') as f:
                f.write(rendered)
    
    def save_animation(
        self,
        frames: List[Union[str, Dict]],
        output_path: Union[str, Path]
    ) -> None:
        """Save animation frames to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Render and save all frames
        rendered_frames = [
            self.render_pattern(frame)
            for frame in frames
        ]
        
        with open(output_path, 'w') as f:
            f.write('\n---FRAME---\n'.join(rendered_frames))
    
    def _render_with_api(self, pattern: Union[str, Dict]) -> Optional[str]:
        """Attempt to render pattern using external API."""
        pattern_text = pattern if isinstance(pattern, str) else pattern.get('pattern', '')
        
        for attempt in range(self.config.max_retries):
            try:
                # Try first API
                response = requests.post(
                    ASCII_ART_API,
                    data={'text': pattern_text},
                    timeout=self.config.api_timeout
                )
                if response.ok:
                    return response.text
                
                # Try backup API
                response = requests.post(
                    TEXT_TO_ASCII_API,
                    json={
                        'text': pattern_text,
                        'font': self.config.font
                    },
                    timeout=self.config.api_timeout
                )
                if response.ok:
                    return response.json()['ascii']
                    
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.warning(f"API attempt {attempt + 1} failed: {e}")
                time.sleep(1)  # Wait before retry
        
        return None
    
    def _render_local(self, pattern: Union[str, Dict]) -> str:
        """Render pattern locally."""
        if isinstance(pattern, dict):
            return pattern.get('pattern', '')
        return pattern
    
    def _check_color_support(self) -> bool:
        """Check if terminal supports color."""
        return (
            hasattr(sys.stdout, 'isatty')
            and sys.stdout.isatty()
            and 'COLORTERM' in os.environ
        )
    
    def _apply_color(self, text: str, color: str) -> str:
        """Apply ANSI color to text."""
        if not self.supports_color:
            return text
            
        color_codes = {
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
        }
        
        reset_code = '\033[0m'
        color_code = color_codes.get(color.lower(), '')
        
        if not color_code:
            return text
            
        return f"{color_code}{text}{reset_code}" 