"""Visualization themes and color schemes."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ColorScheme:
    """Color scheme configuration."""
    name: str
    primary: str
    secondary: str
    accent: str
    background: str
    text: str
    highlight: str
    error: str
    success: str
    warning: str
    neutral: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'primary': self.primary,
            'secondary': self.secondary,
            'accent': self.accent,
            'background': self.background,
            'text': self.text,
            'highlight': self.highlight,
            'error': self.error,
            'success': self.success,
            'warning': self.warning,
            'neutral': self.neutral
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'ColorScheme':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class Theme:
    """Visualization theme configuration."""
    name: str
    color_scheme: ColorScheme
    font_family: str
    font_size: int
    line_width: float
    marker_size: float
    grid_alpha: float
    background_alpha: float
    animation_duration: float
    animation_fps: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'color_scheme': self.color_scheme.to_dict(),
            'font_family': self.font_family,
            'font_size': self.font_size,
            'line_width': self.line_width,
            'marker_size': self.marker_size,
            'grid_alpha': self.grid_alpha,
            'background_alpha': self.background_alpha,
            'animation_duration': self.animation_duration,
            'animation_fps': self.animation_fps
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Theme':
        """Create from dictionary."""
        color_scheme = ColorScheme.from_dict(data['color_scheme'])
        return cls(color_scheme=color_scheme, **{k: v for k, v in data.items() if k != 'color_scheme'})

class ThemeManager:
    """Manages visualization themes and color schemes."""
    
    def __init__(self, themes_dir: Optional[Path] = None):
        """Initialize theme manager.
        
        Args:
            themes_dir: Directory containing theme definitions
        """
        self.themes_dir = themes_dir or Path(__file__).parent / 'themes'
        self.themes_dir.mkdir(parents=True, exist_ok=True)
        
        # Load built-in themes
        self._load_builtin_themes()
        
        # Load custom themes
        self._load_custom_themes()
    
    def _load_builtin_themes(self):
        """Load built-in themes."""
        self.themes = {
            'default': Theme(
                name='Default',
                color_scheme=ColorScheme(
                    name='Default',
                    primary='#1f77b4',
                    secondary='#ff7f0e',
                    accent='#2ca02c',
                    background='#ffffff',
                    text='#000000',
                    highlight='#d62728',
                    error='#e41a1c',
                    success='#4daf4a',
                    warning='#ff7f00',
                    neutral='#7f7f7f'
                ),
                font_family='sans-serif',
                font_size=12,
                line_width=1.5,
                marker_size=6,
                grid_alpha=0.3,
                background_alpha=1.0,
                animation_duration=5.0,
                animation_fps=30
            ),
            'dark': Theme(
                name='Dark',
                color_scheme=ColorScheme(
                    name='Dark',
                    primary='#1f77b4',
                    secondary='#ff7f0e',
                    accent='#2ca02c',
                    background='#1a1a1a',
                    text='#ffffff',
                    highlight='#d62728',
                    error='#e41a1c',
                    success='#4daf4a',
                    warning='#ff7f00',
                    neutral='#7f7f7f'
                ),
                font_family='sans-serif',
                font_size=12,
                line_width=1.5,
                marker_size=6,
                grid_alpha=0.2,
                background_alpha=1.0,
                animation_duration=5.0,
                animation_fps=30
            ),
            'light': Theme(
                name='Light',
                color_scheme=ColorScheme(
                    name='Light',
                    primary='#1f77b4',
                    secondary='#ff7f0e',
                    accent='#2ca02c',
                    background='#f5f5f5',
                    text='#000000',
                    highlight='#d62728',
                    error='#e41a1c',
                    success='#4daf4a',
                    warning='#ff7f00',
                    neutral='#7f7f7f'
                ),
                font_family='sans-serif',
                font_size=12,
                line_width=1.5,
                marker_size=6,
                grid_alpha=0.3,
                background_alpha=1.0,
                animation_duration=5.0,
                animation_fps=30
            )
        }
    
    def _load_custom_themes(self):
        """Load custom themes from themes directory."""
        for theme_file in self.themes_dir.glob('*.json'):
            try:
                with open(theme_file, 'r') as f:
                    theme_data = json.load(f)
                theme = Theme.from_dict(theme_data)
                self.themes[theme.name.lower()] = theme
                logger.info(f"Loaded custom theme: {theme.name}")
            except Exception as e:
                logger.error(f"Error loading theme {theme_file}: {e}")
    
    def get_theme(self, name: str) -> Theme:
        """Get theme by name.
        
        Args:
            name: Theme name
            
        Returns:
            Theme configuration
        """
        theme = self.themes.get(name.lower())
        if theme is None:
            logger.warning(f"Theme '{name}' not found, using default")
            theme = self.themes['default']
        return theme
    
    def save_theme(self, theme: Theme):
        """Save custom theme.
        
        Args:
            theme: Theme to save
        """
        theme_file = self.themes_dir / f"{theme.name.lower()}.json"
        with open(theme_file, 'w') as f:
            json.dump(theme.to_dict(), f, indent=2)
        self.themes[theme.name.lower()] = theme
        logger.info(f"Saved custom theme: {theme.name}")
    
    def apply_theme(self, theme: Theme):
        """Apply theme to matplotlib.
        
        Args:
            theme: Theme to apply
        """
        plt.rcParams.update({
            'font.family': theme.font_family,
            'font.size': theme.font_size,
            'lines.linewidth': theme.line_width,
            'lines.markersize': theme.marker_size,
            'grid.alpha': theme.grid_alpha,
            'figure.facecolor': theme.color_scheme.background,
            'axes.facecolor': theme.color_scheme.background,
            'axes.edgecolor': theme.color_scheme.text,
            'axes.labelcolor': theme.color_scheme.text,
            'xtick.color': theme.color_scheme.text,
            'ytick.color': theme.color_scheme.text,
            'text.color': theme.color_scheme.text
        })
        
        # Set color cycle
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            color=[
                theme.color_scheme.primary,
                theme.color_scheme.secondary,
                theme.color_scheme.accent,
                theme.color_scheme.highlight,
                theme.color_scheme.error,
                theme.color_scheme.success,
                theme.color_scheme.warning,
                theme.color_scheme.neutral
            ]
        )
    
    def get_color_palette(self, theme: Theme, n_colors: int) -> List[str]:
        """Get color palette from theme.
        
        Args:
            theme: Theme to use
            n_colors: Number of colors needed
            
        Returns:
            List of hex color codes
        """
        if n_colors <= 8:
            return [
                theme.color_scheme.primary,
                theme.color_scheme.secondary,
                theme.color_scheme.accent,
                theme.color_scheme.highlight,
                theme.color_scheme.error,
                theme.color_scheme.success,
                theme.color_scheme.warning,
                theme.color_scheme.neutral
            ][:n_colors]
        else:
            # Generate additional colors using seaborn
            return sns.color_palette(
                theme.color_scheme.primary,
                n_colors=n_colors
            ).as_hex() 