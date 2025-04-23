"""
Semantic Shape Visualization Module.

This module creates dynamic visualizations of semantic shapes,
showing how meaning flows and transforms in the manifold space.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import colorsys
import logging
from typing import List, Optional, Tuple, Dict, Union, Any
from pathlib import Path
import datetime
import os

from ..manifold.semantic_shape import SemanticShape
from ..embeddings.phrase_embeddings import PhraseEmbedding, PhraseEmbedder

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default to INFO level

# Create formatters and handlers if they don't exist
if not logger.handlers:
    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler for warnings and errors
    file_handler = logging.FileHandler('visualization.log')
    file_handler.setLevel(logging.WARNING)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

class ExportConfig:
    """Configuration for exporting visualizations."""
    def __init__(
        self,
        format: str = "mp4",
        dpi: int = 300,
        fps: int = 60,
        bitrate: int = 2000,
        save_frames: bool = True,
        output_dir: Optional[str] = None
    ):
        self.format = format
        self.dpi = dpi
        self.fps = fps
        self.bitrate = bitrate
        self.save_frames = save_frames
        self.output_dir = Path(output_dir) if output_dir else Path("visualizations")
        
        # Create output directories
        self.frames_dir = self.output_dir / "frames"
        self.animations_dir = self.output_dir / "animations"
        self.static_dir = self.output_dir / "static"
        
        for dir_path in [self.frames_dir, self.animations_dir, self.static_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

class ShapeVisualizer:
    """
    Advanced visualization class for semantic shapes with enhanced visual encoding.
    """
    
    def __init__(
        self,
        color_scheme: str = "semantic",
        use_textures: bool = True,
        export_config: Optional[ExportConfig] = None
    ):
        """Initialize the visualizer with enhanced visual options."""
        self.color_scheme = color_scheme
        self.use_textures = use_textures
        self.export_config = export_config or ExportConfig()
        
        # Initialize color maps for different semantic properties
        self.color_maps = {
            'emotion': plt.cm.get_cmap('winter'),    # Red-Yellow-Blue for emotional valence
            'complexity': plt.cm.get_cmap('viridis'), # Viridis for complexity
            'abstraction': plt.cm.get_cmap('magma'),  # Magma for concrete-abstract spectrum
            'energy': plt.cm.get_cmap('plasma'),      # Plasma for energy/intensity
            'delta': plt.cm.get_cmap('coolwarm')     # Coolwarm for transformation deltas
        }
        
        # Initialize texture patterns
        if use_textures:
            self.textures = self._create_texture_patterns()
            
        # Track transformation history
        self.previous_state = None
        self.transformation_type = None
        
        logger.info(f"Initialized ShapeVisualizer with {color_scheme} color scheme")

    def _create_texture_patterns(self) -> Dict[str, np.ndarray]:
        """Create texture patterns for different semantic properties."""
        textures = {}
        
        # Create basic patterns
        size = 64  # Texture size
        
        # Lines pattern for flow/direction
        lines = np.zeros((size, size))
        for i in range(size):
            lines[i, :] = i % 8 < 4
        textures['flow'] = lines
        
        # Dots pattern for emphasis
        dots = np.zeros((size, size))
        for i in range(0, size, 8):
            for j in range(0, size, 8):
                dots[i:i+4, j:j+4] = 1
        textures['emphasis'] = dots
        
        # Waves pattern for rhythm
        x = np.linspace(0, 4*np.pi, size)
        y = np.linspace(0, 4*np.pi, size)
        X, Y = np.meshgrid(x, y)
        waves = np.sin(X) * np.cos(Y)
        textures['rhythm'] = (waves + 1) / 2
        
        # Gradient pattern for transitions
        gradient = np.linspace(0, 1, size)
        textures['transition'] = np.tile(gradient, (size, 1))
        
        # Numerological transformation pattern (sacred geometry)
        theta = np.linspace(0, 2*np.pi, size)
        r = np.linspace(0, 1, size)
        R, T = np.meshgrid(r, theta)
        sacred = np.sin(7*T) * R  # Heptagram pattern
        textures['numerological'] = (sacred + 1) / 2
        
        # Alignment transformation pattern (concentric circles)
        R = np.sqrt(X**2 + Y**2) / size
        align = np.cos(R * 10 * np.pi)
        textures['align'] = (align + 1) / 2
        
        # Contrast transformation pattern (radiating lines)
        T = np.arctan2(Y, X)
        contrast = np.cos(T * 8)
        textures['contrast'] = (contrast + 1) / 2
        
        # Repulsion transformation pattern (expanding waves)
        repel = np.cos(R * 8 * np.pi - T)
        textures['repel'] = (repel + 1) / 2
        
        return textures
    
    def _apply_texture(self, base_img: np.ndarray, texture: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Apply a texture pattern to a base image."""
        if base_img.shape[:2] != texture.shape:
            # Resize texture to match base image
            from scipy.ndimage import zoom
            zoom_factor = (base_img.shape[0] / texture.shape[0], 
                         base_img.shape[1] / texture.shape[1])
            texture = zoom(texture, zoom_factor, order=1)
        
        # Add channel dimension if needed
        if len(texture.shape) == 2:
            texture = texture[..., np.newaxis]
        
        # Blend texture with base image
        return base_img * (1 - alpha) + texture * alpha
    
    def _normalize_color_data(self, data: Union[np.ndarray, tuple]) -> np.ndarray:
        """Normalize color data to ensure it's in valid range [0,1]."""
        # Convert tuple to numpy array if needed
        if isinstance(data, tuple):
            data = np.array(data)
            
        elif data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Handle NaN and infinity values
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize to [0,1] range
        if data.min() < 0 or data.max() > 1:
            data = (data - data.min()) / (data.max() - data.min() + 1e-10)
        
        return np.clip(data, 0, 1)

    def _create_color_gradient(
        self,
        shape_params: Dict[str, float],
        size: Tuple[int, int]
    ) -> np.ndarray:
        """Create a color gradient based on shape parameters."""
        # Extract and normalize parameters
        emotion = self._normalize_color_data(np.array(shape_params.get('emotional_valence', 0)))
        complexity = self._normalize_color_data(np.array(shape_params.get('syntax_complexity', 0)))
        abstraction = self._normalize_color_data(np.array(shape_params.get('concrete_abstract_ratio', 0)))
        energy = self._normalize_color_data(np.array(shape_params.get('semantic_density', 0)))
        
        # Create base gradients
        x = np.linspace(0, 1, size[1])
        y = np.linspace(0, 1, size[0])
        X, Y = np.meshgrid(x, y)
        
        # Initialize color layers
        color_layers = np.zeros((4, size[0], size[1], 3))
        
        # Generate each color layer
        emotion_color = np.array(self.color_maps['emotion'](emotion))[:3]
        color_layers[0] = np.broadcast_to(
            self._normalize_color_data(emotion_color),
            (size[0], size[1], 3)
        )
        
        complexity_values = self._normalize_color_data(complexity * X + (1-complexity) * Y)
        complexity_colors = np.array(self.color_maps['complexity'](complexity_values))[..., :3]
        color_layers[1] = self._normalize_color_data(complexity_colors)
        
        abstraction_values = self._normalize_color_data(abstraction * np.sqrt(X**2 + Y**2))
        abstraction_colors = np.array(self.color_maps['abstraction'](abstraction_values))[..., :3]
        color_layers[2] = self._normalize_color_data(abstraction_colors)
        
        energy_values = self._normalize_color_data(energy * (X + Y) / 2)
        energy_colors = np.array(self.color_maps['energy'](energy_values))[..., :3]
        color_layers[3] = self._normalize_color_data(energy_colors)
        
        # Blend layers
        weights = np.array([0.4, 0.3, 0.2, 0.1])[:, np.newaxis, np.newaxis, np.newaxis]
        final_color = np.sum(color_layers * weights, axis=0)
        
        # Add alpha channel
        alpha = np.ones((*size, 1))
        final_color = np.concatenate([final_color, alpha], axis=-1)
        
        return self._normalize_color_data(final_color)

    def _create_visualization_frame(
        self,
        current_shape: SemanticShape,
        next_shape: Optional[SemanticShape],
        alpha: float,
        size: Tuple[int, int]
    ) -> np.ndarray:
        """Create a single visualization frame with interpolation."""
        # Get base visualization
        img = self._create_color_gradient(current_shape.shape_params, size)
        
        if next_shape and alpha > 0:
            # Interpolate between shapes
            next_img = self._create_color_gradient(next_shape.shape_params, size)
            img = self._normalize_color_data(img * (1 - alpha) + next_img * alpha)
        
        if self.use_textures:
            img = self._apply_textures_to_frame(img, current_shape, next_shape, alpha)
        
        return self._normalize_color_data(img)

    def _apply_textures_to_frame(
        self,
        img: np.ndarray,
        current_shape: SemanticShape,
        next_shape: Optional[SemanticShape],
        alpha: float
    ) -> np.ndarray:
        """Apply textures to a visualization frame based on shape properties and transformation type."""
        # Apply textures based on shape properties and transformation type
        rhythm_score = current_shape.shape_params.get('sentence_rhythm_score', 0)
        if rhythm_score > 0.5:
            img = self._apply_texture(img, self.textures['rhythm'], rhythm_score * 0.3)
        
        flow_score = current_shape.shape_params.get('transition_smoothness', 0)
        if flow_score > 0.5:
            img = self._apply_texture(img, self.textures['flow'], flow_score * 0.3)
        
        emphasis = current_shape.shape_params.get('semantic_density', 0)
        if emphasis > 0.7:
            img = self._apply_texture(img, self.textures['emphasis'], emphasis * 0.2)
        
        # Add transformation-specific textures
        if self.transformation_type:
            texture_alpha = min(alpha * 2, 0.5)
            if self.transformation_type == 'numerological':
                img = self._apply_texture(img, self.textures['numerological'], texture_alpha)
            elif self.transformation_type == 'align':
                img = self._apply_texture(img, self.textures['align'], texture_alpha)
            elif self.transformation_type == 'contrast':
                img = self._apply_texture(img, self.textures['contrast'], texture_alpha)
            elif self.transformation_type == 'repel':
                img = self._apply_texture(img, self.textures['repel'], texture_alpha)
        
        return img

    def set_transformation_type(self, transform_type: str):
        """Set the current transformation type for visualization."""
        self.transformation_type = transform_type

    def _create_glow_effect(self, color: str, n_layers: int = 5) -> List[Tuple[str, float]]:
        """Create a glowing effect by layering colors with different alphas."""
        base_color = mcolors.to_rgb(color)
        glow_layers = []
        
        # Create inner glow
        for i in range(n_layers):
            alpha = 0.8 * (1 - i/n_layers)
            # Shift color slightly towards white for inner layers
            inner_color = [
                c + (1 - c) * (1 - i/n_layers) * 0.3
                for c in base_color
            ]
            glow_layers.append((mcolors.rgb2hex(inner_color), alpha))
        
        # Create outer glow
        for i in range(n_layers):
            alpha = 0.4 * (1 - i/n_layers)
            # Shift color slightly towards black for outer layers
            outer_color = [
                c * (1 - i/n_layers) * 0.7
                for c in base_color
            ]
            glow_layers.append((mcolors.rgb2hex(outer_color), alpha))
        
        return glow_layers
    
    def _create_flow_lines(
        self,
        shape: SemanticShape,
        n_lines: int = 50,
        n_points: int = 10
    ) -> np.ndarray:
        """Create flow lines showing semantic direction."""
        # Generate starting points in a spiral pattern
        theta = np.linspace(0, 4*np.pi, n_lines)
        r = np.linspace(0.1, 1.0, n_lines)
        points = np.array([
            [r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i])]
            for i in range(n_lines)
        ])
        
        # Create flow lines with varying step sizes
        lines = []
        for start_point in points:
            line = [start_point]
            point = start_point.copy()
            
            # Vary step size based on distance from center
            base_step = 0.1
            for i in range(n_points):
                # Get flow direction at current point
                flow = shape.get_flow_field(point[None, :])[0]
                
                # Calculate step size based on distance and complexity
                dist = np.linalg.norm(point)
                step_size = base_step * (1 + 0.5 * np.sin(dist * np.pi))
                
                # Move point along flow
                point = point + flow * step_size
                line.append(point.copy())
            
            lines.append(line)
            
        return np.array(lines)
    
    def _create_particle_effect(
        self,
        shape: SemanticShape,
        n_particles: int = 100
    ) -> np.ndarray:
        """Create particle effect based on shape properties."""
        # Generate particles in a circular pattern
        theta = np.random.uniform(0, 2*np.pi, n_particles)
        r = np.random.uniform(0, 1.5, n_particles)
        particles = np.array([
            [r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i])]
            for i in range(n_particles)
        ])
        
        # Calculate particle sizes based on distance from shape boundary
        sizes = []
        for p in particles:
            dist = min(np.linalg.norm(p - b) for b in shape.boundary)
            size = 50 * np.exp(-dist * 2)  # Exponential falloff
            sizes.append(size)
            
        return particles, np.array(sizes)
    
    def create_shape_field(
        self,
        text: str,
        chunk_size: int = 3,
        embedder: Optional[PhraseEmbedder] = None
    ) -> None:
        """
        Create a visualization of the semantic shape field for a text.
        
        Args:
            text: Text to analyze
            chunk_size: Number of sentences per chunk
            embedder: Optional PhraseEmbedder instance
        """
        # Create embedder if not provided
        if embedder is None:
            embedder = PhraseEmbedder()
        
        # Embed text chunks
        chunk_embeddings = embedder.embed_text(text, chunk_size)
        
        # Create shapes
        shapes = [SemanticShape(embedding) for embedding in chunk_embeddings]
        
        # Visualize evolution
        self.visualize_shape_evolution(shapes)
        
    def create_comparative_visualization(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None
    ) -> None:
        """
        Create a comparative visualization of semantic shapes for different texts.
        
        Args:
            texts: List of texts to compare
            labels: Optional labels for the texts
        """
        embedder = PhraseEmbedder()
        
        # Create shapes for each text
        all_shapes = []
        for text in texts:
            embeddings = embedder.embed_text(text)
            shapes = [SemanticShape(embedding) for embedding in embeddings]
            all_shapes.extend(shapes)
        
        # Create visualization
        self.visualize_shape_evolution(
            all_shapes,
            duration=len(all_shapes) * 2.0  # 2 seconds per shape
        )

    def export_animation(
        self,
        animation: FuncAnimation,
        name: str,
        frame_data: Optional[List[Path]] = None
    ) -> Path:
        """Export animation to file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.export_config.animations_dir / f"{name}_{timestamp}.{self.export_config.format}"
        
        try:
            if self.export_config.format == 'mp4':
                writer = FFMpegWriter(
                    fps=self.export_config.fps,
                    metadata=dict(title=name),
                    bitrate=self.export_config.bitrate
                )
                animation.save(str(output_path), writer=writer)
            else:
                animation.save(
                    str(output_path),
                    fps=self.export_config.fps,
                    dpi=self.export_config.dpi
                )
            
            logger.info(f"Animation exported to {output_path}")
            
            if frame_data:
                logger.info(f"Individual frames saved to {self.export_config.frames_dir}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export animation: {str(e)}", exc_info=True)
            raise

    def visualize_shape_evolution(
        self,
        shapes: List[SemanticShape],
        duration: float = 5.0,
        size: Tuple[int, int] = (1920, 1080),
        interpolation_steps: int = 30
    ) -> Tuple[FuncAnimation, Optional[List[Path]]]:
        """Create an animated visualization of shape evolution."""
        logger.info(f"Creating shape evolution visualization for {len(shapes)} shapes")
        
        # Create figure
        fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=self.export_config.dpi)
        ax = plt.gca()
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        
        # Calculate total frames
        total_frames = int(duration * self.export_config.fps)
        frame_data = [] if self.export_config.save_frames else None
        
        def update(frame):
            ax.clear()
            ax.set_facecolor('black')
            
            # Calculate interpolation
            t = frame / total_frames
            seq_pos = t * (len(shapes) - 1) * interpolation_steps
            idx = int(seq_pos / interpolation_steps)
            alpha = (seq_pos % interpolation_steps) / interpolation_steps
            
            current_shape = shapes[min(idx, len(shapes) - 1)]
            next_shape = shapes[idx + 1] if idx < len(shapes) - 1 else None
            
            # Generate visualization
            img = self._create_visualization_frame(current_shape, next_shape, alpha, size)
            ax.imshow(img)
            
            # Add text overlays
            self._add_text_overlays(ax, current_shape, next_shape, alpha, size)
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            if frame_data is not None:
                frame_path = self.export_config.frames_dir / f"frame_{frame:04d}.png"
                plt.savefig(frame_path, dpi=self.export_config.dpi, bbox_inches='tight')
                frame_data.append(frame_path)
            
            return ax.images
        
        # Create animation
        animation = FuncAnimation(
            fig,
            update,
            frames=total_frames,
            interval=1000/self.export_config.fps,
            blit=True
        )
        
        return animation, frame_data

    def _add_text_overlays(
        self,
        ax: plt.Axes,
        current_shape: SemanticShape,
        next_shape: Optional[SemanticShape],
        alpha: float,
        size: Tuple[int, int]
    ) -> None:
        """Add enhanced text overlays with shape information."""
        # Add main text with shadow for better visibility
        text_pos = (size[0] * 0.05, size[1] * 0.95)
        text = current_shape.text
        if next_shape and alpha > 0:
            text = f"{text} → {next_shape.text}"
        
        # Add shadow
        ax.text(*text_pos, text,
                color='black', fontsize=14, alpha=0.8,
                bbox=dict(facecolor='black', alpha=0.6),
                zorder=2)
        
        # Add actual text
        ax.text(*text_pos, text,
                color='white', fontsize=14, alpha=0.9,
                bbox=dict(facecolor='none', alpha=0),
                zorder=3)
        
        # Add metrics with enhanced formatting
        metrics = self._format_metrics(current_shape, next_shape, alpha)
        ax.text(size[0] * 0.95, size[1] * 0.05, metrics,
                color='white', fontsize=12, alpha=0.8,
                bbox=dict(facecolor='black', alpha=0.6),
                ha='right', va='bottom')

    def _format_metrics(
        self,
        shape: SemanticShape,
        next_shape: Optional[SemanticShape],
        alpha: float
    ) -> str:
        """Format shape metrics for display with interpolation."""
        metrics = []
        for key, value in shape.shape_params.items():
            if isinstance(value, (int, float)):
                if next_shape and alpha > 0:
                    next_val = next_shape.shape_params.get(key, value)
                    interpolated = value * (1 - alpha) + next_val * alpha
                    metrics.append(f"{key}: {interpolated:.2f}")
                else:
                    metrics.append(f"{key}: {value:.2f}")
        return "\n".join(metrics)
