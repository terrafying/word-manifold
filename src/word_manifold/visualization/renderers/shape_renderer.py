"""
Shape Renderer for Geometric Visualizations

This module provides rendering capabilities for geometric shape visualizations,
including static plots, animations, and interactive visualizations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from pathlib import Path

from ..engines.shape_engine import Shape, ShapeField

class ShapeRenderer:
    """Renderer for geometric shape visualizations."""
    
    def __init__(self):
        # Default style configuration
        self.style = {
            'figure.figsize': (10, 10),
            'figure.dpi': 100,
            'figure.facecolor': 'white',
            'axes.facecolor': '#f0f0f0',
            'axes.grid': True,
            'grid.alpha': 0.3
        }
        
        # Apply style
        plt.style.use('default')
        plt.rcParams.update(self.style)
    
    def render_shape_field(
        self,
        field: ShapeField,
        title: Optional[str] = None,
        show_grid: bool = True,
        show_labels: bool = False
    ) -> plt.Figure:
        """Render a shape field as a static plot."""
        fig, ax = plt.subplots()
        
        # Create patches for all shapes
        patches = [shape.to_patch() for shape in field.shapes]
        collection = plt.PatchCollection(patches, match_original=True)
        ax.add_collection(collection)
        
        # Set bounds
        bounds = field.get_bounds()
        margin = 0.1 * max(bounds[1] - bounds[0], bounds[3] - bounds[2])
        ax.set_xlim(bounds[0] - margin, bounds[1] + margin)
        ax.set_ylim(bounds[2] - margin, bounds[3] + margin)
        
        # Add labels if requested
        if show_labels:
            for shape in field.shapes:
                if shape.properties and 'word' in shape.properties:
                    ax.annotate(
                        shape.properties['word'],
                        shape.center,
                        ha='center',
                        va='center',
                        color='black',
                        alpha=0.7
                    )
        
        # Configure axes
        ax.set_aspect('equal')
        if not show_grid:
            ax.grid(False)
        
        if title:
            ax.set_title(title)
        
        return fig
    
    def create_animation(
        self,
        fields: List[ShapeField],
        duration: float = 10.0,
        fps: int = 30,
        title: Optional[str] = None
    ) -> FuncAnimation:
        """Create an animation from a sequence of shape fields."""
        fig, ax = plt.subplots()
        
        # Get global bounds
        all_bounds = [field.get_bounds() for field in fields]
        xmin = min(bounds[0] for bounds in all_bounds)
        xmax = max(bounds[1] for bounds in all_bounds)
        ymin = min(bounds[2] for bounds in all_bounds)
        ymax = max(bounds[3] for bounds in all_bounds)
        
        margin = 0.1 * max(xmax - xmin, ymax - ymin)
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
        
        # Configure axes
        ax.set_aspect('equal')
        if title:
            ax.set_title(title)
        
        # Create update function for animation
        def update(frame):
            ax.clear()
            field = fields[frame]
            
            # Create patches for current field
            patches = [shape.to_patch() for shape in field.shapes]
            collection = plt.PatchCollection(patches, match_original=True)
            ax.add_collection(collection)
            
            # Maintain bounds and style
            ax.set_xlim(xmin - margin, xmax + margin)
            ax.set_ylim(ymin - margin, ymax + margin)
            ax.set_aspect('equal')
            if title:
                ax.set_title(f"{title} - Frame {frame+1}/{len(fields)}")
            
            return collection,
        
        # Create animation
        n_frames = len(fields)
        interval = duration * 1000 / n_frames  # Convert to milliseconds
        
        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval,
            blit=True
        )
        
        return anim
    
    def create_geometric_relationships(
        self,
        shapes: List[ShapeField],
        labels: Optional[List[str]] = None,
        show_connections: bool = True
    ) -> go.Figure:
        """Create an interactive visualization of geometric relationships."""
        # Create plotly figure
        fig = go.Figure()
        
        # Process each shape field
        for i, field in enumerate(shapes):
            # Calculate field center
            centers = np.array([shape.center for shape in field.shapes])
            field_center = np.mean(centers, axis=0)
            
            # Add shapes as scatter points
            for shape in field.shapes:
                fig.add_trace(go.Scatter(
                    x=[shape.center[0]],
                    y=[shape.center[1]],
                    mode='markers',
                    marker=dict(
                        size=shape.size * 20,  # Scale for visibility
                        color=shape.color,
                        symbol=self._get_plotly_symbol(shape.type),
                        opacity=shape.alpha
                    ),
                    name=labels[i] if labels else f"Field {i+1}"
                ))
            
            # Add connections between fields if requested
            if show_connections and i > 0:
                prev_center = np.mean([
                    shape.center for shape in shapes[i-1].shapes
                ], axis=0)
                
                fig.add_trace(go.Scatter(
                    x=[prev_center[0], field_center[0]],
                    y=[prev_center[1], field_center[1]],
                    mode='lines',
                    line=dict(
                        color='gray',
                        width=1,
                        dash='dot'
                    ),
                    showlegend=False
                ))
        
        # Configure layout
        fig.update_layout(
            title="Geometric Relationships",
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='#f0f0f0',
            xaxis=dict(
                showgrid=True,
                gridcolor='white',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='white',
                zeroline=False,
                scaleanchor='x',
                scaleratio=1
            )
        )
        
        return fig
    
    def save_visualization(
        self,
        fig: Any,
        output_path: Path,
        **kwargs
    ) -> None:
        """Save visualization to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(fig, plt.Figure):
            fig.savefig(output_path, **kwargs)
            plt.close(fig)
        elif isinstance(fig, go.Figure):
            fig.write_html(output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported figure type: {type(fig)}")
    
    def _get_plotly_symbol(self, shape_type: str) -> str:
        """Convert shape type to plotly symbol."""
        symbol_map = {
            'circle': 'circle',
            'square': 'square',
            'triangle': 'triangle-up',
            'rectangle': 'square'
        }
        return symbol_map.get(shape_type, 'circle') 