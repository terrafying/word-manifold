"""
Interactive Renderer for Dynamic Visualizations

This module provides rendering capabilities for interactive visualizations,
including real-time updates, animations, and user interaction features.
"""

import plotly.graph_objects as go
from typing import List, Dict, Optional, Any, Union
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class InteractiveRenderer:
    """Renderer for interactive visualizations."""
    
    def __init__(self):
        # Default configuration
        self.config = {
            'plot_bgcolor': '#111111',
            'paper_bgcolor': '#111111',
            'font': {
                'color': '#ffffff',
                'family': 'Arial, sans-serif'
            },
            'showlegend': True,
            'margin': dict(l=40, r=40, t=40, b=40),
            'hovermode': 'closest'
        }
        
        # Animation defaults
        self.animation_defaults = {
            'frame': {'duration': 500, 'redraw': True},
            'transition': {'duration': 300}
        }
    
    def create_figure(self, title: Optional[str] = None) -> go.Figure:
        """Create a new figure with default styling."""
        fig = go.Figure()
        
        # Apply default styling
        fig.update_layout(
            **self.config,
            title=title or ''
        )
        
        return fig
    
    def add_scatter_trace(
        self,
        fig: go.Figure,
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray],
        name: str,
        mode: str = 'markers',
        marker: Optional[Dict[str, Any]] = None,
        line: Optional[Dict[str, Any]] = None,
        text: Optional[List[str]] = None,
        hovertext: Optional[List[str]] = None,
        **kwargs
    ) -> go.Figure:
        """Add a scatter trace to the figure."""
        trace = go.Scatter(
            x=x,
            y=y,
            name=name,
            mode=mode,
            marker=marker or dict(size=8),
            line=line,
            text=text,
            hovertext=hovertext,
            **kwargs
        )
        fig.add_trace(trace)
        return fig
    
    def add_animation_frames(
        self,
        fig: go.Figure,
        frames: List[Dict[str, Any]],
        slider_steps: Optional[List[Dict[str, Any]]] = None
    ) -> go.Figure:
        """Add animation frames and slider to the figure."""
        # Add frames
        fig.frames = frames
        
        # Add slider if steps provided
        if slider_steps:
            sliders = [{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 16},
                    'prefix': 'Frame: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 300},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': slider_steps
            }]
            fig.update_layout(sliders=sliders)
        
        # Update animation settings
        fig.update(frames=[go.Frame(**frame) for frame in frames])
        fig.update_layout(updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                dict(label='Play',
                     method='animate',
                     args=[None, self.animation_defaults]),
                dict(label='Pause',
                     method='animate',
                     args=[[None], {
                         'frame': {'duration': 0, 'redraw': False},
                         'mode': 'immediate',
                         'transition': {'duration': 0}
                     }])
            ],
            'x': 0.1,
            'y': 0,
            'xanchor': 'right',
            'yanchor': 'top'
        }])
        
        return fig
    
    def add_hover_data(
        self,
        fig: go.Figure,
        hover_data: Dict[str, List[Any]],
        template: str = "{key}: {value}"
    ) -> go.Figure:
        """Add hover data to the figure."""
        # Create hover text from data
        hover_text = []
        n_points = len(next(iter(hover_data.values())))
        
        for i in range(n_points):
            point_text = []
            for key, values in hover_data.items():
                if i < len(values):
                    point_text.append(template.format(
                        key=key,
                        value=values[i]
                    ))
            hover_text.append("<br>".join(point_text))
        
        # Update all traces with hover text
        for trace in fig.data:
            if len(trace.x) == n_points:  # Only update matching traces
                trace.hovertext = hover_text
                trace.hoverinfo = "text"
        
        return fig
    
    def add_colorbar(
        self,
        fig: go.Figure,
        values: Union[List[float], np.ndarray],
        colorscale: str = 'Viridis',
        title: Optional[str] = None
    ) -> go.Figure:
        """Add a colorbar to the figure."""
        fig.update_traces(
            marker=dict(
                color=values,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=title or '',
                    titleside='right',
                    thickness=15,
                    len=0.75
                )
            )
        )
        return fig
    
    def save_figure(
        self,
        fig: go.Figure,
        output_path: Union[str, Path],
        format: str = 'html',
        **kwargs
    ) -> None:
        """Save the figure to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'html':
                fig.write_html(output_path, **kwargs)
            elif format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(fig.to_dict(), f)
            else:
                fig.write_image(output_path, **kwargs)
            
            logger.info(f"Saved figure to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save figure: {e}")
            raise
    
    def update_figure_style(
        self,
        fig: go.Figure,
        style: Dict[str, Any]
    ) -> go.Figure:
        """Update figure styling."""
        fig.update_layout(**style)
        return fig
    
    def create_subplot_figure(
        self,
        rows: int,
        cols: int,
        titles: Optional[List[str]] = None,
        shared_xaxes: bool = False,
        shared_yaxes: bool = False
    ) -> go.Figure:
        """Create a figure with subplots."""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=titles,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes
        )
        
        # Apply default styling
        fig.update_layout(**self.config)
        
        return fig
    
    def add_annotations(
        self,
        fig: go.Figure,
        annotations: List[Dict[str, Any]]
    ) -> go.Figure:
        """Add annotations to the figure."""
        current = list(fig.layout.annotations or [])
        current.extend(annotations)
        fig.update_layout(annotations=current)
        return fig
    
    def create_interactive_symbolic_field(
        self,
        field: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """Create an interactive visualization of a symbolic field.
        
        Args:
            field: ASCII art field to visualize
            title: Optional title for the visualization
            
        Returns:
            Plotly figure with interactive features
        """
        # Convert ASCII field to 2D array
        lines = field.split('\n')
        height = len(lines)
        width = max(len(line) for line in lines)
        
        # Create character matrix and density matrix
        char_matrix = np.zeros((height, width), dtype=str)
        density_matrix = np.zeros((height, width))
        
        # Special characters have higher density values
        special_chars = '♠♡♢♣★☆⚝✧✦❈❉❊❋✺✹✸✷✶✵✴✳'
        
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                char_matrix[i, j] = char
                # Assign density based on character type
                if char in special_chars:
                    density_matrix[i, j] = 0.8
                elif char != ' ':
                    density_matrix[i, j] = 0.5
        
        # Create figure
        fig = self.create_figure(title)
        
        # Add heatmap for density
        fig.add_trace(go.Heatmap(
            z=density_matrix,
            text=char_matrix,
            texttemplate="%{text}",
            textfont={"size": 12},
            showscale=False,
            colorscale='Viridis'
        ))
        
        # Update layout for better text visibility
        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor='x',
                scaleratio=1
            )
        )
        
        # Add hover data
        hover_data = {
            'Character': char_matrix.flatten(),
            'Density': density_matrix.flatten()
        }
        self.add_hover_data(fig, hover_data)
        
        return fig 