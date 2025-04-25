"""
Three.js renderer for geometric visualizations with hyperdimensional projection support.
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Sequence, Tuple
import numpy as np
from flask import Flask, render_template, jsonify, request

from word_manifold.visualization.engines.projection import ProjectionEngine

logger = logging.getLogger(__name__)

# Type aliases
Point3D = Sequence[float]  # (x, y, z)
Point = Sequence[float]    # n-dimensional point
Color = str               # hex color string

class ThreeRenderer:
    """Renderer for 3D visualizations using Three.js."""
    
    def __init__(self,
                 template_dir: Optional[str] = None,
                 port: int = 5000,
                 background_color: Color = "#000000",
                 ambient_intensity: float = 0.4,
                 directional_intensity: float = 0.6,
                 projection_type: str = "stereographic"):
        """
        Initialize the renderer.
        
        Args:
            template_dir: Directory containing HTML templates
            port: Port for the Flask server
            background_color: Scene background color
            ambient_intensity: Ambient light intensity
            directional_intensity: Directional light intensity
            projection_type: Type of projection for high-dimensional data
        """
        self.port = port
        self.background_color = background_color
        self.ambient_intensity = ambient_intensity
        self.directional_intensity = directional_intensity
        
        # Set up projection engine
        self.projection_engine = ProjectionEngine(
            projection_type=projection_type
        )
        
        # Set up Flask app with proper template directory
        if template_dir is None:
            # Default to the templates directory in the visualization package
            template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
        
        self.app = Flask(__name__, 
                        template_folder=template_dir,
                        static_folder=os.path.join(os.path.dirname(template_dir), 'static'))
            
        # Register routes
        self._register_routes()
        
        # Store shape data
        self.shape_data: Optional[Dict[str, Any]] = None
        self.animation_data: Optional[Dict[str, Any]] = None
        
    def _register_routes(self):
        """Register Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template(
                'three_viewer.html',
                shape_data=self.shape_data,
                background_color=self.background_color,
                ambient_intensity=self.ambient_intensity,
                directional_intensity=self.directional_intensity
            )
            
        @self.app.route('/data')
        def get_data():
            return jsonify({
                "shape": self.shape_data,
                "animation": self.animation_data,
                "settings": {
                    "background": self.background_color,
                    "ambient": self.ambient_intensity,
                    "directional": self.directional_intensity
                }
            })
            
    def prepare_shape_data(self,
                          points: np.ndarray,
                          edges: Optional[List[Tuple[int, int]]] = None,
                          colors: Optional[np.ndarray] = None,
                          sizes: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Prepare shape data for visualization.
        
        Args:
            points: Point coordinates (n_points, n_dimensions)
            edges: Optional list of (start, end) vertex indices
            colors: Optional array of RGB colors
            sizes: Optional array of point sizes
            
        Returns:
            Dictionary of prepared shape data
        """
        # Project points to 3D
        projected_points = self.projection_engine.project_points(points)
        
        # Convert to list format
        shape_data = {
            "points": projected_points.tolist()
        }
        
        # Add optional data
        if edges is not None:
            shape_data["edges"] = self.projection_engine.project_edges(
                points, edges
            )
            
        if colors is not None:
            shape_data["colors"] = colors.tolist()
            
        if sizes is not None:
            shape_data["sizes"] = sizes.tolist()
            
        return shape_data
        
    def render_shapes(self,
                     points: np.ndarray,
                     edges: Optional[List[Tuple[int, int]]] = None,
                     colors: Optional[np.ndarray] = None,
                     sizes: Optional[np.ndarray] = None):
        """
        Render shapes in 3D viewer.
        
        Args:
            points: Point coordinates
            edges: Optional edge connections
            colors: Optional point colors
            sizes: Optional point sizes
        """
        self.shape_data = self.prepare_shape_data(
            points=points,
            edges=edges,
            colors=colors,
            sizes=sizes
        )
        
        logger.info(f"Starting Three.js server on port {self.port}")
        self.app.run(port=self.port)
        
    def save_static(self,
                   points: np.ndarray,
                   output_path: Union[str, Path],
                   edges: Optional[List[Tuple[int, int]]] = None,
                   colors: Optional[np.ndarray] = None,
                   sizes: Optional[np.ndarray] = None):
        """
        Save static visualization to HTML file.
        
        Args:
            points: Point coordinates
            output_path: Path to save HTML file
            edges: Optional edge connections
            colors: Optional point colors
            sizes: Optional point sizes
        """
        shape_data = self.prepare_shape_data(
            points=points,
            edges=edges,
            colors=colors,
            sizes=sizes
        )
        
        with self.app.app_context():
            html = render_template(
                'three_viewer.html',
                shape_data=shape_data,
                background_color=self.background_color,
                ambient_intensity=self.ambient_intensity,
                directional_intensity=self.directional_intensity
            )
        
        output_path = Path(output_path)
        output_path.write_text(html)
        
    def create_animation(self,
                        points: np.ndarray,
                        n_steps: int = 10,
                        edges: Optional[List[Tuple[int, int]]] = None,
                        colors: Optional[np.ndarray] = None,
                        sizes: Optional[np.ndarray] = None):
        """
        Create animation sequence for dimensional transitions.
        
        Args:
            points: Point coordinates
            n_steps: Number of animation steps
            edges: Optional edge connections
            colors: Optional point colors
            sizes: Optional point sizes
        """
        # Generate projection sequence
        point_sequence = self.projection_engine.create_projection_sequence(
            points,
            n_steps
        )
        
        # Prepare frame data
        frames = []
        for projected_points in point_sequence:
            frame_data = self.prepare_shape_data(
                points=projected_points,
                edges=edges,
                colors=colors,
                sizes=sizes
            )
            frames.append(frame_data)
            
        self.animation_data = {
            "frames": frames,
            "frame_delay": 100  # milliseconds
        }
        
        logger.info(f"Starting Three.js animation server on port {self.port}")
        self.app.run(port=self.port)
        
    def update_shapes(self,
                     points: np.ndarray,
                     edges: Optional[List[Tuple[int, int]]] = None,
                     colors: Optional[np.ndarray] = None,
                     sizes: Optional[np.ndarray] = None):
        """
        Update shape data for real-time visualization.
        
        Args:
            points: Point coordinates
            edges: Optional edge connections
            colors: Optional point colors
            sizes: Optional point sizes
        """
        self.shape_data = self.prepare_shape_data(
            points=points,
            edges=edges,
            colors=colors,
            sizes=sizes
        ) 

    def project_points(self, points: List[Point]) -> List[Point3D]:
        """
        Project high-dimensional points to 3D.
        
        Args:
            points: List of n-dimensional points to project
            
        Returns:
            List of projected 3D points
        """
        return self.projection_engine.project(points)

    def add_shape(self, 
                  vertices: List[Point],
                  edges: List[Sequence[int]] = None,
                  colors: List[Color] = None,
                  name: str = "shape") -> None:
        """
        Add a shape to the scene.
        
        Args:
            vertices: List of n-dimensional vertices
            edges: List of vertex index pairs defining edges
            colors: List of colors for vertices
            name: Name of the shape
        """
        projected = self.project_points(vertices)
        
        shape_data = {
            "vertices": projected,
            "edges": edges or [],
            "colors": colors or ["#ffffff"] * len(vertices),
            "name": name
        }
        
        self.shape_data = shape_data

    def add_animation(self,
                     keyframes: List[List[Point]],
                     edges: List[Sequence[int]] = None,
                     colors: List[Color] = None,
                     duration: float = 5.0,
                     name: str = "animation") -> None:
        """
        Add an animation sequence to the scene.
        
        Args:
            keyframes: List of frames, each containing n-dimensional points
            edges: List of vertex index pairs defining edges
            colors: List of colors for vertices
            duration: Duration of animation in seconds
            name: Name of the animation
        """
        projected_frames = [self.project_points(frame) for frame in keyframes]
        
        animation_data = {
            "frames": projected_frames,
            "edges": edges or [],
            "colors": colors or ["#ffffff"] * len(keyframes[0]),
            "duration": duration,
            "name": name
        }
        
        self.animation_data = animation_data

    def serve(self, debug: bool = False):
        """
        Start the Flask server.
        
        Args:
            debug: Whether to run Flask in debug mode
        """
        self.app.run(port=self.port, debug=debug)

    def save_html(self, output_path: Union[str, Path]) -> None:
        """
        Save the visualization as a standalone HTML file.
        
        Args:
            output_path: Path to save the HTML file
        """
        if isinstance(output_path, str):
            output_path = Path(output_path)
            
        # Create a temporary directory for the template
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "visualization.html"
            
            # Write data to the template
            with open(temp_path, 'w') as f:
                template = render_template('three_viewer.html')
                data = {
                    "shape": self.shape_data,
                    "animation": self.animation_data,
                    "settings": {
                        "background": self.background_color,
                        "ambient": self.ambient_intensity,
                        "directional": self.directional_intensity
                    }
                }
                
                # Insert data into the template
                template = template.replace(
                    '// DATA_PLACEHOLDER',
                    f'const data = {json.dumps(data)};'
                )
                
                f.write(template)
            
            # Copy to final destination
            output_path.write_text(temp_path.read_text()) 