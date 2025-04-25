"""
Advanced Symbolic Visualizations

This module demonstrates advanced symbolic visualization capabilities,
including animated ASCII art, interactive symbolic fields,
and semantic pattern generation.
"""

import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import time
from tqdm import tqdm

from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.visualization.symbolic_visualizer import SymbolicVisualizer, SymbolicPattern
from word_manifold.visualization.engines.ascii_engine import ASCIIEngine
from word_manifold.visualization.renderers.ascii_renderer import ASCIIRenderer
from word_manifold.visualization.renderers.interactive import InteractiveRenderer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedSymbolicVisualization:
    """Advanced symbolic visualization capabilities."""
    
    def __init__(self, output_dir: str = "visualizations/advanced_symbolic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embeddings = WordEmbeddings()
        self.ascii_engine = ASCIIEngine()
        self.ascii_renderer = ASCIIRenderer()
        self.interactive_renderer = InteractiveRenderer()
        
        # Configure symbolic visualizer with larger canvas
        self.visualizer = SymbolicVisualizer(
            word_embeddings=self.embeddings,
            width=120,
            height=60
        )
        
        # Active patterns and their states
        self.active_patterns: Dict[str, SymbolicPattern] = {}
        self.pattern_states: Dict[str, float] = {}  # Energy levels
    
    def create_animated_mandala(
        self,
        central_term: str,
        related_terms: List[str],
        n_frames: int = 60,
        frame_delay: float = 0.1,
        use_color: bool = True,
        add_glow: bool = True
    ) -> str:
        """Create an animated ASCII mandala with advanced effects."""
        
        # Load terms and create patterns
        self.embeddings.load_terms([central_term] + related_terms)
        central_pattern = self._create_resonant_pattern(central_term)
        related_patterns = [self._create_resonant_pattern(term) for term in related_terms]
        
        # Generate mandala frames with effects
        frames = []
        for i in tqdm(range(n_frames), desc="Generating mandala frames"):
            # Create base pattern with phase shift
            pattern = self.ascii_engine.generate_mandala_pattern(
                central_term,
                radius=min(self.visualizer.width, self.visualizer.height) // 4,
                rotation=2 * np.pi * i / n_frames
            )
            
            # Add related term influences with varying energy
            for term_pattern in related_patterns:
                phase = 2 * np.pi * i / n_frames
                energy = 0.5 + 0.5 * np.sin(phase)  # Pulsing energy level
                influence = self.ascii_engine.generate_term_influence(
                    term_pattern.base_symbols,
                    phase=phase
                )
                pattern = self.ascii_engine.combine_patterns(pattern, influence)
            
            # Apply color if requested
            if use_color:
                pattern = self.ascii_renderer.apply_color(
                    pattern,
                    self._get_semantic_color(central_term)
                )
            
            # Add glow effect if requested
            if add_glow:
                pattern = self.ascii_renderer.create_gradient(
                    pattern,
                    start_color='blue',
                    end_color='cyan'
                )
            
            frames.append(pattern)
        
        # Save animation with metadata
        output_path = self.output_dir / f"{central_term}_mandala.txt"
        with open(output_path, 'w') as f:
            # Write animation metadata
            f.write(f"Animated Mandala for '{central_term}'\n")
            f.write(f"Related terms: {', '.join(related_terms)}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write frames with effects
            for i, frame in enumerate(frames):
                f.write(f"\033[2J\033[H")  # Clear screen and move cursor to top
                f.write(f"Frame {i+1}/{n_frames}\n")
                f.write(frame)
                f.write("\n" + "=" * 80 + "\n")
                time.sleep(frame_delay)
        
        return str(output_path)
    
    def create_symbolic_field(
        self,
        text: str,
        field_size: Tuple[int, int] = (80, 40),
        density: float = 0.7,
        interactive: bool = True,
        show_energy_flow: bool = True
    ) -> Tuple[str, Optional[str]]:
        """Create a symbolic field visualization with interactive features."""
        
        # Generate base symbolic field
        field = self.ascii_engine.generate_symbolic_field(
            text,
            width=field_size[0],
            height=field_size[1],
            density=density
        )
        
        # Add energy flow patterns if requested
        if show_energy_flow:
            # Create flow lines based on semantic relationships
            words = text.split()
            embeddings = [self.embeddings.get_embedding(word) for word in words]
            valid_embeddings = [e for e in embeddings if e is not None]
            
            if len(valid_embeddings) >= 2:
                # Use PCA to get main flow directions
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                flow_directions = pca.fit_transform(valid_embeddings)
                
                # Add flow patterns
                for direction in flow_directions:
                    angle = np.arctan2(direction[1], direction[0])
                    flow_pattern = self._create_flow_pattern(angle, field_size)
                    field = self.ascii_engine.combine_patterns(field, flow_pattern)
        
        # Save static field
        static_path = self.output_dir / "symbolic_field.txt"
        with open(static_path, 'w') as f:
            f.write(field)
        
        # Create interactive version if requested
        interactive_path = None
        if interactive:
            fig = self.interactive_renderer.create_interactive_symbolic_field(
                field,
                title="Interactive Symbolic Field"
            )
            
            interactive_path = self.output_dir / "symbolic_field.html"
            fig.write_html(str(interactive_path))
        
        return str(static_path), str(interactive_path) if interactive_path else None
    
    def create_transformation_sequence(
        self,
        source_text: str,
        target_text: str,
        n_steps: int = 10,
        pattern_size: Tuple[int, int] = (60, 30),
        add_intermediates: bool = True
    ) -> str:
        """Create a sequence of symbolic transformations with intermediate states."""
        
        # Generate source and target patterns
        source_pattern = self.ascii_engine.generate_pattern(
            source_text,
            width=pattern_size[0],
            height=pattern_size[1]
        )
        
        target_pattern = self.ascii_engine.generate_pattern(
            target_text,
            width=pattern_size[0],
            height=pattern_size[1]
        )
        
        # Create transformation sequence
        transformations = self.ascii_engine.create_transformation_sequence(
            source_pattern,
            target_pattern,
            n_steps=n_steps
        )
        
        # Add intermediate resonant patterns if requested
        if add_intermediates:
            # Get semantic midpoint
            source_embedding = self.embeddings.get_embedding(source_text)
            target_embedding = self.embeddings.get_embedding(target_text)
            if source_embedding is not None and target_embedding is not None:
                midpoint = (source_embedding + target_embedding) / 2
                # Find terms near midpoint
                similar_terms = self.embeddings.find_similar_terms_by_vector(midpoint, k=3)
                
                for term, _ in similar_terms:
                    intermediate = self.ascii_engine.generate_pattern(
                        term,
                        width=pattern_size[0],
                        height=pattern_size[1]
                    )
                    # Insert at middle of sequence
                    mid_idx = len(transformations) // 2
                    transformations.insert(mid_idx, intermediate)
        
        # Save sequence with enhanced formatting
        output_path = self.output_dir / "transformation_sequence.txt"
        with open(output_path, 'w') as f:
            f.write(f"Transformation: '{source_text}' → '{target_text}'\n")
            f.write("=" * 80 + "\n\n")
            
            for i, pattern in enumerate(transformations):
                # Add frame decoration
                frame = self.ascii_renderer.render_frame(
                    pattern,
                    title=f"Step {i+1}/{len(transformations)}",
                    frame_char='█'
                )
                f.write(frame)
                f.write("\n" + "-" * 80 + "\n")
        
        return str(output_path)
    
    def _create_resonant_pattern(self, term: str) -> SymbolicPattern:
        """Create a pattern that resonates with a term's semantic properties."""
        # Get semantic properties
        embedding = self.embeddings.get_embedding(term)
        if embedding is None:
            return SymbolicPattern(
                base_symbols="○□△",
                transformations=[],
                meaning=term,
                energy=0.5,
                resonance=set()
            )
        
        # Find similar terms for resonance
        similar_terms = {term for term, _ in self.embeddings.find_similar_terms(term, k=5)}
        
        # Select symbols based on semantic properties
        abstract_ratio = np.mean(np.abs(embedding[:10]))  # Use first 10 dimensions
        if abstract_ratio > 0.7:
            symbols = self.visualizer.ABSTRACT_SYMBOLS
        elif abstract_ratio > 0.4:
            symbols = self.visualizer.SACRED_SYMBOLS
        else:
            symbols = self.visualizer.ORGANIC_SYMBOLS
        
        # Create transformation sequence
        transformations = []
        n_transforms = 5
        for i in range(n_transforms):
            # Shift and rotate symbols
            shifted = symbols[i:] + symbols[:i]
            if i % 2 == 0:
                shifted = shifted[::-1]  # Reverse on even steps
            transformations.append(shifted)
        
        return SymbolicPattern(
            base_symbols=symbols,
            transformations=transformations,
            meaning=term,
            energy=abstract_ratio,
            resonance=similar_terms
        )
    
    def _create_flow_pattern(
        self,
        angle: float,
        size: Tuple[int, int]
    ) -> str:
        """Create a flow pattern in the specified direction."""
        width, height = size
        pattern = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Flow characters based on angle
        if -np.pi/4 <= angle <= np.pi/4:  # Right
            flow_chars = '→⇢⟶'
        elif np.pi/4 < angle <= 3*np.pi/4:  # Down
            flow_chars = '↓⇣⟱'
        elif -3*np.pi/4 <= angle < -np.pi/4:  # Up
            flow_chars = '↑⇡⟰'
        else:  # Left
            flow_chars = '←⇠⟵'
        
        # Add flow characters sparsely
        for y in range(height):
            for x in range(width):
                if np.random.random() < 0.1:  # 10% density
                    pattern[y][x] = np.random.choice(list(flow_chars))
        
        return '\n'.join(''.join(row) for row in pattern)
    
    def _get_semantic_color(self, term: str) -> str:
        """Get a color based on term's semantic properties."""
        embedding = self.embeddings.get_embedding(term)
        if embedding is None:
            return 'white'
        
        # Map first three dimensions to RGB
        rgb = embedding[:3]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # Normalize to [0,1]
        
        # Convert to color name
        if np.mean(rgb) > 0.7:
            return 'white'
        elif rgb[0] > 0.6:
            return 'red'
        elif rgb[1] > 0.6:
            return 'green'
        elif rgb[2] > 0.6:
            return 'blue'
        else:
            return 'cyan'  # Default color

def main():
    """Run advanced symbolic visualization examples."""
    try:
        # Initialize visualization
        viz = AdvancedSymbolicVisualization()
        
        # Example 1: Animated Mandala
        central_term = "enlightenment"
        related_terms = [
            "wisdom", "understanding", "knowledge",
            "illumination", "awakening", "consciousness"
        ]
        
        mandala_path = viz.create_animated_mandala(
            central_term=central_term,
            related_terms=related_terms,
            n_frames=120,
            frame_delay=0.08,
            use_color=True,
            add_glow=True
        )
        
        logger.info(f"Created animated mandala: {mandala_path}")
        
        # Example 2: Symbolic Field
        mystical_text = """
        In the sacred patterns of existence,
        symbols dance with eternal meaning.
        Each form reveals a hidden truth,
        as above, so below, in perfect symmetry.
        """
        
        field_path, interactive_path = viz.create_symbolic_field(
            text=mystical_text,
            field_size=(100, 50),
            density=0.8,
            interactive=True,
            show_energy_flow=True
        )
        
        logger.info(f"Created symbolic field: {field_path}")
        if interactive_path:
            logger.info(f"Created interactive visualization: {interactive_path}")
        
        # Example 3: Transformation Sequence
        source = "The shadow self"
        target = "The illuminated being"
        
        sequence_path = viz.create_transformation_sequence(
            source_text=source,
            target_text=target,
            n_steps=15,
            pattern_size=(80, 40),
            add_intermediates=True
        )
        
        logger.info(f"Created transformation sequence: {sequence_path}")
        
    except Exception as e:
        logger.error("Error in advanced symbolic visualization example", exc_info=e)

if __name__ == "__main__":
    main() 