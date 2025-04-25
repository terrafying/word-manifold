"""
CLI command for magic structure visualization.

This module provides a command line interface for generating and visualizing
magic squares, cubes, and higher-dimensional magic structures.
"""

import click
from pathlib import Path
import logging
from typing import List, Optional
from ...visualization.magic_visualizer import MagicVisualizer
from ...embeddings.word_embeddings import WordEmbeddings

logger = logging.getLogger(__name__)

@click.command()
@click.option(
    '--dimension', '-d',
    type=int,
    default=2,
    help='Number of dimensions for the magic structure (default: 2)'
)
@click.option(
    '--size', '-s',
    type=int,
    default=3,
    help='Size in each dimension (default: 3)'
)
@click.option(
    '--terms',
    type=str,
    multiple=True,
    help='Terms to use for semantic weighting'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(),
    default='visualizations/magic',
    help='Output directory for visualizations'
)
@click.option(
    '--interactive/--static',
    default=False,
    help='Create interactive visualization (default: False)'
)
@click.option(
    '--show-values/--hide-values',
    default=True,
    help='Show numeric values in visualization (default: True)'
)
@click.option(
    '--show-connections/--hide-connections',
    default=True,
    help='Show connections between cells (default: True)'
)
@click.option(
    '--color-scheme',
    type=str,
    default='viridis',
    help='Color scheme for visualization (default: viridis)'
)
@click.option(
    '--semantic-weighting/--no-semantic-weighting',
    default=True,
    help='Apply semantic weighting to values (default: True)'
)
@click.option(
    '--embeddings-model',
    type=str,
    default='en_core_web_lg',
    help='Word embeddings model to use (default: en_core_web_lg)'
)
def magic(
    dimension: int,
    size: int,
    terms: List[str],
    output_dir: str,
    interactive: bool,
    show_values: bool,
    show_connections: bool,
    color_scheme: str,
    semantic_weighting: bool,
    embeddings_model: str
) -> None:
    """
    Generate and visualize magic structures.
    
    This command creates magic squares (2D), cubes (3D), or higher-dimensional
    magic structures. The structures can be weighted by semantic meaning of
    provided terms using word embeddings.
    
    Example usage:
    \b
    # Create a 3x3 magic square
    magic -d 2 -s 3
    
    \b
    # Create a 4x4x4 magic cube with semantic weighting
    magic -d 3 -s 4 --terms love wisdom power
    
    \b
    # Create an interactive 5D visualization
    magic -d 5 -s 3 --interactive
    """
    try:
        # Initialize word embeddings if needed
        word_embeddings = None
        if terms and semantic_weighting:
            try:
                word_embeddings = WordEmbeddings(model_name=embeddings_model)
                logger.info(f"Loaded word embeddings model: {embeddings_model}")
            except Exception as e:
                logger.warning(f"Failed to load word embeddings: {e}")
                logger.warning("Continuing without semantic weighting")
        
        # Create visualizer
        visualizer = MagicVisualizer(
            word_embeddings=word_embeddings,
            output_dir=output_dir,
            enable_semantic_weighting=semantic_weighting,
            color_scheme=color_scheme
        )
        
        # Generate magic structure
        structure = visualizer.generate_magic_structure(
            dimension=dimension,
            size=size,
            terms=list(terms) if terms else None
        )
        
        # Validate structure
        if not structure.is_magic():
            logger.warning("Generated structure does not satisfy magic properties")
        
        # Create output filename
        dim_str = f"{size}{'d' * dimension}"
        terms_str = '_'.join(terms) if terms else 'no_terms'
        filename = f"magic_{dim_str}_{terms_str}"
        
        if interactive:
            save_path = str(Path(output_dir) / f"{filename}.html")
        else:
            save_path = str(Path(output_dir) / f"{filename}.png")
        
        # Create visualization
        visualizer.visualize(
            structure=structure,
            title=f"{size}{'D' * dimension} Magic Structure",
            show_values=show_values,
            show_connections=show_connections,
            interactive=interactive,
            save_path=save_path
        )
        
        logger.info(f"Saved visualization to: {save_path}")
        
        # Print magic constant
        click.echo(f"Magic constant: {structure.magic_constant:.2f}")
        
    except Exception as e:
        logger.error(f"Failed to create magic structure visualization: {e}")
        raise click.ClickException(str(e)) 