#!/usr/bin/env python3
"""
Resonance Explorer

Explores morphic resonance fields through mandala evolution,
with automatic node discovery and collective resonance.
"""

import ray
import click
import logging
import asyncio
from pathlib import Path
import numpy as np
from datetime import datetime
import random
from typing import Set, Optional

from word_manifold.visualization.engines.resonant_mandala import (
    ResonantMandalaEngine, ResonanceConfig, ResonanceField,
    MandalaConfig, MandalaStyle
)
from word_manifold.discovery.resonance_network import (
    create_resonance_network, CollectiveState
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResonanceExplorer:
    """Explores resonant mandala patterns."""
    
    def __init__(self, output_dir: Path, port: int = 5000):
        self.output_dir = output_dir
        self.port = port
        self.engine = ResonantMandalaEngine()
        self.active_fields: Set[ResonanceField] = set()
        self.discovery = None
        self.network_manager = None
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "patterns").mkdir(exist_ok=True)
        (self.output_dir / "evolution").mkdir(exist_ok=True)
        
    async def initialize_network(self):
        """Initialize resonance network."""
        self.discovery, self.network_manager = await create_resonance_network(
            port=self.port,
            active_fields=self.active_fields,
            metadata={
                "node_type": "explorer",
                "start_time": datetime.now().isoformat()
            }
        )
        
        # Set up callbacks
        self.discovery.on_node_join = self._handle_node_join
        self.discovery.on_node_leave = self._handle_node_leave
        self.discovery.on_field_update = self._handle_field_update
        
        logger.info("Resonance network initialized")
    
    def _handle_node_join(self, node):
        """Handle new node joining the network."""
        logger.info(f"Node joined: {node.node_id} with fields {node.active_fields}")
        
    def _handle_node_leave(self, node_id):
        """Handle node leaving the network."""
        logger.info(f"Node left: {node_id}")
        
    def _handle_field_update(self, node_id, fields):
        """Handle node updating its active fields."""
        logger.info(f"Node {node_id} updated fields: {fields}")
    
    def explore_field(
        self,
        field: ResonanceField,
        secondary_fields: Optional[Set[ResonanceField]] = None,
        meditation_depth: int = 5,
        evolution_steps: int = 3
    ):
        """Explore a resonance field."""
        # Update active fields
        self.active_fields.add(field)
        if secondary_fields:
            self.active_fields.update(secondary_fields)
        
        # Update network
        if self.discovery:
            self.discovery.update_fields(
                self.active_fields,
                field_strength=random.random()  # Could be more sophisticated
            )
        
        # Create base configuration
        base_config = MandalaConfig(
            radius=200,
            resolution=(1024, 1024),
            style=MandalaStyle.MYSTICAL,
            symmetry=8,
            complexity=1.0
        )
        
        # Configure resonance
        resonance_config = ResonanceConfig(
            primary_field=field,
            secondary_fields=secondary_fields or set(),
            intensity=0.8,
            meditation_depth=meditation_depth,
            collective_resonance=True
        )
        
        # Generate initial mandala
        result = self.engine.generate_resonant_mandala(base_config, resonance_config)
        
        # Save initial pattern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pattern_path = self.output_dir / "patterns" / f"{field.value}_{timestamp}.npy"
        np.save(pattern_path, result["pattern"])
        
        logger.info(f"\nExploring {field.value}:")
        logger.info(f"Primary Theme: {result['narrative']['primary_theme']}")
        logger.info("Secondary Themes:")
        for theme in result['narrative']['secondary_themes']:
            logger.info(f"  - {theme}")
        logger.info(f"\nDepth: {result['narrative']['depth_interpretation']}")
        logger.info(f"Resonance: {result['narrative']['resonance_quality']}")
        
        # Evolve through resonance
        evolution = self.engine.evolve_mandala(
            result,
            resonance_config,
            evolution_steps=evolution_steps
        )
        
        # Save evolution
        evolution_path = self.output_dir / "evolution" / f"{field.value}_{timestamp}"
        evolution_path.mkdir(exist_ok=True)
        
        logger.info("\nEvolution Journey:")
        for i, step in enumerate(evolution):
            np.save(evolution_path / f"step_{i:03d}.npy", step["pattern"])
            logger.info(f"\nStage {step['evolution_metrics']['step']}:")
            logger.info(f"Primary Movement: {step['narrative']['primary_movement']}")
            logger.info("Secondary Movements:")
            for movement in step['narrative']['secondary_movements']:
                logger.info(f"  - {movement}")
            logger.info(f"Field Strength: {step['evolution_metrics']['field_strength']:.2f}")
        
        return result, evolution

@click.group()
def cli():
    """Resonance exploration tools."""
    pass

@cli.command()
@click.option('--field', type=click.Choice([f.value for f in ResonanceField]), help='Primary resonance field')
@click.option('--secondary', multiple=True, type=click.Choice([f.value for f in ResonanceField]), help='Secondary fields')
@click.option('--depth', default=5, help='Meditation depth')
@click.option('--steps', default=3, help='Evolution steps')
@click.option('--port', default=5000, help='Network port')
@click.option('--output-dir', type=click.Path(path_type=Path), default='resonance_output', help='Output directory')
def explore(field: str, secondary: tuple, depth: int, steps: int, port: int, output_dir: Path):
    """Explore resonance fields."""
    explorer = ResonanceExplorer(output_dir, port=port)
    
    # Initialize network
    asyncio.run(explorer.initialize_network())
    
    try:
        # Convert field names to enums
        primary_field = ResonanceField(field)
        secondary_fields = {ResonanceField(f) for f in secondary}
        
        # Start exploration
        explorer.explore_field(
            field=primary_field,
            secondary_fields=secondary_fields,
            meditation_depth=depth,
            evolution_steps=steps
        )
        
    except KeyboardInterrupt:
        logger.info("Exploration interrupted")
    finally:
        if explorer.discovery:
            explorer.discovery.stop()

@cli.command()
@click.option('--port', default=5000, help='Network port')
def discover(port: int):
    """Discover active resonance nodes."""
    explorer = ResonanceExplorer(Path('resonance_output'), port=port)
    
    async def run_discovery():
        await explorer.initialize_network()
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            if explorer.discovery:
                explorer.discovery.stop()
    
    asyncio.run(run_discovery())

if __name__ == '__main__':
    cli() 