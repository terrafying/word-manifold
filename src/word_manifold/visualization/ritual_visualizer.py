"""
Ritual Visualizer Module

This module provides visualization capabilities for ritual transformations,
integrating semantic, geometric, and energetic aspects of ritual states.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..automata.hermetic_principles import (HermeticPrinciple, PRINCIPLE_COLORS,
                                          PRINCIPLE_ENERGY_PATTERNS,
                                          PRINCIPLE_FREQUENCIES,
                                          PRINCIPLE_GEOMETRIES)
from ..embeddings.word_embeddings import WordEmbeddings
from .hypertools_visualizer import HyperToolsVisualizer


class RitualPhase(Enum):
    """Phases of ritual transformation."""
    PREPARATION = auto()
    INVOCATION = auto()
    TRANSFORMATION = auto()
    INTEGRATION = auto()
    COMPLETION = auto()


@dataclass
class RitualState:
    """Represents the current state of a ritual transformation."""
    phase: RitualPhase
    active_terms: Set[str]
    transformed_terms: Set[str]
    energy_level: float
    resonance_pattern: Dict[HermeticPrinciple, float]
    dominant_principle: HermeticPrinciple
    timestamp: datetime


class RitualVisualizer:
    """
    Visualizes ritual transformations by combining semantic, geometric,
    and energetic aspects into cohesive visual representations.
    """

    def __init__(
        self,
        word_embeddings: WordEmbeddings,
        output_dir: str = "visualizations/rituals",
        n_dims: int = 3,
        frame_duration: int = 500,
        energy_threshold: float = 0.7,
        resonance_threshold: float = 0.8
    ):
        """
        Initialize the RitualVisualizer.

        Args:
            word_embeddings: WordEmbeddings instance for semantic analysis
            output_dir: Directory to save visualization outputs
            n_dims: Number of dimensions for visualization (2 or 3)
            frame_duration: Duration of each frame in animations (ms)
            energy_threshold: Threshold for significant energy level changes
            resonance_threshold: Threshold for principle resonance
        """
        self.word_embeddings = word_embeddings
        self.output_dir = output_dir
        self.n_dims = n_dims
        self.frame_duration = frame_duration
        self.energy_threshold = energy_threshold
        self.resonance_threshold = resonance_threshold

        self.hypertools_viz = HyperToolsVisualizer(
            word_embeddings=word_embeddings,
            output_dir=output_dir,
            n_dims=n_dims
        )

        os.makedirs(output_dir, exist_ok=True)

        # Initialize state tracking
        self.states: List[RitualState] = []
        self.term_evolution: Dict[str, List[str]] = {}

    def visualize_ritual_sequence(
        self,
        ritual_name: str,
        initial_terms: Set[str],
        transformation_sequence: List[Tuple[str, str]],
        save_frames: bool = True
    ) -> None:
        """
        Visualize a complete ritual transformation sequence.

        Args:
            ritual_name: Name of the ritual for file naming
            initial_terms: Initial set of terms in the ritual
            transformation_sequence: List of (source_term, target_term) pairs
            save_frames: Whether to save individual frames
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ritual_dir = os.path.join(self.output_dir, f"{ritual_name}_{timestamp}")
        os.makedirs(ritual_dir, exist_ok=True)

        # Initialize first state
        initial_state = self._create_initial_state(initial_terms)
        self.states.append(initial_state)

        # Process each transformation
        for idx, (source, target) in enumerate(transformation_sequence):
            # Update state
            new_state = self._process_transformation(source, target)
            self.states.append(new_state)

            if save_frames:
                frame_path = os.path.join(ritual_dir, f"frame_{idx:03d}.html")
                self._save_visualization_frame(new_state, frame_path)

        # Create final visualization
        self._create_final_visualization(ritual_dir)

    def _create_initial_state(self, initial_terms: Set[str]) -> RitualState:
        """Create the initial ritual state."""
        return RitualState(
            phase=RitualPhase.PREPARATION,
            active_terms=initial_terms,
            transformed_terms=set(),
            energy_level=self._calculate_energy_level(initial_terms),
            resonance_pattern=self._calculate_resonance_pattern(initial_terms),
            dominant_principle=self._determine_dominant_principle(initial_terms),
            timestamp=datetime.now()
        )

    def _process_transformation(self, source: str, target: str) -> RitualState:
        """Process a single term transformation and return new state."""
        current_state = self.states[-1]
        
        # Update term sets
        new_active_terms = current_state.active_terms - {source} | {target}
        new_transformed_terms = current_state.transformed_terms | {source}

        # Track term evolution
        if source not in self.term_evolution:
            self.term_evolution[source] = []
        self.term_evolution[source].append(target)

        # Calculate new state properties
        new_energy = self._calculate_energy_level(new_active_terms)
        new_resonance = self._calculate_resonance_pattern(new_active_terms)
        new_principle = self._determine_dominant_principle(new_active_terms)
        new_phase = self._determine_ritual_phase(
            new_energy,
            len(new_transformed_terms),
            len(current_state.active_terms)
        )

        return RitualState(
            phase=new_phase,
            active_terms=new_active_terms,
            transformed_terms=new_transformed_terms,
            energy_level=new_energy,
            resonance_pattern=new_resonance,
            dominant_principle=new_principle,
            timestamp=datetime.now()
        )

    def _calculate_energy_level(self, terms: Set[str]) -> float:
        """Calculate the current energy level based on term embeddings."""
        if not terms:
            return 0.0

        # Get embeddings for terms
        embeddings = np.array([
            self.word_embeddings.get_embedding(term)
            for term in terms
            if self.word_embeddings.get_embedding(term) is not None
        ])

        if len(embeddings) == 0:
            return 0.0

        # Calculate energy based on embedding properties
        magnitude = np.linalg.norm(embeddings, axis=1).mean()
        coherence = np.abs(np.corrcoef(embeddings)).mean()
        
        return (magnitude * coherence) / 2

    def _calculate_resonance_pattern(
        self,
        terms: Set[str]
    ) -> Dict[HermeticPrinciple, float]:
        """Calculate resonance with each hermetic principle."""
        resonance = {principle: 0.0 for principle in HermeticPrinciple}
        
        if not terms:
            return resonance

        # Get term embeddings
        term_embeddings = np.array([
            self.word_embeddings.get_embedding(term)
            for term in terms
            if self.word_embeddings.get_embedding(term) is not None
        ])

        if len(term_embeddings) == 0:
            return resonance

        # Calculate resonance for each principle
        for principle in HermeticPrinciple:
            # Use principle frequency as weight
            frequency = PRINCIPLE_FREQUENCIES[principle]
            pattern = PRINCIPLE_ENERGY_PATTERNS[principle]
            
            # Calculate weighted similarity
            principle_resonance = np.mean([
                np.sin(frequency * t + pattern['phase']) * pattern['amplitude']
                for t in range(len(term_embeddings))
            ])
            
            resonance[principle] = abs(principle_resonance)

        return resonance

    def _determine_dominant_principle(
        self,
        terms: Set[str]
    ) -> HermeticPrinciple:
        """Determine the currently dominant hermetic principle."""
        resonance = self._calculate_resonance_pattern(terms)
        return max(resonance.items(), key=lambda x: x[1])[0]

    def _determine_ritual_phase(
        self,
        energy_level: float,
        num_transformed: int,
        total_terms: int
    ) -> RitualPhase:
        """Determine the current phase of the ritual."""
        progress = num_transformed / total_terms if total_terms > 0 else 0

        if progress == 0:
            return RitualPhase.PREPARATION
        elif progress < 0.3:
            return RitualPhase.INVOCATION
        elif progress < 0.7:
            return RitualPhase.TRANSFORMATION
        elif progress < 1.0:
            return RitualPhase.INTEGRATION
        else:
            return RitualPhase.COMPLETION

    def _save_visualization_frame(
        self,
        state: RitualState,
        output_path: str
    ) -> None:
        """Save a single frame of the visualization."""
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scene" if self.n_dims == 3 else "xy"}, {"type": "xy"}],
                  [{"type": "xy"}, {"type": "xy"}]],
            subplot_titles=(
                "Term Relationships",
                "Energy Level",
                "Principle Resonance",
                "Phase Progression"
            )
        )

        # Plot term relationships using HyperTools
        self.hypertools_viz.plot_terms(
            list(state.active_terms),
            color_by=lambda t: PRINCIPLE_COLORS[state.dominant_principle],
            fig=fig,
            row=1,
            col=1
        )

        # Plot energy level
        energy_history = [s.energy_level for s in self.states]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(energy_history))),
                y=energy_history,
                mode='lines+markers',
                name='Energy Level'
            ),
            row=1,
            col=2
        )

        # Plot principle resonance
        principles = list(state.resonance_pattern.keys())
        resonance_values = list(state.resonance_pattern.values())
        fig.add_trace(
            go.Bar(
                x=[p.name for p in principles],
                y=resonance_values,
                marker_color=[PRINCIPLE_COLORS[p] for p in principles]
            ),
            row=2,
            col=1
        )

        # Plot phase progression
        phases = [s.phase for s in self.states]
        phase_values = [p.value for p in phases]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(phases))),
                y=phase_values,
                mode='lines+markers',
                name='Ritual Phase'
            ),
            row=2,
            col=2
        )

        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            showlegend=False,
            title_text=f"Ritual State - {state.phase.name}"
        )

        # Save figure
        fig.write_html(output_path)

    def _create_final_visualization(self, ritual_dir: str) -> None:
        """Create and save the final visualization summary."""
        # Create summary plots
        self._plot_term_evolution(ritual_dir)
        self._plot_energy_progression(ritual_dir)
        self._plot_principle_dominance(ritual_dir)
        self._plot_phase_timeline(ritual_dir)

    def _plot_term_evolution(self, ritual_dir: str) -> None:
        """Plot the evolution of terms through the ritual."""
        fig = go.Figure()

        for source, targets in self.term_evolution.items():
            # Plot evolution path
            x = list(range(len(targets) + 1))
            y = [source] + targets
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers+text',
                name=source,
                text=[source] + targets,
                textposition="top center"
            ))

        fig.update_layout(
            title="Term Evolution Through Ritual",
            xaxis_title="Transformation Step",
            yaxis_title="Terms",
            height=800
        )

        fig.write_html(os.path.join(ritual_dir, "term_evolution.html"))

    def _plot_energy_progression(self, ritual_dir: str) -> None:
        """Plot the progression of energy levels."""
        energy_levels = [state.energy_level for state in self.states]
        timestamps = [state.timestamp for state in self.states]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=energy_levels,
            mode='lines+markers',
            name='Energy Level'
        ))

        fig.update_layout(
            title="Ritual Energy Progression",
            xaxis_title="Time",
            yaxis_title="Energy Level",
            height=600
        )

        fig.write_html(os.path.join(ritual_dir, "energy_progression.html"))

    def _plot_principle_dominance(self, ritual_dir: str) -> None:
        """Plot the dominance of hermetic principles over time."""
        principles = list(HermeticPrinciple)
        timestamps = [state.timestamp for state in self.states]
        
        fig = go.Figure()

        for principle in principles:
            dominance = [
                1 if state.dominant_principle == principle else 0
                for state in self.states
            ]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=dominance,
                mode='lines',
                name=principle.name,
                line=dict(color=PRINCIPLE_COLORS[principle])
            ))

        fig.update_layout(
            title="Hermetic Principle Dominance",
            xaxis_title="Time",
            yaxis_title="Dominance",
            height=600
        )

        fig.write_html(os.path.join(ritual_dir, "principle_dominance.html"))

    def _plot_phase_timeline(self, ritual_dir: str) -> None:
        """Plot the timeline of ritual phases."""
        phases = [state.phase for state in self.states]
        timestamps = [state.timestamp for state in self.states]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[phase.value for phase in phases],
            mode='lines+markers',
            name='Ritual Phase',
            text=[phase.name for phase in phases],
            textposition="top center"
        ))

        fig.update_layout(
            title="Ritual Phase Timeline",
            xaxis_title="Time",
            yaxis_title="Phase",
            height=600
        )

        fig.write_html(os.path.join(ritual_dir, "phase_timeline.html")) 