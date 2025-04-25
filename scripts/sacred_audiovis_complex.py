"""
Generate complex synchronized audio-visual patterns based on sacred geometry.
"""

import os
import time
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple

from word_manifold.core.audio import (
    AudioParams, generate_sacred_audio, get_audio_features, save_audio,
    WaveShape, ModulationType
)
from word_manifold.visualization.engines.ascii import ASCIIEngine, Pattern
from word_manifold.visualization.renderers.image_renderer import ImageRenderer

@dataclass
class VisualizationParams:
    """Parameters for visualization generation."""
    radius: int = 20
    symmetry: int = 8
    style: str = "mystical"
    layers: int = 3
    width: int = 120
    height: int = 60

@dataclass
class AudioVisualParams:
    """Combined parameters for audio-visual generation."""
    sample_rate: int = 44100
    duration: float = 10.0
    base_freq: float = 432.0  # Sacred A
    harmonics: List[float] = None
    display_size: Tuple[int, int] = (120, 60)
    frame_rate: int = 12
    wave_shapes: List[WaveShape] = None
    modulation_types: List[ModulationType] = None
    use_fibonacci: bool = True
    use_phi: bool = True

    def __post_init__(self):
        if self.harmonics is None:
            # Using sacred ratios including golden ratio
            self.harmonics = [1.0, 1.618034, 2.0, 2.618034, 3.0, 4.0]
        if self.wave_shapes is None:
            self.wave_shapes = [
                WaveShape.SINE,
                WaveShape.TRIANGLE,
                WaveShape.SQUARE,
                WaveShape.SAWTOOTH,
                WaveShape.PULSE
            ]
        if self.modulation_types is None:
            self.modulation_types = [
                ModulationType.AMPLITUDE,
                ModulationType.FREQUENCY,
                ModulationType.PHASE,
                ModulationType.RING
            ]

def map_audio_to_visual(audio_features: dict, base_params: VisualizationParams) -> VisualizationParams:
    """Maps audio features to visual parameters."""
    # Extract audio features
    freq = audio_features.get('frequency', 432.0)
    amplitude = audio_features.get('amplitude', 0.5)
    harmonic_content = audio_features.get('harmonic_content', 0.5)
    spectral_centroid = audio_features.get('spectral_centroid', 1000)
    
    # Map frequency to radius (logarithmic scaling)
    radius = int(10 + 20 * np.log10(freq / 432.0 + 1))
    radius = max(5, min(40, radius))
    
    # Map spectral centroid to symmetry (even numbers)
    symmetry = int(4 + 4 * (spectral_centroid / 2000))
    symmetry = max(4, min(12, symmetry * 2)) # Ensure even numbers
    
    # Map harmonic content to number of layers
    layers = int(2 + 4 * harmonic_content)
    layers = max(2, min(6, layers))
    
    # Create new params
    return VisualizationParams(
        radius=radius,
        symmetry=symmetry,
        style=base_params.style,
        layers=layers,
        width=base_params.width,
        height=base_params.height
    )

def generate_synchronized_patterns(params: AudioVisualParams) -> Tuple[str, List[str]]:
    """Generates synchronized audio and visual patterns."""
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "data/audiovis/complex"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate multiple audio segments with different parameters
    segments = []
    segment_duration = params.duration / 4  # Split into 4 segments
    
    for i in range(4):
        # Vary parameters for each segment
        wave_shape = params.wave_shapes[i % len(params.wave_shapes)]
        mod_type = params.modulation_types[i % len(params.modulation_types)]
        
        # Create segment-specific AudioParams
        audio_params = AudioParams(
            sample_rate=params.sample_rate,
            duration=segment_duration,
            base_freq=params.base_freq * (1 + 0.1 * np.sin(i * np.pi/2)),  # Slight frequency variation
            harmonics=params.harmonics,
            wave_shape=wave_shape,
            modulation_type=mod_type,
            modulation_depth=0.3 + 0.2 * np.sin(i * np.pi/2),  # Varying modulation depth
            modulation_freq=0.5 * (i + 1),  # Increasing modulation frequency
            use_fibonacci=params.use_fibonacci,
            use_phi=params.use_phi,
            resonance=1.0 + 0.5 * np.sin(i * np.pi/2),  # Varying resonance
            phase_shift=i * np.pi/4  # Phase variation
        )
        
        # Generate segment
        _, segment_audio = generate_sacred_audio(audio_params)
        segments.append(segment_audio)
    
    # Concatenate segments with crossfade
    crossfade_samples = int(0.1 * params.sample_rate)  # 100ms crossfade
    audio_data = np.zeros(int(params.duration * params.sample_rate))
    
    for i, segment in enumerate(segments):
        start_idx = int(i * segment_duration * params.sample_rate)
        end_idx = start_idx + len(segment)
        
        if i > 0:  # Apply crossfade with previous segment
            fade_in = np.linspace(0, 1, crossfade_samples)
            fade_out = np.linspace(1, 0, crossfade_samples)
            audio_data[start_idx:start_idx + crossfade_samples] *= fade_out
            segment[:crossfade_samples] *= fade_in
        
        audio_data[start_idx:end_idx] += segment
    
    # Save audio file
    audio_file = os.path.join(output_dir, f"sacred_audio_{timestamp}.wav")
    save_audio(audio_data, params.sample_rate, audio_file)
    
    # Initialize ASCII engine
    ascii_engine = ASCIIEngine()
    
    # Calculate frame parameters
    num_frames = int(params.duration * params.frame_rate)
    frame_files = []
    
    # Base visualization parameters
    base_vis_params = VisualizationParams(
        width=params.display_size[0],
        height=params.display_size[1]
    )
    
    # Generate frames
    for frame in range(num_frames):
        # Get audio features for current frame
        frame_start = int((frame / params.frame_rate) * params.sample_rate)
        frame_end = int(((frame + 1) / params.frame_rate) * params.sample_rate)
        frame_audio = audio_data[frame_start:frame_end]
        
        # Get audio features
        features = get_audio_features(frame_audio, params.sample_rate)
        
        # Map audio features to visual parameters
        vis_params = base_vis_params
        vis_params.radius = int(20 + features['amplitude'] * 20)  # Scale radius with amplitude
        vis_params.symmetry = max(6, int(features['frequency'] / 50))  # Map frequency to symmetry
        vis_params.layers = max(2, int(features['harmonic_content'] * 8))  # Map harmonic content to layers
        
        # Generate mandala pattern
        pattern = ascii_engine.generate_mandala(
            radius=vis_params.radius,
            symmetry=vis_params.symmetry,
            style=vis_params.style,
            layers=vis_params.layers
        )
        
        # Save frame
        frame_file = os.path.join(output_dir, f"frame_{frame:04d}.txt")
        with open(frame_file, 'w') as f:
            f.write(str(pattern))
        frame_files.append(frame_file)
        
        print(f"Generated frame {frame+1}/{num_frames}")
    
    return audio_file, frame_files

def main():
    """Main function to generate synchronized patterns."""
    # Set up parameters
    params = AudioVisualParams(
        sample_rate=44100,
        duration=10.0,
        base_freq=432.0,
        display_size=(120, 60),
        frame_rate=12,
        use_fibonacci=True,
        use_phi=True
    )
    
    print("Generating synchronized patterns...")
    audio_file, frame_files = generate_synchronized_patterns(params)
    
    print(f"\nGeneration complete!")
    print(f"Audio file: {audio_file}")
    print(f"Generated {len(frame_files)} frames")
    print("\nTo view the animation, you can:")
    print("1. Play the audio file")
    print("2. Display frames in sequence:")
    print(f"   for f in {frame_files[0].replace('0000', '*')}; do cat $f; sleep {1/12}; clear; done")

if __name__ == "__main__":
    main() 