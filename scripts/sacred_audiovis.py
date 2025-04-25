"""Generate synchronized sacred geometry audio and ASCII visualizations."""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import math

from sacred_audio import (
    AudioParams, SacredRatios, SacredFrequencies, WaveShape,
    generate_vesica_piscis, generate_metatrons_cube,
    normalize_audio, convert_to_16bit, save_audio_file
)

from word_manifold.visualization.engines.ascii import ASCIIEngine
from word_manifold.visualization.renderers.ascii import ASCIIRenderer

@dataclass(frozen=True)
class VisualizationParams:
    """Parameters for ASCII visualization."""
    width: int = 80
    height: int = 40
    density: float = 0.6
    style: str = 'mystical'
    symmetry: int = 8
    layers: int = 5

@dataclass(frozen=True)
class AudioVisualParams:
    """Combined parameters for audio-visual generation."""
    audio: AudioParams
    visual: VisualizationParams
    duration: float = 5.0  # seconds
    frames: int = 60      # number of frames for animation

def map_frequency_to_radius(freq: float, base_freq: float = 432.0, min_radius: int = 10, max_radius: int = 30) -> int:
    """Map frequency to mandala radius using logarithmic scaling."""
    ratio = math.log2(freq / base_freq)
    # Map ratio to radius range
    radius = min_radius + (max_radius - min_radius) * (ratio + 1) / 2
    return int(radius)

def map_frequency_to_symmetry(freq: float, base_freq: float = 432.0, min_sym: int = 4, max_sym: int = 12) -> int:
    """Map frequency to mandala symmetry using harmonic relationships."""
    ratio = freq / base_freq
    # Use musical ratios to determine symmetry
    harmonics = [1/2, 2/3, 3/4, 1, 4/3, 3/2, 2]
    closest_harmonic = min(harmonics, key=lambda x: abs(ratio - x))
    # Map harmonic to symmetry range
    sym_range = max_sym - min_sym
    sym = min_sym + int(sym_range * harmonics.index(closest_harmonic) / (len(harmonics) - 1))
    return sym

def map_amplitude_to_density(amplitude: float, min_density: float = 0.3, max_density: float = 0.9) -> float:
    """Map audio amplitude to visual density."""
    return min_density + (max_density - min_density) * amplitude

def map_spectrum_to_style(spectrum: np.ndarray) -> str:
    """Map spectral characteristics to visual style."""
    # Calculate spectral centroid
    freqs = np.linspace(0, 22050, len(spectrum))
    centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    
    # Map centroid to styles
    styles = ['mystical', 'cosmic', 'ethereal', 'ancient', 'digital']
    style_idx = int((centroid / 22050) * len(styles))
    return styles[min(style_idx, len(styles) - 1)]

def generate_frame_parameters(
    audio_segment: np.ndarray,
    sample_rate: int,
    base_params: AudioVisualParams
) -> VisualizationParams:
    """Generate visualization parameters based on audio segment analysis."""
    # Calculate frequency spectrum
    spectrum = np.abs(np.fft.rfft(audio_segment))
    
    # Find dominant frequency
    freq_bins = np.fft.rfftfreq(len(audio_segment), 1/sample_rate)
    dom_freq_idx = np.argmax(spectrum)
    dom_freq = freq_bins[dom_freq_idx]
    
    # Calculate amplitude
    amplitude = np.max(np.abs(audio_segment))
    
    # Map audio features to visual parameters
    radius = map_frequency_to_radius(dom_freq)
    symmetry = map_frequency_to_symmetry(dom_freq)
    density = map_amplitude_to_density(amplitude)
    style = map_spectrum_to_style(spectrum)
    
    return VisualizationParams(
        width=base_params.visual.width,
        height=base_params.visual.height,
        density=density,
        style=style,
        symmetry=symmetry,
        layers=base_params.visual.layers
    )

def generate_synchronized_patterns(
    params: AudioVisualParams,
    output_dir: Path,
    pattern_type: str = 'mandala'
) -> Tuple[Path, List[str]]:
    """Generate synchronized audio and visual patterns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize engines
    ascii_engine = ASCIIEngine()
    ascii_renderer = ASCIIRenderer()
    
    # Generate base audio
    if pattern_type == 'vesica':
        audio = generate_vesica_piscis(params.audio, SacredRatios(), SacredFrequencies())
    else:  # mandala/metatron
        audio = generate_metatrons_cube(params.audio, SacredFrequencies())
    
    # Normalize and save audio
    audio_normalized = normalize_audio(audio)
    audio_pcm = convert_to_16bit(audio_normalized)
    audio_path = output_dir / f"sacred_{pattern_type}_{timestamp}.wav"
    save_audio_file(audio_pcm, params.audio.sample_rate, audio_path)
    
    # Generate synchronized frames
    frames = []
    samples_per_frame = len(audio) // params.frames
    
    for i in range(params.frames):
        # Analyze audio segment
        start = i * samples_per_frame
        end = start + samples_per_frame
        segment = audio[start:end]
        
        # Generate visualization parameters
        vis_params = generate_frame_parameters(segment, params.audio.sample_rate, params)
        
        # Generate ASCII frame
        if pattern_type == 'mandala':
            pattern = ascii_engine.generate_mandala(
                radius=map_frequency_to_radius(SacredFrequencies.OM),
                complexity=vis_params.density,
                style=vis_params.style,
                layers=vis_params.layers,
                symmetry=vis_params.symmetry
            )
        else:  # metatron/vesica
            pattern = ascii_engine.generate_field(
                width=vis_params.width,
                height=vis_params.height,
                density=vis_params.density,
                style=vis_params.style,
                pattern_type='crystalline'
            )
        
        # Save frame
        frame_path = output_dir / f"frame_{i:03d}.txt"
        ascii_renderer.save_pattern(pattern, frame_path)
        frames.append(str(frame_path))
        
        # Save frame parameters
        params_path = output_dir / f"frame_{i:03d}_params.json"
        with open(params_path, 'w') as f:
            json.dump({
                'audio': {
                    'amplitude': float(np.max(np.abs(segment))),
                    'dominant_frequency': float(np.fft.fftfreq(len(segment))[np.argmax(np.abs(np.fft.fft(segment)))]),
                },
                'visual': vis_params.__dict__
            }, f, indent=2)
    
    return audio_path, frames

if __name__ == '__main__':
    # Example usage
    params = AudioVisualParams(
        audio=AudioParams(
            sample_rate=44100,
            duration=5.0,
            amplitude=0.8
        ),
        visual=VisualizationParams(
            width=80,
            height=40,
            density=0.6,
            style='mystical',
            symmetry=8,
            layers=5
        ),
        duration=5.0,
        frames=60
    )
    
    output_dir = Path('data/audiovis')
    audio_path, frame_paths = generate_synchronized_patterns(params, output_dir)
    
    print(f"Generated synchronized patterns:")
    print(f"Audio: {audio_path}")
    print(f"Generated {len(frame_paths)} visual frames")
    
    # Example of how to play/display
    print("\nTo view animation, you can:")
    print("1. Play the audio file")
    print("2. Display frames in sequence (e.g., using 'cat' with sleep):")
    print(f"   for f in {output_dir}/frame_*.txt; do cat $f; sleep 0.0167; clear; done") 