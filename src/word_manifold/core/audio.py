"""
Functions for generating audio based on sacred geometry principles and ratios.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
from enum import Enum
import math

class WaveShape(Enum):
    """Different wave shapes for audio generation."""
    SINE = "sine"
    TRIANGLE = "triangle"
    SQUARE = "square"
    SAWTOOTH = "sawtooth"
    PULSE = "pulse"

class ModulationType(Enum):
    """Types of modulation to apply."""
    AMPLITUDE = "amplitude"
    FREQUENCY = "frequency"
    PHASE = "phase"
    RING = "ring"

@dataclass
class AudioParams:
    sample_rate: int = 44100
    duration: float = 10.0  # seconds
    base_freq: float = 432.0  # Hz
    harmonics: List[float] = None
    ratios: List[float] = None
    amplitude: float = 0.5
    fade_duration: float = 0.1  # seconds for fade in/out
    wave_shape: WaveShape = WaveShape.SINE
    modulation_type: ModulationType = ModulationType.AMPLITUDE
    modulation_depth: float = 0.2
    modulation_freq: float = 0.5  # Hz
    use_fibonacci: bool = True
    use_phi: bool = True
    pulse_width: float = 0.5
    resonance: float = 1.0
    phase_shift: float = 0.0
    sacred_intervals: List[float] = None

    def __post_init__(self):
        if self.harmonics is None:
            # Extended harmonics based on sacred ratios
            self.harmonics = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        if self.ratios is None:
            # Extended sacred geometry ratios
            self.ratios = [
                1.618034,  # phi (golden ratio)
                1.414214,  # sqrt(2) (sacred cut)
                1.732051,  # sqrt(3) (vesica piscis)
                2.236068,  # sqrt(5)
                2.618034,  # phi^2
                3.141593,  # pi
                4.236068   # phi^3
            ]
        if self.sacred_intervals is None:
            # Solfeggio frequencies ratios
            self.sacred_intervals = [
                1.0,        # UT - 432 Hz
                1.125,      # RE - 480 Hz
                1.25,       # MI - 528 Hz
                1.333333,   # FA - 576 Hz
                1.5,        # SOL - 648 Hz
                1.666667    # LA - 720 Hz
            ]

def generate_wave(t: np.ndarray, freq: float, shape: WaveShape, amplitude: float = 1.0, 
                 pulse_width: float = 0.5, phase: float = 0.0) -> np.ndarray:
    """Generate different wave shapes."""
    phase_term = 2 * np.pi * freq * t + phase
    
    if shape == WaveShape.SINE:
        return amplitude * np.sin(phase_term)
    elif shape == WaveShape.TRIANGLE:
        return amplitude * (2/np.pi) * np.arcsin(np.sin(phase_term))
    elif shape == WaveShape.SQUARE:
        return amplitude * np.sign(np.sin(phase_term))
    elif shape == WaveShape.SAWTOOTH:
        return amplitude * (2/np.pi) * np.arctan(np.tan(phase_term/2))
    elif shape == WaveShape.PULSE:
        return amplitude * (np.sin(phase_term) > (2 * pulse_width - 1)).astype(float)
    return np.zeros_like(t)

def generate_sacred_audio(params: AudioParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate audio based on sacred geometry principles.
    
    Args:
        params: AudioParams object containing generation parameters
        
    Returns:
        Tuple of (time_array, audio_data)
    """
    # Create time array
    t = np.linspace(0, params.duration, int(params.sample_rate * params.duration))
    
    # Initialize audio data
    audio = np.zeros_like(t)
    
    # Generate Fibonacci sequence for additional harmonics if requested
    if params.use_fibonacci:
        fib = [1, 1]
        while fib[-1] < 13:
            fib.append(fib[-1] + fib[-2])
        fib_ratios = [f/fib[0] for f in fib]
        params.harmonics.extend([r for r in fib_ratios if r not in params.harmonics])
    
    # Add golden ratio harmonics if requested
    if params.use_phi:
        phi = (1 + np.sqrt(5)) / 2
        phi_powers = [phi**n for n in range(1, 5)]
        params.harmonics.extend([r for r in phi_powers if r not in params.harmonics])
    
    # Generate base harmonics with selected wave shape
    for harmonic in params.harmonics:
        freq = params.base_freq * harmonic
        wave = generate_wave(
            t, freq, params.wave_shape,
            amplitude=params.amplitude / len(params.harmonics),
            pulse_width=params.pulse_width,
            phase=params.phase_shift
        )
        audio += wave * params.resonance
    
    # Apply modulation
    mod_wave = np.sin(2 * np.pi * params.modulation_freq * t)
    if params.modulation_type == ModulationType.AMPLITUDE:
        audio *= (1 + params.modulation_depth * mod_wave)
    elif params.modulation_type == ModulationType.FREQUENCY:
        time_mod = t + params.modulation_depth * np.cumsum(mod_wave) / params.sample_rate
        audio = np.interp(t, time_mod, audio)
    elif params.modulation_type == ModulationType.PHASE:
        phase_mod = params.modulation_depth * mod_wave
        audio = np.interp(t, t + phase_mod, audio)
    elif params.modulation_type == ModulationType.RING:
        audio *= (mod_wave + 1) / 2
    
    # Apply sacred ratios as additional modulation
    for ratio in params.ratios:
        mod_freq = params.base_freq / ratio
        mod = np.sin(2 * np.pi * mod_freq * t)
        audio *= (1 + 0.1 * mod)
    
    # Apply sacred intervals
    for interval in params.sacred_intervals:
        freq = params.base_freq * interval
        interval_wave = generate_wave(
            t, freq, params.wave_shape,
            amplitude=0.2 * params.amplitude / len(params.sacred_intervals),
            pulse_width=params.pulse_width,
            phase=params.phase_shift
        )
        audio += interval_wave
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Apply fade in/out
    fade_samples = int(params.fade_duration * params.sample_rate)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    
    return t, audio

def get_audio_features(audio_data: np.ndarray, sample_rate: int, window_size: int = 2048) -> dict:
    """
    Extract audio features from a segment of audio data.
    
    Args:
        audio_data: Audio signal array
        sample_rate: Sample rate in Hz
        window_size: Size of the FFT window
        
    Returns:
        Dictionary containing audio features:
        - amplitude: RMS amplitude
        - frequency: Dominant frequency
        - spectral_centroid: Spectral centroid
        - harmonic_content: Measure of harmonic content
    """
    # Calculate amplitude (RMS)
    amplitude = np.sqrt(np.mean(audio_data**2))
    
    # Calculate spectrum
    spectrum = np.abs(np.fft.rfft(audio_data * np.hanning(len(audio_data))))
    freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
    
    # Find dominant frequency
    dominant_freq = freqs[np.argmax(spectrum)]
    
    # Calculate spectral centroid
    spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    
    # Calculate harmonic content (ratio of harmonic to non-harmonic energy)
    harmonic_mask = np.zeros_like(freqs, dtype=bool)
    for i in range(1, 8):  # Check first 7 harmonics
        harmonic_freq = dominant_freq * i
        idx = np.argmin(np.abs(freqs - harmonic_freq))
        harmonic_mask[max(0, idx-2):min(len(freqs), idx+3)] = True
    
    harmonic_energy = np.sum(spectrum[harmonic_mask])
    total_energy = np.sum(spectrum)
    harmonic_content = harmonic_energy / total_energy if total_energy > 0 else 0
    
    return {
        'amplitude': amplitude,
        'frequency': dominant_freq,
        'spectral_centroid': spectral_centroid,
        'harmonic_content': harmonic_content
    }

def save_audio(audio_data: np.ndarray, sample_rate: int, filename: str):
    """
    Save audio data to a WAV file.
    
    Args:
        audio_data: Audio signal array
        sample_rate: Sample rate in Hz
        filename: Output filename
    """
    import soundfile as sf
    sf.write(filename, audio_data, sample_rate) 