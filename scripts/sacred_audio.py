"""Generate audio patterns based on sacred geometry ratios and relationships."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import math

# Type aliases for clarity
Seconds = float
Hertz = float
Sample = np.ndarray  # 1D array of float64
AudioSignal = Sample  # Alias for semantic clarity
Phase = float
Amplitude = float

class WaveShape(Enum):
    """Available waveform shapes."""
    SINE = auto()
    TRIANGLE = auto()
    SQUARE = auto()
    SAWTOOTH = auto()

@dataclass(frozen=True)
class SacredRatios:
    """Immutable collection of sacred geometric ratios."""
    PHI: float = (1 + 5 ** 0.5) / 2  # Golden ratio
    SQRT2: float = 2 ** 0.5         # Sacred cut
    SQRT3: float = 3 ** 0.5         # Vesica Piscis ratio
    PI: float = math.pi             # Circle ratio
    E: float = math.e               # Natural growth

@dataclass(frozen=True)
class SacredFrequencies:
    """Immutable collection of sacred frequencies in Hz."""
    OM: Hertz = 432.0
    SOLFEGGIO_UT: Hertz = 396.0   # Liberation from fear
    SOLFEGGIO_RE: Hertz = 417.0   # Change/transformation
    SOLFEGGIO_MI: Hertz = 528.0   # DNA repair
    SOLFEGGIO_FA: Hertz = 639.0   # Connection/relationships
    SOLFEGGIO_SOL: Hertz = 741.0  # Awakening intuition
    SOLFEGGIO_LA: Hertz = 852.0   # Spiritual order

@dataclass(frozen=True)
class AudioParams:
    """Parameters for audio generation."""
    sample_rate: int = 44100
    duration: Seconds = 1.0
    amplitude: Amplitude = 1.0

def generate_time_array(params: AudioParams) -> np.ndarray:
    """Generate time array for the given duration and sample rate."""
    return np.linspace(0, params.duration, int(params.sample_rate * params.duration))

def generate_waveform(
    frequency: Hertz,
    shape: WaveShape,
    t: np.ndarray,
    amplitude: Amplitude = 1.0,
    phase: Phase = 0.0
) -> Sample:
    """Generate a waveform of specified shape and frequency."""
    angular_freq = 2 * np.pi * frequency
    
    if shape == WaveShape.SINE:
        return amplitude * np.sin(angular_freq * t + phase)
    elif shape == WaveShape.TRIANGLE:
        return amplitude * (2 * np.abs(2 * ((frequency * t + phase/(2*np.pi)) % 1) - 1) - 1)
    elif shape == WaveShape.SQUARE:
        return amplitude * np.sign(np.sin(angular_freq * t + phase))
    elif shape == WaveShape.SAWTOOTH:
        return amplitude * (2 * ((frequency * t + phase/(2*np.pi)) % 1) - 1)
    
    raise ValueError(f"Unsupported waveform shape: {shape}")

def apply_phase_modulation(
    carrier_freq: Hertz,
    mod_freq: Hertz,
    mod_index: float,
    t: np.ndarray
) -> Sample:
    """Apply phase modulation to a carrier frequency."""
    phase = mod_index * np.sin(2 * np.pi * mod_freq * t)
    return np.sin(2 * np.pi * carrier_freq * t + phase)

def generate_vesica_piscis(
    params: AudioParams,
    ratios: SacredRatios,
    freqs: SacredFrequencies
) -> Sample:
    """Generate sound based on Vesica Piscis geometry."""
    t = generate_time_array(params)
    
    # Two overlapping circles - represented by two frequencies
    f1 = freqs.OM
    f2 = f1 * ratios.SQRT3
    
    # Generate primary tones
    circle1 = generate_waveform(f1, WaveShape.SINE, t, amplitude=0.5)
    circle2 = generate_waveform(f2, WaveShape.SINE, t, amplitude=0.5)
    
    # Generate interference pattern
    modulation = generate_waveform(ratios.SQRT3, WaveShape.SINE, t)
    
    return (circle1 + circle2) * modulation

def generate_flower_of_life(
    params: AudioParams,
    freqs: SacredFrequencies,
    num_petals: int = 6
) -> Sample:
    """Generate sound based on Flower of Life geometry."""
    t = generate_time_array(params)
    signal = np.zeros_like(t)
    
    # Create overlapping circles
    for i in range(num_petals):
        angle = 2 * np.pi * i / num_petals
        freq = freqs.OM * (1 + 0.1 * np.sin(angle))
        signal += generate_waveform(freq, WaveShape.SINE, t, phase=angle)
    
    # Add central circle
    signal += generate_waveform(freqs.OM, WaveShape.SINE, t)
    
    return signal / (num_petals + 1)  # Normalize

def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number using dynamic programming."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    fib = [0] * (n + 1)
    fib[1] = 1
    
    for i in range(2, n + 1):
        fib[i] = fib[i-1] + fib[i-2]
    
    return fib[n]

def fibonacci_ratio(n: int) -> float:
    """Calculate ratio of consecutive Fibonacci numbers."""
    if n <= 1:
        return 1.0
    return fibonacci(n) / fibonacci(n-1)

def generate_metatrons_cube(
    params: AudioParams,
    freqs: SacredFrequencies,
    platonic_faces: List[int] = [4, 6, 8, 12, 20]  # Tetrahedron to Icosahedron
) -> Sample:
    """Generate sound based on Metatron's Cube geometry."""
    t = generate_time_array(params)
    signal = np.zeros_like(t)
    
    # 13 circles of creation
    for i in range(13):
        # Base frequency modulated by Fibonacci ratios
        freq = freqs.OM * fibonacci_ratio(i + 1)
        
        # Add platonic solid frequencies
        if i < len(platonic_faces):
            # Add harmonics based on platonic solid faces
            harmonic = freq * (platonic_faces[i] / platonic_faces[0])  # Normalize to first solid
            signal += 0.3 * generate_waveform(harmonic, WaveShape.SINE, t)
        
        signal += generate_waveform(freq, WaveShape.SINE, t)
    
    return signal / 13  # Normalize

def generate_sri_yantra(
    params: AudioParams,
    freqs: SacredFrequencies,
    ratios: SacredRatios,
    num_triangles: int = 9
) -> Sample:
    """Generate sound based on Sri Yantra geometry."""
    t = generate_time_array(params)
    signal = np.zeros_like(t)
    
    for i in range(num_triangles):
        # Base frequency modulated by golden ratio
        freq = freqs.OM * (ratios.PHI ** (i / num_triangles))
        
        # Generate triangle wave with phase shifts
        phase = 2 * np.pi * i / num_triangles
        signal += generate_waveform(freq, WaveShape.TRIANGLE, t, phase=phase)
    
    return signal / num_triangles  # Normalize

def apply_crossfade(
    signals: List[Sample],
    fade_duration: Seconds,
    sample_rate: int
) -> Sample:
    """Apply crossfades between multiple audio signals."""
    if not signals:
        return np.array([])
    
    fade_len = int(sample_rate * fade_duration)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    
    # Calculate total length needed
    segment_len = len(signals[0])
    total_len = segment_len * len(signals)
    result = np.zeros(total_len)
    
    for i, signal in enumerate(signals):
        start = i * segment_len
        end = start + segment_len
        
        # Apply fades
        if i > 0:  # Fade in
            signal[:fade_len] *= fade_in
        if i < len(signals) - 1:  # Fade out
            signal[-fade_len:] *= fade_out
        
        result[start:end] += signal
    
    return result

def normalize_audio(signal: Sample) -> Sample:
    """Normalize audio to [-1, 1] range."""
    return signal / (np.max(np.abs(signal)) + 1e-10)  # Avoid division by zero

def convert_to_16bit(signal: Sample) -> np.ndarray:
    """Convert float64 audio to 16-bit PCM."""
    return (signal * 32767).astype(np.int16)

def save_audio_file(
    signal: Sample,
    sample_rate: int,
    output_path: Union[str, Path]
) -> Path:
    """Save audio to WAV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_path), sample_rate, signal)
    return output_path

def generate_sacred_geometry_suite(
    params: AudioParams = AudioParams(),
    fade_duration: Seconds = 0.5
) -> Path:
    """Generate a suite of sacred geometry sounds."""
    ratios = SacredRatios()
    freqs = SacredFrequencies()
    
    # Generate individual patterns
    patterns = [
        generate_vesica_piscis(params, ratios, freqs),
        generate_flower_of_life(params, freqs),
        generate_metatrons_cube(params, freqs),
        generate_sri_yantra(params, freqs, ratios)
    ]
    
    # Combine patterns with crossfades
    combined = apply_crossfade(patterns, fade_duration, params.sample_rate)
    
    # Process final audio
    normalized = normalize_audio(combined)
    pcm = convert_to_16bit(normalized)
    
    # Save to file
    output_path = save_audio_file(
        pcm,
        params.sample_rate,
        Path('data/audio/sacred_geometry.wav')
    )
    print(f"Generated sacred geometry audio suite: {output_path}")
    
    return output_path

if __name__ == '__main__':
    # Example usage with custom parameters
    params = AudioParams(
        sample_rate=44100,
        duration=2.5,  # 2.5 seconds per pattern
        amplitude=0.8
    )
    generate_sacred_geometry_suite(params) 