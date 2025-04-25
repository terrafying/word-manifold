"""
Sacred Audio Explorer

Advanced sacred geometry audio pattern generator with:
- Recursive rhythm generation
- Feedback and delay effects
- Harmonic synthesis
- Pattern evolution
"""

import numpy as np
from pathlib import Path
import soundfile as sf
from typing import List, Dict, Tuple, Optional, Union
import time
from dataclasses import dataclass
import json
from scipy import signal as sig

from word_manifold.core.audio import (
    AudioParams, generate_sacred_audio, get_audio_features,
    WaveShape, ModulationType
)

@dataclass
class RhythmParams:
    """Parameters for rhythm generation."""
    tempo: float = 120.0  # BPM
    subdivision: int = 4   # Subdivisions per beat
    pattern_length: int = 16  # Beats in pattern
    accent_probability: float = 0.3
    swing: float = 0.0  # 0.0 to 0.33
    phase_shift: float = 0.0
    euclidean_steps: int = 4
    euclidean_pulses: int = 3

@dataclass
class DelayParams:
    """Parameters for delay effects."""
    delay_time: float = 0.25  # seconds
    feedback: float = 0.4     # 0-1
    mix: float = 0.3         # dry/wet mix
    filter_freq: float = 2000.0  # Hz
    resonance: float = 0.7    # filter Q

@dataclass
class RecursiveParams:
    """Parameters for recursive pattern generation."""
    depth: int = 3
    decay: float = 0.7
    mutation_rate: float = 0.2
    self_similarity: float = 0.8
    evolution_rate: float = 0.3

@dataclass
class HarmonicParams:
    """Parameters for harmonic synthesis."""
    fundamental: float = 432.0
    overtones: List[float] = None
    ratios: List[float] = None
    phase_coherence: float = 0.8
    spectral_tilt: float = -6.0  # dB/octave

    def __post_init__(self):
        if self.overtones is None:
            # Harmonic series with golden ratio influences
            phi = (1 + np.sqrt(5)) / 2
            self.overtones = [
                1.0,            # fundamental
                phi,            # golden ratio
                2.0,           # octave
                phi * 2,       # golden octave
                3.0,           # perfect fifth + octave
                phi * 3,       # golden fifth
                5.0,           # major third + 2 octaves
                phi * 5,       # golden third
                8.0,           # triple octave
                phi * 8        # golden triple octave
            ]
        if self.ratios is None:
            # Sacred geometry ratios
            self.ratios = [
                1.0,        # unison
                1.618034,   # phi
                2.236068,   # √5
                2.618034,   # phi²
                3.141593,   # π
                3.732051,   # √(1+phi)
                4.236068    # phi³
            ]

def generate_euclidean_rhythm(steps: int, pulses: int) -> List[int]:
    """Generate Euclidean rhythm pattern (even distribution of pulses)."""
    if pulses > steps:
        pulses = steps
    if pulses == 0:
        return [0] * steps
    
    pattern = []
    counts = []
    remainder = [1] * pulses + [0] * (steps - pulses)
    
    while len(remainder) > 1:
        counts.append(len(remainder))
        new_remainder = []
        for i in range(0, len(remainder) - 1, 2):
            pattern.append(remainder[i])
            new_remainder.append(remainder[i + 1])
        if len(remainder) % 2 == 1:
            new_remainder.append(remainder[-1])
        remainder = new_remainder
    
    pattern.extend(remainder)
    
    # Rotate pattern to start with a pulse
    first_pulse = pattern.index(1)
    return pattern[first_pulse:] + pattern[:first_pulse]

def generate_fibonacci_rhythm(length: int) -> List[int]:
    """Generate rhythm based on Fibonacci sequence."""
    # Generate Fibonacci numbers up to length
    fib = [1, 1]
    while fib[-1] < length:
        fib.append(fib[-1] + fib[-2])
    
    # Create rhythm pattern
    pattern = [0] * length
    for i in fib:
        if i < length:
            pattern[i] = 1
    return pattern

def generate_golden_rhythm(length: int) -> List[int]:
    """Generate rhythm based on golden ratio."""
    phi = (1 + np.sqrt(5)) / 2
    pattern = [0] * length
    
    # Place accents at positions based on phi
    pos = 0
    while pos < length:
        pattern[int(pos)] = 1
        pos += phi
    
    return pattern

def generate_spiral_rhythm(sample_rate: int, duration: float, 
                         frequency: float = 1.0, decay: float = 0.5) -> np.ndarray:
    """Generate a rhythm based on logarithmic spiral."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    spiral = np.exp(-decay * t) * np.sin(2 * np.pi * frequency * t * np.exp(t/duration))
    return (spiral > 0.3).astype(float)  # Threshold to create rhythm

def apply_swing(pattern: List[int], swing_amount: float) -> List[int]:
    """Apply swing feel to rhythm pattern."""
    if len(pattern) % 2 != 0:
        pattern.append(0)
    
    swung_pattern = []
    for i in range(0, len(pattern), 2):
        swung_pattern.append(pattern[i])
        if i + 1 < len(pattern):
            # Delay every second eighth note
            swung_pattern.append(0)
            swung_pattern.append(pattern[i + 1])
            swung_pattern.append(0)
    
    return swung_pattern[:len(pattern)]

def create_rhythm_variation(base_rhythm: List[int], variation_type: str,
                          params: RhythmParams) -> List[int]:
    """Create variation of base rhythm pattern."""
    pattern = base_rhythm.copy()
    
    if variation_type == "density":
        # Add notes between existing ones
        for i in range(len(pattern)-1):
            if pattern[i] == 1 and pattern[i+1] == 0:
                if np.random.random() < params.accent_probability:
                    pattern[i+1] = 0.7
    
    elif variation_type == "displacement":
        # Shift some notes slightly
        for i in range(len(pattern)):
            if pattern[i] == 1 and np.random.random() < 0.2:
                if i > 0 and pattern[i-1] == 0:
                    pattern[i-1], pattern[i] = pattern[i], pattern[i-1]
    
    elif variation_type == "euclidean":
        # Combine with Euclidean rhythm
        euclidean = generate_euclidean_rhythm(
            params.euclidean_steps,
            params.euclidean_pulses
        )
        # Extend to match length
        while len(euclidean) < len(pattern):
            euclidean.extend(euclidean)
        euclidean = euclidean[:len(pattern)]
        # Combine patterns
        pattern = [max(a, b) for a, b in zip(pattern, euclidean)]
    
    return pattern

def pattern_to_signal(pattern: List[int], sample_rate: int, tempo: float,
                     wave_shape: WaveShape = WaveShape.SINE) -> np.ndarray:
    """Convert rhythm pattern to audio signal."""
    # Calculate samples per beat
    beat_duration = 60.0 / tempo  # seconds per beat
    samples_per_beat = int(beat_duration * sample_rate)
    
    # Create base signal
    signal = np.zeros(len(pattern) * samples_per_beat)
    
    # Generate each beat
    for i, accent in enumerate(pattern):
        if accent > 0:
            # Create beat with attack and decay
            beat_signal = np.zeros(samples_per_beat)
            attack_samples = int(0.005 * sample_rate)  # 5ms attack
            decay_samples = int(0.1 * sample_rate)    # 100ms decay
            
            # Generate beat with selected wave shape
            t = np.linspace(0, 1, attack_samples)
            if wave_shape == WaveShape.SINE:
                beat_signal[:attack_samples] = np.sin(2 * np.pi * 1000 * t)
            elif wave_shape == WaveShape.TRIANGLE:
                beat_signal[:attack_samples] = 2 * np.abs(2 * (t - np.floor(t + 0.5))) - 1
            elif wave_shape == WaveShape.SQUARE:
                beat_signal[:attack_samples] = np.sign(np.sin(2 * np.pi * 1000 * t))
            
            # Apply envelope
            beat_signal[:attack_samples] *= np.linspace(0, 1, attack_samples)
            beat_signal[attack_samples:attack_samples+decay_samples] *= \
                np.exp(-3 * np.linspace(0, 1, decay_samples))
            
            # Add to main signal with accent level
            start = i * samples_per_beat
            end = start + len(beat_signal)
            signal[start:end] += beat_signal * accent
    
    return signal

def apply_delay(signal: np.ndarray, sample_rate: int, params: DelayParams) -> np.ndarray:
    """Apply delay effect with feedback and filtering."""
    delay_samples = int(params.delay_time * sample_rate)
    output = signal.copy()
    feedback_signal = np.zeros_like(signal)
    
    # Create resonant filter
    b, a = sig.butter(2, params.filter_freq / (sample_rate/2), btype='low', analog=False)
    
    # Apply delay with feedback
    for i in range(len(signal)):
        if i >= delay_samples:
            feedback_signal[i] = output[i - delay_samples]
            # Apply filter to feedback
            if i >= 2:
                feedback_signal[i] = (
                    b[0] * feedback_signal[i] +
                    b[1] * feedback_signal[i-1] +
                    b[2] * feedback_signal[i-2] -
                    a[1] * feedback_signal[i-1] -
                    a[2] * feedback_signal[i-2]
                ) * params.resonance
    
    # Mix dry and wet signals
    output = (1 - params.mix) * signal + params.mix * (signal + params.feedback * feedback_signal)
    return output

def generate_recursive_rhythm(
    base_pattern: List[int],
    params: RecursiveParams
) -> List[int]:
    """Generate recursive variations of a rhythm pattern."""
    if params.depth <= 0:
        return base_pattern
    
    pattern_length = len(base_pattern)
    new_pattern = base_pattern.copy()
    
    # Apply recursive subdivision
    for depth in range(params.depth):
        subdivision = []
        scale = params.decay ** depth
        
        for i, val in enumerate(new_pattern):
            if val > 0:
                # Create self-similar subpattern
                if np.random.random() < params.self_similarity:
                    sub = [x * scale for x in base_pattern[:pattern_length//2]]
                else:
                    # Mutate subpattern
                    sub = [
                        scale if np.random.random() < params.mutation_rate else 0
                        for _ in range(pattern_length//2)
                    ]
                subdivision.extend(sub)
            else:
                subdivision.extend([0] * (pattern_length//2))
        
        # Evolve pattern
        if np.random.random() < params.evolution_rate:
            new_pattern = [
                max(a, b) for a, b in zip(new_pattern, subdivision[:pattern_length])
            ]
        else:
            new_pattern = subdivision[:pattern_length]
    
    return new_pattern

def synthesize_harmonics(
    frequency: float,
    duration: float,
    sample_rate: int,
    params: HarmonicParams
) -> np.ndarray:
    """Synthesize rich harmonic content based on sacred ratios."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    signal = np.zeros_like(t)
    
    # Generate harmonic series
    for i, overtone in enumerate(params.overtones):
        # Calculate amplitude using spectral tilt
        amplitude = 10 ** (params.spectral_tilt * np.log2(i + 1) / 20)
        
        # Generate phase-coherent overtones
        phase = (
            2 * np.pi * np.random.random()
            if np.random.random() > params.phase_coherence
            else 0
        )
        
        signal += amplitude * np.sin(2 * np.pi * frequency * overtone * t + phase)
    
    # Apply sacred ratio modulation
    for ratio in params.ratios:
        mod_freq = frequency / ratio
        mod_depth = 0.1 * (1 / ratio)  # Deeper modulation for lower ratios
        signal *= (1 + mod_depth * np.sin(2 * np.pi * mod_freq * t))
    
    return signal

def generate_variation_suite(
    duration: float = 5.0,
    sample_rate: int = 44100,
    base_freq: float = 432.0,
    tempo: float = 120.0
) -> Dict[str, np.ndarray]:
    """Generate a suite of rhythm variations with effects."""
    
    # Initialize parameters
    rhythm_params = RhythmParams(
        tempo=tempo,
        subdivision=4,
        pattern_length=16,
        accent_probability=0.3,
        swing=0.2
    )
    
    recursive_params = RecursiveParams(
        depth=3,
        decay=0.7,
        mutation_rate=0.2,
        self_similarity=0.8,
        evolution_rate=0.3
    )
    
    delay_params = DelayParams(
        delay_time=60/tempo/4,  # 16th note delay
        feedback=0.4,
        mix=0.3,
        filter_freq=2000.0,
        resonance=0.7
    )
    
    harmonic_params = HarmonicParams(
        fundamental=base_freq
    )
    
    # Create base patterns
    fib_rhythm = generate_fibonacci_rhythm(rhythm_params.pattern_length)
    golden_rhythm = generate_golden_rhythm(rhythm_params.pattern_length)
    euclidean_rhythm = generate_euclidean_rhythm(
        rhythm_params.pattern_length,
        int(rhythm_params.pattern_length * 0.618)
    )
    
    # Apply recursive generation
    fib_recursive = generate_recursive_rhythm(fib_rhythm, recursive_params)
    golden_recursive = generate_recursive_rhythm(golden_rhythm, recursive_params)
    euclidean_recursive = generate_recursive_rhythm(euclidean_rhythm, recursive_params)
    
    # Apply swing to patterns
    if rhythm_params.swing > 0:
        fib_recursive = apply_swing(fib_recursive, rhythm_params.swing)
        golden_recursive = apply_swing(golden_recursive, rhythm_params.swing)
        euclidean_recursive = apply_swing(euclidean_recursive, rhythm_params.swing)
    
    # Create variations dictionary
    variations = {}
    
    # Generate base signals with different wave shapes
    variations['fibonacci'] = pattern_to_signal(
        fib_recursive, sample_rate, tempo, WaveShape.SINE
    )
    variations['golden'] = pattern_to_signal(
        golden_recursive, sample_rate, tempo, WaveShape.TRIANGLE
    )
    variations['euclidean'] = pattern_to_signal(
        euclidean_recursive, sample_rate, tempo, WaveShape.SQUARE
    )
    
    # Generate spiral rhythm
    variations['spiral'] = generate_spiral_rhythm(
        sample_rate, duration, frequency=tempo/60.0
    )
    
    # Add harmonic synthesis
    harmonic_signal = synthesize_harmonics(
        base_freq, duration, sample_rate, harmonic_params
    )
    
    # Ensure all signals are the same length
    target_length = int(duration * sample_rate)
    for key in variations:
        if len(variations[key]) > target_length:
            variations[key] = variations[key][:target_length]
        elif len(variations[key]) < target_length:
            padding = np.zeros(target_length - len(variations[key]))
            variations[key] = np.concatenate([variations[key], padding])
    
    # Create complex combinations
    variations['fib_golden'] = variations['fibonacci'] * 0.6 + variations['golden'] * 0.4
    variations['spiral_euclidean'] = variations['spiral'] * 0.7 + variations['euclidean'] * 0.3
    
    # Add delay effects
    variations['fib_golden_delay'] = apply_delay(
        variations['fib_golden'], sample_rate, delay_params
    )
    variations['spiral_euclidean_delay'] = apply_delay(
        variations['spiral_euclidean'], sample_rate, delay_params
    )
    
    # Create harmonic variations
    variations['harmonic_base'] = harmonic_signal
    variations['harmonic_rhythm'] = harmonic_signal * variations['fib_golden']
    variations['harmonic_delay'] = apply_delay(
        variations['harmonic_rhythm'], sample_rate, delay_params
    )
    
    # Add modulated versions
    audio_params = AudioParams(
        sample_rate=sample_rate,
        duration=duration,
        base_freq=base_freq,
        wave_shape=WaveShape.SINE,
        modulation_type=ModulationType.AMPLITUDE,
        modulation_depth=0.3,
        modulation_freq=tempo/60.0
    )
    
    _, modulated_signal = generate_sacred_audio(audio_params)
    variations['modulated'] = modulated_signal * variations['harmonic_rhythm']
    
    # Normalize all variations
    for key in variations:
        variations[key] = variations[key] / np.max(np.abs(variations[key]))
    
    return variations

def save_variations(variations: Dict[str, np.ndarray], 
                   output_dir: str = "data/audio/explorations",
                   sample_rate: int = 44100):
    """Save variation suite to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save individual variations
    for name, signal in variations.items():
        filename = output_path / f"{name}_{timestamp}.wav"
        sf.write(filename, signal, sample_rate)
    
    # Save analysis
    analysis = {}
    for name, signal in variations.items():
        features = get_audio_features(signal, sample_rate)
        analysis[name] = {
            'amplitude': float(features['amplitude']),
            'frequency': float(features['frequency']),
            'harmonic_content': float(features['harmonic_content'])
        }
    
    # Save analysis to JSON
    with open(output_path / f"analysis_{timestamp}.json", 'w') as f:
        json.dump(analysis, f, indent=2)

def main():
    """Generate and save rhythm variations."""
    print("Generating sacred rhythm variations...")
    
    variations = generate_variation_suite(
        duration=5.0,
        sample_rate=44100,
        base_freq=432.0,
        tempo=120.0
    )
    
    print("\nSaving variations...")
    save_variations(variations)
    
    print("\nVariations generated and saved!")
    print("Check the data/audio/explorations directory for the files.")

if __name__ == "__main__":
    main() 