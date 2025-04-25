"""Generate test audio files for visualization testing."""

import numpy as np
from scipy.io import wavfile
import os
from pathlib import Path

def generate_test_audio():
    """Generate a test audio file with various patterns."""
    # Set parameters
    sample_rate = 44100
    duration = 10  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate base frequencies
    f1, f2, f3 = 440, 880, 220  # A4, A5, A3
    
    # Create interesting patterns
    signal = np.zeros_like(t)
    
    # Add sine waves with varying amplitudes
    signal += 0.3 * np.sin(2 * np.pi * f1 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    signal += 0.2 * np.sin(2 * np.pi * f2 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.3 * t))
    signal += 0.1 * np.sin(2 * np.pi * f3 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    
    # Add some rhythmic pulses
    pulse = np.sin(2 * np.pi * 2 * t)  # 2 Hz pulse
    signal += 0.2 * pulse * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t))
    
    # Add frequency sweeps
    sweep = np.sin(2 * np.pi * (f1 + 100 * np.sin(2 * np.pi * 0.1 * t)) * t)
    signal += 0.2 * sweep
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # Convert to 16-bit PCM
    signal_16bit = (signal * 32767).astype(np.int16)
    
    # Create data directory if it doesn't exist
    data_dir = Path('data/audio')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the file
    output_path = data_dir / 'test_pattern.wav'
    wavfile.write(str(output_path), sample_rate, signal_16bit)
    print(f"Generated test audio file: {output_path}")

if __name__ == '__main__':
    generate_test_audio() 