"""
Audio-reactive ASCII Viewer

Combines audio generation with real-time ASCII visualization.
Features real-time audio analysis and pattern generation.
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging
from dataclasses import dataclass
import json
import asyncio
import websockets
from scipy.fft import fft
from scipy import signal
import colorsys
from .renderers.ascii_renderer import ASCIIRenderer, RenderConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio configuration settings."""
    sample_rate: int = 44100
    block_size: int = 2048
    channels: int = 1
    device: Optional[int] = None
    
@dataclass
class VisualizerConfig:
    """Visualizer configuration settings."""
    width: int = 80
    height: int = 40
    fps: int = 30
    color_mode: str = "spectrum"  # spectrum, intensity, frequency, rainbow, custom
    pattern_style: str = "wave"   # wave, mandala, field
    high_res: bool = False
    interference: bool = False
    blend: bool = False
    complexity: int = 5
    density: int = 50
    speed: int = 50
    effects: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.effects is None:
            self.effects = {
                'glow': False,
                'pulse': False,
                'mirror': False
            }
    
class AudioVisualizer:
    """Real-time audio-reactive ASCII visualizer."""
    
    def __init__(
        self,
        audio_config: Optional[AudioConfig] = None,
        vis_config: Optional[VisualizerConfig] = None
    ):
        """Initialize the visualizer."""
        self.audio_config = audio_config or AudioConfig()
        self.vis_config = vis_config or VisualizerConfig()
        
        # Initialize components
        self.renderer = ASCIIRenderer(RenderConfig(
            width=self.vis_config.width,
            height=self.vis_config.height,
            color_support=True
        ))
        
        # Audio processing state
        self.audio_queue = queue.Queue()
        self.fft_data = np.zeros(self.audio_config.block_size // 2)
        self.intensity = 0.0
        self.dominant_freq = 0.0
        
        # Enhanced audio analysis
        self.beat_detected = False
        self.onset_detected = False
        self.spectral_flux = 0.0
        self.prev_spectrum = None
        self.smoothed_spectrum = None
        self.energy_history = []
        
        # Pattern state
        self.phase = 0.0
        self.pattern_buffer = None
        self.last_pattern = None
        
        # Visualization state
        self.frame_data = {}
        self.running = False
        self.clients = set()
        
        # Audio file state
        self.audio_file = None
        self.audio_data = None
        self.audio_pos = 0
        
    async def start_server(self, host: str = 'localhost', port: int = 8765):
        """Start the WebSocket server for real-time visualization."""
        async with websockets.serve(self._handle_client, host, port):
            logger.info(f"Visualization server running at ws://{host}:{port}")
            await asyncio.Future()  # run forever
            
    async def _handle_client(self, websocket):
        """Handle WebSocket client connection."""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if 'config' in data:
                        self._update_config(data['config'])
                except json.JSONDecodeError:
                    logger.warning(f"Invalid message format: {message}")
        finally:
            self.clients.remove(websocket)
            
    def _update_config(self, config: Dict[str, Any]):
        """Update visualization configuration."""
        if 'width' in config:
            self.vis_config.width = config['width']
        if 'height' in config:
            self.vis_config.height = config['height']
        if 'color_mode' in config:
            self.vis_config.color_mode = config['color_mode']
        if 'pattern_style' in config:
            self.vis_config.pattern_style = config['pattern_style']
        if 'high_res' in config:
            self.vis_config.high_res = config['high_res']
        if 'interference' in config:
            self.vis_config.interference = config['interference']
        if 'blend' in config:
            self.vis_config.blend = config['blend']
        if 'complexity' in config:
            self.vis_config.complexity = config['complexity']
        if 'density' in config:
            self.vis_config.density = config['density']
        if 'speed' in config:
            self.vis_config.speed = config['speed']
        if 'effects' in config:
            self.vis_config.effects.update(config['effects'])
            
    def start(self):
        """Start audio processing and visualization."""
        self.running = True
        
        # Start audio input stream
        self.audio_stream = sd.InputStream(
            channels=self.audio_config.channels,
            samplerate=self.audio_config.sample_rate,
            blocksize=self.audio_config.block_size,
            device=self.audio_config.device,
            callback=self._audio_callback
        )
        
        # Start processing threads
        self.audio_thread = threading.Thread(target=self._process_audio)
        self.vis_thread = threading.Thread(target=self._update_visualization)
        
        self.audio_stream.start()
        self.audio_thread.start()
        self.vis_thread.start()
        
    def stop(self):
        """Stop audio processing and visualization."""
        self.running = False
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        if hasattr(self, 'vis_thread'):
            self.vis_thread.join()
            
    def set_audio_file(self, file_path: str):
        """Set audio file for playback."""
        try:
            import soundfile as sf
            self.audio_data, file_sr = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(self.audio_data.shape) > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)
            
            # Resample if needed
            if file_sr != self.audio_config.sample_rate:
                from scipy import signal
                samples = int(len(self.audio_data) * self.audio_config.sample_rate / file_sr)
                self.audio_data = signal.resample(self.audio_data, samples)
            
            self.audio_file = file_path
            self.audio_pos = 0
            logger.info(f"Loaded audio file: {file_path}")
            
        except ImportError:
            logger.error("soundfile not found. Please install with: pip install soundfile")
            raise
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise
            
    def _audio_callback(self, indata, frames, time, status):
        """Handle incoming audio data."""
        if status:
            logger.warning(f"Audio callback status: {status}")
            
        if self.audio_file and self.audio_data is not None:
            # Use audio file data instead of input
            end_pos = self.audio_pos + frames
            if end_pos > len(self.audio_data):
                # Loop back to start
                self.audio_pos = 0
                end_pos = frames
            
            data = self.audio_data[self.audio_pos:end_pos]
            self.audio_pos = end_pos
            
            # Reshape to match input format
            data = data.reshape(-1, 1)
        else:
            # Use live input
            data = indata.copy()
            
        self.audio_queue.put(data)
        
    def _process_audio(self):
        """Process audio data and update analysis."""
        while self.running:
            try:
                data = self.audio_queue.get(timeout=1.0)
                
                # Compute FFT
                fft_data = np.abs(fft(data[:, 0])) / len(data)
                self.fft_data = fft_data[:len(fft_data)//2]
                
                # Calculate intensity
                self.intensity = np.mean(np.abs(data))
                
                # Find dominant frequency
                freq_range = np.fft.fftfreq(len(data), 1/self.audio_config.sample_rate)
                pos_freq_range = freq_range[:len(freq_range)//2]
                max_idx = np.argmax(self.fft_data)
                self.dominant_freq = pos_freq_range[max_idx]
                
                # Enhanced audio analysis
                self._analyze_audio(data[:, 0], self.fft_data)
                
            except queue.Empty:
                continue
                
    def _analyze_audio(self, audio_data: np.ndarray, spectrum: np.ndarray):
        """Perform enhanced audio analysis."""
        # Spectral flux
        if self.prev_spectrum is not None:
            diff = spectrum - self.prev_spectrum
            self.spectral_flux = np.sum(diff[diff > 0])
        self.prev_spectrum = spectrum.copy()
        
        # Smoothed spectrum
        if self.smoothed_spectrum is None:
            self.smoothed_spectrum = spectrum
        else:
            self.smoothed_spectrum = 0.8 * self.smoothed_spectrum + 0.2 * spectrum
        
        # Energy and beat detection
        energy = np.sum(spectrum)
        self.energy_history.append(energy)
        if len(self.energy_history) > 43:  # About 1 second at 44.1kHz
            self.energy_history.pop(0)
        
        # Beat detection
        if len(self.energy_history) > 1:
            local_energy = np.mean(self.energy_history[-4:])
            hist_energy = np.mean(self.energy_history[:-4])
            variance = np.std(self.energy_history[:-4])
            
            self.beat_detected = (
                local_energy > hist_energy + 0.5 * variance
                and local_energy > 0.01  # Minimum threshold
            )
        
        # Onset detection using spectral flux
        if self.spectral_flux > 0.5:  # Adjustable threshold
            self.onset_detected = True
        else:
            self.onset_detected = False
                
    def _update_visualization(self):
        """Update visualization frames."""
        frame_interval = 1.0 / self.vis_config.fps
        
        while self.running:
            start_time = time.time()
            
            # Generate frame based on audio analysis
            frame = self._generate_frame()
            
            # Update frame data
            self.frame_data = {
                'pattern': frame,
                'intensity': float(self.intensity),
                'dominant_freq': float(self.dominant_freq),
                'spectrum': self.fft_data.tolist(),
                'beat_detected': self.beat_detected,
                'onset_detected': self.onset_detected,
                'spectral_flux': float(self.spectral_flux)
            }
            
            # Send frame to connected clients
            if self.clients:
                asyncio.run(self._broadcast_frame())
                
            # Maintain frame rate
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                
    async def _broadcast_frame(self):
        """Send frame data to all connected clients."""
        if not self.clients:
            return
            
        message = json.dumps(self.frame_data)
        await asyncio.gather(
            *[client.send(message) for client in self.clients]
        )
        
    def _generate_frame(self) -> str:
        """Generate ASCII frame based on audio analysis."""
        # Map audio features to visual parameters
        intensity_norm = np.clip(self.intensity * 10, 0, 1)
        freq_norm = np.clip(self.dominant_freq / 2000, 0, 1)
        
        # Update phase based on speed setting
        phase_increment = self.vis_config.speed / 500.0  # 0.002 to 0.2
        self.phase = (self.phase + phase_increment) % (2 * np.pi)
        
        # Generate base pattern
        if self.vis_config.pattern_style == 'wave':
            pattern = self._generate_wave_pattern(intensity_norm, freq_norm)
        elif self.vis_config.pattern_style == 'mandala':
            pattern = self._generate_mandala_pattern(intensity_norm, freq_norm)
        else:  # field
            pattern = self._generate_field_pattern(intensity_norm, freq_norm)
            
        # Apply effects
        if self.vis_config.effects['mirror']:
            pattern = self._apply_mirror(pattern)
        if self.vis_config.effects['pulse'] and self.beat_detected:
            pattern = self._apply_pulse(pattern)
            
        # Apply color based on mode
        if self.vis_config.color_mode == 'spectrum':
            color = self._get_spectrum_color(freq_norm)
        elif self.vis_config.color_mode == 'intensity':
            color = self._get_intensity_color(intensity_norm)
        elif self.vis_config.color_mode == 'rainbow':
            color = self._get_rainbow_color(self.phase)
        else:  # frequency
            color = self._get_frequency_color(freq_norm)
            
        # Store last pattern for transitions
        self.last_pattern = pattern
            
        # Render final frame
        return self.renderer.render_pattern(pattern, color=color)
        
    def _generate_wave_pattern(self, intensity: float, frequency: float) -> str:
        """Generate wave-based pattern."""
        width = self.vis_config.width
        height = self.vis_config.height
        t = np.linspace(0, 2*np.pi, width)
        
        # Create base wave
        wave = np.sin(frequency * 10 * t + self.phase)
        if self.vis_config.interference:
            wave += 0.5 * np.sin(frequency * 20 * t - self.phase)
            wave += 0.25 * np.sin(frequency * 30 * t + self.phase * 2)
            wave = wave / 1.75  # Normalize
        
        wave = wave * (height/2 * intensity)
        wave += height/2
        
        # Create pattern
        pattern = np.zeros((height, width), dtype=str)
        pattern.fill(' ')
        
        # High-res mode uses more ASCII characters
        chars = ' ░▒▓█' if self.vis_config.high_res else ' █'
        
        for x, y in enumerate(wave.astype(int)):
            if 0 <= y < height:
                # Add "thickness" in high-res mode
                if self.vis_config.high_res:
                    for dy in range(-1, 2):
                        if 0 <= y + dy < height:
                            intensity = 1 - abs(dy) * 0.3
                            char_idx = int(intensity * (len(chars) - 1))
                            pattern[y + dy, x] = chars[char_idx]
                else:
                    pattern[y, x] = chars[-1]
                    
        return '\n'.join(''.join(row) for row in pattern)
        
    def _generate_mandala_pattern(self, intensity: float, frequency: float) -> str:
        """Generate mandala-based pattern."""
        width = self.vis_config.width
        height = self.vis_config.height
        center_x = width // 2
        center_y = height // 2
        
        # Multiple layers with different radii
        pattern = np.zeros((height, width), dtype=str)
        pattern.fill(' ')
        
        # High-res mode uses more ASCII characters
        chars = '.·•●○◎◉★✦✧✶' if self.vis_config.high_res else '●○'
        
        layers = self.vis_config.complexity
        for layer in range(layers):
            radius = (min(center_x, center_y) * intensity * 
                     (1 - layer/layers) * (1 + 0.2 * np.sin(self.phase)))
            
            points = int(16 * frequency * (layer + 1))
            for i in range(points):
                angle = 2 * np.pi * i / points + self.phase * (layer % 2 * 2 - 1)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                
                if 0 <= x < width and 0 <= y < height:
                    char_idx = int((layer/layers) * (len(chars) - 1))
                    pattern[y, x] = chars[char_idx]
                    
        return '\n'.join(''.join(row) for row in pattern)
        
    def _generate_field_pattern(self, intensity: float, frequency: float) -> str:
        """Generate field-based pattern."""
        width = self.vis_config.width
        height = self.vis_config.height
        
        # Create noise field
        x = np.linspace(0, 4*np.pi, width)
        y = np.linspace(0, 4*np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        # Generate complex field pattern
        field = np.zeros((height, width))
        
        # Layer multiple waves
        for i in range(self.vis_config.complexity):
            freq_mod = frequency * (i + 1) * 2
            phase_mod = self.phase * (i % 2 * 2 - 1)
            field += np.sin(X * freq_mod + Y * freq_mod + phase_mod) / (i + 1)
            
        field = (field + self.vis_config.complexity) / (2 * self.vis_config.complexity)
        
        # Apply density threshold
        threshold = 1 - (self.vis_config.density / 100)
        pattern = np.zeros((height, width), dtype=str)
        
        # High-res mode uses more ASCII characters
        if self.vis_config.high_res:
            chars = ' ░▒▓█'
            levels = np.linspace(0, 1, len(chars))
            for i, level in enumerate(levels[:-1]):
                mask = (field > level) & (field <= levels[i + 1])
                pattern[mask] = chars[i + 1]
        else:
            pattern[field > threshold] = '█'
            pattern[field <= threshold] = ' '
            
        return '\n'.join(''.join(row) for row in pattern)
        
    def _apply_mirror(self, pattern: str) -> str:
        """Apply mirror effect to pattern."""
        lines = pattern.split('\n')
        mirrored_lines = []
        for line in lines:
            mirrored_lines.append(line + line[::-1])
        return '\n'.join(mirrored_lines)
        
    def _apply_pulse(self, pattern: str) -> str:
        """Apply pulse effect to pattern."""
        lines = pattern.split('\n')
        pulsed_lines = []
        pulse_char = '✶' if self.vis_config.high_res else '*'
        
        for line in lines:
            if np.random.random() < 0.2:  # Random pulse effect
                line = line.replace(' ', pulse_char)
            pulsed_lines.append(line)
            
        return '\n'.join(pulsed_lines)
        
    def _get_spectrum_color(self, freq_norm: float) -> str:
        """Get color based on frequency spectrum."""
        hue = freq_norm
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
        
    def _get_intensity_color(self, intensity: float) -> str:
        """Get color based on audio intensity."""
        value = int(255 * intensity)
        return f"#{value:02x}{value:02x}{value:02x}"
        
    def _get_rainbow_color(self, phase: float) -> str:
        """Get color based on phase for rainbow effect."""
        hue = (phase / (2 * np.pi)) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
        
    def _get_frequency_color(self, freq_norm: float) -> str:
        """Get color based on dominant frequency."""
        if freq_norm < 0.33:
            return "#0000ff"  # Blue for low frequencies
        elif freq_norm < 0.66:
            return "#00ff00"  # Green for mid frequencies
        else:
            return "#ff0000"  # Red for high frequencies 