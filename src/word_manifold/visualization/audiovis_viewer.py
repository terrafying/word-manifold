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
from typing import Optional, Dict, Any, List
import logging
from dataclasses import dataclass
import json
import asyncio
import websockets
from scipy.fft import fft
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
    color_mode: str = "spectrum"  # spectrum, intensity, frequency
    pattern_style: str = "wave"   # wave, mandala, field
    
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
        
        # Visualization state
        self.frame_data = {}
        self.running = False
        self.clients = set()
        
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
                # Handle client messages (e.g., configuration updates)
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
            
    def _audio_callback(self, indata, frames, time, status):
        """Handle incoming audio data."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())
        
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
                
            except queue.Empty:
                continue
                
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
                'spectrum': self.fft_data.tolist()
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
        
        # Generate base pattern
        if self.vis_config.pattern_style == 'wave':
            pattern = self._generate_wave_pattern(intensity_norm, freq_norm)
        elif self.vis_config.pattern_style == 'mandala':
            pattern = self._generate_mandala_pattern(intensity_norm, freq_norm)
        else:  # field
            pattern = self._generate_field_pattern(intensity_norm, freq_norm)
            
        # Apply color based on mode
        if self.vis_config.color_mode == 'spectrum':
            color = self._get_spectrum_color(freq_norm)
        elif self.vis_config.color_mode == 'intensity':
            color = self._get_intensity_color(intensity_norm)
        else:  # frequency
            color = self._get_frequency_color(freq_norm)
            
        # Render final frame
        return self.renderer.render_pattern(pattern, color=color)
        
    def _generate_wave_pattern(self, intensity: float, frequency: float) -> str:
        """Generate wave-based pattern."""
        width = self.vis_config.width
        height = self.vis_config.height
        t = np.linspace(0, 2*np.pi, width)
        
        # Create wave with frequency and intensity modulation
        wave = np.sin(frequency * 10 * t) * (height/2 * intensity)
        wave += height/2
        
        # Convert to ASCII
        pattern = np.zeros((height, width), dtype=str)
        pattern.fill(' ')
        
        for x, y in enumerate(wave.astype(int)):
            if 0 <= y < height:
                pattern[y, x] = '█'
                
        return '\n'.join(''.join(row) for row in pattern)
        
    def _generate_mandala_pattern(self, intensity: float, frequency: float) -> str:
        """Generate mandala-based pattern."""
        width = self.vis_config.width
        height = self.vis_config.height
        center_x = width // 2
        center_y = height // 2
        radius = min(center_x, center_y) * intensity
        
        pattern = np.zeros((height, width), dtype=str)
        pattern.fill(' ')
        
        # Generate mandala
        points = int(16 * frequency)
        for i in range(points):
            angle = 2 * np.pi * i / points
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            if 0 <= x < width and 0 <= y < height:
                pattern[y, x] = '●'
                
        return '\n'.join(''.join(row) for row in pattern)
        
    def _generate_field_pattern(self, intensity: float, frequency: float) -> str:
        """Generate field-based pattern."""
        width = self.vis_config.width
        height = self.vis_config.height
        
        # Create noise field
        x = np.linspace(0, 4*np.pi, width)
        y = np.linspace(0, 4*np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        field = np.sin(X * frequency + Y * frequency + time.time())
        field = (field + 1) / 2  # Normalize to [0, 1]
        
        # Apply intensity threshold
        threshold = 1 - intensity
        pattern = np.zeros((height, width), dtype=str)
        pattern[field > threshold] = '█'
        pattern[field <= threshold] = ' '
        
        return '\n'.join(''.join(row) for row in pattern)
        
    def _get_spectrum_color(self, freq_norm: float) -> str:
        """Get color based on frequency spectrum."""
        hue = freq_norm
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
        
    def _get_intensity_color(self, intensity: float) -> str:
        """Get color based on audio intensity."""
        value = int(255 * intensity)
        return f"#{value:02x}{value:02x}{value:02x}"
        
    def _get_frequency_color(self, freq_norm: float) -> str:
        """Get color based on dominant frequency."""
        if freq_norm < 0.33:
            return "#0000ff"  # Blue for low frequencies
        elif freq_norm < 0.66:
            return "#00ff00"  # Green for mid frequencies
        else:
            return "#ff0000"  # Red for high frequencies 