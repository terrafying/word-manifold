"""
Image Renderer for ASCII Patterns with Audio-Reactive and Subliminal Effects

Converts ASCII patterns into high-quality images with various rendering styles.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
from typing import Optional, Tuple, List, Dict, Union
import colorsys
from pathlib import Path
import math
import random
import librosa
import soundfile as sf
import numpy.fft as fft
from dataclasses import dataclass
import time

from ..engines.ascii import ASCIIEngine
from ...types.patterns import Pattern, Mandala, Field

@dataclass
class AudioFeatures:
    """Audio analysis features for reactive visualization."""
    waveform: np.ndarray
    spectrum: np.ndarray
    onset_env: np.ndarray
    tempo: float
    beat_frames: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    mfcc: np.ndarray

class SubliminalEffect:
    """Manages subliminal and superliminal visual effects."""
    
    def __init__(self, duration_ms: float = 16.7):  # One frame at 60fps
        self.duration_ms = duration_ms
        self.symbols = {
            'protection': '⛊',
            'wisdom': '⛤',
            'power': '⛥',
            'harmony': '⛧',
            'transformation': '⛮',
            'unity': '⛯',
            'transcendence': '⛶',
            'infinity': '∞',
            'consciousness': '☯',
            'enlightenment': '☸',
        }
        
        # Load or generate sigils
        self.sigils = self._generate_sigils()
        
    def _generate_sigils(self) -> Dict[str, np.ndarray]:
        """Generate or load pre-made sigils."""
        sigils = {}
        
        # Basic geometric sigils
        for intent in ['focus', 'clarity', 'energy', 'peace', 'growth']:
            sigil = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
            draw = ImageDraw.Draw(sigil)
            
            if intent == 'focus':
                # Triangular focus sigil
                draw.polygon([(50, 10), (90, 90), (10, 90)], outline=(255, 255, 255, 127))
                draw.line([(50, 10), (50, 90)], fill=(255, 255, 255, 127))
            elif intent == 'clarity':
                # Crystalline clarity sigil
                points = [(50, 10), (90, 50), (50, 90), (10, 50)]
                draw.polygon(points, outline=(255, 255, 255, 127))
                draw.line([(10, 50), (90, 50)], fill=(255, 255, 255, 127))
            elif intent == 'energy':
                # Spiral energy sigil
                for i in range(0, 360, 30):
                    r = i / 360 * 40
                    x = 50 + r * math.cos(math.radians(i))
                    y = 50 + r * math.sin(math.radians(i))
                    draw.point((x, y), fill=(255, 255, 255, 127))
            elif intent == 'peace':
                # Circular peace sigil
                draw.ellipse([(20, 20), (80, 80)], outline=(255, 255, 255, 127))
                draw.line([(50, 20), (50, 80)], fill=(255, 255, 255, 127))
            elif intent == 'growth':
                # Branching growth sigil
                draw.line([(50, 90), (50, 10)], fill=(255, 255, 255, 127))
                draw.line([(50, 50), (10, 20)], fill=(255, 255, 255, 127))
                draw.line([(50, 50), (90, 20)], fill=(255, 255, 255, 127))
            
            sigils[intent] = np.array(sigil)
            
        return sigils
    
    def embed_symbol(self, frame: Image.Image, symbol: str, position: Tuple[int, int], alpha: int = 20) -> Image.Image:
        """Embed a symbolic character with controlled opacity."""
        draw = ImageDraw.Draw(frame)
        draw.text(position, self.symbols.get(symbol, '?'), fill=(255, 255, 255, alpha))
        return frame
    
    def embed_sigil(self, frame: Image.Image, intent: str, position: Tuple[int, int], scale: float = 1.0) -> Image.Image:
        """Embed a sigil into the frame."""
        if intent in self.sigils:
            sigil = Image.fromarray(self.sigils[intent])
            if scale != 1.0:
                new_size = tuple(int(x * scale) for x in sigil.size)
                sigil = sigil.resize(new_size, Image.Resampling.LANCZOS)
            
            # Calculate position to center sigil
            x = position[0] - sigil.width // 2
            y = position[1] - sigil.height // 2
            
            # Paste sigil with transparency
            frame.paste(sigil, (x, y), sigil)
            
        return frame
    
    def create_subliminal_frame(self, base_frame: Image.Image, message: str) -> Image.Image:
        """Create a subliminal frame with embedded message."""
        frame = base_frame.copy()
        draw = ImageDraw.Draw(frame)
        
        # Calculate text size and position
        font = ImageFont.load_default()
        msg_width = sum(font.getbbox(c)[2] for c in message)
        x = (frame.width - msg_width) // 2
        y = frame.height // 2
        
        # Draw text with very low opacity
        draw.text((x, y), message, fill=(255, 255, 255, 10))
        
        return frame

class AudioReactiveRenderer:
    """Handles audio analysis and reactive effects."""
    
    def __init__(self, sr: int = 44100, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
        self.current_frame = 0
        self.features = None
        
    def load_audio(self, audio_path: str) -> AudioFeatures:
        """Load and analyze audio file."""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Extract features
        spectrum = np.abs(librosa.stft(y, hop_length=self.hop_length))
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        
        self.features = AudioFeatures(
            waveform=y,
            spectrum=spectrum,
            onset_env=onset_env,
            tempo=tempo,
            beat_frames=beat_frames,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            mfcc=mfcc
        )
        
        return self.features
    
    def get_frame_features(self, frame_idx: int) -> Dict[str, float]:
        """Get audio features for a specific frame."""
        if self.features is None:
            return {}
            
        # Calculate time-based indices
        spec_idx = min(frame_idx, self.features.spectrum.shape[1] - 1)
        cent_idx = min(frame_idx, len(self.features.spectral_centroid) - 1)
        
        # Get current beat intensity
        is_beat = frame_idx in self.features.beat_frames
        beat_intensity = self.features.onset_env[min(frame_idx, len(self.features.onset_env) - 1)]
        
        # Calculate energy in different frequency bands
        spec_frame = self.features.spectrum[:, spec_idx]
        bass = np.mean(spec_frame[:10])
        mids = np.mean(spec_frame[10:30])
        highs = np.mean(spec_frame[30:])
        
        return {
            'is_beat': is_beat,
            'beat_intensity': float(beat_intensity),
            'bass': float(bass),
            'mids': float(mids),
            'highs': float(highs),
            'centroid': float(self.features.spectral_centroid[cent_idx]),
            'rolloff': float(self.features.spectral_rolloff[cent_idx])
        }

class ImageRenderer:
    """Renders ASCII patterns as high-quality images with audio reactivity and subliminal effects."""
    
    def __init__(self, font_size: int = 20, padding: int = 2):
        """Initialize the image renderer.
        
        Args:
            font_size: Base font size for characters
            padding: Padding between characters
        """
        self.font_size = font_size
        self.padding = padding
        self.font = ImageFont.load_default()
        
        # Try to load a better font if available
        try:
            # Look for fonts in common locations
            font_paths = [
                "/System/Library/Fonts/Monaco.ttf",  # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
                "C:\\Windows\\Fonts\\consola.ttf",  # Windows
            ]
            for path in font_paths:
                if Path(path).exists():
                    self.font = ImageFont.truetype(path, font_size)
                    break
        except Exception:
            pass
        
        # Color palettes for different themes
        self.palettes = {
            'mystical': ['#FF00FF', '#8A2BE2', '#4B0082'],
            'cosmic': ['#00FFFF', '#FF69B4', '#9400D3'],
            'ethereal': ['#F0F8FF', '#E6E6FA', '#B0C4DE'],
            'fire': ['#FF4500', '#FF8C00', '#FFD700'],
            'nature': ['#228B22', '#32CD32', '#90EE90'],
            'cyber': ['#00FF00', '#7FFF00', '#00FF7F'],
            'void': ['#000000', '#1A1A1A', '#333333'],
            # New palettes
            'aurora': ['#00FF87', '#00B8FF', '#8A2BE2', '#FF1493'],
            'sunset': ['#FF0000', '#FF8C00', '#FFD700', '#4B0082'],
            'ocean': ['#000080', '#0000FF', '#00FFFF', '#87CEEB'],
            'forest': ['#006400', '#228B22', '#32CD32', '#90EE90'],
            'neon': ['#FF00FF', '#00FF00', '#00FFFF', '#FF0000'],
            'pastel': ['#FFB6C1', '#98FB98', '#87CEFA', '#DDA0DD'],
            'monochrome': ['#000000', '#333333', '#666666', '#999999'],
            'matrix': ['#003B00', '#008F11', '#00FF41'],
            'cyberpunk': ['#FF00FF', '#00FFFF', '#FF0000', '#FFFF00'],
            'retro': ['#EE4B2B', '#7FFFD4', '#DEB887', '#FF69B4']
        }
        
        # Add new psychedelic palettes
        self.palettes.update({
            'rainbow_pulse': ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#8F00FF'],
            'acid': ['#FF00FF', '#00FFFF', '#FFFF00', '#FF00FF'],
            'plasma': ['#FF3800', '#FF00FF', '#0000FF', '#00FFFF'],
            'fractal': ['#000764', '#206BCB', '#EDFFFF', '#FFAA00', '#310230'],
            'quantum': ['#001F3F', '#7FDBFF', '#39CCCC', '#01FF70', '#FFDC00'],
            'neural': ['#FF4136', '#85144b', '#F012BE', '#B10DC9', '#FFFFFF'],
            'dream': ['#FF1493', '#9400D3', '#4B0082', '#0000FF', '#00FF00']
        })
        
        # Add new effects
        self.effect_generators = {
            'fractals': self._generate_fractal_overlay,
            'neural': self._generate_neural_pattern,
            'quantum': self._generate_quantum_noise,
            'flow_field': self._generate_flow_field,
            'reaction_diffusion': self._generate_reaction_diffusion
        }
        
        # Initialize audio and subliminal components
        self.audio_renderer = AudioReactiveRenderer()
        self.subliminal = SubliminalEffect()
        
        # Add new audio-reactive palettes
        self.palettes.update({
            'spectrum': ['#FF0000', '#00FF00', '#0000FF'],  # RGB spectrum
            'intensity': ['#000000', '#FFFFFF'],  # Black to white
            'frequency': ['#FF00FF', '#00FFFF', '#FFFF00']  # Frequency-based
        })
    
    def _get_char_size(self, char: str) -> Tuple[int, int]:
        """Get the pixel dimensions of a character."""
        bbox = self.font.getbbox(char)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    def _create_base_image(self, pattern: Pattern, scale: float = 1.0) -> Image.Image:
        """Create a base image for the pattern."""
        char_width, char_height = self._get_char_size('█')
        
        # Calculate image dimensions with padding
        width = int((char_width + self.padding) * pattern.width * scale)
        height = int((char_height + self.padding) * pattern.height * scale)
        
        return Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    def _apply_effects(
        self,
        img: Image.Image,
        style: str,
        glow: bool = True,
        blur: bool = True,
        enhance: bool = True,
        chromatic_aberration: bool = False,
        scanlines: bool = False,
        noise: float = 0.0,
        vignette: bool = False,
        psychedelic_effects: List[str] = None
    ) -> Image.Image:
        """Apply visual effects to the image.
        
        Args:
            img: Input image
            style: Visual style
            glow: Whether to add glow effect
            blur: Whether to add blur effect
            enhance: Whether to enhance contrast/brightness
            chromatic_aberration: Whether to add RGB split effect
            scanlines: Whether to add scanline effect
            noise: Amount of noise to add (0.0 to 1.0)
            vignette: Whether to add vignette effect
            psychedelic_effects: List of psychedelic effects to apply
        """
        if glow:
            # Create glow effect
            glow = img.filter(ImageFilter.GaussianBlur(radius=3))
            glow = ImageEnhance.Brightness(glow).enhance(1.2)
            img = Image.alpha_composite(glow, img)
        
        if blur:
            # Add slight blur for smoothing
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        if enhance:
            # Enhance contrast and brightness
            img = ImageEnhance.Contrast(img).enhance(1.1)
            img = ImageEnhance.Brightness(img).enhance(1.1)
        
        if chromatic_aberration:
            # Split RGB channels
            r, g, b, a = img.split()
            img = Image.merge('RGBA', (
                ImageOps.offset(r, 2, 0),  # Red channel
                g,                         # Green channel
                ImageOps.offset(b, -2, 0), # Blue channel
                a
            ))
        
        if scanlines:
            # Add scanline effect
            width, height = img.size
            scanline = Image.new('RGBA', (width, 2), (0, 0, 0, 0))
            for y in range(0, height, 4):
                img.paste((0, 0, 0, 64), (0, y, width, y + 1))
        
        if noise > 0:
            # Add noise
            noise_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            noise_data = []
            for _ in range(img.width * img.height):
                v = int(np.random.normal(0, noise * 255))
                v = max(-255, min(255, v))
                noise_data.extend([v, v, v, int(abs(v) * 0.3)])
            noise_layer.putdata(noise_data)
            img = Image.alpha_composite(img, noise_layer)
        
        if vignette:
            # Add vignette effect
            width, height = img.size
            vignette = Image.new('RGBA', img.size, (0, 0, 0, 0))
            radius = min(width, height) // 2
            for y in range(height):
                for x in range(width):
                    distance = ((x - width/2)**2 + (y - height/2)**2) ** 0.5
                    alpha = int(255 * (1 - min(1, distance/radius)))
                    vignette.putpixel((x, y), (0, 0, 0, 255 - alpha))
            img = Image.alpha_composite(img, vignette)
        
        # Apply psychedelic effects if requested
        if psychedelic_effects:
            width, height = img.size
            for effect in psychedelic_effects:
                if effect in self.effect_generators:
                    overlay = self.effect_generators[effect](width, height)
                    img = Image.alpha_composite(img, overlay)
        
        return img
    
    def _get_color_gradient(self, style: str, density: float) -> Tuple[int, int, int, int]:
        """Get color for a character based on style and density."""
        if style not in self.palettes:
            style = 'mystical'
            
        colors = self.palettes[style]
        
        # Convert hex colors to RGB
        rgb_colors = [tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for c in colors]
        
        # Interpolate between colors based on density
        idx = min(int(density * (len(rgb_colors) - 1)), len(rgb_colors) - 2)
        t = (density * (len(rgb_colors) - 1)) % 1
        
        c1 = rgb_colors[idx]
        c2 = rgb_colors[idx + 1]
        
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        
        # Add alpha based on density
        alpha = int(255 * min(1.0, density * 1.2))
        
        return (r, g, b, alpha)
    
    def _generate_fractal_overlay(self, width: int, height: int) -> Image.Image:
        """Generate a Mandelbrot-inspired fractal overlay."""
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        pixels = overlay.load()
        
        max_iter = 100
        for x in range(width):
            for y in range(height):
                # Map pixel coordinates to complex plane
                zx = 1.5 * (x - width/2) / (0.5 * width)
                zy = 1.0 * (y - height/2) / (0.5 * height)
                
                c = complex(zx, zy)
                z = complex(0, 0)
                
                for i in range(max_iter):
                    if abs(z) > 2:
                        break
                    z = z*z + c
                
                # Color based on escape time
                if i < max_iter - 1:
                    hue = i / max_iter
                    rgb = tuple(int(x*255) for x in colorsys.hsv_to_rgb(hue, 0.8, 0.9))
                    pixels[x, y] = rgb + (int(127 * (1 - i/max_iter)),)
                else:
                    pixels[x, y] = (0, 0, 0, 0)
                    
        return overlay

    def _generate_neural_pattern(self, width: int, height: int) -> Image.Image:
        """Generate a neural network-inspired pattern."""
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Create nodes
        nodes = []
        for _ in range(50):
            x = random.randint(0, width)
            y = random.randint(0, height)
            nodes.append((x, y))
        
        # Draw connections
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if random.random() < 0.2:  # 20% chance to connect
                    distance = math.sqrt((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2)
                    if distance < width/3:  # Only connect nearby nodes
                        alpha = int(255 * (1 - distance/(width/3)))
                        draw.line([node1, node2], fill=(255, 255, 255, alpha), width=1)
        
        # Draw nodes
        for x, y in nodes:
            radius = random.randint(2, 6)
            draw.ellipse(
                [(x-radius, y-radius), (x+radius, y+radius)],
                fill=(255, 255, 255, 127)
            )
            
        return overlay

    def _generate_quantum_noise(self, width: int, height: int) -> Image.Image:
        """Generate quantum-inspired noise pattern."""
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        pixels = overlay.load()
        
        # Generate wave function
        for x in range(width):
            for y in range(height):
                # Quantum probability wave
                phi = math.sin(x/20) * math.cos(y/20)
                psi = math.cos(x/15) * math.sin(y/15)
                probability = abs(phi * psi)
                
                if random.random() < probability:
                    # Quantum color
                    hue = (x + y) / (width + height)
                    rgb = tuple(int(x*255) for x in colorsys.hsv_to_rgb(hue, 0.8, probability))
                    pixels[x, y] = rgb + (int(127 * probability),)
                    
        return overlay

    def _generate_flow_field(self, width: int, height: int) -> Image.Image:
        """Generate a flow field pattern."""
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Generate flow field
        particles = [(random.randint(0, width), random.randint(0, height)) for _ in range(1000)]
        
        for _ in range(50):  # Number of steps
            new_particles = []
            for x, y in particles:
                if 0 <= x < width and 0 <= y < height:
                    # Calculate flow direction based on position
                    angle = (math.sin(x/50) + math.cos(y/50)) * math.pi
                    dx = math.cos(angle) * 2
                    dy = math.sin(angle) * 2
                    
                    # Draw particle trail
                    new_x = x + dx
                    new_y = y + dy
                    alpha = int(127 * (1 - math.sqrt((x-width/2)**2 + (y-height/2)**2)/(width/2)))
                    if alpha > 0:
                        draw.line([(x, y), (new_x, new_y)], fill=(255, 255, 255, alpha), width=1)
                    
                    new_particles.append((new_x, new_y))
            
            particles = new_particles
            
        return overlay

    def _generate_reaction_diffusion(self, width: int, height: int) -> Image.Image:
        """Generate a reaction-diffusion pattern."""
        # Initialize chemical concentrations
        A = np.ones((height, width))
        B = np.zeros((height, width))
        
        # Add random seed
        B[height//2-10:height//2+10, width//2-10:width//2+10] = 1
        
        # Simulation parameters
        Da = 1.0  # Diffusion rate A
        Db = 0.5  # Diffusion rate B
        f = 0.055  # Feed rate
        k = 0.062  # Kill rate
        
        # Run simulation
        for _ in range(50):
            # Compute Laplacian
            LA = np.roll(A, 1, 0) + np.roll(A, -1, 0) + np.roll(A, 1, 1) + np.roll(A, -1, 1) - 4*A
            LB = np.roll(B, 1, 0) + np.roll(B, -1, 0) + np.roll(B, 1, 1) + np.roll(B, -1, 1) - 4*B
            
            # Update concentrations
            AB2 = A * B * B
            A += Da*LA - AB2 + f*(1-A)
            B += Db*LB + AB2 - (k+f)*B
            
            # Clip values
            A = np.clip(A, 0, 1)
            B = np.clip(B, 0, 1)
        
        # Convert to image
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        pixels = overlay.load()
        
        for y in range(height):
            for x in range(width):
                v = B[y, x]
                pixels[x, y] = (255, 255, 255, int(v * 127))
                
        return overlay

    def _apply_audio_reactive_effects(
        self,
        img: Image.Image,
        audio_features: Dict[str, float],
        intensity: float = 1.0
    ) -> Image.Image:
        """Apply audio-reactive effects to the image."""
        if not audio_features:
            return img
            
        # Create working copy
        result = img.copy()
        
        # Apply beat-based scaling
        if audio_features['is_beat']:
            scale_factor = 1.0 + (audio_features['beat_intensity'] * 0.2 * intensity)
            new_size = tuple(int(x * scale_factor) for x in img.size)
            result = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Re-center the scaled image
            offset_x = (result.width - img.width) // 2
            offset_y = (result.height - img.height) // 2
            result = result.crop((
                offset_x, offset_y,
                offset_x + img.width,
                offset_y + img.height
            ))
        
        # Apply frequency-based effects
        bass_blur = ImageFilter.GaussianBlur(radius=audio_features['bass'] * intensity)
        result = result.filter(bass_blur)
        
        # Apply spectral effects
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(1.0 + (audio_features['highs'] * 0.5 * intensity))
        
        # Apply color modulation based on spectral centroid
        if audio_features['centroid'] > 0:
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(1.0 + (audio_features['centroid'] * 0.3 * intensity))
        
        return result
    
    def render_pattern(
        self,
        pattern: Pattern,
        style: str = 'mystical',
        scale: float = 1.0,
        glow: bool = True,
        blur: bool = True,
        enhance: bool = True,
        chromatic_aberration: bool = False,
        scanlines: bool = False,
        noise: float = 0.0,
        vignette: bool = False,
        psychedelic_effects: List[str] = None,
        audio_file: Optional[str] = None,
        audio_frame: int = 0,
        audio_intensity: float = 1.0,
        subliminal_message: Optional[str] = None,
        subliminal_symbols: List[str] = None,
        subliminal_sigils: List[str] = None
    ) -> Image.Image:
        """Render a pattern with audio reactivity and subliminal effects."""
        # Create base image
        img = self._create_base_image(pattern, scale)
        draw = ImageDraw.Draw(img)
        
        char_width, char_height = self._get_char_size('█')
        char_width = int(char_width * scale)
        char_height = int(char_height * scale)
        
        # Draw each character
        for y in range(pattern.height):
            for x in range(pattern.width):
                char = pattern.data[y, x]
                if char == ' ':
                    continue
                    
                # Calculate character position
                pos_x = x * (char_width + self.padding)
                pos_y = y * (char_height + self.padding)
                
                # Get color based on character density
                density = pattern.symbols.find(char) / len(pattern.symbols)
                color = self._get_color_gradient(style, density)
                
                # Draw character
                draw.text((pos_x, pos_y), char, fill=color, font=self.font)
        
        # Apply effects
        img = self._apply_effects(
            img,
            style,
            glow=glow,
            blur=blur,
            enhance=enhance,
            chromatic_aberration=chromatic_aberration,
            scanlines=scanlines,
            noise=noise,
            vignette=vignette,
            psychedelic_effects=psychedelic_effects
        )
        
        # Apply audio-reactive effects if audio file provided
        if audio_file and not hasattr(self, '_audio_features'):
            self._audio_features = self.audio_renderer.load_audio(audio_file)
        
        if hasattr(self, '_audio_features'):
            frame_features = self.audio_renderer.get_frame_features(audio_frame)
            img = self._apply_audio_reactive_effects(img, frame_features, audio_intensity)
        
        # Apply subliminal effects
        if subliminal_message:
            img = self.subliminal.create_subliminal_frame(img, subliminal_message)
        
        if subliminal_symbols:
            width, height = img.size
            for symbol in subliminal_symbols:
                # Place symbols at pseudo-random positions
                x = random.randint(0, width)
                y = random.randint(0, height)
                img = self.subliminal.embed_symbol(img, symbol, (x, y))
        
        if subliminal_sigils:
            width, height = img.size
            for sigil in subliminal_sigils:
                # Place sigils at golden ratio positions
                phi = (1 + math.sqrt(5)) / 2
                x = width / phi
                y = height / phi
                img = self.subliminal.embed_sigil(img, sigil, (int(x), int(y)))
        
        return img
    
    def render_animation_frames(
        self,
        frames: List[Pattern],
        style: str = 'mystical',
        scale: float = 1.0,
        glow: bool = True,
        blur: bool = True,
        enhance: bool = True,
        chromatic_aberration: bool = False,
        scanlines: bool = False,
        noise: float = 0.0,
        vignette: bool = False
    ) -> List[Image.Image]:
        """Render animation frames as images."""
        return [
            self.render_pattern(
                frame,
                style=style,
                scale=scale,
                glow=glow,
                blur=blur,
                enhance=enhance,
                chromatic_aberration=chromatic_aberration,
                scanlines=scanlines,
                noise=noise,
                vignette=vignette
            )
            for frame in frames
        ]
    
    def save_pattern(
        self,
        pattern: Pattern,
        output_path: str,
        style: str = 'mystical',
        scale: float = 1.0,
        glow: bool = True,
        blur: bool = True,
        enhance: bool = True,
        chromatic_aberration: bool = False,
        scanlines: bool = False,
        noise: float = 0.0,
        vignette: bool = False,
        background_color: Optional[Tuple[int, int, int]] = None,
        format: str = 'PNG'
    ) -> None:
        """Save pattern as an image file.
        
        Args:
            pattern: Pattern to render
            output_path: Path to save the image
            style: Visual style to use
            scale: Scale factor for the image
            glow: Whether to add glow effect
            blur: Whether to add blur effect
            enhance: Whether to enhance contrast/brightness
            chromatic_aberration: Whether to add RGB split effect
            scanlines: Whether to add scanline effect
            noise: Amount of noise to add (0.0 to 1.0)
            vignette: Whether to add vignette effect
            background_color: Optional background color (RGB)
            format: Output format (PNG, JPEG, etc.)
        """
        img = self.render_pattern(
            pattern,
            style=style,
            scale=scale,
            glow=glow,
            blur=blur,
            enhance=enhance,
            chromatic_aberration=chromatic_aberration,
            scanlines=scanlines,
            noise=noise,
            vignette=vignette
        )
        img.save(output_path, format)
    
    def save_animation(
        self,
        frames: List[Pattern],
        output_path: str,
        style: str = 'mystical',
        scale: float = 1.0,
        glow: bool = True,
        blur: bool = True,
        enhance: bool = True,
        chromatic_aberration: bool = False,
        scanlines: bool = False,
        noise: float = 0.0,
        vignette: bool = False,
        psychedelic_effects: List[str] = None,
        audio_file: Optional[str] = None,
        audio_intensity: float = 1.0,
        subliminal_message: Optional[str] = None,
        subliminal_symbols: List[str] = None,
        subliminal_sigils: List[str] = None,
        duration: int = 100  # ms per frame
    ) -> None:
        """Save animation with audio reactivity and subliminal effects."""
        images = []
        for i, frame in enumerate(frames):
            img = self.render_pattern(
                frame,
                style=style,
                scale=scale,
                glow=glow,
                blur=blur,
                enhance=enhance,
                chromatic_aberration=chromatic_aberration,
                scanlines=scanlines,
                noise=noise,
                vignette=vignette,
                psychedelic_effects=psychedelic_effects,
                audio_file=audio_file,
                audio_frame=i,
                audio_intensity=audio_intensity,
                subliminal_message=subliminal_message,
                subliminal_symbols=subliminal_symbols,
                subliminal_sigils=subliminal_sigils
            )
            images.append(img)
        
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        ) 