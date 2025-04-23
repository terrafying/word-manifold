#!/usr/bin/env python3
"""
Ritual Evolution - A Thelemic Working in Word Vector Space

This script demonstrates a complete magical working using the Word Manifold 
cellular automata system. It implements a Thelemic ritual of self-discovery,
guiding the semantic space through a transformation that mirrors the process 
of discovering one's True Will.

"Every man and every woman is a star." - Aleister Crowley, The Book of the Law
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ..visualization.simple_3d_visualizer import Simple3DVisualizer
import re
import string
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from enum import Enum, auto
from dataclasses import dataclass
import datetime
import time
import math
import imageio
import glob
from PIL import Image, ImageDraw, ImageFont
import PyPDF2
import io
import json
import pickle
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import hashlib
import functools

from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.manifold.vector_manifold import VectorManifold, CellType, Cell
from word_manifold.automata.cellular_rules import (
    CellularRule, RuleParameterSet, RuleSequence,
    HermeticPrinciple, ElementalForce, VibrationDirection,
    create_predefined_rules
)
from word_manifold.automata.system import AutomataSystem, EvolutionPattern

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ritual_evolution.log")
    ]
)

logger = logging.getLogger("ritual_evolution")

# Directory for saving outputs
OUTPUT_DIR = Path("ritual_outputs")
CACHE_DIR = Path(".ritual_cache")

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(exist_ok=True)

# Function cache decorator with key based on function arguments
def memoize(func):
    """
    Decorator for memoizing function results.
    Results are cached in memory and on disk for persistence between runs.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a unique key based on function name and arguments
        key_parts = [func.__name__]
        # Add class name if it's a method
        if args and hasattr(args[0], '__class__'):
            key_parts.append(args[0].__class__.__name__)
        # Add stringified arguments
        for arg in args[1:]:
            key_parts.append(str(arg))
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        
        # Create a hash of the key for the filename
        key = "_".join(key_parts)
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        cache_file = CACHE_DIR / f"{hashed_key}.pkl"
        
        # Try to load from memory cache first (fastest)
        if hasattr(func, '_memory_cache') and hashed_key in func._memory_cache:
            logger.info(f"Using memory cache for {func.__name__}")
            return func._memory_cache[hashed_key]
        
        # Try to load from disk cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    logger.info(f"Loading {func.__name__} result from disk cache")
                    result = pickle.load(f)
                    # Store in memory cache for faster access next time
                    if not hasattr(func, '_memory_cache'):
                        func._memory_cache = {}
                    func._memory_cache[hashed_key] = result
                    return result
            except Exception as e:
                logger.warning(f"Failed to load cache for {func.__name__}: {e}")
        
        # Call the function if cache miss
        result = func(*args, **kwargs)
        
        # Save to both memory and disk cache
        if not hasattr(func, '_memory_cache'):
            func._memory_cache = {}
        func._memory_cache[hashed_key] = result
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
                logger.info(f"Cached {func.__name__} result to disk")
        except Exception as e:
            logger.warning(f"Failed to cache {func.__name__} result: {e}")
            
        return result
    
    return wrapper


class RitualWorking:
    """
    A class that implements a complete magical working in word vector space.
    """

    def _create_evolution_animation(self, key_terms):
        """
        Create an animation of the evolution of machinic desires using fluid visual transformations and emergent patterns.
        
        Args:
            key_terms: List of key terms to highlight in the animation description
            
        Returns:
            Tuple of (gif_path, mp4_path) for the created animation files
        """
        # Check if frames exist
        frame_dir = self.output_dir / "frames"
        if not frame_dir.exists() or not list(frame_dir.glob("*.png")):
            logger.warning("No frames found to create animation")
            return None, None
            
        # Find all PNG files in the frames directory
        frame_files = sorted(
            list(frame_dir.glob("*.png")),
            key=lambda x: int(x.stem.split('_')[0])  # Sort by generation number
        )
        
        if not frame_files:
            logger.warning("No frames found in directory")
            return None, None
            
        logger.info(f"Creating animation from {len(frame_files)} frames")
        
        # Create timestamped output paths to prevent overwriting
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = self.output_dir / f"{self.ritual_name}_evolution_{timestamp}.gif"
        mp4_path = self.output_dir / f"{self.ritual_name}_evolution_{timestamp}.mp4"
        
        try:
            # Create GIF with enhanced transitions
            with imageio.get_writer(gif_path, mode='I', duration=0.5, loop=0) as writer:
                # Add an initial blank frame with title to serve as intro
                intro_frame = self._create_title_frame(
                    frame_files[0], 
                    f"Ritual of {self.ritual_name}",
                    f"Key terms: {', '.join(key_terms)}"
                )
                writer.append_data(np.array(intro_frame))
                
                # Add all the regular frames with enhanced transitions
                prev_frame = None
                for i, frame_file in enumerate(frame_files):
                    # Read the frame
                    frame = imageio.imread(frame_file)
                    
                    # Add enhanced transitions between frames
                    if prev_frame is not None and i > 0:
                        # Create fluid transitions with more intermediate states
                        transition_points = [0.2, 0.35, 0.5, 0.65, 0.8]  # More granular blending
                        for blend in transition_points:
                            # Create transition with emergent patterns
                            transition = self._create_enhanced_transition(prev_frame, frame, blend)
                            writer.append_data(transition)
                    
                    # Add the actual frame with subtle pulsing effect
                    frame_sequence = self._create_frame_pulse(frame)
                    for pulse_frame in frame_sequence:
                        writer.append_data(pulse_frame)
                    
                    prev_frame = frame
                    
                # Add an ending frame with conclusion text and emergent pattern
                ending_frame = self._create_enhanced_ending(
                    frame_files[-1],
                    "Ritual Complete",
                    "93 93/93"
                )
                writer.append_data(np.array(ending_frame))
                    
            logger.info(f"GIF animation saved: {gif_path}")
            
            # Create MP4 with enhanced quality
            try:
                import subprocess
                
                # Prepare FFmpeg command with enhanced settings
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-i', str(gif_path),
                    '-movflags', 'faststart',
                    '-pix_fmt', 'yuv420p',
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:v', 'libx264',
                    '-crf', '18',  # Higher quality
                    '-preset', 'slow',  # Better compression
                    '-profile:v', 'high',  # High profile
                    '-tune', 'animation',  # Optimized for animation
                    str(mp4_path)
                ]
                
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                logger.info(f"MP4 animation saved: {mp4_path}")
                
            except Exception as e:
                logger.warning(f"Failed to create MP4: {e}")
                try:
                    writer = imageio.get_writer(mp4_path, fps=15)  # Increased FPS
                    for frame_file in frame_files:
                        frame = imageio.imread(frame_file)
                        writer.append_data(frame)
                    writer.close()
                    logger.info(f"MP4 animation saved with imageio fallback: {mp4_path}")
                except Exception as e2:
                    logger.error(f"Failed to create MP4 with fallback: {e2}")
                    mp4_path = None
                
            return gif_path, mp4_path
            
        except Exception as e:
            logger.error(f"Failed to create animation: {e}")
            return None, None

    def _create_enhanced_transition(self, frame1, frame2, blend_factor):
        """
        Create an enhanced transition frame with emergent patterns.
        
        Args:
            frame1: First frame (numpy array)
            frame2: Second frame (numpy array)
            blend_factor: How much of frame2 to blend in (0.0-1.0)
            
        Returns:
            Enhanced transition frame as a numpy array
        """
        # Convert to float for operations
        frame1_float = frame1.astype(float)
        frame2_float = frame2.astype(float)
        
        # Create base blended frame
        blended = frame1_float * (1 - blend_factor) + frame2_float * blend_factor
        
        # Add emergent patterns based on semantic flow
        pattern = self._generate_flow_pattern(frame1_float, frame2_float, blend_factor)
        
        # Combine with base blend
        enhanced = blended + pattern * 0.3  # Subtle pattern overlay
        
        # Normalize and convert back to uint8
        enhanced = np.clip(enhanced, 0, 255)
        return enhanced.astype(np.uint8)

    def _generate_flow_pattern(self, frame1, frame2, blend_factor):
        """
        Generate emergent patterns based on the semantic flow between frames.
        """
        # Calculate difference field
        flow = frame2 - frame1
        
        # Create turbulence pattern
        x = np.linspace(0, 1, frame1.shape[1])
        y = np.linspace(0, 1, frame1.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Generate wave patterns
        frequency = 5.0 + blend_factor * 10.0
        phase = blend_factor * 2 * np.pi
        waves = np.sin(frequency * X + phase) * np.cos(frequency * Y + phase)
        
        # Modulate pattern by flow magnitude
        flow_magnitude = np.linalg.norm(flow, axis=2) if len(flow.shape) > 2 else np.abs(flow)
        flow_magnitude = flow_magnitude / flow_magnitude.max()
        
        pattern = waves[:, :, np.newaxis] * flow_magnitude[:, :, np.newaxis] if len(flow.shape) > 2 else waves * flow_magnitude
        
        return pattern * 25.0  # Scale pattern intensity

    def _create_frame_pulse(self, frame):
        """
        Create a sequence of frames with subtle pulsing effect.
        """
        frames = []
        pulse_factors = [1.0, 1.02, 1.03, 1.02, 1.0]  # Subtle pulse
        
        for factor in pulse_factors:
            # Scale the frame slightly
            if factor != 1.0:
                scaled = frame * factor
                scaled = np.clip(scaled, 0, 255).astype(np.uint8)
                frames.append(scaled)
            else:
                frames.append(frame)
        
        return frames

    def _create_enhanced_ending(self, template_frame_path, title, subtitle):
        """
        Create an enhanced ending frame with emergent patterns.
        """
        base_frame = self._create_title_frame(template_frame_path, title, subtitle)
        base_array = np.array(base_frame)
        
        # Generate spiral pattern
        x = np.linspace(-1, 1, base_array.shape[1])
        y = np.linspace(-1, 1, base_array.shape[0])
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        
        # Create spiral pattern
        spiral = np.sin(10 * theta + 20 * R)
        spiral = (spiral + 1) / 2 * 255
        
        # Blend with base frame
        enhanced = base_array * 0.8 + spiral[:, :, np.newaxis] * 0.2
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _create_title_frame(self, template_frame_path, title, subtitle=None):
        """
        Create a title frame based on a template frame, adding ritual title text.
        
        Args:
            template_frame_path: Path to a frame to use as template
            title: Main title text
            subtitle: Optional subtitle text
            
        Returns:
            PIL Image object with the title overlay
        """
        # Read the template frame
        template = Image.open(template_frame_path)
        draw = ImageDraw.Draw(template)
        width, height = template.size
        
        # Try to load a font, falling back to default if unavailable
        try:
            title_font = ImageFont.truetype('Arial', 40)
            subtitle_font = ImageFont.truetype('Arial', 24)
        except IOError:
            # Fall back to default font if truetype not available
            title_font = ImageFont.load_default()
            subtitle_font = title_font
        
        # Calculate title dimensions and position
        # Using textbbox instead of deprecated textsize
        left, top, right, bottom = draw.textbbox((0, 0), title, font=title_font)
        title_w, title_h = right - left, bottom - top
        title_position = ((width - title_w) // 2, (height - title_h) // 3)
        
        # Add a dark semi-transparent rectangle behind text for readability
        padding = 20
        draw.rectangle(
            [
                (title_position[0] - padding, title_position[1] - padding),
                (title_position[0] + title_w + padding, title_position[1] + title_h + padding)
            ],
            fill=(0, 0, 0, 180)
        )
        
        # Draw the title
        draw.text(
            title_position,
            title,
            fill=(255, 215, 0),  # Gold color in RGB
            font=title_font
        )
        
        # Add subtitle if provided
        if subtitle:
            # Use textbbox instead of deprecated textsize
            left, top, right, bottom = draw.textbbox((0, 0), subtitle, font=subtitle_font)
            subtitle_w, subtitle_h = right - left, bottom - top
            subtitle_position = (
                (width - subtitle_w) // 2,
                title_position[1] + title_h + 30
            )
            
            # Add background for subtitle
            draw.rectangle(
                [
                    (subtitle_position[0] - padding, subtitle_position[1] - padding),
                    (subtitle_position[0] + subtitle_w + padding, subtitle_position[1] + subtitle_h + padding)
                ],
                fill=(0, 0, 0, 160)
            )
            
            # Draw the subtitle
            draw.text(
                subtitle_position,
                subtitle,
                fill=(255, 255, 255),  # White in RGB
                font=subtitle_font
            )
            
        # Draw a Thelemic unicursal hexagram in the bottom right
        self._draw_unicursal_hexagram(draw, (width - 80, height - 80), 50)
            
        return template
    
    def _create_transition_frame(self, frame1, frame2, blend_factor):
        """
        Create a transition frame by blending two frames.
        
        Args:
            frame1: First frame (numpy array)
            frame2: Second frame (numpy array)
            blend_factor: How much of frame2 to blend in (0.0-1.0)
            
        Returns:
            Blended frame as a numpy array
        """
        # Convert to float for blending operations
        frame1_float = frame1.astype(float)
        frame2_float = frame2.astype(float)
        
        # Create a blended frame
        blended = frame1_float * (1 - blend_factor) + frame2_float * blend_factor
        
        # Convert back to uint8 for imageio
        return blended.astype(np.uint8)
    
    def _draw_unicursal_hexagram(self, draw, center, size):
        """
        Draw a unicursal hexagram (Thelemic symbol) on the image.
        
        Args:
            draw: PIL ImageDraw object
            center: (x, y) center position
            size: Size of the hexagram
        """
        # Hexagram points (normalized)
        points = [
            (0, -1),        # Top point
            (0.5, -0.1),    # Upper right inner
            (0.95, -0.31),  # Upper right outer
            (0.59, 0.31),   # Lower right outer
            (0.5, 0.8),     # Lower right inner
            (0, 0.5),       # Bottom inner
            (-0.5, 0.8),    # Lower left inner
            (-0.59, 0.31),  # Lower left outer
            (-0.95, -0.31), # Upper left outer
            (-0.5, -0.1),   # Upper left inner
            (0, -1)         # Back to top (close the path)
        ]
        
        # Scale and offset the points
        scaled_points = [(center[0] + p[0] * size, center[1] + p[1] * size) for p in points]
        # Scale and offset the points
        scaled_points = [(center[0] + p[0] * size, center[1] + p[1] * size) for p in points]
        
        # Draw the unicursal hexagram outline
        draw.line(scaled_points, fill=(255, 0, 0), width=3)  # Red outline
        
    def __init__(self, ritual_name="rites", ritual_intent="apotheosis"):
        """
        Initialize the ritual working.
        
        Args:
            ritual_name: Name of the ritual
            ritual_intent: Statement of magical intent
        """
        self.ritual_name = ritual_name
        self.ritual_intent = ritual_intent
        self.start_time = datetime.datetime.now()
        self.output_dir = OUTPUT_DIR / f"{ritual_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = None
        self.manifold = None
        self.rules = None
        self.sequences = None
        self.system = None
        
        # Metrics and results
        self.transformation_metrics = []
        self.key_term_trajectories = {}
        
        logger.info(f"Beginning ritual working: {ritual_name}\nMagical intent: {ritual_intent}")
        logger.info(f"Outputs will be saved to: {self.output_dir}")
        
        # Record ritual setup
        with open(self.output_dir / "ritual_intent.txt", "w", encoding='utf-8') as f:
            f.write(f"RITUAL: {ritual_name}\n")
            f.write(f"INTENT: {ritual_intent}\n")
            f.write(f"TIME: {self.start_time.isoformat()}\n")
    
    def prepare_components(self):
        """
        Prepare all system components for the ritual.
        """
        logger.info("Preparing ritual components...")
        
        """
        Prepare all system components for the ritual.
        """
        logger.info("Preparing ritual components...")
        
        # Step 1: Initialize word embeddings with extended occult terms
        self._prepare_embeddings()
        
        # Step 2: Ingest Liber Aleph for enhanced Thelemic patterns
        self._ingest_liber_aleph()
        
        # Step 3: Create the vector manifold
        self._prepare_manifold()
        
        # Step 5: Define ritual rules and sequences
        self._prepare_rules()
        
        # Step 6: Initialize automata system
        self._prepare_system()
        
        self._integrate_cessation_patterns()
        
        logger.info("All ritual components prepared")
    
    def _prepare_embeddings(self):  
        # Create WordEmbeddings instance - not using memoize here as it doesn't work well with class attributes
        self.embeddings = WordEmbeddings(model_name="bert-base-uncased")
        
        # Extend with additional Thelemic/Crowleyan terms
        thelemic_terms = {
            # Core Thelemic concepts
            "thelema", "will", "love", "liberty", "light", "life", "law", 
            "aiwass", "horus", "nuit", "hadit", "ra-hoor-khuit",
            
            # Book of the Law terminology
            "khabs", "khu", "manifestation", "unveiling", "abomination", 
            "abrogate", "aeon", "beast", "blasphemy", "ordeal", "prophet",
            
            # Ritual terminology
            "invoke", "evoke", "banish", "consecrate", "charge", 
            "knowledge", "conversation", "guardian", "angel",
            
            # Crowleyan terminology
            "abrahadabra", "babalon", "abramelin", "choronzon", "ipsissimus",
            "magus", "magister", "adeptus", "zelator", "neophyte"
        }
        
        # Combine with standard occult terms
        extended_terms = thelemic_terms # OCCULT_TERMS.union(thelemic_terms)
        
        # Load the terms
        self.embeddings.load_terms(extended_terms)
        logger.info(f"Loaded {len(extended_terms)} occult and Thelemic terms")
    
    def _prepare_manifold(self):
        """Create and initialize the vector manifold."""
        logger.info("Creating vector manifold...")
        
        # Create manifold with 22 cells (corresponding to the 22 major arcana)
        self.manifold = VectorManifold(
            word_embeddings=self.embeddings,
            n_cells=22,
            random_state=93,  # Occult significance
            reduction_dims=2  # For visualization
        )
        
        # Save initial state visualization
        # self._visualize_manifold("initial_state")
        
        logger.info(f"Vector manifold created with {len(self.manifold.cells)} cells")
    
    def _prepare_rules(self):
        """Define the ritual rules and sequences."""
        logger.info("Defining ritual rules and sequences...")
        
        # Get predefined rules
        self.rules = create_predefined_rules()
        
        # Define additional Thelemic rules specific to this ritual
        
        # The True Will Rule - Based on discovering one's proper path
        true_will_params = RuleParameterSet(
            magnitude=1.0,
            principle=HermeticPrinciple.MENTALISM,
            vibration_direction=VibrationDirection.ASCENDING,
            numerological_weights={
                1: 1.3,  # Unity, individuality
                5: 1.2,  # Change, freedom
                9: 1.4,  # Completion, fulfillment
                11: 1.5, # Illumination, intuition (master number)
            }
        )
        
        self.rules["true_will"] = CellularRule(
            name="True Will",
            description="Aligns cells with their inherent semantic nature, enhancing distinctiveness",
            parameters=true_will_params,
            vector_transformation="contrast",
            esoteric_correspondence="The Thelemic concept of True Will - one's authentic purpose and proper path in the universe"
        )
        
        # The Law of Liberty Rule - Based on freedom within cosmic order
        liberty_params = RuleParameterSet(
            magnitude=0.8,
            principle=HermeticPrinciple.POLARITY,
            vibration_direction=VibrationDirection.EXPANDING,
            elemental_influence={
                ElementalForce.EARTH: 0.7,
                ElementalForce.AIR: 1.5,  # Strong air (freedom) influence
                ElementalForce.FIRE: 1.3,
                ElementalForce.WATER: 0.8
            }
        )
        
        self.rules["liberty"] = CellularRule(
            name="Law of Liberty",
            description="Expands cells' range of influence while maintaining overall harmony",
            parameters=liberty_params,
            vector_transformation="repel",
            esoteric_correspondence="The Thelemic Law of Liberty - 'Do what thou wilt shall be the whole of the Law, Love is the law, love under will'"
        )
        
        # Define ritual sequences
        
        # Thelemic Initiation Sequence
        initiation_sequence = RuleSequence(
            name="Thelemic Initiation",
            description="A sequence embodying the initiation into discovering one's True Will",
            rules=[
                self.rules["tower"],     # Breaking down preconceptions
                self.rules["hermit"],    # Inner contemplation
                self.rules["true_will"], # Discovering one's path
                self.rules["liberty"],   # Embracing freedom
                self.rules["star"]       # Finding higher guidance
            ],
            esoteric_correspondence="The Thelemic initiatory process of breaking down false conceptions, discovering one's True Will, and attaining spiritual illumination"
        )
        
        # Knowledge and Conversation Sequence
        k_and_c_sequence = RuleSequence(
            name="Knowledge and Conversation",
            description="A sequence representing attainment of the Knowledge and Conversation of the Holy Guardian Angel",
            rules=[
                self.rules["hermit"],    # Solitary preparation
                self.rules["lovers"],    # Union with higher self
                self.rules["true_will"], # Alignment with divine purpose
                self.rules["star"],      # Divine guidance
                self.rules["sun"]        # Illumination and clarity
            ],
            esoteric_correspondence="The mystical attainment of the Knowledge and Conversation of the Holy Guardian Angel - communion with one's higher self"
        )
        
        # Store sequences
        self.sequences = {
            "initiation": initiation_sequence,
            "k_and_c": k_and_c_sequence
        }
        
        logger.info(f"Defined {len(self.rules)} rules and {len(self.sequences)} sequences")
    
    def _prepare_system(self):
        """Initialize the automata system."""
        logger.info("Initializing automata system...")
        
        # Create system with Thelemic evolution pattern
        self.system = AutomataSystem(
            manifold=self.manifold,
            rules_dict=self.rules,
            sequences_dict=self.sequences,
            evolution_pattern=EvolutionPattern.THELEMIC,
            save_path=str(self.output_dir / "system_states")
        )
        
        logger.info("Automata system initialized")
    
    def perform_ritual(self):
        """
        Perform the complete ritual working.
        
        This executes the sequence of transformations that embody
        the ritual intent.
        """
        if not self.system:
            raise ValueError("System not initialized. Call prepare_components() first.")
            
        logger.info(f"Beginning ritual performance: {self.ritual_name}")
        
        # Track key terms to observe their transformation
        key_terms = [
            "will", "thelema", "love", "magick", "ritual", 
            "knowledge", "angel", "spirit", "truth", "light"
        ]
        self._initialize_term_tracking(key_terms)
        
        # Phase 1: Preparation - Apply the Tower rule to break down existing structures
        logger.info("Phase 1: Preparation - Breaking down existing structures")
        self.system.rules["tower"].apply(self.manifold, self.system.generation)
        self.system.generation += 1
        self._update_term_positions()
        self._visualize_manifold("phase1_preparation")
        
        # Phase 2: Contemplation - Apply the Hermit rule for introspection
        logger.info("Phase 2: Contemplation - Introspection and self-analysis")
        self.system.rules["hermit"].apply(self.manifold, self.system.generation)
        self.system.generation += 1
        self._update_term_positions()
        self._visualize_manifold("phase2_contemplation")
        
        # Phase 3: Invocation - Apply the Knowledge and Conversation sequence
        logger.info("Phase 3: Invocation - Knowledge and Conversation")
        self.system.apply_sequence("k_and_c")
        self._update_term_positions()
        self._visualize_manifold("phase3_invocation")
        
        # Phase 4: Transformation - Apply True Will rule repeatedly
        logger.info("Phase 4: Transformation - Discovering True Will")
        for i in range(3):  # Apply three times to emphasize the transformation
            self.system.rules["true_will"].apply(self.manifold, self.system.generation)
            self.system.generation += 1
            self._update_term_positions()
        self._visualize_manifold("phase4_transformation")
        
        # Phase 5: Integration - Apply Liberty rule for expansion
        logger.info("Phase 5: Integration - Embracing the Law of Liberty")
        self.system.rules["liberty"].apply(self.manifold, self.system.generation)
        self.system.generation += 1
        self._update_term_positions()
        self._visualize_manifold("phase5_integration")
        
        # Phase 6: Illumination - Apply Star rule for guidance
        logger.info("Phase 6: Illumination - Receiving astral guidance")
        self.system.rules["star"].apply(self.manifold, self.system.generation)
        self.system.generation += 1
        self._update_term_positions()
        self._visualize_manifold("phase6_illumination")
        # Final state
        logger.info("Ritual working completed")
        self._visualize_manifold("final_state")
        
        # Create animation of the entire evolution
        self._create_evolution_animation(key_terms)
        
        # Analyze semantic shifts
        self._analyze_semantic_shifts(key_terms)
    
    def _initialize_term_tracking(self, terms: List[str]):
        """Initialize tracking of term positions through the ritual."""
        self.key_term_trajectories = {term: [] for term in terms}
        for term in terms:
            cell = self.manifold.get_term_cell(term)
            if cell:
                # Store initial reduced position if available
                if self.manifold.reduced:
                    term_idx = self.manifold.get_term_cell(term).id
                    if term_idx is not None:
                        position = self.manifold.reduced.points[term_idx].copy()
                        self.key_term_trajectories[term].append(position)
                    else:
                        logger.warning(f"Term '{term}' not found in index")
                        # Add null position for consistency
                        self.key_term_trajectories[term].append(None)
                else:
                    logger.warning(f"Manifold has no reduced representation. Cannot track '{term}' position.")
                    self.key_term_trajectories[term].append(None)
            else:
                logger.warning(f"Term '{term}' not found in any cell, cannot track trajectory")
                # Keep the term in the dictionary but mark as untrackable
                self.key_term_trajectories[term].append(None)
        
        logger.info(f"Initialized tracking for {len(terms)} terms, found {sum(1 for t in terms if self.key_term_trajectories[t][0] is not None)} valid positions")
        
    def _update_term_positions(self):
        """
        Update term positions after a manifold evolution step.
        This method appends the new positions to the term trajectories.
        """
        # Skip if manifold has no reduced representation
        if not self.manifold.reduced:
            logger.warning("Cannot update term positions: manifold has no reduced representation")
            return
            
        # Update position for each tracked term
        tracked_count = 0
        for term in self.key_term_trajectories.keys():
            cell = self.manifold.get_term_cell(term)
            if cell:
                term_idx = self.manifold.get_term_cell(term).id
                if term_idx is not None:
                    # Get the current position and append to trajectory
                    position = self.manifold.reduced.points[term_idx].copy()
                    self.key_term_trajectories[term].append(position)
                    tracked_count += 1
                else:
                    logger.warning(f"Cannot update position for '{term}': term not found in index")
                    # Add null position to maintain list structure
                    self.key_term_trajectories[term].append(None)
            else:
                logger.warning(f"Cannot update position for '{term}': term not found in any cell")
                # Add null position to maintain list structure
                self.key_term_trajectories[term].append(None)
                
        logger.debug(f"Updated positions for {tracked_count}/{len(self.key_term_trajectories)} tracked terms")
        
    def _visualize_manifold(self, phase_name):
        """
        Create a visualization of the current manifold state and tracked terms.
        
        Args:
            phase_name: Name of the ritual phase for labeling the visualization
        
        Returns:
            Path to the saved visualization file
        """
        if not self.manifold.reduced:
            logger.warning("Cannot visualize manifold: no reduced representation available")
            return None
            
        # Create frame directory if it doesn't exist
        frame_dir = self.output_dir / "frames"
        if not frame_dir.exists():
            frame_dir.mkdir(parents=True, exist_ok=True)
            
        # Define Thelemic color scheme
        # Red (Mars/Energy), Gold (Sun/Will), Blue (Jupiter/Wisdom), Purple (Saturn/Binah)
        colors = {
            "background": "#111111",  # Dark background representing the void
            "points": "#FFD700",      # Gold for general terms
            "cells": "#3A5FCD",       # Royal blue for cell boundaries
            "tracked": "#FF3300",     # Bright red for tracked terms
            "trajectories": "#9932CC"  # Purple for trajectories
        }
        
        # Set up the figure with dark background
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111)
        ax.set_facecolor(colors["background"])
        
        # Plot all reduced points
        points = self.manifold.reduced.points
        plt.scatter(
            points[:, 0], points[:, 1], 
            c=colors["points"], alpha=0.6, s=15, 
            edgecolors='none'
        )
        
        # Plot cell boundaries if available
        if hasattr(self.manifold.reduced, 'boundaries') and self.manifold.reduced.boundaries:
            boundaries = self.manifold.reduced.boundaries
            if hasattr(boundaries, 'ridge_vertices'):
                # For Voronoi diagrams
                for simplex in boundaries.ridge_vertices:
                    if -1 not in simplex:  # Skip if ridge extends to infinity
                        vertices = boundaries.vertices[simplex]
                        plt.plot(
                            vertices[:, 0], vertices[:, 1], 
                            '-', color=colors["cells"], linewidth=0.7, alpha=0.7
                        )
        
        # Highlight and label the key tracked terms
        for term, positions in self.key_term_trajectories.items():
            if positions and positions[-1] is not None:
                current_pos = positions[-1]
                
                # Plot the current position with a larger marker
                plt.scatter(
                    current_pos[0], current_pos[1], 
                    c=colors["tracked"], s=100, 
                    edgecolors='white', linewidth=1, zorder=10
                )
                
                # Add a text label
                plt.annotate(
                    term,
                    (current_pos[0], current_pos[1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    weight='bold',
                    color='white',
                    backgroundcolor=colors["background"]
                )
                
                # Plot trajectory if there are multiple positions
                if len(positions) > 1:
                    # Filter out None values
                    valid_positions = [p for p in positions if p is not None]
                    if len(valid_positions) > 1:
                        trajectory = np.array(valid_positions)
                        plt.plot(
                            trajectory[:, 0], trajectory[:, 1],
                            '-', color=colors["trajectories"], 
                            linewidth=1.5, alpha=0.8, zorder=8
                        )
        
        # Add title and generation info
        title = f"Ritual of {self.ritual_name} - {phase_name.replace('_', ' ').title()}"
        subtitle = f"Generation {self.system.generation}"
        
        plt.title(
            f"{title}\n{subtitle}", 
            fontsize=16, 
            color='white', 
            pad=20
        )
        
        # Add signature (93 93/93 is a Thelemic greeting)
        plt.figtext(
            0.01, 0.01, "93 93/93", 
            color='white', 
            fontsize=8, 
            alpha=0.7
        )
        
        # Set infinity as axis limits to auto-scale
        plt.axis('equal')
        plt.tight_layout()
        
        # Create filename based on generation and phase
        filename = f"{self.system.generation:04d}_{phase_name}.png"
        filepath = frame_dir / filename
        
        # Save the figure
        plt.savefig(filepath, dpi=150, facecolor=colors["background"])
        plt.close()
        logger.info(f"Visualization saved: {filepath}")
        return filepath
        
    def _analyze_semantic_shifts(self, key_terms):
        """
        Analyze semantic shifts of key terms during the ritual evolution.
        
        Computes distances traveled in vector space, directional changes,
        and creates visualizations of the transformations.
        
        Args:
            key_terms: List of key terms to analyze
            
        Returns:
            Path to the analysis report file
        """
        logger.info("Analyzing semantic shifts of key terms during the ritual...")
        # Check if we have trajectories for the terms
        if not self.key_term_trajectories:
            logger.warning("No term trajectories available for analysis")
            return None
            
        # Create a directory for analysis outputs
        analysis_dir = self.output_dir / "analysis"
        if not analysis_dir.exists():
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
        # Create a report file
        report_path = analysis_dir / "semantic_shifts_report.txt"
        html_report_path = analysis_dir / "semantic_shifts_report.html"
        
        # Prepare data for analysis
        valid_terms = []
        trajectories = []
        
        for term in key_terms:
            positions = self.key_term_trajectories.get(term, [])
            # Filter out None values and ensure we have at least start and end points
            valid_positions = [p for p in positions if p is not None]
            
            if len(valid_positions) >= 2:
                valid_terms.append(term)
                trajectories.append(valid_positions)
                
        if not valid_terms:
            logger.warning("No valid term trajectories available for analysis")
            return None
            
        # Calculate metrics for each term
        results = []
        
        for i, term in enumerate(valid_terms):
            positions = trajectories[i]
            
            # 1. Calculate total distance traveled
            total_distance = 0.0
            for j in range(1, len(positions)):
                segment_distance = np.linalg.norm(positions[j] - positions[j-1])
                total_distance += segment_distance
                
            # 2. Calculate direct displacement (start to end)
            direct_displacement = np.linalg.norm(positions[-1] - positions[0])
            
            # 3. Calculate efficiency ratio (direct / total)
            efficiency = direct_displacement / total_distance if total_distance > 0 else 0.0
            
            # 4. Calculate directional changes
            direction_changes = 0
            prev_direction = None
            
            for j in range(1, len(positions) - 1):
                # Calculate direction vectors between consecutive points
                dir1 = positions[j] - positions[j-1]
                dir2 = positions[j+1] - positions[j]
                
                # Normalize vectors
                norm1 = np.linalg.norm(dir1)
                norm2 = np.linalg.norm(dir2)
                
                if norm1 > 0 and norm2 > 0:
                    dir1 = dir1 / norm1
                    dir2 = dir2 / norm2
                    
                    # Calculate angle between directions (cosine similarity)
                    cos_similarity = np.dot(dir1, dir2)
                    # Clamp to [-1, 1] to avoid numerical issues
                    cos_similarity = max(-1.0, min(1.0, cos_similarity))
                    
                    # If angle > 30 degrees, consider it a direction change
                    if cos_similarity < 0.866:  # cos(30°) ≈ 0.866
                        direction_changes += 1
            
            # 5. Calculate semantic neighborhood changes - which other terms became closer/farther
            initial_neighbors = self._find_nearest_terms(term, positions[0], valid_terms, 3)
            final_neighbors = self._find_nearest_terms(term, positions[-1], valid_terms, 3)
            
            # Find terms that moved closer or farther
            moved_closer = [t for t in final_neighbors if t not in initial_neighbors]
            moved_away = [t for t in initial_neighbors if t not in final_neighbors]
            
            # Store results
            results.append({
                'term': term,
                'total_distance': total_distance,
                'direct_displacement': direct_displacement,
                'efficiency': efficiency,
                'direction_changes': direction_changes,
                'initial_position': positions[0],
                'final_position': positions[-1],
                'initial_neighbors': initial_neighbors,
                'final_neighbors': final_neighbors,
                'moved_closer': moved_closer,
                'moved_away': moved_away
            })
        
        # Generate text report
        with open(report_path, 'w') as f:
            f.write(f"SEMANTIC SHIFTS ANALYSIS: {self.ritual_name}\n")
            f.write(f"===================================================\n\n")
            f.write(f"Ritual intent: {self.ritual_intent}\n")
            f.write(f"Time: {self.start_time} to {datetime.datetime.now()}\n")
            f.write(f"Generations: {self.system.generation}\n\n")
            
            f.write("TERM TRANSFORMATIONS:\n")
            f.write("--------------------\n\n")
            
            # Sort terms by displacement (highest first)
            sorted_results = sorted(results, key=lambda x: x['direct_displacement'], reverse=True)
            
            for result in sorted_results:
                f.write(f"TERM: {result['term'].upper()}\n")
                f.write(f"  Total path length: {result['total_distance']:.4f}\n")
                f.write(f"  Direct displacement: {result['direct_displacement']:.4f}\n")
                f.write(f"  Path efficiency: {result['efficiency']:.2f}\n")
                f.write(f"  Direction changes: {result['direction_changes']}\n")
                f.write(f"  Initial neighbors: {', '.join(result['initial_neighbors'])}\n")
                f.write(f"  Final neighbors: {', '.join(result['final_neighbors'])}\n")
                f.write(f"  Terms moved closer: {', '.join(result['moved_closer'])}\n")
                f.write(f"  Terms moved away: {', '.join(result['moved_away'])}\n\n")
            
            # Summary statistics
            avg_displacement = np.mean([r['direct_displacement'] for r in results])
            max_displacement = max([r['direct_displacement'] for r in results])
            min_displacement = min([r['direct_displacement'] for r in results])
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("------------------\n")
            f.write(f"Average displacement: {avg_displacement:.4f}\n")
            f.write(f"Maximum displacement: {max_displacement:.4f} (term: {sorted_results[0]['term']})\n")
            f.write(f"Minimum displacement: {min_displacement:.4f}\n\n")
            
            f.write("RITUAL INTERPRETATION:\n")
            f.write("---------------------\n")
            f.write("The transformations of key terms reflect the energetic currents and Will-forces\n")
            f.write("active during the ritual. Terms with high displacement or directional changes\n")
            f.write("were most affected by the ritual current, indicating areas of maximal influence.\n\n")
            
            f.write("93 93/93\n")  # Thelemic closing
        
        # Create visualization of the term transformations
        self._create_transformation_visualization(valid_terms, trajectories, analysis_dir)
        
        # Create higher-dimensional visualizations using HyperTools
        self._create_hypertools_visualizations(valid_terms, trajectories, analysis_dir)
    
    def _create_hypertools_visualizations(self, terms, trajectories, output_dir):
        """
        Create high-dimensional visualizations of term transformations.
        
        Args:
            terms: List of term names
            trajectories: List of trajectory arrays
            output_dir: Directory to save visualizations
        """
        logger.info("Creating high-dimensional ritual visualizations")
        
        # Initialize Simple3D visualizer
        visualizer_dir = self.output_dir / "3d_visualizations" 
        visualizer_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizer with 3D output
        visualizer = Simple3DVisualizer(
            output_dir=str(visualizer_dir),
            color_palette="plasma",
            n_dimensions=3
        )
        
        # Format trajectories for visualization
        term_trajectory_dict = {terms[i]: trajectories[i] for i in range(len(terms))}
        
        # Create a 3D visualization of the term evolutions
        phase_names = [
            "Initial State", 
            "Preparation",
            "Contemplation", 
            "Invocation", 
            "Transformation", 
            "Integration", 
            "Illumination"
        ]
        
        # 1. Create static visualization of term evolution paths
        html_path = visualizer.visualize_term_evolution(
            term_trajectory_dict, 
            phase_names=phase_names,
            title=f"3D Semantic Evolution in {self.ritual_name}",
            color_by_phase=False
        )
        
        # 2. Create static visualization by phase for comparison
        html_path_phases = visualizer.visualize_term_evolution(
            term_trajectory_dict, 
            phase_names=phase_names,
            title=f"Ritual Phases in {self.ritual_name}",
            color_by_phase=True
        )
        
        # 3. Create animated visualization
        animation_path = visualizer.create_animated_ritual(
            term_trajectory_dict,
            phase_names=phase_names,
            title=f"Animated Ritual Evolution - {self.ritual_name}",
            duration=15.0,  # 15 seconds animation
            fps=30,
            add_trails=True
        )
        
        # Create links to these visualizations in the analysis directory
        with open(output_dir / "hypertools_visualizations.html", 'w') as f:
            f.write(f"""<!DOCTYPE html>
            <html>
            <head>
                <title>HyperTools Visualizations - {self.ritual_name}</title>
                <style>
                    body {{ background-color: #111; color: #DDD; font-family: sans-serif; padding: 20px; }}
                    h1, h2 {{ color: #FFD700; }}
                    .viz-link {{ 
                        display: block; 
                        background-color: #222; 
                        padding: 15px; 
                        margin: 15px 0; 
                        color: #FFF; 
                        text-decoration: none;
                        border-radius: 5px;
                        border-left: 5px solid #FFD700;
                    }}
                    .viz-link:hover {{ background-color: #333; }}
                </style>
            </head>
            <body>
                <h1>High-Dimensional Visualizations of {self.ritual_name}</h1>
                <p>These visualizations provide deeper insights into the semantic transformations occurring during the ritual.</p>
                
                <h2>Available Visualizations</h2>
                <a class="viz-link" href="../hypertools_visualizations/term_evolution_*.html" target="_blank">
                    3D Term Evolution - View the transformation of key terms in 3D space
                </a>
                
                <a class="viz-link" href="../hypertools_visualizations/ritual_animation_*.mp4" target="_blank">
                    Animated Ritual Evolution - Watch the dynamic evolution of terms through ritual phases
                </a>
                
                <p style="margin-top: 30px; font-style: italic; color: #666; text-align: center;">
                    "Do what thou wilt shall be the whole of the Law.<br>
                    Love is the law, love under will."<br>
                    93 93/93
                </p>
            </body>
            </html>""")
            
        logger.info(f"Created 3D visualizations in {visualizer_dir}")
        
        # Get a reference to the report files created in the analysis_dir
        report_path = output_dir / "semantic_shifts_report.txt"
        html_report_path = output_dir / "semantic_shifts_report.html"
        
        # Calculate metrics for each term (as in _analyze_semantic_shifts)
        # This is needed for the HTML report
        results = []
        for i, term in enumerate(terms):
            positions = trajectories[i]
            
            # Calculate metrics similar to _analyze_semantic_shifts
            total_distance = sum(np.linalg.norm(positions[j] - positions[j-1]) for j in range(1, len(positions)))
            direct_displacement = np.linalg.norm(positions[-1] - positions[0])
            efficiency = direct_displacement / total_distance if total_distance > 0 else 0.0
            
            # For simplicity, we'll skip the detailed calculations for the HTML report
            results.append({
                'term': term,
                'total_distance': total_distance,
                'direct_displacement': direct_displacement,
                'efficiency': efficiency,
                'direction_changes': 0,  # simplified
                'initial_neighbors': [],
                'final_neighbors': [],
                'moved_closer': [],
                'moved_away': []
            })
            
        # Sort results by displacement (highest first) for the HTML report
        sorted_results = sorted(results, key=lambda x: x['direct_displacement'], reverse=True)
        
        # Create HTML report with the semantic shift results
        self._create_html_report(sorted_results, html_report_path)
        
        logger.info(f"Semantic shifts analysis saved to: {report_path}")
        return report_path
    
    def _find_nearest_terms(self, target_term, position, all_terms, n=3):
        """
        Find the n nearest terms to a given position.
        
        Args:
            target_term: The term to exclude from the search
            position: The position vector to search around
            all_terms: List of all terms to consider
            n: Number of neighbors to find
            
        Returns:
            List of nearest term names
        """
        distances = []
        
        for term in all_terms:
            if term == target_term:
                continue
                
            # Get the current position of this term
            term_positions = self.key_term_trajectories.get(term, [])
            valid_positions = [p for p in term_positions if p is not None]
            
            if valid_positions:
                # Use the latest position
                term_pos = valid_positions[-1]
                distance = np.linalg.norm(position - term_pos)
                distances.append((term, distance))
        
        # Sort by distance and take top n
        distances.sort(key=lambda x: x[1])
        return [term for term, _ in distances[:n]]
    
    def _create_transformation_visualization(self, terms, trajectories, output_dir):
        """
        Create visualizations of term transformations.
        
        Args:
            terms: List of term names
            trajectories: List of trajectory arrays
            output_dir: Directory to save visualizations
        """
        # Create a vector field visualization showing the transformations
        plt.figure(figsize=(15, 12))
        ax = plt.subplot(111)
        ax.set_facecolor("#111111")  # Dark background
        
        # Plot all terms and their trajectories
        for i, term in enumerate(terms):
            positions = np.array(trajectories[i])
            
            # Plot trajectory
            plt.plot(
                positions[:, 0], positions[:, 1],
                '-', linewidth=1.5, alpha=0.7,
                label=None
            )
            
            # Plot start point (with different color)
            plt.scatter(
                positions[0, 0], positions[0, 1],
                c='green', s=80, alpha=0.7,
                edgecolors='white', linewidth=1
            )
            
            # Plot end point
            plt.scatter(
                positions[-1, 0], positions[-1, 1],
                c='red', s=100, alpha=0.8,
                edgecolors='white', linewidth=1
            )
            
            # Add arrow to show direction
            arrow_start = positions[-2] if len(positions) > 1 else positions[0]
            arrow_end = positions[-1]
            
            # Calculate the direction vector
            direction = arrow_end - arrow_start
            norm = np.linalg.norm(direction)
            
            if norm > 0.01:  # Only draw meaningful arrows
                # Scale the arrow
                arrow_len = norm * 0.3
                direction = direction / norm * arrow_len
                
                # Draw the arrow
                plt.arrow(
                    arrow_end[0] - direction[0], arrow_end[1] - direction[1],
                    direction[0], direction[1],
                    head_width=0.1, head_length=0.15,
                    color='white', alpha=0.7
                )
            
            # Add term label at end position
            plt.annotate(
                term,
                (positions[-1, 0], positions[-1, 1]),
                xytext=(10, 5),
                textcoords='offset points',
                fontsize=11,
                weight='bold',
                color='white',
                backgroundcolor="#111111"
            )
        
        # Add title and legend
        plt.title(
            f"Semantic Transformations in {self.ritual_name}",
            fontsize=18,
            color='white',
            pad=20
        )
        
        # Add some interpretive text
        plt.figtext(
            0.02, 0.02,
            "Green: Initial positions | Red: Final positions\n"
            "Arrows indicate transformation direction and magnitude",
            color='white',
            fontsize=10
        )
        
        # Add "Do what thou wilt" quote
        plt.figtext(
            0.98, 0.02,
            "Do what thou wilt shall be the whole of the Law",
            color='#FFD700',  # Gold
            fontsize=10,
            ha='right'
        )
        
        plt.axis('equal')
        plt.tight_layout()
        
        # Save the visualization
        viz_path = output_dir / "term_transformations.png"
        plt.savefig(viz_path, dpi=150, facecolor="#111111")
        plt.close()
        
        logger.info(f"Transformation visualization saved to: {viz_path}")
        
    def _create_html_report(self, results, output_path):
        """
        Create an HTML report with interactive visualizations.
        
        Args:
            results: Analysis results
            output_path: Path to save the HTML file
        """
        # Basic HTML report with styling
        html = f'''<!DOCTYPE html>
        <html>
        <head>
            <title>Thelemic Ritual Analysis: {self.ritual_name}</title>
            <style>
                body {{
                    background-color: #111;
                    color: #DDD;
                    font-family: "Helvetica Neue", Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #FFD700;
                    border-bottom: 1px solid #333;
                    padding-bottom: 10px;
                }}
                .container {{
                    max-width: 900px;
                    margin: 0 auto;
                }}
                .term-card {{
                    background-color: #222;
                    border: 1px solid #444;
                    border-radius: 5px;
                    margin: 15px 0;
                    padding: 15px;
                }}
                .term-header {{
                    font-size: 1.5em;
                    color: #FF3300;
                    margin-bottom: 10px;
                }}
                .metric {{
                    margin: 5px 0;
                }}
                .highlight {{
                    color: #FFD700;
                }}
                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    font-style: italic;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Thelemic Ritual Analysis: {self.ritual_name}</h1>
                <p>Ritual intent: {self.ritual_intent}</p>
                <p>Time: {self.start_time} to {datetime.datetime.now()}</p>
                <p>Generations: {self.system.generation}</p>
                
                <h2>Term Transformations</h2>
'''
        
        # Add term cards
        for result in sorted(results, key=lambda x: x['direct_displacement'], reverse=True):
            html += f"""
                <div class="term-card">
                    <div class="term-header">{result['term'].upper()}</div>
                    <div class="metric">Total path length: <span class="highlight">{result['total_distance']:.4f}</span></div>
                    <div class="metric">Direct displacement: <span class="highlight">{result['direct_displacement']:.4f}</span></div>
                    <div class="metric">Path efficiency: <span class="highlight">{result['efficiency']:.2f}</span></div>
                    <div class="metric">Direction changes: <span class="highlight">{result['direction_changes']}</span></div>
                    <div class="metric">Initial neighbors: {', '.join(result['initial_neighbors'])}</div>
                    <div class="metric">Final neighbors: {', '.join(result['final_neighbors'])}</div>
                    <div class="metric">Terms moved closer: <span class="highlight">{', '.join(result['moved_closer'])}</span></div>
                    <div class="metric">Terms moved away: <span class="highlight">{', '.join(result['moved_away'])}</span></div>
                </div>
"""

        # Add summary and closing
        avg_displacement = np.mean([r['direct_displacement'] for r in results])
        max_displacement = max([r['direct_displacement'] for r in results])
        max_term = max(results, key=lambda x: x['direct_displacement'])['term']
        min_displacement = min([r['direct_displacement'] for r in results])

        html += f"""
                <h2>Summary Statistics</h2>
                <div class="term-card">
                    <div class="metric">Average displacement: <span class="highlight">{avg_displacement:.4f}</span></div>
                    <div class="metric">Maximum displacement: <span class="highlight">{max_displacement:.4f}</span> (term: {max_term})</div>
                    <div class="metric">Minimum displacement: <span class="highlight">{min_displacement:.4f}</span></div>
                </div>
                
                <h2>Ritual Interpretation</h2>
                <div class="term-card">
                    <p>The transformations of key terms reflect the energetic currents and Will-forces
                    active during the ritual. Terms with high displacement or directional changes
                    were most affected by the ritual current, indicating areas of maximal influence.</p>
                </div>
                
                <div class="footer">
                    "Do what thou wilt shall be the whole of the Law.<br>
                    Love is the law, love under will."<br>
                    93 93/93
                </div>
            </div>
        </body>
        </html>
"""

        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html)
            
        logger.info(f"HTML report saved to: {output_path}")
        
    def get_cell_neighbors(self, idx, grid_shape, radius=1, wrap=True):
        """
        Get the indices of neighboring cells in a grid.
        
        Args:
            idx: 1D index of the cell
            grid_shape: Tuple of (rows, columns) defining the grid shape
            radius: Radius of neighborhood (default=1 for Moore neighborhood)
            wrap: Whether to wrap around at edges (default=True)
            
        Returns:
            List of indices of neighboring cells
        """
        rows, cols = grid_shape
        # Convert 1D index to 2D coordinates
        row = idx // cols
        col = idx % cols
        
        neighbors = []
        
        # Check all positions within the radius
        for r in range(-radius, radius + 1):
            for c in range(-radius, radius + 1):
                # Skip the center cell itself
                if r == 0 and c == 0:
                    continue
                    
                # Calculate neighbor coordinates
                neighbor_row = row + r
                neighbor_col = col + c
                
                # Apply wrapping if enabled
                if wrap:
                    neighbor_row = neighbor_row % rows
                    neighbor_col = neighbor_col % cols
                # Otherwise check if the neighbor is within bounds
                elif neighbor_row < 0 or neighbor_row >= rows or neighbor_col < 0 or neighbor_col >= cols:
                    continue
                
                # Convert 2D coordinates back to 1D index
                neighbor_idx = neighbor_row * cols + neighbor_col
                neighbors.append(neighbor_idx)
        
        return neighbors
        
    def _ingest_liber_aleph(self):
        """
        Parse and extract semantic patterns from Liber Aleph PDF.
        
        This method extracts key concepts, relationships, and Thelemic correspondences
        from Liber Aleph to enhance the manifold structure and ritual rules.
        """
        logger.info("Ingesting Liber Aleph for enhanced Thelemic patterns...")
        
        pdf_path = Path("./data/Libera Aleph vel cxi The book of Wisdom or Folly.pdf")
        if not pdf_path.exists():
            logger.warning(f"Liber Aleph PDF not found at {pdf_path}. Skipping ingestion.")
            return
            
        # Initialize NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK resources...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        # Extract text from PDF
        extracted_text = ""
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    extracted_text += page.extract_text()
            
            # Create a directory for storing extracted content
            aleph_dir = self.output_dir / "liber_aleph"
            aleph_dir.mkdir(parents=True, exist_ok=True)
            
            # Save raw extracted text
            with open(aleph_dir / "extracted_text.txt", 'w', encoding='utf-8') as f:
                f.write(extracted_text)
                
            logger.info(f"Extracted {len(extracted_text)} characters from Liber Aleph")
            
            # Process the text to extract semantic patterns
            self._process_liber_aleph_text(extracted_text, aleph_dir)
            
        except Exception as e:
            logger.error(f"Error extracting text from Liber Aleph: {e}")
    
    def _process_liber_aleph_text(self, text, output_dir):
        """
        Process the extracted Liber Aleph text to identify semantic patterns.
        
        Args:
            text: The extracted text from Liber Aleph
            output_dir: Directory to save processing results
        """
        # Tokenize text
        sentences = sent_tokenize(text)
        
        # Extract chapters and verses (Liber Aleph is structured in short chapters)
        chapters = []
        current_chapter = {"title": "", "content": ""}
        
        for sentence in sentences:
            # Typical chapter titles in Liber Aleph are in uppercase and short
            if sentence.isupper() and len(sentence.split()) <= 7:
                # Save previous chapter if it has content
                if current_chapter["title"] and current_chapter["content"]:
                    chapters.append(current_chapter)
                    
                # Start new chapter
                current_chapter = {"title": sentence.strip(), "content": ""}
            else:
                current_chapter["content"] += sentence + " "
        
        # Add the last chapter
        if current_chapter["title"] and current_chapter["content"]:
            chapters.append(current_chapter)
        
        # Save structured chapters
        with open(output_dir / "structured_chapters.json", 'w', encoding='utf-8') as f:
            json.dump(chapters, f, indent=2)
            
        logger.info(f"Extracted {len(chapters)} chapters from Liber Aleph")
        
        # Extract key terms and their contexts
        thelemic_terms = set([
            "thelema", "will", "love", "law", "liberty", "light", "life", "stars", 
            "aiwass", "horus", "nuit", "hadit", "beast", "babalon", "aeon", 
            "magick", "ritual", "knowledge", "wisdom", "folly", "truth"
        ])
        
        # Identify term relationships and contexts
        term_contexts = defaultdict(list)
        term_relationships = defaultdict(lambda: defaultdict(int))
        
        # Process each chapter to extract semantic patterns
        for chapter in chapters:
            # Tokenize content
            words = word_tokenize(chapter["content"].lower())
            words = [w for w in words if w.isalpha() and w not in stopwords.words('english')]
            
            # Find key terms in this chapter
            chapter_terms = set(words).intersection(thelemic_terms)
            
            # Record contexts for each term
            for term in chapter_terms:
                term_contexts[term].append({
                    "chapter": chapter["title"],
                    "excerpt": chapter["content"][:200] + "..." if len(chapter["content"]) > 200 else chapter["content"]
                })
                
            # Record co-occurrences (relationships between terms)
            term_list = list(chapter_terms)
            for i in range(len(term_list)):
                for j in range(i + 1, len(term_list)):
                    term1, term2 = term_list[i], term_list[j]
                    term_relationships[term1][term2] += 1
                    term_relationships[term2][term1] += 1
        
        # Save term contexts and relationships
        with open(output_dir / "term_contexts.json", 'w', encoding='utf-8') as f:
            json.dump(dict(term_contexts), f, indent=2)
            
        with open(output_dir / "term_relationships.json", 'w', encoding='utf-8') as f:
            json.dump({k: dict(v) for k, v in term_relationships.items()}, f, indent=2)
            
        # Enhance the embedding model with these relationships
        self._enhance_embeddings_with_liber_aleph(term_relationships)
        
        logger.info(f"Processed Liber Aleph: found {len(term_contexts)} key terms with {sum(len(v) for v in term_contexts.values())} contexts")
    
    def _enhance_embeddings_with_liber_aleph(self, term_relationships):
        """
        Enhance word embeddings based on term relationships from Liber Aleph.
        
        Args:
            term_relationships: Dict mapping terms to their related terms with co-occurrence counts
        """
        # Define Thelemic correspondences based on Liber Aleph
        thelemic_correspondences = {
            "will": {"type": CellType.PLANETARY, "value": 9},  # Solar principle
            "love": {"type": CellType.ELEMENTAL, "value": 6},  # Venus principle
            "wisdom": {"type": CellType.SEPHIROTIC, "value": 2},  # Chokmah
            "folly": {"type": CellType.TAROT, "value": 0},  # The Fool
            "liberty": {"type": CellType.ZODIACAL, "value": 11},  # Aquarius
            "law": {"type": CellType.TAROT, "value": 8},  # Justice/Adjustment
            "truth": {"type": CellType.PLANETARY, "value": 1},  # Mercury
        }
        
        # Store these correspondences for later use in the manifold
        self.thelemic_correspondences = thelemic_correspondences
        
        # Add additional terms to embeddings based on Liber Aleph's unique vocabulary
        liber_aleph_terms = set(term_relationships.keys())
        existing_terms = set(self.embeddings.terms)
        new_terms = liber_aleph_terms - existing_terms
        
        if new_terms:
            logger.info(f"Adding {len(new_terms)} new terms from Liber Aleph to embeddings")
            self.embeddings.load_terms(new_terms)
    
    def _integrate_cessation_patterns(self):
        """
        Parse and integrate wave patterns and recursion structures from the cessation simulations.
        
        This enhances the manifold's topology with standing wave patterns and fractal coherence
        structures from the cessation simulation models.
        """
        logger.info("Integrating cessation simulation patterns...")
        
        cessation_dir = Path("./data/cessation-simulations")
        if not cessation_dir.exists():
            logger.warning(f"Cessation simulations directory not found at {cessation_dir}. Skipping integration.")
            return
            
        # Create directory for extracted patterns
        pattern_dir = self.output_dir / "cessation_patterns"
        # Create directory for extracted patterns
        pattern_dir = self.output_dir / "cessation_patterns"
        pattern_dir.mkdir(parents=True, exist_ok=True)
        
        # Maps to store extracted patterns
        wave_patterns = []
        recursive_patterns = []
        coupled_kernels = []
        
        # Process HTML files to extract wave patterns and fractal structures
        html_files = {
            "standing_waves": cessation_dir / "standing_waves.html",
            "fractal_recursive": cessation_dir / "fractal_recursive_coherence.html",
            "coupled_kernels": cessation_dir / "coupled_kernels.html",
            "wave_control": cessation_dir / "4D_wave_control.html"
        }
        
        # Extract patterns from each file
        for pattern_type, file_path in html_files.items():
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            logger.info(f"Processing {pattern_type} pattern from {file_path}")
            
            try:
                # Parse the HTML file
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Use BeautifulSoup to parse the HTML
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract embedded JavaScript that contains pattern data
                script_tags = soup.find_all('script')
                js_content = ""
                for script in script_tags:
                    if script.string and any(term in script.string for term in ['pattern', 'wave', 'fractal', 'kernel', 'frequency']):
                        js_content += script.string
                
                # Save the extracted script content
                with open(pattern_dir / f"{pattern_type}_script.js", 'w', encoding='utf-8') as f:
                    f.write(js_content)
                
                # Extract pattern data based on pattern type
                if pattern_type == "standing_waves":
                    # Extract wave parameters (frequency, amplitude, phase)
                    wave_params = self._extract_wave_parameters(js_content)
                    wave_patterns.extend(wave_params)
                    
                    # Save extracted patterns
                    with open(pattern_dir / "wave_patterns.json", 'w', encoding='utf-8') as f:
                        json.dump(wave_params, f, indent=2)
                    
                elif pattern_type == "fractal_recursive":
                    # Extract fractal parameters (depth, branching factor, scaling)
                    fractal_params = self._extract_fractal_parameters(js_content)
                    recursive_patterns.extend(fractal_params)
                    
                    # Save extracted patterns
                    with open(pattern_dir / "fractal_patterns.json", 'w', encoding='utf-8') as f:
                        json.dump(fractal_params, f, indent=2)
                
                elif pattern_type == "coupled_kernels":
                    # Extract coupling parameters (strength, topology, resonance)
                    coupling_params = self._extract_coupling_parameters(js_content)
                    coupled_kernels.extend(coupling_params)
                    
                    # Save extracted patterns
                    with open(pattern_dir / "kernel_patterns.json", 'w', encoding='utf-8') as f:
                        json.dump(coupling_params, f, indent=2)
                
                elif pattern_type == "wave_control":
                    # Extract 4D wave control parameters
                    control_params = self._extract_control_parameters(js_content)
                    
                    # Save extracted patterns
                    with open(pattern_dir / "control_patterns.json", 'w', encoding='utf-8') as f:
                        json.dump(control_params, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error processing {pattern_type}: {e}")
        
        # Apply the extracted patterns to the manifold topology
        self._apply_patterns_to_manifold(wave_patterns, recursive_patterns, coupled_kernels)
        
        logger.info(f"Integrated cessation patterns: {len(wave_patterns)} wave patterns, {len(recursive_patterns)} recursive patterns, {len(coupled_kernels)} coupled kernels")
    
    def _extract_wave_parameters(self, js_content):
        """
        Extract standing wave parameters from JavaScript content.
        
        Args:
            js_content: JavaScript code from the standing_waves.html file
            
        Returns:
            List of wave parameter dictionaries
        """
        wave_params = []
        
        # Look for wave parameter patterns in the JavaScript code
        # Example patterns: frequency = X, amplitude = Y, phase = Z
        freq_pattern = re.compile(r'frequency\s*=\s*([\d.]+)')
        ampl_pattern = re.compile(r'amplitude\s*=\s*([\d.]+)')
        phase_pattern = re.compile(r'phase\s*=\s*([\d.]+)')
        wave_pattern = re.compile(r'createWave\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)')
        
        # Extract individual parameters
        frequencies = [float(match) for match in freq_pattern.findall(js_content)]
        amplitudes = [float(match) for match in ampl_pattern.findall(js_content)]
        phases = [float(match) for match in phase_pattern.findall(js_content)]
        
        # Extract wave creation calls
        wave_calls = wave_pattern.findall(js_content)
        
        # Process explicit wave creation calls
        for freq, ampl, phase in wave_calls:
            wave_params.append({
                "frequency": float(freq),
                "amplitude": float(ampl),
                "phase": float(phase),
                "type": "explicit"
            })
            
        # If we found individual parameters but not complete wave calls,
        # make reasonable combinations
        if not wave_calls and frequencies and amplitudes:
            for i, freq in enumerate(frequencies):
                ampl = amplitudes[i % len(amplitudes)]
                phase = phases[i % len(phases)] if phases else 0.0
                
                wave_params.append({
                    "frequency": freq,
                    "amplitude": ampl,
                    "phase": phase,
                    "type": "inferred"
                })
        
        # If nothing specific found, create default standing wave patterns
        if not wave_params:
            # Create default harmonic series (common in Thelemic vibration theory)
            for i in range(1, 8):  # First 7 harmonics
                wave_params.append({
                    "frequency": i,
                    "amplitude": 1.0 / i,
                    "phase": 0.0,
                    "type": "default"
                })
                
        return wave_params
    
    def _extract_fractal_parameters(self, js_content):
        """
        Extract fractal recursion parameters from JavaScript content.
        
        Args:
            js_content: JavaScript code from the fractal_recursive_coherence.html file
            
        Returns:
            List of fractal parameter dictionaries
        """
        fractal_params = []
        
        # Look for fractal parameter patterns
        depth_pattern = re.compile(r'depth\s*=\s*(\d+)')
        branch_pattern = re.compile(r'branches\s*=\s*(\d+)')
        scale_pattern = re.compile(r'scale\s*=\s*([\d.]+)')
        angle_pattern = re.compile(r'angle\s*=\s*([\d.]+)')
        
        # Extract parameters
        depths = [int(match) for match in depth_pattern.findall(js_content)]
        branches = [int(match) for match in branch_pattern.findall(js_content)]
        scales = [float(match) for match in scale_pattern.findall(js_content)]
        angles = [float(match) for match in angle_pattern.findall(js_content)]
        
        # Create parameter sets
        if depths and branches:
            for i, depth in enumerate(depths):
                branch = branches[i % len(branches)]
                scale = scales[i % len(scales)] if scales else 0.7  # Default scale
                angle = angles[i % len(angles)] if angles else 45.0  # Default angle
                
                fractal_params.append({
                    "depth": depth,
                    "branches": branch,
                    "scale": scale,
                    "angle": angle,
                    "type": "extracted"
                })
        
        # If nothing found, create default fractal parameters
        if not fractal_params:
            # Create Thelemic fractal parameters (based on Tree of Life structure)
            fractal_params.append({
                "depth": 3,
                "branches": 3,  # Triadic structure
                "scale": 0.618,  # Golden ratio
                "angle": 120.0,  # 360/3
                "type": "thelemic_triad"
            })
            
            fractal_params.append({
                "depth": 4,
                "branches": 4,  # Tetragrammaton structure
                "scale": 0.5,
                "angle": 90.0,  # 360/4
                "type": "thelemic_tetragrammaton"
            })
            
            fractal_params.append({
                "depth": 7,
                "branches": 2,  # Binary tree (pillars of the Tree of Life)
                "scale": 0.7,
                "angle": 45.0,
                "type": "thelemic_tree"
            })
            
        return fractal_params
    
    def _extract_coupling_parameters(self, js_content):
        """
        Extract coupled kernel parameters from JavaScript content.
        
        Args:
            js_content: JavaScript code from the coupled_kernels.html file
            
        Returns:
            List of coupling parameter dictionaries
        """
        coupling_params = []
        
        # Look for coupling parameter patterns
        strength_pattern = re.compile(r'couplingStrength\s*=\s*([\d.]+)')
        nodes_pattern = re.compile(r'nodes\s*=\s*(\d+)')
        topology_pattern = re.compile(r'topology\s*=\s*[\'"](\w+)[\'"]')
        
        # Extract parameters
        strengths = [float(match) for match in strength_pattern.findall(js_content)]
        nodes_list = [int(match) for match in nodes_pattern.findall(js_content)]
        topologies = topology_pattern.findall(js_content)
        
        # Create parameter sets
        if strengths:
            for i, strength in enumerate(strengths):
                nodes = nodes_list[i % len(nodes_list)] if nodes_list else 7  # Default nodes
                topology = topologies[i % len(topologies)] if topologies else "ring"  # Default topology
                
                coupling_params.append({
                    "coupling_strength": strength,
                    "nodes": nodes,
                    "topology": topology,
                    "type": "extracted"
                })
        
        # If nothing found, create default coupling parameters
        if not coupling_params:
            # Create Thelemic coupling parameters
            coupling_params.append({
                "coupling_strength": 0.93,  # Thelemic number significance
                "nodes": 11,  # Master number in numerology
                "topology": "star",  # Star pattern (reference to the Star card)
                "type": "thelemic_star"
            })
            
            coupling_params.append({
                "coupling_strength": 0.777,  # Significant number in Thelema
                "nodes": 10,  # Number of Sephiroth
                "topology": "tree",  # Tree of Life structure
                "type": "thelemic_tree"
            })
            
            coupling_params.append({
                "coupling_strength": 0.418,  # Golden ratio (reversed)
                "nodes": 22,  # Number of paths on the Tree of Life / Major Arcana
                "topology": "mesh",  # Interconnected network
                "type": "thelemic_mesh"
            })
            
        return coupling_params

    def _extract_control_parameters(self, js_content):
        """
        Extract 4D wave control parameters from JavaScript content.
        
        Args:
            js_content: JavaScript code from the 4D_wave_control.html file
            
        Returns:
            Dictionary of control parameters
        """
        # Look for control parameter patterns
        dimension_pattern = re.compile(r'dimensions\s*=\s*(\d+)')
        control_pattern = re.compile(r'controlPoints\s*=\s*(\d+)')
        frequency_pattern = re.compile(r'controlFrequency\s*=\s*([\d.]+)')
        
        # Extract parameters
        dimensions = [int(match) for match in dimension_pattern.findall(js_content)]
        control_points = [int(match) for match in control_pattern.findall(js_content)]
        frequencies = [float(match) for match in frequency_pattern.findall(js_content)]
        
        # Create parameter dictionary
        control_params = {
            "dimensions": dimensions[0] if dimensions else 4,
            "control_points": control_points[0] if control_points else 7,
            "control_frequency": frequencies[0] if frequencies else 0.93,
            "harmonic_series": [1.0, 2.0, 3.0, 5.0, 7.0, 11.0],  # Prime number harmonics
            "type": "extracted" if dimensions or control_points or frequencies else "default"
        }
        
        return control_params
    
    def _apply_patterns_to_manifold(self, wave_patterns, recursive_patterns, coupled_kernels):
        """
        Apply the extracted cessation patterns to the manifold topology.
        
        This method uses the extracted wave, recursive, and coupling patterns
        to inform and enhance the manifold's structure, adjusting cell relationships
        and evolution rules based on the pattern properties.
        
        Args:
            wave_patterns: List of wave parameter dictionaries
            recursive_patterns: List of fractal parameter dictionaries
            coupled_kernels: List of coupling parameter dictionaries
        """
        if not (wave_patterns or recursive_patterns or coupled_kernels):
            logger.warning("No patterns to apply to manifold")
            return
            
        logger.info("Applying cessation patterns to manifold topology")
        
        # 1. Apply wave patterns to cell vibration frequencies
        if wave_patterns:
            # Get representative wave pattern parameters
            wave_freqs = [w["frequency"] for w in wave_patterns]
            wave_ampls = [w["amplitude"] for w in wave_patterns]
            wave_phases = [w["phase"] for w in wave_patterns]
            
            # Configure frequency-based cell relationships
            for i, cell_id in enumerate(self.manifold.cells.keys()):
                # Assign wave parameters to cells cyclically
                freq_idx = i % len(wave_freqs)
                cell = self.manifold.cells[cell_id]
                
                # Store wave properties in cell metadata
                if not hasattr(cell, 'metadata'):
                    cell.metadata = {}
                    
                cell.metadata["wave_frequency"] = wave_freqs[freq_idx]
                cell.metadata["wave_amplitude"] = wave_ampls[freq_idx]
                cell.metadata["wave_phase"] = wave_phases[freq_idx]
                
                # Adjust numerological value based on wave frequency
                # (Higher frequencies correlate with higher vibration states in Thelemic theory)
                freq_factor = wave_freqs[freq_idx] / max(wave_freqs)
                if freq_factor > 0.8 and cell.numerological_value not in [11, 22, 33]:
                    # Boost to a master number for high frequency cells
                    cell.numerological_value = 11
                    
        # 2. Apply fractal patterns to define hierarchical cell relationships
        if recursive_patterns:
            # Select a primary fractal pattern model
            primary_pattern = max(recursive_patterns, key=lambda p: p["depth"] * p["branches"])
            
            # Use the fractal structure to define special relationships between cells
            depth = primary_pattern["depth"]
            branches = primary_pattern["branches"]
            
            # Create a fractal-inspired neighborhood map
            fractal_neighborhoods = {}
            
            # Root cell (typically center of manifold)
            root_cell_id = list(self.manifold.cells.keys())[0]
            
            # Build fractal tree levels
            current_level = [root_cell_id]
            all_cells = list(self.manifold.cells.keys())
            
            # Use remaining cells for branches
            remaining_cells = all_cells[1:]
            
            for level in range(1, depth + 1):
                next_level = []
                
                # For each cell in the current level, add branch children
                for parent in current_level:
                    # Add up to 'branches' children for this parent
                    children = []
                    for _ in range(branches):
                        if not remaining_cells:
                            break
                        children.append(remaining_cells.pop(0))
                    
                    # Store parent-child relationships
                    fractal_neighborhoods[parent] = children
                    next_level.extend(children)
                    
                current_level = next_level
                if not remaining_cells:
                    break
                    
            # Store this fractal structure for use in ritual rules
            self.fractal_neighborhoods = fractal_neighborhoods
                
        # 3. Apply coupled kernel patterns to evolution rules
        if coupled_kernels:
            # Extract coupling topologies for cell interaction patterns
            coupling_strengths = [c["coupling_strength"] for c in coupled_kernels]
            coupling_topologies = [c["topology"] for c in coupled_kernels]
            
            # Create weighted edge lists based on coupling patterns
            edge_weights = {}
            
            # Process each coupling pattern
            for i, topology in enumerate(coupling_topologies):
                strength = coupling_strengths[i % len(coupling_strengths)]
                
                # Create edge weights based on topology
                if topology == "ring":
                    # Ring topology: each cell connected to its neighbors in a circle
                    cells = list(self.manifold.cells.keys())
                    for j in range(len(cells)):
                        next_idx = (j + 1) % len(cells)
                        edge = (cells[j], cells[next_idx])
                        edge_weights[edge] = strength
                        
                elif topology == "star":
                    # Star topology: central cell connected to all others
                    cells = list(self.manifold.cells.keys())
                    center = cells[0]
                    for other in cells[1:]:
                        edge = (center, other)
                        edge_weights[edge] = strength
                        
                elif topology == "mesh" or topology == "tree":
                    # Use existing neighborhood relationships but with coupling weights
                    for cell_id, neighbors in self.manifold.cell_neighbors.items():
                        for neighbor in neighbors:
                            edge = (cell_id, neighbor)
                            reverse_edge = (neighbor, cell_id)
                            
                            # Ensure we don't duplicate edges
                            if edge not in edge_weights and reverse_edge not in edge_weights:
                                edge_weights[edge] = strength
            
            # Store coupling edge weights for ritual rule application
            self.coupling_edge_weights = edge_weights
            
            # Modify the True Will rule to incorporate these coupling patterns
            if "true_will" in self.rules:
                # Enhanced parameters based on coupling patterns
                self.rules["true_will"].parameters.coupling_weights = edge_weights
                logger.info("Enhanced True Will rule with cessation coupling patterns")

    def _get_term_position(self, term: str) -> Optional[np.ndarray]:
        """Get the position of a term in the reduced space."""
        term_idx = self.manifold.term_to_index.get(term)
        if term_idx is not None:
            return self.manifold.reduced.points[term_idx]
        return None

    def _get_term_vector(self, term: str) -> Optional[np.ndarray]:
        """Get the vector representation of a term."""
        term_idx = self.manifold.term_to_index.get(term)
        if term_idx is not None:
            return self.embeddings.vectors[term_idx]
        return None

def main():
    """ Demonstrate the ritual evolution process. """
    ritual = RitualWorking(
        ritual_name="True Will Discovery",
        ritual_intent="To discover and align with one's True Will through semantic transformation"
    )
    
    # Prepare components
    ritual.prepare_components()
    
    # Perform the ritual
    ritual.perform_ritual()
    
    logger.info("Ritual demonstration completed")
    

if __name__ == "__main__":
    main()
