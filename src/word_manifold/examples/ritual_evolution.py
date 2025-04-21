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
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Dict, List, Set, Tuple, Any
import datetime

# Add the parent directory to sys.path if running as a standalone script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from word_manifold.embeddings.word_embeddings import WordEmbeddings, OCCULT_TERMS
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


class RitualWorking:
    """
    A class that implements a complete magical working in word vector space.
    """
    
    def __init__(self, ritual_name: str, ritual_intent: str):
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
        
        # Components to be initialized
        self.embeddings = None
        self.manifold = None
        self.rules = None
        self.sequences = None
        self.system = None
        
        # Metrics and results
        self.transformation_metrics = []
        self.key_term_trajectories = {}
        
        logger.info(f"Beginning ritual working: {ritual_name}")
        logger.info(f"Magical intent: {ritual_intent}")
        logger.info(f"Outputs will be saved to: {self.output_dir}")
        
        # Record ritual setup
        with open(self.output_dir / "ritual_intent.txt", "w") as f:
            f.write(f"RITUAL: {ritual_name}\n")
            f.write(f"INTENT: {ritual_intent}\n")
            f.write(f"TIME: {self.start_time.isoformat()}\n")
    
    def prepare_components(self):
        """
        Prepare all system components for the ritual.
        """
        logger.info("Preparing ritual components...")
        
        # Step 1: Initialize word embeddings with extended occult terms
        self._prepare_embeddings()
        
        # Step 2: Create the vector manifold
        self._prepare_manifold()
        
        # Step 3: Define ritual rules and sequences
        self._prepare_rules()
        
        # Step 4: Initialize automata system
        self._prepare_system()
        
        logger.info("All ritual components prepared")
    
    def _prepare_embeddings(self):
        """Prepare word embeddings with occult terminology."""
        logger.info("Initializing word embeddings...")
        
        # Create WordEmbeddings instance
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
        extended_terms = OCCULT_TERMS.union(thelemic_terms)
        
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
        self._visualize_manifold("initial_state")
        
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
            "knowledge", "angel", "spirit", "self", "light"
        ]
        self._initialize_term_tracking(key_terms)
        
        # Phase 1: Preparation - Apply the Tower rule to break down existing structures
        logger.info("Phase 1: Preparation - Breaking down existing structures")
        self.system.rules["tower"].apply(self.manifold, self.system.generation)
        self.system.generation += 1
        self._update_term_positions(key_terms)
        self._visualize_manifold("phase1_preparation")
        
        # Phase 2: Contemplation - Apply the Hermit rule for introspection
        logger.info("Phase 2: Contemplation - Introspection and self-analysis")
        self.system.rules["hermit"].apply(self.manifold, self.system.generation)
        self.system.generation += 1
        self._update_term_positions(key_terms)
        self._visualize_manifold("phase2_contemplation")
        
        # Phase 3: Invocation - Apply the Knowledge and Conversation sequence
        logger.info("Phase 3: Invocation - Knowledge and Conversation")
        self.system.apply_sequence("k_and_c")
        self._update_term_positions(key_terms)
        self._visualize_manifold("phase3_invocation")
        
        # Phase 4: Transformation - Apply True Will rule repeatedly
        logger.info("Phase 4: Transformation - Discovering True Will")
        for i in range(3):  # Apply three times to emphasize the transformation
            self.system.rules["true_will"].apply(self.manifold, self.system.generation)
            self.system.generation += 1
            self._update_term_positions(key_terms)
        self._visualize_manifold("phase4_transformation")
        
        # Phase 5: Integration - Apply Liberty rule for expansion
        logger.info("Phase 5: Integration - Embracing the Law of Liberty")
        self.system.rules["liberty"].apply(self.manifold, self.system.generation)
        self.system.generation += 1
        self._update_term_positions(key_terms)
        self._visualize_manifold("phase5_integration")
        
        # Phase 6: Illumination - Apply Star rule for guidance
        logger.info("Phase 6: Illumination - Receiving astral guidance")
        self.system.rules["star"].apply(self.manifold, self.system.generation)
        self.system.generation += 1
        self._update_term_positions(key_terms)
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
                    term_idx = self.manifold.term_to_idx.get(term)
                    if term_idx is not None:
                        position = self.manifold.reduced.points[term_idx].copy()
                        self.key_term_trajectories[term].append(position)
                    else:
                        logger.warning(f"Term '{term}' not found in index")
            else:

