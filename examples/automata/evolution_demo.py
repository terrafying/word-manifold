"""
Demonstration of the cellular automata evolution system.

This example shows:
1. Setting up an automata system
2. Applying different evolution rules
3. Running rule sequences
4. Capturing and analyzing system states
"""

import logging
from pathlib import Path
from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.manifold.vector_manifold import VectorManifold
from word_manifold.automata.system import AutomataSystem, EvolutionPattern
from word_manifold.automata.cellular_rules import create_predefined_rules, create_predefined_sequences

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_system():
    """Set up the automata system with initial terms and rules."""
    # Initialize embeddings with occult terms
    embeddings = WordEmbeddings()
    terms = {
        "thelema", "will", "love", "magick", "ritual",
        "knowledge", "wisdom", "power", "light", "dark",
        "earth", "air", "fire", "water", "spirit",
        "sun", "moon", "mercury", "venus", "mars"
    }
    embeddings.load_terms(terms)
    
    # Create vector manifold
    manifold = VectorManifold(embeddings)
    
    # Create automata system
    rules = create_predefined_rules()
    sequences = create_predefined_sequences()
    
    system = AutomataSystem(
        manifold=manifold,
        rules_dict=rules,
        sequences_dict=sequences,
        evolution_pattern=EvolutionPattern.THELEMIC,
        save_path="ritual_outputs"
    )
    
    return system

def demonstrate_evolution(system: AutomataSystem):
    """Demonstrate different evolution patterns."""
    # Individual rule evolution
    logger.info("\nApplying individual rules:")
    for rule_name in ["great_work", "equilibrium", "tower"]:
        rule = system.rules[rule_name]
        logger.info("Applying rule: %s" % rule.name)
        logger.info("Description: %s" % rule.description)
        rule.apply(system.manifold, system.generation)
        state = system.manifold.get_manifold_state()
        # logger.info(f"System state after rule: {state}")
        
    # Sequence evolution
    logger.info("\nApplying rule sequences:")
    for seq_name in ["great_work", "thelemic"]:
        sequence = system.sequences[seq_name]
        logger.info("Applying sequence: %s" % sequence.name)
        logger.info("Description: %s" % sequence.description)
        sequence.apply(system.manifold, system.generation)
        state = system.get_history_summary()

        logger.info("System state after sequence: %s" % state)
        
    # Multiple generation evolution
    logger.info("\nEvolving system for multiple generations:")
    system.evolve(generations=5)
    logger.info(f"Final generation: {system.generation}")
    logger.info("Final state: %s" % system.get_history_summary())

def main():
    # Set up system
    system = setup_system()
    logger.info("Initialized automata system")
    
    # Demonstrate evolution
    demonstrate_evolution(system)
    
    logger.info("\nEvolution demonstration complete")

if __name__ == "__main__":
    main() 