"""
Demonstration of the WordEmbeddings class functionality.

This example shows:
1. Loading and initializing word embeddings
2. Computing embeddings for occult terms
3. Finding similar terms using FAISS
4. Computing numerological significance
5. Saving and loading embeddings
"""

import logging
from pathlib import Path
from word_manifold.embeddings.word_embeddings import WordEmbeddings, OCCULT_TERMS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize embeddings
    embeddings = WordEmbeddings()
    logger.info("Initialized word embeddings model")
    
    # Load standard occult terms
    embeddings.load_terms(OCCULT_TERMS)
    logger.info(f"Loaded {len(OCCULT_TERMS)} occult terms")
    
    # Demonstrate similarity search
    query_terms = ["magic", "thelema", "ritual"]
    k = 5  # Number of similar terms to find
    
    for query in query_terms:
        logger.info(f"\nFinding terms similar to '{query}':")
        similar_terms = embeddings.find_similar_terms(query, k=k)
        for term, distance in similar_terms:
            logger.info(f"  {term}: {distance:.4f}")
            
    # Demonstrate numerological significance
    test_terms = ["abrahadabra", "thelema", "magic", "ritual"]
    logger.info("\nNumerological significance:")
    for term in test_terms:
        value = embeddings.find_numerological_significance(term)
        logger.info(f"  {term}: {value}")
        
    # Save embeddings
    save_path = Path("data/embeddings/occult_embeddings.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings.save_embeddings(save_path)
    logger.info(f"\nSaved embeddings to {save_path}")
    
    # Load embeddings in a new instance
    new_embeddings = WordEmbeddings()
    new_embeddings.load_saved_embeddings(save_path)
    logger.info(f"Loaded {len(new_embeddings.terms)} terms from saved embeddings")
    
    # Verify loaded embeddings work
    query = "magic"
    similar_terms = new_embeddings.find_similar_terms(query, k=3)
    logger.info(f"\nVerifying loaded embeddings with query '{query}':")
    for term, distance in similar_terms:
        logger.info(f"  {term}: {distance:.4f}")

if __name__ == "__main__":
    main() 