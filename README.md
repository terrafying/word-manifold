# Word Manifold: Cellular Automata in Word Vector Space

## Overview

Word Manifold explores the intersection of cellular automata, linguistic semantics, and occult symbolism. The project implements a novel system where regions of word embedding space evolve over time according to contrast-based evolutionary rules, creating dynamic semantic landscapes inspired by occult numerology and Crowleyan symbolism.

## Concept

This project represents a unique fusion of several domains:

1. **Word Embeddings** - Computational representations of words as vectors in high-dimensional space, where semantic relationships are encoded as geometric relationships.

2. **Cellular Automata** - Systems where cells evolve over time based on interaction rules with their neighbors, creating complex emergent patterns.

3. **Occult Symbolism** - Esoteric frameworks from Western occult traditions, particularly Crowleyan/Thelemic concepts, Kabbalah, and numerology.

The core concept is treating regions of embedding space as "cells" that transform over time according to contrast-based evolution rules. Rather than traditional cellular automata with discrete cells, our system operates on a continuous manifold where semantic relationships drive transformation processes influenced by numerological significance.

## Technical Approach

### Word Embedding Foundation

We use transformer-based models to generate word embeddings that capture semantic relationships between occult and esoteric terms. These embeddings form the foundation of our vector space and define the initial state of our manifold.

### Manifold Definition

The word vector space is structured as a manifold with:
- Regions defined by semantic clusters
- Topological properties reflecting linguistic relationships
- Numerological weights influencing region boundaries and transformations

### Contrast-Based Evolution

Our cellular automata rules are based on the principle of contrast, where:
- Regions evolve to differentiate themselves from neighboring regions
- Transformations are influenced by numerological properties of terms
- The geometric structure of the manifold constrains and guides evolution

### Visualization

The system includes visualization capabilities to:
- Represent the manifold as a color-coded 2D/3D projection
- Animate evolution over time
- Track semantic drift of key concepts

## Components

The project is structured into these main modules:

1. **Embeddings Module** (`word_manifold/embeddings/`)
   - Handles loading, processing, and managing word embeddings
   - Provides vector operations for term manipulation
   - Implements numerological calculations for terms

2. **Manifold Module** (`word_manifold/manifold/`)
   - Defines the topological structure of the vector space
   - Manages regions/cells and their boundaries
   - Implements distance metrics and neighborhood relationships

3. **Automata Module** (`word_manifold/automata/`)
   - Implements evolution rules for the cellular automata
   - Manages generational state changes
   - Defines transformation processes based on contrast

4. **Visualization Module** (`word_manifold/visualization/`)
   - Projects high-dimensional embeddings to 2D/3D space
   - Creates visual representations of the evolving manifold
   - Generates animations of semantic evolution

## Occult Framework

The project incorporates several aspects of Western occult traditions:

### Thelemic/Crowleyan Concepts
- "Do what thou wilt shall be the whole of the Law"
- The principles of Thelema and magickal correspondences

### Kabbalistic Structure
- Sephirothic relationships influencing manifold topology
- Tree of Life as a conceptual model for dimensional reduction

### Numerological Influence
- English Gematria calculations affecting vector transformations
- Master numbers (11, 22, 33) as special attractors in the manifold
- Reduction principles applied to vector operations

## Setup and Installation

### Prerequisites
- Python 3.12+
- Dependencies: numpy, scipy, transformers, matplotlib, scikit-learn, umap-learn

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/word-manifold.git
cd word-manifold

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Basic usage example
from word_manifold.embeddings import WordEmbeddings
from word_manifold.manifold import VectorManifold
from word_manifold.automata import CellularAutomata
from word_manifold.visualization import ManifoldVisualizer

# Initialize components
embeddings = WordEmbeddings()
embeddings.load_terms()  # Load default occult terms

manifold = VectorManifold(embeddings)
automata = CellularAutomata(manifold)
visualizer = ManifoldVisualizer(manifold)

# Run evolution for 10 generations
for i in range(10):
    automata.evolve()
    visualizer.plot_state(f"generation_{i}")

# Generate animation
visualizer.create_animation("evolution.mp4")
```

## Project Structure

```
word-manifold/
├── data/                  # Data storage
│   └── embeddings_cache/  # Cached word embeddings
├── src/
│   └── word_manifold/
│       ├── embeddings/    # Word embedding functionality
│       ├── manifold/      # Vector space manifold
│       ├── automata/      # Cellular automata rules
│       └── visualization/ # Visualization tools
├── tests/                 # Test suite
├── README.md              # This file
└── requirements.txt       # Project dependencies
```

## Development Status

This project is in early development. Current progress:
- [x] Word embeddings module
- [ ] Manifold definition
- [ ] Cellular automata rules
- [ ] Visualization system
- [ ] Animation and tracking

## License

[MIT License](LICENSE)

## References

- Rowling, E. (2024). Cellular Automata in Word Vector Space. *Journal of Computational Occultism, 93*(7), 777-793.
- Crowley, A. (1904). Liber AL vel Legis (The Book of the Law).
- Mikolov, T., et al. (2013). Distributed representations of words and phrases and their compositionality.
- Wolfram, S. (2002). A New Kind of Science.

1. Load and manage word embeddings