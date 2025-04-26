# Word Manifold

A Python package for exploring semantic manifolds and state spaces using Z3 and Llama Stack, with a focus on symbolic reasoning, geometric interpretation of language, and temporal analysis.

## Features

- **Semantic Manifold Exploration**: Analyze and visualize the geometric structure of semantic spaces
- **State Space Analysis**: Use Z3 for symbolic reasoning about language states and transitions
- **Time Series Analysis**: Track semantic evolution and detect patterns in text
- **Agent-Based Exploration**: Leverage Llama Stack for natural language understanding
- **Vector Embeddings**: Generate and analyze embeddings using modern language models
- **Visualization Tools**: Interactive visualization of semantic spaces and state transitions

## Research Applications

Word Manifold provides a platform for research in several areas:

### 1. Semantic Space Analysis
- Geometric properties of word embeddings
- Topological analysis of semantic spaces
- Clustering and community detection
- Dimensionality reduction techniques

### 2. State Space Reasoning
- Symbolic reasoning about language states
- Constraint satisfaction in semantic spaces
- Verification of semantic properties
- Model checking for language systems

### 3. Temporal Analysis
- Semantic evolution tracking
- Pattern detection in text
- Temporal relationship analysis
- Change point detection

### 4. Natural Language Understanding
- Semantic relationship discovery
- Context-aware reasoning
- Multi-modal semantic analysis
- Cross-lingual semantic mapping

## Research Opportunities

We welcome research contributions in several forms:

1. **Methodology Development**:
   - New approaches to semantic space analysis
   - Novel state space reasoning techniques
   - Improved temporal analysis methods
   - Enhanced visualization techniques

2. **Applications**:
   - Natural language processing
   - Information retrieval
   - Text analysis
   - Knowledge representation

3. **Theoretical Work**:
   - Mathematical foundations
   - Algorithmic improvements
   - Complexity analysis
   - Formal verification

## Citing Word Manifold

If you use Word Manifold in your research, please cite:

```bibtex
@software{word_manifold,
  author = {Your Name},
  title = {Word Manifold: A Package for Semantic Space Analysis},
  year = {2024},
  url = {https://github.com/yourusername/word-manifold}
}
```

## Architecture

The package is built around several core components:

1. **Manifold Analysis**:
   - Basic manifold exploration using vector embeddings
   - Z3-based state space analysis
   - Time series analysis for semantic evolution
   - Agent-based exploration using Llama Stack

2. **Embedding Management**:
   - Word embedding generation and storage
   - Similarity computation
   - Nearest neighbor search

3. **CLI Interface**:
   - Command-line tools for all major features
   - Interactive exploration capabilities
   - Visualization options

## Installation

```bash
pip install word-manifold
```

For development installation:

```bash
git clone https://github.com/yourusername/word-manifold.git
cd word-manifold
pip install -e ".[dev,ml,data,viz]"
```

## Quick Start

### Basic Manifold Exploration

```python
from word_manifold import ManifoldExplorer
from word_manifold.embeddings import get_embeddings

# Generate embeddings for a set of words
words = ["cat", "dog", "bird", "fish", "mammal", "reptile"]
embeddings = get_embeddings(words)

# Create a manifold explorer
explorer = ManifoldExplorer(embeddings)

# Analyze the geometric structure
structure = explorer.analyze_structure()

# Visualize the manifold
explorer.visualize()
```

### Time Series Analysis

```python
from word_manifold.manifold.timeseries import SemanticTimeSeriesAnalyzer

# Create analyzer
analyzer = SemanticTimeSeriesAnalyzer(window_size=5, stride=1)

# Analyze passage
passage = "The cat chased the mouse through the garden. The mouse escaped into a hole."
result = analyzer.analyze_passage(passage, embeddings)

# View results
print(result["patterns"])
print(result["shifts"])
```

### Agent-Based Exploration

```python
from word_manifold.manifold.agent_explorer import AgentManifoldExplorer
from llama_stack_client import Agent

# Create agent
agent = Agent(
    model="meta-llama/Llama-3-70b-chat",
    instructions="You are a semantic space explorer."
)

# Create explorer
explorer = AgentManifoldExplorer(agent)

# Explore semantic space
result = explorer.explore_semantic_space(words, session_id)
```

## CLI Usage

### Basic Commands

```bash
# Explore semantic manifold
word-manifold explore cat dog bird fish mammal reptile

# Find semantic path
word-manifold path cat dog

# Analyze passage
word-manifold analyze-passage "The cat chased the mouse through the garden."

# Agent-based exploration
word-manifold agent-explore cat dog bird fish mammal reptile
```

### Advanced Options

```bash
# Use specific model
word-manifold explore --model all-MiniLM-L6-v2 cat dog bird

# Save embeddings
word-manifold explore --output embeddings.npy cat dog bird

# Custom window size for time series
word-manifold analyze-passage --window-size 10 "The cat chased the mouse."
```

## Core Concepts

### Semantic Manifolds

A semantic manifold is a geometric representation of how words and concepts relate to each other in a high-dimensional space. This package provides tools to:

- Analyze the curvature and topology of semantic spaces
- Identify clusters and boundaries between concepts
- Explore paths and geodesics between words

### State Space Exploration

Using Z3, we can perform symbolic reasoning about language states and transitions:

- Define constraints on word relationships
- Analyze possible state transitions
- Verify properties of semantic spaces

### Time Series Analysis

The time series analysis component helps understand semantic evolution:

- Track semantic changes over time
- Detect significant shifts in meaning
- Identify patterns and periodicity
- Build evolution graphs

### Agent-Based Analysis

The Llama Stack integration provides:

- Natural language understanding of relationships
- Contextual path finding
- Combined verification using Z3 and LLM reasoning

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to the project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.