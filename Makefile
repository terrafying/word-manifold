# Word Manifold Project Makefile

.PHONY: setup test clean examples docs lint

# Python environment settings
PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin

# Installation settings
REQUIREMENTS := requirements.txt

# Output directories
VISUALIZATIONS := visualizations
TEST_OUTPUTS := test_outputs
DOCS := docs

# Default target
all: setup test

# Setup virtual environment and install dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e .
	$(BIN)/pip install -r $(REQUIREMENTS)

# Run tests
test:
	$(BIN)/pytest tests/ -v --cov=word_manifold

# Run linting
lint:
	$(BIN)/flake8 src/word_manifold
	$(BIN)/mypy src/word_manifold

# Clean generated files and caches
clean:
	rm -rf $(VISUALIZATIONS)/* $(TEST_OUTPUTS)/*
	rm -rf .pytest_cache .coverage
	rm -rf **/__pycache__
	rm -rf build/ dist/ *.egg-info/

# Run examples
examples: force-field ritual semantic-crystal

force-field:
	$(BIN)/python src/word_manifold/examples/force_field_demo.py

ritual:
	$(BIN)/python src/word_manifold/examples/hyperdimensional_ritual.py

semantic-crystal:
	$(BIN)/python src/word_manifold/examples/semantic_crystallization.py

# Generate documentation
docs:
	$(BIN)/pdoc --html --output-dir $(DOCS) src/word_manifold

# Development watch mode - auto-rerun tests on file changes
watch:
	$(BIN)/python dev_watch.py

# Install development dependencies
dev-setup: setup
	$(BIN)/pip install -r requirements-dev.txt 