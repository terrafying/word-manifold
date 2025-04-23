#!/usr/bin/env python3
"""
Setup for word-manifold package.

This package implements cellular automata in word vector space,
exploring the intersection of linguistics, cellular automata,
and occult symbolism.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="word-manifold",
    version="0.1.0",
    author="E. Van till",
    author_email="example@example.com",
    description="Cellular Automata in Word Vector Space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terrafying/word-manifold",
    package_dir={"": "src"},
    # Find all packages under src (word_manifold and its subpackages)
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=2.0.2",
        "torch>=2.6.0",
        "transformers>=4.51.3",
        "spacy>=3.7.4",
        "scikit-learn>=1.6.1",
        "pandas>=2.2.3",
        "matplotlib>=3.9.4",
        "seaborn>=0.13.2",
        "plotly>=6.0.1",
        "networkx>=3.2.1",
        "nltk>=3.9.1",
        "beautifulsoup4>=4.13.4",
        "requests>=2.32.3",
        "tqdm>=4.67.1",
        "safetensors>=0.5.3",
        "huggingface-hub>=0.30.2",
        # Enhanced visualization dependencies
        "dash>=2.15.0",
        "dash-core-components>=2.0.0",
        "dash-html-components>=2.0.0",
        "dash-renderer>=1.9.1",
        "dash-table>=5.0.0",
        "scipy>=1.13.1",  # For advanced texture generation
        "pillow>=11.2.1",  # For image processing
        "ffmpeg-python>=0.2.0",  # For video generation
        "imageio>=2.37.0",  # For GIF creation
        "umap-learn>=0.5.7",  # For dimensionality reduction
        "sounddevice>=0.5.1",  # For audio feedback (optional)
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.5",
            "pytest-cov>=6.1.1",
            "black>=25.1.0",
            "isort>=6.0.1",
            "flake8>=7.2.0",
            "mypy>=1.15.0"
        ],
        "interactive": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.1.2",
            "ipython>=8.22.2",
            "jupyterlab>=4.1.5"
        ]
    },
    entry_points={
        "console_scripts": [
            "word-manifold=word_manifold.visualization.cli:cli",
            "word-manifold-visualize=word_manifold.visualization.cli:visualize",
            "word-manifold-animate=word_manifold.visualization.cli:animate_ritual",
            "word-manifold-serve=word_manifold.visualization.cli:serve"
        ]
    }
)

