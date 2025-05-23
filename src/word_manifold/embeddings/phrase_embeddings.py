"""
Phrase and Sentence Embedding Module.

This module handles the embedding of complete phrases and sentences,
extracting both their semantic content and structural/emotional shape.
Uses state-of-the-art language models for improved accuracy and efficiency.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import spacy
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from collections import Counter, defaultdict
from spacy.tokens import Token
from functools import lru_cache
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer
import gc

# Model configuration
BACKUP_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Using smaller model as backup

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatters and handlers if they don't exist
if not logger.handlers:
    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create debug file handler
    debug_handler = logging.FileHandler('debug_embeddings.log')
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(funcName)s:%(lineno)d - %(message)s'
    )
    debug_handler.setFormatter(debug_formatter)
    logger.addHandler(debug_handler)

# Lazy loading of spaCy model
_spacy_model = None

def get_spacy_model(model_name: str = 'en_core_web_lg') -> spacy.language.Language:
    """Load spaCy model with fallback options."""
    global _spacy_model
    if _spacy_model is None:
        try:
            logger.info(f"Loading spaCy model '{model_name}'")
            _spacy_model = spacy.load(model_name)
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not found, downloading...")
            try:
                spacy.cli.download(model_name)
                _spacy_model = spacy.load(model_name)
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
                logger.info("Falling back to en_core_web_sm")
                try:
                    _spacy_model = spacy.load('en_core_web_sm')
                except OSError:
                    spacy.cli.download('en_core_web_sm')
                    _spacy_model = spacy.load('en_core_web_sm')
    return _spacy_model

# Enhanced emotion anchors with more nuanced categories
EMOTION_CATEGORIES = {
    'joy': ['happy', 'joyful', 'delighted', 'elated', 'ecstatic', 'content', 'pleased'],
    'sadness': ['sad', 'depressed', 'gloomy', 'melancholy', 'heartbroken', 'grieving', 'unhappy'],
    'anger': ['angry', 'furious', 'enraged', 'hostile', 'irritated', 'outraged', 'mad'],
    'fear': ['afraid', 'scared', 'terrified', 'anxious', 'panicked', 'worried', 'frightened'],
    'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled', 'stunned', 'unexpected'],
    'disgust': ['disgusted', 'repulsed', 'revolted', 'appalled', 'nauseated', 'offended', 'gross'],
    'trust': ['trusting', 'confident', 'secure', 'reliable', 'faithful', 'assured', 'believing'],
    'anticipation': ['expectant', 'eager', 'excited', 'hopeful', 'optimistic', 'ready', 'awaiting']
}

# Initialize emotion vectors with caching
@lru_cache(maxsize=None)
def get_emotion_vector(emotion: str) -> np.ndarray:
    """Get cached emotion vector for a category."""
    words = EMOTION_CATEGORIES[emotion]
    vectors = []
    
    nlp = get_spacy_model()
    
    # Try each word in the category
    for word in words:
        token = nlp(word)[0]
        if token.has_vector:
            vectors.append(token.vector)
    
    if not vectors:
        logger.warning(f"No vectors found for emotion {emotion}, using fallback method")
        # Try getting vector directly from word embeddings
        try:
            # Use spaCy's vocab directly
            vector = nlp.vocab.get_vector(words[0])
            if np.any(vector):
                return vector / np.linalg.norm(vector)
        except KeyError:
            logger.error(f"Could not find vector for any words in {emotion} category")
            # Return normalized random vector as last resort
            vector = np.random.randn(nlp.vocab.vectors_length)
            return vector / np.linalg.norm(vector)
    
    # Average the vectors and normalize
    avg_vector = np.mean(vectors, axis=0)
    return avg_vector / np.linalg.norm(avg_vector)

# Initialize emotion anchors lazily
_EMOTION_ANCHORS = None

def get_emotion_anchors():
    """Get emotion anchors, initializing them if needed."""
    global _EMOTION_ANCHORS
    if _EMOTION_ANCHORS is None:
        _EMOTION_ANCHORS = {}
        for emotion in EMOTION_CATEGORIES:
            vector = get_emotion_vector(emotion)
            if np.any(vector):  # Only add emotions with non-zero vectors
                _EMOTION_ANCHORS[emotion] = vector
    return _EMOTION_ANCHORS

class PhraseEmbedding:
    """
    A class representing the embedding of a phrase or sentence,
    including both its semantic content and structural shape.
    """
    def __init__(self, text: str, embedding: np.ndarray, shape_params: Dict):
        self.text = text
        self.embedding = embedding
        self.shape_params = shape_params
        
    def __repr__(self) -> str:
        return f"PhraseEmbedding(text='{self.text[:50]}...', shape_params={list(self.shape_params.keys())})"

class PhraseEmbedder:
    """
    A class for embedding phrases and sentences into a semantic manifold,
    extracting both meaning and structural patterns.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",  # Using smaller default model
        emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base",  # Smaller emotion model
        cache_size: int = 500,  # Reduced cache size
        use_gpu: bool = True,
        batch_size: int = 16,  # Smaller batch size
        max_length: int = 512  # Reduced max length
    ):
        """
        Initialize the PhraseEmbedder.
        
        Args:
            model_name: Name of the embedding model
            emotion_model_name: Name of the emotion analysis model
            cache_size: Size of the LRU cache
            use_gpu: Whether to use GPU if available
            batch_size: Default batch size for processing
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.emotion_model_name = emotion_model_name
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        
        # Initialize models
        self._initialize_models()
        
        # Configure caches with size limits
        self.get_embedding = lru_cache(maxsize=cache_size)(self._get_embedding_uncached)
        self.get_emotion_vector = lru_cache(maxsize=cache_size)(self._get_emotion_vector_uncached)
        
        # Use the globally loaded spaCy model
        self.nlp = get_spacy_model()
        
        # Initialize emotion cache with size limit
        self._emotion_cache = {}
        self._max_emotion_cache = 1000
        
        # Memory management
        self._memory_warning_threshold = 0.8
    
    def _initialize_models(self) -> None:
        """Initialize embedding and emotion models with memory optimization."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.tokenizer = self.model.tokenizer  # Get tokenizer from the model
            self.model.to(self.device)
            
            # Clear GPU cache after model loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}")
            logger.info(f"Attempting to load backup model: {BACKUP_MODEL}")
            try:
                self.model_name = BACKUP_MODEL
                self.model = SentenceTransformer(BACKUP_MODEL)
                self.tokenizer = self.model.tokenizer  # Get tokenizer from backup model
                self.model.to(self.device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e2:
                logger.error(f"Failed to load backup model: {e2}")
                raise RuntimeError("Could not initialize embedding model")
        
        # Initialize emotion analysis model
        try:
            logger.info(f"Loading emotion model: {self.emotion_model_name}")
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.emotion_model_name,
                device=0 if torch.cuda.is_available() else -1,
                top_k=None
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            raise RuntimeError("Could not initialize emotion analysis model")
            
    def _check_memory(self) -> bool:
        """Check if memory usage is above threshold."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            if memory_allocated / memory_reserved > self._memory_warning_threshold:
                return True
        return False
        
    def _clear_caches(self) -> None:
        """Clear caches to free memory."""
        self.get_embedding.cache_clear()
        self.get_emotion_vector.cache_clear()
        self._emotion_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _manage_emotion_cache(self) -> None:
        """Manage emotion cache size."""
        if len(self._emotion_cache) > self._max_emotion_cache:
            # Remove oldest entries
            num_to_remove = len(self._emotion_cache) - self._max_emotion_cache
            for _ in range(num_to_remove):
                self._emotion_cache.popitem(last=False)
                
    def get_embeddings_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> Dict[str, np.ndarray]:
        """Get embeddings for multiple texts efficiently using batching."""
        batch_size = batch_size or self.batch_size
        results = {}
        
        # Create iterator
        batches = range(0, len(texts), batch_size)
        if show_progress:
            batches = tqdm(batches, desc="Getting embeddings")
        
        for i in batches:
            # Check memory before processing batch
            if self._check_memory():
                logger.warning("High memory usage detected, clearing caches")
                self._clear_caches()
                
            batch = texts[i:i + batch_size]
            
            # Process batch with memory optimization
            with torch.no_grad():
                embeddings = self.model.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
            # Store results
            for text, embedding in zip(batch, embeddings):
                results[text] = embedding
                
        return results
        
    def get_emotion_vectors_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> Dict[str, np.ndarray]:
        """Get emotion vectors for multiple texts efficiently using batching."""
        batch_size = batch_size or self.batch_size
        results = {}
        
        # Create iterator
        batches = range(0, len(texts), batch_size)
        if show_progress:
            batches = tqdm(batches, desc="Getting emotion vectors")
        
        for i in batches:
            # Check memory
            if self._check_memory():
                self._clear_caches()
                
            batch = texts[i:i + batch_size]
            
            # Get emotion predictions for batch
            predictions = self.emotion_pipeline(batch, batch_size=batch_size)
            
            # Process each prediction
            for text, result in zip(batch, predictions):
                emotions = {label: score for label, score in zip(result['labels'], result['scores'])}
                emotion_vector = np.array(list(emotions.values()))
                emotion_vector = emotion_vector / np.linalg.norm(emotion_vector)
                results[text] = emotion_vector
                
            # Manage emotion cache
            self._manage_emotion_cache()
                
        return results
        
    def __del__(self):
        """Cleanup when object is deleted."""
        self._clear_caches()

    def _register_extensions(self):
        """Register all custom spaCy extensions."""
        extensions = {
            "is_negated": {"getter": self._get_is_negated},
            "is_coreferent": {"default": False},
            "semantic_role": {"default": ""},
            "is_semantic_root": {"default": False},
            "discourse_relation": {"default": ""},
            "discourse_markers": {"default": []},
            "emotion_scores": {"default": {}},
            "intensity_score": {"default": 0.0}
        }
        
        for name, config in extensions.items():
            if not Token.has_extension(name):
                if "getter" in config:
                    Token.set_extension(name, getter=config["getter"])
                else:
                    Token.set_extension(name, default=config["default"])
    
    def _get_is_negated(self, token) -> bool:
        """Check if a token is negated using dependency parsing."""
        for child in token.children:
            if child.dep_ == "neg":
                return True
        return False

    @torch.no_grad()
    def _get_embedding_uncached(self, text: str) -> np.ndarray:
        """Get embedding for text without caching.
        
        Args:
            text: The text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        # Add instruction for query-style embedding
        query = f'Instruct: Represent this text for retrieval\nQuery: {text}'
        
        # Get embedding directly from the model
        embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        return embedding.cpu().numpy()
    
    def _get_emotion_vector_uncached(self, text: str) -> np.ndarray:
        """Get emotion vector for text without caching.
        
        Args:
            text: The text to analyze
            
        Returns:
            Numpy array containing emotion scores
        """
        # Get emotion predictions
        result = self.emotion_pipeline(text)[0]
        
        # Convert to normalized vector
        emotions = {label: score for label, score in zip(result['labels'], result['scores'])}
        emotion_vector = np.array(list(emotions.values()))
        
        # Normalize to unit vector
        emotion_vector = emotion_vector / np.linalg.norm(emotion_vector)
        return emotion_vector
    
    def analyze_text(
        self,
        text: str,
        return_emotions: bool = True
    ) -> Dict[str, Any]:
        """Analyze text for semantic and emotional content.
        
        Args:
            text: The text to analyze
            return_emotions: Whether to include emotion analysis
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'embedding': self.get_embedding(text),
            'length': len(text.split())
        }
        
        if return_emotions:
            results['emotions'] = self.get_emotion_vector(text)
        
        return results
    
    def analyze_texts_batch(
        self,
        texts: List[str],
        return_emotions: bool = True,
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze multiple texts efficiently using batching.
        
        Args:
            texts: List of texts to analyze
            return_emotions: Whether to include emotion analysis
            batch_size: Optional custom batch size
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping texts to their analysis results
        """
        # Get embeddings
        embeddings = self.get_embeddings_batch(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        # Get emotion vectors if requested
        emotion_vectors = {}
        if return_emotions:
            emotion_vectors = self.get_emotion_vectors_batch(
                texts,
                batch_size=batch_size,
                show_progress=show_progress
            )
        
        # Combine results
        results = {}
        for text in texts:
            result = {
                'embedding': embeddings[text],
                'length': len(text.split())
            }
            
            if return_emotions:
                result['emotions'] = emotion_vectors[text]
            
            results[text] = result
        
        return results

    def embed_phrase(self, text: str) -> PhraseEmbedding:
        """
        Create a complete embedding for a phrase, including both semantic
        and structural information.
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Get transformer embedding
        with torch.no_grad():
            embedding = self.get_embedding(text)
        
        # Analyze emotional content
        emotion_scores, emotional_valence = self.analyze_emotions(doc)
        
        # Analyze structural properties
        structure = self.analyze_structure(doc)
        
        # Calculate tree depth
        max_depth = 0
        for token in doc:
            depth = len(list(token.ancestors))
            max_depth = max(max_depth, depth)
        
        # Calculate syntactic properties
        syntax_tree = [(token.dep_, token.head.i) for token in doc]
        
        # Analyze clause structure
        clauses = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp", "xcomp"]:
                    clause = {
                        "verb": token.text,
                        "subject": [child.text for child in token.children if child.dep_ == "nsubj"],
                        "object": [child.text for child in token.children if child.dep_ in ["dobj", "pobj"]]
                    }
                    clauses.append(clause)
        # Combine all parameters
        shape_params = {
            'emotions': emotion_scores,
            'emotional_valence': emotional_valence,
            'structure': structure,
            'tree_depth': max_depth,
            'syntax_complexity': len(syntax_tree) / max(len(doc), 1),
            'clause_structure': clauses,
            'clause_count': len(clauses),
            'length': len(doc),
            'syllable_pattern': [self._count_syllables(token.text) for token in doc],
            'complexity': {
                'sentence_length': len(doc),
                'unique_words': len(set(token.lower_ for token in doc)),
                'avg_word_length': float(np.mean([len(token.text) for token in doc if token.is_alpha]) if doc else 0)
            }
        }
        
        return PhraseEmbedding(text, embedding, shape_params)

    def embed_phrases(self, texts: List[str], show_progress: bool = True) -> List[PhraseEmbedding]:
        """
        Efficiently embed multiple phrases using batching.
        """
        embeddings = []
        iterator = tqdm(texts, desc="Embedding phrases") if show_progress else texts
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.get_embeddings_batch(batch)
            
            # Process each text in the batch
            for text, embedding in zip(batch, batch_embeddings):
                doc = self.nlp(text)
                emotion_scores, emotional_valence = self.analyze_emotions(doc)
                structure = self.analyze_structure(doc)
                
                # Calculate tree depth
                max_depth = 0
                for token in doc:
                    depth = len(list(token.ancestors))
                    max_depth = max(max_depth, depth)
                
                # Calculate syntactic properties
                syntax_tree = [(token.dep_, token.head.i) for token in doc]
                
                # Analyze clause structure
                clauses = []
                for sent in doc.sents:
                    for token in sent:
                        if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp", "xcomp"]:
                            clause = {
                                "verb": token.text,
                                "subject": [child.text for child in token.children if child.dep_ == "nsubj"],
                                "object": [child.text for child in token.children if child.dep_ in ["dobj", "pobj"]]
                            }
                            clauses.append(clause)
                
                shape_params = {
                    'emotions': emotion_scores,
                    'emotional_valence': emotional_valence,
                    'structure': structure,
                    'tree_depth': max_depth,
                    'syntax_complexity': len(syntax_tree) / len(doc),
                    'clause_structure': clauses,
                    'clause_count': len(clauses),
                    'length': len(doc),
                    'complexity': {
                        'sentence_length': len(doc),
                        'unique_words': len(set(token.lower_ for token in doc)),
                        'avg_word_length': np.mean([len(token.text) for token in doc if token.is_alpha]) if doc else 0
                    }
                }
                
                embeddings.append(PhraseEmbedding(text, embedding, shape_params))
        
        return embeddings

    def embed_text(self, text: str, chunk_size: int = 3) -> List[PhraseEmbedding]:
        """
        Embed a longer text by breaking it into meaningful chunks.
        
        Args:
            text: The text to embed
            chunk_size: Number of sentences per chunk
            
        Returns:
            List of PhraseEmbedding objects
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        # Create chunks of sentences
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            chunk_text = ' '.join(sent.text for sent in chunk)
            chunks.append(self.embed_phrase(chunk_text))
        
        return chunks

    def analyze_emotions(self, doc) -> Tuple[Dict[str, float], float]:
        """Analyze emotional content of text."""
        emotion_scores = {}
        for token in doc:
            if token.has_vector:
                vector = token.vector
                for emotion, anchor in get_emotion_anchors().items():
                    similarity = np.dot(vector, anchor)
                    emotion_scores[emotion] = emotion_scores.get(emotion, 0) + max(0, similarity)
        
        # Normalize scores
        total = sum(emotion_scores.values()) or 1
        emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        # Calculate overall emotional valence
        emotional_valence = sum(score * (1 if emotion in ['joy', 'trust'] else -1) 
                              for emotion, score in emotion_scores.items())
        
        return emotion_scores, emotional_valence

    def analyze_structure(self, doc) -> Dict[str, Any]:
        """Analyze structural properties of text."""
        return {
            'sentence_count': len(list(doc.sents)),
            'word_count': len([token for token in doc if token.is_alpha]),
            'dependency_tree': [(token.text, token.dep_, token.head.text) for token in doc],
            'pos_tags': Counter(token.pos_ for token in doc),
            'root_verbs': [token.text for token in doc if token.dep_ == 'ROOT' and token.pos_ == 'VERB']
        }

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
            
        # Handle common cases
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        if count == 0:
            count = 1
            
        return count 