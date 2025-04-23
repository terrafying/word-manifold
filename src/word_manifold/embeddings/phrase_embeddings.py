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
from typing import List, Dict, Tuple, Optional, Union
import logging
from collections import Counter, defaultdict
from spacy.tokens import Token
from functools import lru_cache
from tqdm import tqdm

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

# Load spaCy model with improved error handling
def load_spacy_model(model_name: str = 'en_core_web_trf') -> spacy.language.Language:
    """Load spaCy model with fallback options."""
    try:
        logger.info(f"Loading spaCy model '{model_name}'")
        return spacy.load(model_name)
    except OSError:
        logger.warning(f"spaCy model '{model_name}' not found, downloading...")
        try:
            spacy.cli.download(model_name)
            return spacy.load(model_name)
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            logger.info("Falling back to en_core_web_lg")
            try:
                return spacy.load('en_core_web_lg')
            except OSError:
                spacy.cli.download('en_core_web_lg')
                return spacy.load('en_core_web_lg')

nlp = load_spacy_model()

# Enhanced emotion anchors with more nuanced categories
EMOTION_CATEGORIES = {
    'joy': ['happy', 'joyful', 'delighted', 'elated', 'ecstatic', 'content'],
    'sadness': ['sad', 'depressed', 'gloomy', 'melancholy', 'heartbroken', 'grieving'],
    'anger': ['angry', 'furious', 'enraged', 'hostile', 'irritated', 'outraged'],
    'fear': ['afraid', 'scared', 'terrified', 'anxious', 'panicked', 'worried'],
    'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled', 'stunned'],
    'disgust': ['disgusted', 'repulsed', 'revolted', 'appalled', 'nauseated', 'offended'],
    'trust': ['trusting', 'confident', 'secure', 'reliable', 'faithful', 'assured'],
    'anticipation': ['expectant', 'eager', 'excited', 'hopeful', 'optimistic', 'ready']
}

# Initialize emotion vectors with caching
@lru_cache(maxsize=None)
def get_emotion_vector(emotion: str) -> np.ndarray:
    """Get cached emotion vector for a category."""
    words = EMOTION_CATEGORIES[emotion]
    vectors = [nlp(word).vector for word in words if nlp(word).has_vector]
    if not vectors:
        logger.warning(f"No vectors found for emotion {emotion}")
        return np.zeros(nlp.vocab.vectors_length)
    return np.mean(vectors, axis=0)

# Initialize emotion anchors with normalization
EMOTION_ANCHORS = {
    emotion: get_emotion_vector(emotion) / np.linalg.norm(get_emotion_vector(emotion))
    for emotion in EMOTION_CATEGORIES
}

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
        model_name: str = "BAAI/bge-large-en-v1.5",  # Updated to use a more recent model
        use_gpu: bool = True,
        batch_size: int = 32
    ):
        """Initialize the embedder with specified model."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        
        # Load models with error handling
        logger.info(f"Loading {model_name} model and tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            logger.info("Falling back to all-MiniLM-L6-v2")
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
        
        # Use the globally loaded spaCy model
        self.nlp = nlp
        
        # Register custom extensions
        self._register_extensions()
        
        # Initialize emotion cache
        self._emotion_cache = {}
    
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
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Efficiently embed a batch of texts using the transformer model."""
        # Tokenize all texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Get model outputs
        outputs = self.model(**encoded)
        
        # Use mean pooling
        attention_mask = encoded['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Convert to numpy and normalize
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings

    def analyze_emotions(self, doc) -> Dict[str, float]:
        """
        Analyze the emotional content of a document using improved vector comparison
        and contextual analysis.
        """
        # Get document vector
        doc_vector = doc.vector / np.linalg.norm(doc.vector)
        
        # Calculate emotion scores using cosine similarity
        emotion_scores = {}
        for emotion, anchor in EMOTION_ANCHORS.items():
            similarity = np.dot(doc_vector, anchor)
            # Apply sigmoid to bound scores between 0 and 1
            emotion_scores[emotion] = 1 / (1 + np.exp(-5 * (similarity - 0.5)))
        
        # Adjust scores based on negation and intensity
        for token in doc:
            if token._.is_negated:
                # Invert emotion scores for negated contexts
                for emotion in emotion_scores:
                    emotion_scores[emotion] = 1 - emotion_scores[emotion]
            
            # Adjust for intensity modifiers
            intensity_words = {'very', 'extremely', 'incredibly', 'barely', 'slightly', 'somewhat'}
            if token.lower_ in intensity_words:
                next_token = token.nbor() if token.i + 1 < len(doc) else None
                if next_token:
                    modifier = 1.5 if token.lower_ in {'very', 'extremely', 'incredibly'} else 0.5
                    for emotion in emotion_scores:
                        emotion_scores[emotion] = np.clip(emotion_scores[emotion] * modifier, 0, 1)
        
        return emotion_scores

    def analyze_structure(self, doc) -> Dict[str, float]:
        """
        Analyze the structural properties of the text, including rhythm,
        complexity, and coherence.
        """
        # Initialize structure parameters
        structure = {
            'rhythm_score': 0.0,
            'complexity_score': 0.0,
            'coherence_score': 0.0,
            'emphasis_score': 0.0
        }
        
        # Analyze rhythm using syllable patterns
        syllable_pattern = []
        for token in doc:
            if token.is_alpha and not token.is_stop:
                syllable_count = self._count_syllables(token.text)
                syllable_pattern.append(syllable_count)
        
        if syllable_pattern:
            # Calculate rhythm score based on pattern regularity
            differences = np.diff(syllable_pattern)
            structure['rhythm_score'] = 1.0 / (1.0 + np.std(differences))
        
        # Analyze complexity
        word_lengths = [len(token.text) for token in doc if token.is_alpha]
        if word_lengths:
            avg_word_length = np.mean(word_lengths)
            structure['complexity_score'] = np.clip(avg_word_length / 12.0, 0, 1)
        
        # Analyze coherence using dependency distances
        distances = []
        for token in doc:
            if token.head != token:  # Skip root
                distance = abs(token.i - token.head.i)
                distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            structure['coherence_score'] = 1.0 / (1.0 + 0.1 * avg_distance)
        
        # Analyze emphasis using caps, punctuation, and special tokens
        emphasis_markers = sum(1 for token in doc if token.is_upper or token.text in "!?*")
        structure['emphasis_score'] = np.clip(emphasis_markers / len(doc), 0, 1)
        
        return structure

    def _count_syllables(self, word: str) -> int:
        """
        Count the number of syllables in a word using improved heuristics.
        """
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_is_vowel = False
        
        # Handle special cases
        if word.endswith('e'):
            word = word[:-1]
        
        # Count vowel groups
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        # Ensure at least one syllable
        return max(1, count)

    def embed_phrase(self, text: str) -> PhraseEmbedding:
        """
        Create a complete embedding for a phrase, including both semantic
        and structural information.
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Get transformer embedding
        with torch.no_grad():
            embedding = self.embed_batch([text])[0]
        
        # Analyze emotional content
        emotion_scores = self.analyze_emotions(doc)
        
        # Analyze structural properties
        structure = self.analyze_structure(doc)
        
        # Combine all parameters
        shape_params = {
            'emotions': emotion_scores,
            'structure': structure,
            'length': len(doc),
            'complexity': {
                'sentence_length': len(doc),
                'unique_words': len(set(token.lower_ for token in doc)),
                'avg_word_length': np.mean([len(token.text) for token in doc if token.is_alpha]) if doc else 0
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
            batch_embeddings = self.embed_batch(batch)
            
            # Process each text in the batch
            for text, embedding in zip(batch, batch_embeddings):
                doc = self.nlp(text)
                emotion_scores = self.analyze_emotions(doc)
                structure = self.analyze_structure(doc)
                
                shape_params = {
                    'emotions': emotion_scores,
                    'structure': structure,
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