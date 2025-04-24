"""
Example demonstrating recursive dimensionality reduction and multimodal visualization.

This example shows how to:
1. Load pre-trained word embeddings
2. Select semantically meaningful term clusters
3. Apply recursive UMAP reduction
4. Generate visual representations for semantic clusters
5. Create multimodal visualizations combining semantic and visual information
6. Provide interactive controls and animations
7. Visualize semantic source cube structure
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.manifold.vector_manifold import VectorManifold
from word_manifold.manifold.reduction import RecursiveReducer
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging
from PIL import Image
import io
import requests
from typing import Dict, List, Optional, Tuple, Any
import base64
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.animation import FuncAnimation
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_level_data(result: Dict[str, Any], target_level: int, current_level: int = 0) -> Optional[Dict[str, Any]]:
    """Find data for a specific level in the recursive reduction results."""
    if current_level == target_level:
        return result
    for child in result['children']:
        found = find_level_data(child, target_level, current_level + 1)
        if found is not None:
            return found
    return None

class MultimodalVisualizer:
    """Class for handling multimodal visualization with text and images."""
    
    def __init__(self, image_size=128):  # Reduced from 512 to 128 for memory efficiency
        """Initialize the multimodal visualizer with CLIP model."""
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP with memory optimization
        logger.info("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Move model to device and clear cache
        self.clip_model.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Cache for embeddings
        self._text_embedding_cache = {}
        self._image_embedding_cache = {}
        
    def _clear_caches(self):
        """Clear embedding caches to free memory."""
        self._text_embedding_cache.clear()
        self._image_embedding_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_category_prompt(self, category: str, terms: List[str]) -> str:
        """Generate a detailed prompt for image generation."""
        term_str = ", ".join(terms[:3])  # Use fewer terms for clearer prompts
        
        prompts = {
            "elements": f"Simple sacred geometry symbol representing {term_str}",
            "celestial": f"Minimalist cosmic diagram of {term_str}",
            "mystical": f"Basic mystical sigil combining {term_str}",
            "natural": f"Simple nature pattern showing {term_str}",
            "emotional": f"Abstract emotional symbol for {term_str}",
            "abstract": f"Geometric pattern representing {term_str}"
        }
        
        return prompts.get(category, f"Simple symbol for {term_str}")
    
    def get_image_for_category(self, category: str, terms: List[str]) -> Optional[Image.Image]:
        """Get a low-resolution image for a category."""
        # Create a simple geometric pattern based on category
        img = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
        
        # Basic patterns for each category
        patterns = {
            "elements": self._draw_elemental_pattern,
            "celestial": self._draw_celestial_pattern,
            "mystical": self._draw_mystical_pattern,
            "natural": self._draw_natural_pattern,
            "emotional": self._draw_emotional_pattern,
            "abstract": self._draw_abstract_pattern
        }
        
        draw_func = patterns.get(category, self._draw_abstract_pattern)
        return draw_func(img, terms)
    
    def _draw_elemental_pattern(self, img: Image.Image, terms: List[str]) -> Image.Image:
        """Draw a simple elemental pattern."""
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw a triangle for fire, circle for water, square for earth, spiral for air
        center = self.image_size // 2
        size = self.image_size // 3
        
        draw.regular_polygon((center, center, size), 3, fill="#FF6B6B")
        draw.ellipse([center-size//2, center-size//2, center+size//2, center+size//2], 
                    outline="#4ECDC4", width=2)
        
        return img
    
    def _draw_celestial_pattern(self, img: Image.Image, terms: List[str]) -> Image.Image:
        """Draw a simple celestial pattern."""
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw a star pattern
        center = self.image_size // 2
        size = self.image_size // 3
        
        points = []
        for i in range(8):
            angle = i * np.pi / 4
            points.extend([
                center + size * np.cos(angle),
                center + size * np.sin(angle)
            ])
        
        draw.polygon(points, fill="#4ECDC4")
        return img
    
    def _draw_mystical_pattern(self, img: Image.Image, terms: List[str]) -> Image.Image:
        """Draw a simple mystical pattern."""
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw a pentagram
        center = self.image_size // 2
        size = self.image_size // 3
        
        points = []
        for i in range(5):
            angle = i * 4 * np.pi / 5
            points.extend([
                center + size * np.cos(angle),
                center + size * np.sin(angle)
            ])
        
        draw.polygon(points, fill="#9B59B6")
        return img
    
    def _draw_natural_pattern(self, img: Image.Image, terms: List[str]) -> Image.Image:
        """Draw a simple nature-inspired pattern."""
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw a leaf/tree-like pattern
        center = self.image_size // 2
        size = self.image_size // 3
        
        # Draw a simple tree
        draw.rectangle([center-5, center, center+5, center+size], fill="#2ECC71")
        draw.ellipse([center-size//2, center-size//2, center+size//2, center], 
                    fill="#2ECC71")
        
        return img
    
    def _draw_emotional_pattern(self, img: Image.Image, terms: List[str]) -> Image.Image:
        """Draw a simple emotional pattern."""
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw a heart shape
        center = self.image_size // 2
        size = self.image_size // 3
        
        draw.ellipse([center-size, center-size//2, center, center+size//2], 
                    fill="#F1C40F")
        draw.ellipse([center, center-size//2, center+size, center+size//2], 
                    fill="#F1C40F")
        
        return img
    
    def _draw_abstract_pattern(self, img: Image.Image, terms: List[str]) -> Image.Image:
        """Draw a simple abstract pattern."""
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw intersecting circles
        center = self.image_size // 2
        size = self.image_size // 3
        
        for i in range(4):
            offset = size // 2 * np.cos(i * np.pi / 2)
            draw.ellipse([center-size+offset, center-size, center+offset, center], 
                        outline="#95A5A6", width=2)
        
        return img
    
    @torch.no_grad()
    def get_visual_embedding(self, image: Image.Image) -> np.ndarray:
        """Get CLIP visual embedding with caching and memory optimization."""
        # Check cache
        image_hash = hash(image.tobytes())
        if image_hash in self._image_embedding_cache:
            return self._image_embedding_cache[image_hash]
        
        # Process image
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embedding
        image_features = self.clip_model.get_image_features(**inputs)
        embedding = image_features.cpu().numpy()[0]
        
        # Cache result
        self._image_embedding_cache[image_hash] = embedding
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return embedding
    
    @torch.no_grad()
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get CLIP text embedding with caching and memory optimization."""
        # Check cache
        if text in self._text_embedding_cache:
            return self._text_embedding_cache[text]
        
        # Process text
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embedding
        text_features = self.clip_model.get_text_features(**inputs)
        embedding = text_features.cpu().numpy()[0]
        
        # Cache result
        self._text_embedding_cache[text] = embedding
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return embedding

class InteractiveVisualizer:
    """Class for handling interactive visualization controls."""
    
    def __init__(self, manifold, visualizer):
        self.manifold = manifold
        self.visualizer = visualizer
        self.current_level = 0
        self.fig = None
        self.animation = None
        
        # Create interactive controls
        self.setup_controls()
        
    def setup_controls(self):
        """Setup interactive widgets."""
        self.level_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=3,
            description='Level:',
            continuous_update=False
        )
        
        self.rotation_slider = widgets.FloatSlider(
            value=0,
            min=0,
            max=360,
            description='Rotation:',
            continuous_update=True
        )
        
        self.coherence_threshold = widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Coherence:',
            continuous_update=False
        )
        
        self.show_labels = widgets.Checkbox(
            value=True,
            description='Show Labels'
        )
        
        self.show_images = widgets.Checkbox(
            value=True,
            description='Show Images'
        )
        
        self.animation_speed = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=5.0,
            description='Anim Speed:',
            continuous_update=True
        )
        
        self.play_button = widgets.Button(
            description='Play/Pause',
            icon='play'
        )
        
        # Layout controls
        self.controls = widgets.VBox([
            widgets.HBox([self.level_slider, self.rotation_slider]),
            widgets.HBox([self.coherence_threshold, self.animation_speed]),
            widgets.HBox([self.show_labels, self.show_images, self.play_button])
        ])
        
        # Connect callbacks
        self.level_slider.observe(self.update_plot, names='value')
        self.rotation_slider.observe(self.update_rotation, names='value')
        self.coherence_threshold.observe(self.recompute_reduction, names='value')
        self.show_labels.observe(self.update_plot, names='value')
        self.show_images.observe(self.update_plot, names='value')
        self.play_button.on_click(self.toggle_animation)
        
    def display(self):
        """Display the interactive visualization."""
        display(self.controls)
        self.update_plot(None)
        
    def update_plot(self, change):
        """Update the visualization based on current control values."""
        if self.fig is not None:
            plt.close(self.fig)
        
        self.fig = plt.figure(figsize=(20, 15))
        
        if self.show_images.value:
            self.fig = plot_multimodal_clusters(
                self.manifold,
                level=self.level_slider.value,
                visualizer=self.visualizer,
                show_labels=self.show_labels.value
            )
        else:
            self.fig = plot_term_clusters(
                self.manifold,
                level=self.level_slider.value,
                show_labels=self.show_labels.value
            )
        
        plt.show()
        
    def update_rotation(self, change):
        """Update 3D plot rotation."""
        if self.fig is not None:
            for ax in self.fig.get_axes():
                if isinstance(ax, Axes3D):
                    ax.view_init(elev=20, azim=self.rotation_slider.value)
                    self.fig.canvas.draw_idle()
                    
    def recompute_reduction(self, change):
        """Recompute the recursive reduction with new parameters."""
        self.manifold.recursive_reduce(
            depth=3,
            min_cluster_size=5,
            coherence_threshold=self.coherence_threshold.value,
            scale_factor=0.6,
            rotation_symmetry=5
        )
        self.update_plot(None)
        
    def animate_rotation(self, frame):
        """Animation update function."""
        self.rotation_slider.value = (self.rotation_slider.value + 2) % 360
        return self.fig.get_axes()
        
    def toggle_animation(self, button):
        """Toggle the rotation animation."""
        if self.animation is None:
            self.animation = FuncAnimation(
                self.fig,
                self.animate_rotation,
                interval=50/self.animation_speed.value,
                blit=True
            )
            button.icon = 'pause'
        else:
            self.animation.event_source.stop()
            self.animation = None
            button.icon = 'play'

def load_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load pre-trained embeddings and extract meaningful term clusters."""
    logger.info(f"Loading model: {model_name}")
    
    # Initialize word embeddings with background processing
    word_embeddings = WordEmbeddings(model_name=model_name)
    
    # Define semantic clusters with rich vocabulary
    term_clusters = {
        "elements": [
            "fire", "water", "earth", "air", "aether", "void", "plasma",
            "crystal", "metal", "wood", "stone", "lightning", "ice", "steam"
        ],
        "celestial": [
            "sun", "moon", "mars", "venus", "mercury", "jupiter", "saturn",
            "uranus", "neptune", "pluto", "sirius", "orion", "andromeda",
            "galaxy", "nebula", "quasar", "cosmos", "star", "planet"
        ],
        "mystical": [
            "spirit", "soul", "mind", "consciousness", "enlightenment",
            "wisdom", "truth", "infinity", "eternity", "divine", "sacred",
            "mystic", "occult", "esoteric", "arcane", "ethereal"
        ],
        "natural": [
            "mountain", "ocean", "forest", "desert", "river", "valley",
            "canyon", "glacier", "volcano", "island", "reef", "tundra",
            "savanna", "jungle", "wetland", "prairie"
        ],
        "emotional": [
            "joy", "peace", "love", "serenity", "bliss", "ecstasy",
            "wonder", "awe", "tranquility", "harmony", "balance",
            "clarity", "vitality", "radiance"
        ],
        "abstract": [
            "infinity", "unity", "duality", "trinity", "quaternity",
            "pentagram", "hexagon", "spiral", "fractal", "mandala",
            "labyrinth", "vortex", "nexus", "matrix"
        ]
    }
    
    # Collect all terms first
    all_terms = []
    for category, terms in term_clusters.items():
        all_terms.extend(terms)
    
    # Load all terms at once
    word_embeddings.load_terms(all_terms)
    
    # Wait a bit for initial terms to be processed
    time.sleep(2)
    
    # Find related terms through model vocabulary
    for category, terms in term_clusters.items():
        logger.info(f"Processing {category} terms...")
        
        # Get embeddings for current terms
        term_embeddings = [word_embeddings.get_embedding(term) for term in terms]
        term_embeddings = [emb for emb in term_embeddings if emb is not None]
        
        if term_embeddings:
            # Calculate base embedding for category
            base_embedding = np.mean(term_embeddings, axis=0)
            
            # Sample vocabulary and find related terms
            sample_terms = [
                "matter", "energy", "light", "dark", "space", "time",
                "force", "power", "life", "death", "mind", "body",
                "spirit", "nature", "cosmos", "chaos", "order", "wisdom"
            ]
            
            # Get embeddings for sampled terms
            for word in sample_terms:
                try:
                    word_embeddings.load_terms([word])
                    time.sleep(0.1)  # Small delay to allow processing
                    
                    word_embedding = word_embeddings.get_embedding(word)
                    if word_embedding is not None:
                        similarity = np.dot(word_embedding, base_embedding) / (
                            np.linalg.norm(word_embedding) * np.linalg.norm(base_embedding)
                        )
                        
                        if similarity > 0.5:  # Only add if similar enough
                            all_terms.append(word)
                            logger.info(f"Added related term for {category}: {word} (similarity: {similarity:.3f})")
                            
                except Exception as e:
                    logger.warning(f"Error processing term {word}: {e}")
    
    # Final load of all terms including related ones
    word_embeddings.load_terms(all_terms)
    
    # Wait for processing to complete
    time.sleep(2)
    
    logger.info(f"Created embeddings for {len(all_terms)} terms")
    return word_embeddings

def plot_coherence_heatmap(manifold):
    """Plot coherence heatmap across reduction levels."""
    def extract_coherence(result, depth=0, coherence_matrix=None, max_depth=None):
        if coherence_matrix is None:
            # Find max depth first
            def get_max_depth(r, d=0):
                if not r['children']:
                    return d
                return max(get_max_depth(child, d+1) for child in r['children'])
            
            max_depth = get_max_depth(result)
            coherence_matrix = np.zeros((max_depth + 1, max_depth + 1))
            
        coherence_matrix[depth, depth] = result['coherence']
        
        for child in result['children']:
            # Calculate cross-level coherence
            child_vectors = child['points']
            parent_vectors = result['points']
            cross_coherence = manifold._calculate_reduction_coherence(
                parent_vectors, child_vectors
            )
            coherence_matrix[depth, depth+1] = cross_coherence
            coherence_matrix[depth+1, depth] = cross_coherence
            
            extract_coherence(child, depth+1, coherence_matrix, max_depth)
            
        return coherence_matrix
    
    coherence_matrix = extract_coherence(manifold.recursive_reduced)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(coherence_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Coherence')
    plt.title('Cross-level Coherence Heatmap')
    plt.xlabel('Reduction Level')
    plt.ylabel('Reduction Level')
    
    # Add coherence values as text
    for i in range(coherence_matrix.shape[0]):
        for j in range(coherence_matrix.shape[1]):
            plt.text(j, i, f'{coherence_matrix[i,j]:.2f}',
                    ha='center', va='center')
    
    plt.tight_layout()
    return plt.gcf()

def plot_term_clusters(manifold, level=0, show_labels=True):
    """Create a 3D scatter plot colored by semantic clusters."""
    if not hasattr(manifold, 'recursive_reduced'):
        raise ValueError("Must run recursive_reduce() first")
        
    # Get the points for the specified level
    level_data = find_level_data(manifold.recursive_reduced, level)
    if level_data is None:
        raise ValueError(f"Level {level} not found")
    
    points = level_data['points']
    
    # Create category mapping
    term_categories = {}
    for term in manifold.terms:
        if "fire" in term or "water" in term or "earth" in term or "air" in term:
            term_categories[term] = "elements"
        elif "sun" in term or "moon" in term or "star" in term:
            term_categories[term] = "celestial"
        elif "spirit" in term or "soul" in term or "divine" in term:
            term_categories[term] = "mystical"
        elif any(nature in term for nature in ["mountain", "ocean", "forest", "river"]):
            term_categories[term] = "natural"
        elif any(emotion in term for emotion in ["joy", "peace", "love", "serenity"]):
            term_categories[term] = "emotional"
        else:
            term_categories[term] = "abstract"
    
    # Create plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color mapping with more vibrant colors
    colors = {
        "elements": "#FF6B6B",  # Coral red
        "celestial": "#4ECDC4",  # Turquoise
        "mystical": "#9B59B6",  # Purple
        "natural": "#2ECC71",   # Emerald green
        "emotional": "#F1C40F", # Yellow
        "abstract": "#95A5A6"   # Gray
    }
    
    # Plot points by category with enhanced styling
    for category in colors:
        mask = [term_categories.get(term, "abstract") == category for term in manifold.terms]
        if any(mask):
            scatter = ax.scatter(
                points[mask, 0], points[mask, 1], points[mask, 2],
                c=colors[category],
                label=category,
                alpha=0.7,
                s=100,  # Larger points
                edgecolors='white',  # White edges
                linewidth=1
            )
    
    # Add labels for key terms with improved positioning
    if show_labels:
        for i, term in enumerate(manifold.terms):
            if term in ["fire", "water", "earth", "air", "spirit", "soul", "sun", "moon"]:
                ax.text(
                    points[i, 0], points[i, 1], points[i, 2],
                    term,
                    fontsize=10,
                    fontweight='bold',
                    backgroundcolor='white',
                    alpha=0.7
                )
    
    # Enhance the plot style
    ax.set_title(f'Term Clusters (Level {level})', fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.15, 1), fontsize=10)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add grid for better depth perception
    ax.grid(True, alpha=0.3)
    
    # Rotate the view for better perspective
    ax.view_init(elev=20, azim=45)
    
    return fig

def plot_multimodal_clusters(manifold, level=0, visualizer: Optional[MultimodalVisualizer] = None, show_labels=True):
    """Create a multimodal visualization combining semantic and visual embeddings."""
    if not hasattr(manifold, 'recursive_reduced'):
        raise ValueError("Must run recursive_reduce() first")
    
    # Get level data
    level_data = find_level_data(manifold.recursive_reduced, level)
    if level_data is None:
        raise ValueError(f"Level {level} not found")
    
    points = level_data['points']
    
    # Group terms by category
    categories = {
        "elements": [],
        "celestial": [],
        "mystical": [],
        "natural": [],
        "emotional": [],
        "abstract": []
    }
    
    for term in manifold.terms:
        if "fire" in term or "water" in term or "earth" in term or "air" in term:
            categories["elements"].append(term)
        elif "sun" in term or "moon" in term or "star" in term:
            categories["celestial"].append(term)
        elif "spirit" in term or "soul" in term or "divine" in term:
            categories["mystical"].append(term)
        elif any(nature in term for nature in ["mountain", "ocean", "forest", "river"]):
            categories["natural"].append(term)
        elif any(emotion in term for emotion in ["joy", "peace", "love", "serenity"]):
            categories["emotional"].append(term)
        else:
            categories["abstract"].append(term)
    
    # Create a figure with subplots for each category
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 6, figure=fig)
    
    # Color scheme
    colors = {
        "elements": "#FF6B6B",
        "celestial": "#4ECDC4",
        "mystical": "#9B59B6",
        "natural": "#2ECC71",
        "emotional": "#F1C40F",
        "abstract": "#95A5A6"
    }
    
    # Plot each category
    for i, (category, terms) in enumerate(categories.items()):
        # 3D scatter plot
        ax_3d = fig.add_subplot(gs[0:2, i], projection='3d')
        
        # Get indices for terms in this category
        indices = [i for i, term in enumerate(manifold.terms) if term in terms]
        
        if indices:
            category_points = points[indices]
            
            # Plot points
            ax_3d.scatter(
                category_points[:, 0],
                category_points[:, 1],
                category_points[:, 2],
                c=colors[category],
                s=100,
                alpha=0.7,
                edgecolors='white',
                linewidth=1
            )
            
            # Add labels
            if show_labels:
                for point, term in zip(category_points, terms):
                    ax_3d.text(
                        point[0], point[1], point[2],
                        term,
                        fontsize=8,
                        alpha=0.7,
                        backgroundcolor='white'
                    )
        
        ax_3d.set_title(f'{category.title()}', fontsize=12)
        ax_3d.grid(True, alpha=0.3)
        ax_3d.view_init(elev=20, azim=45)
        
        # Image subplot
        if visualizer:
            ax_img = fig.add_subplot(gs[2, i])
            img = visualizer.get_image_for_category(category, terms)
            if img:
                ax_img.imshow(img)
                ax_img.axis('off')
                
                # Get visual embedding and find most similar terms
                visual_emb = visualizer.get_visual_embedding(img)
                term_similarities = []
                for term in terms:
                    text_emb = visualizer.get_text_embedding(term)
                    similarity = cosine_similarity(visual_emb.reshape(1, -1), text_emb.reshape(1, -1))[0][0]
                    term_similarities.append((term, similarity))
                
                # Add top similar terms as subtitle
                top_terms = sorted(term_similarities, key=lambda x: x[1], reverse=True)[:3]
                term_str = ", ".join(f"{term}" for term, _ in top_terms)
                ax_img.set_title(f'Key terms: {term_str}', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_source_cube(manifold: VectorManifold, n_terms: int = 8) -> Dict[str, Any]:
    """
    Create a semantic source cube visualization.
    
    Args:
        manifold: The VectorManifold instance
        n_terms: Number of terms to place at cube vertices (default 8)
        
    Returns:
        Dictionary containing cube visualization data
    """
    # Get the most semantically diverse terms
    terms = list(manifold.terms)
    embeddings = np.array([manifold.embeddings.get_embedding(term) for term in terms])
    
    # Calculate pairwise distances
    distances = 1 - cosine_similarity(embeddings)
    
    # Find maximally distant terms
    selected_indices = []
    current_idx = np.random.randint(len(terms))  # Start with random term
    selected_indices.append(current_idx)
    
    while len(selected_indices) < n_terms:
        # Calculate average distance to selected terms
        avg_distances = np.mean([distances[i] for i in selected_indices], axis=0)
        # Find furthest term that isn't already selected
        mask = np.ones(len(terms), dtype=bool)
        mask[selected_indices] = False
        current_idx = np.argmax(avg_distances * mask)
        selected_indices.append(current_idx)
    
    selected_terms = [terms[i] for i in selected_indices]
    selected_embeddings = embeddings[selected_indices]
    
    # Create cube vertices in 3D
    cube_vertices = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    
    # Map terms to cube vertices based on semantic similarity
    term_positions = np.zeros((n_terms, 3))
    assigned_vertices = set()
    
    for i, embedding in enumerate(selected_embeddings):
        # Find best unassigned vertex
        best_vertex = None
        best_score = float('-inf')
        
        for j, vertex in enumerate(cube_vertices):
            if j not in assigned_vertices:
                # Score based on semantic relationships
                score = 0
                for k, other_embedding in enumerate(selected_embeddings[:i]):
                    if k != i:
                        # Compare semantic distance with geometric distance
                        semantic_dist = distances[selected_indices[i], selected_indices[k]]
                        geometric_dist = np.linalg.norm(vertex - term_positions[k])
                        score -= abs(semantic_dist - geometric_dist/2)  # Scale geometric distance
                
                if score > best_score:
                    best_score = score
                    best_vertex = j
        
        term_positions[i] = cube_vertices[best_vertex]
        assigned_vertices.add(best_vertex)
    
    # Create edges for visualization
    edges = []
    for i in range(8):
        for j in range(i+1, 8):
            # Connect if vertices differ in only one dimension
            if np.sum(np.abs(cube_vertices[i] - cube_vertices[j])) == 2:
                edges.append((i, j))
    
    return {
        'terms': selected_terms,
        'positions': term_positions,
        'edges': edges,
        'embeddings': selected_embeddings
    }

def plot_source_cube(cube_data: Dict[str, Any], ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot the semantic source cube.
    
    Args:
        cube_data: Cube visualization data from create_source_cube
        ax: Optional matplotlib axes to plot on
        
    Returns:
        The matplotlib axes object
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot vertices
    ax.scatter(
        cube_data['positions'][:, 0],
        cube_data['positions'][:, 1],
        cube_data['positions'][:, 2],
        c='r', s=100
    )
    
    # Plot edges
    for edge in cube_data['edges']:
        start = cube_data['positions'][edge[0]]
        end = cube_data['positions'][edge[1]]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            'k-', alpha=0.5
        )
    
    # Add term labels
    for term, pos in zip(cube_data['terms'], cube_data['positions']):
        ax.text(
            pos[0], pos[1], pos[2],
            term,
            fontsize=8,
            bbox=dict(
                facecolor='white',
                alpha=0.7,
                edgecolor='none'
            )
        )
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Set labels
    ax.set_xlabel('Semantic Dimension 1')
    ax.set_ylabel('Semantic Dimension 2')
    ax.set_zlabel('Semantic Dimension 3')
    ax.set_title('Semantic Source Cube')
    
    return ax

def main():
    # Load pre-trained embeddings
    embeddings = load_embeddings()
    
    # Create manifold with reduced dimensionality for visualization
    manifold = VectorManifold(
        embeddings=embeddings,
        n_cells=22,  # Default to 22 cells (major arcana)
        random_state=93,
        reduction_dims=3,  # Use 3D for better visualization
        use_fractals=True,
        fractal_depth=2  # Reduced from 3 to 2 for stability
    )
    
    # Create source cube visualization
    cube_data = create_source_cube(manifold)
    
    # Plot source cube
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_source_cube(cube_data, ax)
    plt.show()

if __name__ == "__main__":
    main() 