import pyglet
import pyglet.graphics
import pyglet.text
import torch
import os
import sys
import time # For a small delay if needed, though animation uses clock
import numpy as np # Added for loading .npz data

# --- Path Setup ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(script_dir) # Assuming script is at project root
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from word_manifold.embeddings.word_embeddings import WordEmbeddings
    from word_manifold.visualization.visualization_manager import VectorDrivenVisualizer
except ImportError as e:
    print(f"Error setting up path or importing modules: {e}")
    print("Ensure 'src' directory and necessary modules are present.")
    sys.exit(1)

class EvolvingMandalaViewer(pyglet.window.Window):
    def __init__(self, width, height, title, text_sequence):
        super().__init__(width, height, title, resizable=True)
        self.text_sequence = text_sequence
        self.current_text_index = -1 # Will be incremented to 0 for the first item

        self.mandala_elements_batch = pyglet.graphics.Batch()
        self.labels = []

        # Animation state variables
        self.current_text_string = None
        self.current_embedding_vector = None
        self.next_text_string = None
        self.next_embedding_vector = None
        
        self.is_animating = False
        self.animation_fps = 30.0
        self.animation_duration = 1.5  # seconds for one transition
        self.animation_total_steps = int(self.animation_fps * self.animation_duration)
        self.animation_current_step = 0
        self.inter_mandala_delay = 0.75 # seconds to pause on a stable mandala

        # Initialize word embeddings manager
        cache_dir_path = os.path.join(project_root, 'data', 'embeddings_cache')
        os.makedirs(cache_dir_path, exist_ok=True)
        try:
            self.embeddings_manager = WordEmbeddings(cache_dir=cache_dir_path)
            print(f"Initialized with model: {self.embeddings_manager.model_name}")
        except Exception as e:
            print(f"Error initializing WordEmbeddings: {e}")
            self.embeddings_manager = None
            # Optionally, close the window or show an error message
            pyglet.app.exit()
            return

        self.visualizer = VectorDrivenVisualizer(mandala_size=21) # Default mandala size

        # Start the process
        pyglet.clock.schedule_once(self.load_and_display_next_mandala, 0) 

    def get_embedding_for_text(self, text_input):
        if not self.embeddings_manager:
            return None
        try:
            # Get embeddings for a list containing a single text
            embeddings_dict = self.embeddings_manager.get_embeddings([text_input])
            embedding_vector = embeddings_dict.get(text_input)
            
            if embedding_vector is not None:
                # Convert numpy array (if it is) to torch tensor
                if not isinstance(embedding_vector, torch.Tensor):
                    embedding_vector = torch.tensor(embedding_vector, dtype=torch.float32)
                return embedding_vector
            else:
                print(f"Could not retrieve embedding for: {text_input}")
                return None
        except Exception as e:
            print(f"Error getting embedding for '{text_input}': {e}")
            return None

    def draw_mandala_for_embedding(self, embedding_tensor):
        for label in self.labels:
            label.delete()
        self.labels = []
        self.mandala_elements_batch = pyglet.graphics.Batch()

        if embedding_tensor is None:
            # Optionally draw a placeholder or clear screen if no embedding
            return

        mandala_elements = self.visualizer.get_mandala_elements(embedding_tensor)
        
        if mandala_elements:
            grid_size = self.visualizer.mandala_size
            # Handle potential division by zero if window size is tiny before first resize event
            scale_x = self.width / grid_size if grid_size else self.width 
            scale_y = self.height / grid_size if grid_size else self.height
            font_size = max(1, int(min(scale_x, scale_y) * 0.65))

            for element in mandala_elements:
                draw_x = (element['x'] + 0.5) * scale_x
                draw_y = self.height - (element['y'] + 0.5) * scale_y 
                
                label = pyglet.text.Label(
                    element['symbol'], font_name='Monospace', font_size=font_size,
                    x=draw_x, y=draw_y, anchor_x='center', anchor_y='center',
                    batch=self.mandala_elements_batch
                )
                self.labels.append(label)

    def load_and_display_next_mandala(self, dt=None):
        self.current_text_index += 1
        if self.current_text_index >= len(self.text_sequence):
            print("Finished all texts in the sequence.")
            # pyglet.app.exit() # Option to close when done
            return

        next_text_candidate = self.text_sequence[self.current_text_index]
        print(f"Preparing for: {next_text_candidate} ({self.current_text_index + 1}/{len(self.text_sequence)})")
        
        new_embedding = self.get_embedding_for_text(next_text_candidate)

        if new_embedding is None:
            print(f"Failed to get embedding for {next_text_candidate}. Skipping.")
            # Try to load the next one immediately without animation or long pause
            pyglet.clock.schedule_once(self.load_and_display_next_mandala, 0.1) 
            return

        if self.current_embedding_vector is None: # First valid mandala
            self.current_text_string = next_text_candidate
            self.current_embedding_vector = new_embedding
            self.set_caption(f"Evolving Mandala - {self.current_text_string}")
            self.draw_mandala_for_embedding(self.current_embedding_vector)
            # Schedule preparation for the *next* animation after a pause
            pyglet.clock.schedule_once(self.load_and_display_next_mandala, self.inter_mandala_delay)
        else: # We have a current mandala, animate to the new one
            self.next_text_string = next_text_candidate
            self.next_embedding_vector = new_embedding
            
            self.is_animating = True
            self.animation_current_step = 0
            self.set_caption(f"Evolving: {self.current_text_string} -> {self.next_text_string}")
            pyglet.clock.schedule_interval(self.animate_frame, 1.0 / self.animation_fps)

    def animate_frame(self, dt):
        if not self.is_animating or self.current_embedding_vector is None or self.next_embedding_vector is None:
            pyglet.clock.unschedule(self.animate_frame)
            self.is_animating = False
            # This case might indicate an issue or end of animation without proper completion
            # Ensure we attempt to load the next mandala if something went wrong here
            if not (self.current_text_index +1 >= len(self.text_sequence)):
                 pyglet.clock.schedule_once(self.load_and_display_next_mandala, self.inter_mandala_delay)
            return

        self.animation_current_step += 1
        t = self.animation_current_step / self.animation_total_steps

        if t >= 1.0:
            t = 1.0
            self.is_animating = False
            pyglet.clock.unschedule(self.animate_frame)
            
            self.current_text_string = self.next_text_string
            self.current_embedding_vector = self.next_embedding_vector
            self.next_text_string = None
            self.next_embedding_vector = None
            
            self.set_caption(f"Evolving Mandala - {self.current_text_string}")
            # Final draw of the target mandala
            self.draw_mandala_for_embedding(self.current_embedding_vector) 

            # Schedule the next cycle (load next, which might lead to another animation)
            pyglet.clock.schedule_once(self.load_and_display_next_mandala, self.inter_mandala_delay)
        else:
            interpolated_vector = torch.lerp(self.current_embedding_vector, self.next_embedding_vector, t)
            self.draw_mandala_for_embedding(interpolated_vector)

    def on_draw(self):
        self.clear()
        self.mandala_elements_batch.draw()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        # If animating, the current frame will adapt. If stable, the existing one is redrawn.
        # For an immediate update of a stable mandala on resize:
        if not self.is_animating and self.current_embedding_vector is not None:
            self.draw_mandala_for_embedding(self.current_embedding_vector)
        # During animation, next animate_frame call will use new width/height for scale. 

    def on_close(self):
        print("Shutting down embeddings manager...")
        if self.embeddings_manager:
            self.embeddings_manager.shutdown()
        super().on_close()


def load_terms_from_npz(npz_path):
    try:
        data = np.load(npz_path, allow_pickle=True)
        terms = data['terms']
        if isinstance(terms, np.ndarray):
            terms = terms.tolist() # Convert numpy array to list of strings
        print(f"Successfully loaded {len(terms)} terms from {npz_path}")
        return terms
    except FileNotFoundError:
        print(f"Warning: Data file not found at {npz_path}. Using default text sequence.")
        return None
    except Exception as e:
        print(f"Warning: Error loading terms from {npz_path}: {e}. Using default text sequence.")
        return None

if __name__ == '__main__':
    project_root_path = os.path.dirname(os.path.abspath(__file__))
    umap_data_path = os.path.join(project_root_path, 'visualizations', 'umap_default_terms_data.npz')
    
    # Attempt to load terms from UMAP data
    visualization_terms = load_terms_from_npz(umap_data_path)

    default_texts = [
        "hello world", 
        "pyglet and python", 
        "evolving matter", 
        "computational creativity",
        "semantic space",
        "vector geometry",
        "neural networks art",
        "algorithmic beauty"
    ]

    if visualization_terms:
        texts_to_use = visualization_terms
    else:
        texts_to_use = default_texts
    
    if not texts_to_use:
        print("Error: No terms available to display. Exiting.")
        sys.exit(1)

    window_width = 600
    window_height = 600
    
    print("Starting Evolving Mandala Viewer...")
    print("This may take a moment to download/load the sentence embedding model on first run.")
    
    viewer = EvolvingMandalaViewer(window_width, window_height, "Evolving Mandala", texts_to_use)
    
    if viewer.embeddings_manager: # Only run if manager initialized
        pyglet.app.run()
    else:
        print("Failed to initialize embeddings manager. Exiting.")
    
    print("Evolving Mandala Viewer closed.") 