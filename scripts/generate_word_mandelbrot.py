import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch # Added for tensor conversion
import pyglet # Added for graphical mandalas
from pyglet import shapes # Added for drawing shapes
from pyglet.gl import GL_POINTS # For drawing points if needed
from pyglet import text as pyglet_text     # For drawing text symbols
from pyglet import graphics as pyglet_graphics # For batch rendering

# --- Path Setup for potential imports from src, if needed later ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    # Import VectorDrivenVisualizer
    from word_manifold.visualization.visualization_manager import VectorDrivenVisualizer
except ImportError as e:
    print(f"Warning: Could not set up src path or import VectorDrivenVisualizer: {e}")
    print("Ensure 'visualization_manager.py' is in 'word_manifold/visualization/'")
    # For simplicity, we'll let it proceed and fail later if VDV is actually used and not found.

# --- Global Variables for Click Handler (alternative to passing around) ---
# These will be populated in generate_and_plot_word_mandelbrot
g_projected_embeddings_2d = None
g_word_embeddings_array = None
g_terms = None
g_vector_visualizer = None
g_fig = None
g_ax = None

# --- Configuration ---
MAX_ITERATIONS = 50          # Number of iterations for the Mandelbrot-like process
STABILITY_THRESHOLD = 0.9    # Cosine similarity threshold to be considered 'stable' with original word
CHAOS_STRENGTH = 0.05        # Magnitude of the random perturbation
CLICK_RADIUS_THRESHOLD = 0.1 # How close a click needs to be to a point (in data coords)

# Pyglet mandala display config
MANDALA_WINDOW_SCALE = 20 # Scale factor for mandala elements in Pyglet window
MANDALA_POINT_SIZE = 10 # Size of the points/shapes in Pyglet

def calculate_word_mandelbrot_depth(hd_embedding, chaos_vector):
    """
    Calculates the 'Mandelbrot depth' for a single high-dimensional word embedding.
    Depth is defined as the number of iterations it remains stable (similar to itself)
    under a perturbed iterative process.
    """
    w_c = hd_embedding
    z = w_c # Start with the word itself
    iterations_stable = 0

    for _ in range(MAX_ITERATIONS):
        # Apply a fixed chaotic perturbation
        z_transformed = z + chaos_vector * CHAOS_STRENGTH
        if np.linalg.norm(z_transformed) > 1e-6: # Avoid division by zero
            z_transformed /= np.linalg.norm(z_transformed)
        else: # Should not happen with typical embeddings + chaos
            z_transformed = np.zeros_like(z)
            z_transformed[0] = 1.0 # Arbitrary unit vector

        # Re-center towards the original word w_c (averaging)
        z_perturbed = (z_transformed + w_c) / 2.0
        if np.linalg.norm(z_perturbed) > 1e-6:
            z_perturbed /= np.linalg.norm(z_perturbed)
        else:
            z_perturbed = w_c # Fallback to w_c if norm is zero

        # Check similarity to the original word
        # Assumes w_c and z_perturbed are normalized or their relative angle is the focus
        similarity_to_origin = np.dot(z_perturbed, w_c)

        if similarity_to_origin > STABILITY_THRESHOLD:
            iterations_stable += 1
        else:
            break # Escaped stability
        
        z = z_perturbed # Update for next iteration
    
    return iterations_stable

def get_color_for_symbol(symbol, layer, max_layers):
    """Maps a symbol and layer to a color (R, G, B)."""
    # Basic color mapping, can be expanded
    base_color = (100, 100, 100) # Default grey
    if symbol == '.': base_color = (255, 255, 255) # White
    elif symbol == ':': base_color = (200, 200, 0)   # Yellow
    elif symbol == '-': base_color = (0, 150, 255)   # Blue
    elif symbol == '=': base_color = (0, 200, 100)   # Green
    elif symbol == '+': base_color = (255, 100, 0)   # Orange
    elif symbol == '*': base_color = (255, 0, 0)     # Red
    elif symbol == '#': base_color = (150, 0, 200)   # Purple
    elif symbol == '%': base_color = (255, 0, 255)   # Magenta
    elif symbol == '@': base_color = (0, 255, 255)   # Cyan
    
    # Modify intensity based on layer (e.g., outer layers brighter)
    intensity_factor = 0.5 + 0.5 * (layer / max_layers)
    return tuple(int(c * intensity_factor) for c in base_color)

def on_click(event):
    """Handles mouse clicks on the Mandelbrot plot to display a mandala."""
    global g_projected_embeddings_2d, g_word_embeddings_array, g_terms, g_vector_visualizer, g_ax

    if event.inaxes != g_ax:
        return # Click was outside our plot axes

    if g_vector_visualizer is None:
        print("VectorVisualizer not initialized. Cannot generate mandala.")
        return

    click_x, click_y = event.xdata, event.ydata
    
    distances = np.sqrt((g_projected_embeddings_2d[:, 0] - click_x)**2 + 
                        (g_projected_embeddings_2d[:, 1] - click_y)**2)
    
    if distances.size == 0:
        return
        
    closest_idx = np.argmin(distances)
    
    if distances[closest_idx] < CLICK_RADIUS_THRESHOLD:
        term_clicked = g_terms[closest_idx]
        hd_embedding_clicked = g_word_embeddings_array[closest_idx]
        
        print(f"\n--- Clicked on: {term_clicked} ---")
        
        tensor_embedding = torch.from_numpy(hd_embedding_clicked).float()
        
        mandala_pyglet_window = None
        modal_event_loop = None
        try:
            print(f"Displaying Pyglet mandala for {term_clicked}. Close the Pyglet window to continue.")
            mandala_elements = g_vector_visualizer.get_mandala_elements(tensor_embedding)

            if not mandala_elements:
                print(f"No mandala elements generated for {term_clicked}.")
                return

            mandala_window_width = 400
            mandala_window_height = 400
            grid_size = g_vector_visualizer.mandala_size 
            scale_x = mandala_window_width / grid_size
            scale_y = mandala_window_height / grid_size
            font_size = max(1, int(min(scale_x, scale_y) * 0.75))

            # Tick Pyglet's clock to process any pending events before creating a new window/loop
            pyglet.clock.tick(poll=True) 

            mandala_pyglet_window = pyglet.window.Window(
                width=mandala_window_width,
                height=mandala_window_height,
                caption=f"Mandala: {term_clicked}"
            )
            mandala_batch = pyglet_graphics.Batch()
            pyglet_labels = [] 

            for element in mandala_elements:
                label_x_pos = (element['x'] + 0.5) * scale_x
                label_y_pos = mandala_window_height - (element['y'] + 0.5) * scale_y
                label = pyglet_text.Label(
                    element['symbol'],
                    font_name='Monospace',
                    font_size=font_size,
                    x=label_x_pos, y=label_y_pos,
                    anchor_x='center', anchor_y='center',
                    batch=mandala_batch
                )
                pyglet_labels.append(label)

            @mandala_pyglet_window.event
            def on_draw():
                mandala_pyglet_window.clear()
                mandala_batch.draw()

            modal_event_loop = pyglet.app.EventLoop()
            mandala_pyglet_window.on_close = modal_event_loop.exit
            
            modal_event_loop.run() 

            # After the loop exits (window closed), explicitly close the window 
            # This ensures resources are released promptly.
            if mandala_pyglet_window and not mandala_pyglet_window.has_exit:
                 # This case should ideally not be hit if on_close triggers exit correctly
                 mandala_pyglet_window.close()
            elif mandala_pyglet_window and mandala_pyglet_window.has_exit:
                 # If loop exited because has_exit is true, ensure close is called if not already.
                 # For pyglet.window.Window, close() can be called multiple times, though it only acts once.
                 # However, some pyglet versions might have issues if close() is called on an already fully destroyed window.
                 # A simple call to close() here is generally safe. If it was closed via UI, has_exit is True.
                 mandala_pyglet_window.close() 

        except Exception as e:
            print(f"Error generating or displaying Pyglet mandala for {term_clicked}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Final cleanup check. The main close should happen before this.
            if mandala_pyglet_window and not mandala_pyglet_window.has_exit:
                 # This is more of a safeguard; the window should have been closed above.
                try:
                    mandala_pyglet_window.close()
                except Exception as close_err:
                    print(f"Error during final cleanup close for {term_clicked}: {close_err}")
            print(f"Pyglet window processing for {term_clicked} finished.")

    else:
        print("No point close enough to click.")

def generate_and_plot_word_mandelbrot():
    """
    Loads UMAP data, calculates Mandelbrot depths, and plots the result interactively.
    """
    global g_projected_embeddings_2d, g_word_embeddings_array, g_terms, g_vector_visualizer, g_fig, g_ax

    print("Loading UMAP data...")
    data_path = os.path.join(project_root, 'visualizations', 'umap_default_terms_data.npz')
    
    try:
        data = np.load(data_path, allow_pickle=True)
        projected_embeddings_2d = data['projected_embeddings_2d']
        word_embeddings_array = data['word_embeddings_array']
        terms = data['terms']
        print(f"Loaded data for {len(terms)} terms.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please run 'generate_umap_canvas.py' first to create this file.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if word_embeddings_array.ndim == 1: # handles case of single embedding
        print("Warning: only one embedding found. Mandelbrot plot will have a single point.")
        # No need to reshape for iteration, but plotting will be trivial

    embedding_dim = word_embeddings_array.shape[-1]
    
    # Create a fixed chaos vector (once for all calculations)
    # Seed for reproducibility of the chaos vector itself, if desired
    # np.random.seed(42) # Uncomment for reproducible chaos vector
    chaos_vector = np.random.rand(embedding_dim) - 0.5
    if np.linalg.norm(chaos_vector) > 1e-6:
        chaos_vector /= np.linalg.norm(chaos_vector)
    else:
        chaos_vector = np.zeros(embedding_dim)
        chaos_vector[0] = 1.0 # Default if somehow zero norm

    print("Calculating Word Mandelbrot depths...")
    depths = np.array([
        calculate_word_mandelbrot_depth(hd_embedding, chaos_vector) 
        for hd_embedding in word_embeddings_array
    ])

    print("Plotting Word Mandelbrot...")
    plt.figure(figsize=(14, 12))
    
    # Use a colormap for the depths
    scatter = plt.scatter(
        projected_embeddings_2d[:, 0], 
        projected_embeddings_2d[:, 1], 
        c=depths, 
        cmap=cm.viridis, # Or cm.magma, cm.plasma, cm.coolwarm etc.
        s=50,         # Size of points
        alpha=0.8
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Mandelbrot Depth (Iterations Stable)', fontsize=12)

    # Annotate points with terms (optional, can be crowded)
    for i, term in enumerate(terms):
        plt.annotate(term, (projected_embeddings_2d[i, 0], projected_embeddings_2d[i, 1]), fontsize=8, alpha=0.7)
    
    plt.title(f'Word Mandelbrot (Stability Iterations, Max={MAX_ITERATIONS})', fontsize=14)
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_plot_dir = os.path.join(project_root, 'visualizations')
    # os.makedirs(output_plot_dir, exist_ok=True) # Already created by previous script
    output_plot_path = os.path.join(output_plot_dir, 'word_mandelbrot_plot.png')
    
    try:
        plt.savefig(output_plot_path)
        print(f"Word Mandelbrot plot saved to: {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # Initialize VectorDrivenVisualizer for click events
    try:
        g_vector_visualizer = VectorDrivenVisualizer() # Using default sizes
    except NameError: # If import failed
        print("Failed to initialize VectorDrivenVisualizer due to import error earlier.")
    except Exception as e:
        print(f"Error initializing VectorDrivenVisualizer: {e}")

    # Store data for click handler
    g_projected_embeddings_2d = projected_embeddings_2d
    g_word_embeddings_array = word_embeddings_array
    g_terms = terms
    g_fig = plt.gcf() # Get current figure
    g_ax = plt.gca()  # Get current axes

    # Connect the click event handler
    g_fig.canvas.mpl_connect('button_press_event', on_click)
    print("Click on a point in the plot to see its mandala graphically rendered.")
    print("Close the plot window to exit the script. Each mandala opens in a new window.")
    plt.show() # Now we show the plot and wait for interactions
    print("Script finished.")

if __name__ == "__main__":
    generate_and_plot_word_mandelbrot() 