import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Path Setup ---
# Add the 'src' directory to sys.path to allow importing 'word_manifold'
# Assumes this script is in 'word-manifold/scripts/' and 'src' is 'word-manifold/src/'
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from word_manifold.embeddings.word_embeddings import WordEmbeddings
except ImportError as e:
    print(f"Error setting up path or importing WordEmbeddings: {e}")
    print("Please ensure the script is in the 'scripts' directory of the 'word-manifold' project,")
    print("and the 'src' directory with 'word_manifold' package is present at the project root.")
    sys.exit(1)

# --- UMAP Import ---
try:
    import umap
except ImportError:
    print("UMAP library not found. Please install it: pip install umap-learn matplotlib")
    sys.exit(1)

def generate_and_plot_umap():
    """
    Loads word embeddings, generates a UMAP projection, and plots it.
    """
    print("Initializing WordEmbeddings manager...")
    cache_dir_path = os.path.join(project_root, 'data', 'embeddings_cache')
    os.makedirs(cache_dir_path, exist_ok=True) # Ensure cache directory exists

    try:
        # Using a cache_dir relative to the project root
        embeddings_manager = WordEmbeddings(cache_dir=cache_dir_path)
        print(f"Initialized with model: {embeddings_manager.model_name}")
    except Exception as e:
        print(f"Error initializing WordEmbeddings: {e}")
        print("This might be due to model download issues or configuration problems.")
        return

    print("Loading default terms...")
    # __init__ of WordEmbeddings loads default terms.
    loaded_terms_set = embeddings_manager.get_terms()

    if not loaded_terms_set:
        print("No terms loaded automatically. Attempting to load default terms explicitly.")
        embeddings_manager.load_terms(embeddings_manager.DEFAULT_TERMS)
        loaded_terms_set = embeddings_manager.get_terms()

    if not loaded_terms_set:
        print("Failed to load any terms. Please check embedding model and configuration.")
        embeddings_manager.shutdown()
        return

    print(f"Successfully loaded {len(loaded_terms_set)} terms.")
    term_list = sorted(list(loaded_terms_set))

    print("Fetching embeddings for UMAP...")
    embeddings_dict = embeddings_manager.get_embeddings(term_list)

    valid_terms = [term for term in term_list if term in embeddings_dict and embeddings_dict[term] is not None]
    embedding_vectors = [embeddings_dict[term] for term in valid_terms]

    if not embedding_vectors:
        print("No embedding vectors retrieved. Cannot proceed with UMAP.")
        embeddings_manager.shutdown()
        return

    if len(valid_terms) < 2:
        print(f"Only {len(valid_terms)} term(s) with embeddings. UMAP requires at least 2 samples to create a projection.")
        embeddings_manager.shutdown()
        return
        
    word_embeddings_array = np.array(embedding_vectors)
    print(f"Embedding array shape for UMAP: {word_embeddings_array.shape}")

    print("Running UMAP...")
    # Adjust UMAP parameters: n_neighbors must be less than the number of samples.
    n_samples = word_embeddings_array.shape[0]
    # Ensure n_neighbors is at least 2 if possible, and less than n_samples.
    n_neighbors = min(max(2, n_samples // 2), 15) # Heuristic: half of samples up to 15, min 2.
    if n_samples <= n_neighbors: # If n_samples is too small (e.g. <4 for the heuristic above)
        n_neighbors = max(1, n_samples - 1) # Fallback: UMAP needs n_neighbors < n_samples

    if n_neighbors == 0 and n_samples == 1: # UMAP can't run with 1 sample.
         print("UMAP cannot run with only 1 sample.")
         embeddings_manager.shutdown()
         return
    if n_samples > 0 and n_neighbors == 0 : # if n_samples is 1, n_neighbors becomes 0 with max(1, n_samples-1)
        n_neighbors = 1 # UMAP might still handle n_neighbors=1 for n_samples > 1, or warn. For safety for n_samples=1, we already exited.

    try:
        reducer = umap.UMAP(
            n_components=2, 
            random_state=42, 
            n_neighbors=n_neighbors, 
            min_dist=0.1,
            metric='cosine' # Good for semantic similarity
        )
        projected_embeddings_2d = reducer.fit_transform(word_embeddings_array)
        print(f"Projected embeddings shape: {projected_embeddings_2d.shape}")
    except Exception as e:
        print(f"Error during UMAP processing: {e}")
        embeddings_manager.shutdown()
        return

    # --- Plotting ---
    print("Plotting UMAP projection...")
    plt.figure(figsize=(14, 12))
    plt.scatter(projected_embeddings_2d[:, 0], projected_embeddings_2d[:, 1], s=15, alpha=0.7)
    for i, term in enumerate(valid_terms):
        plt.annotate(term, (projected_embeddings_2d[i, 0], projected_embeddings_2d[i, 1]), fontsize=9)
    
    plt.title(f'UMAP Projection of Default Word Embeddings ({embeddings_manager.model_name}) - {len(valid_terms)} terms', fontsize=14)
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    output_plot_dir = os.path.join(project_root, 'visualizations')
    os.makedirs(output_plot_dir, exist_ok=True)
    output_plot_path = os.path.join(output_plot_dir, 'umap_default_terms_canvas.png')
    
    try:
        plt.savefig(output_plot_path)
        print(f"UMAP plot saved to: {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # plt.show() # Uncomment if you want to display the plot interactively

    print("Shutting down embeddings manager...")
    embeddings_manager.shutdown()
    print("Script finished.")

if __name__ == "__main__":
    generate_and_plot_umap() 