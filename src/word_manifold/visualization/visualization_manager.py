import torch
import math # Added for mandala calculations
import logging # Added for consistency and good practice

logger = logging.getLogger(__name__)

# You might need to import ASCIIEngine or other utilities if you reuse them
# from ..engines.ascii import ASCIIEngine # Example

class VectorDrivenVisualizer:
    def __init__(self, mandala_size: int = 21, concretion_size: int = 15):
        self.mandala_size = mandala_size  # Conceptual size, influences scaling
        self.concretion_size = concretion_size # Kept for completeness, not used in ASCII mandala
        
        # ASCII symbols from the more complete version
        self.mandala_symbols = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
        # self.concretion_symbols = ['o', '.', ':', '-', '>', '<', '^', 'v', '*'] # Not used now
        
        # Attributes from the more complete version, not strictly needed for ASCII mandala but kept for structure
        # self.mandala_pyglet_shapes = []
        # self.concretion_pyglet_shapes = [] 
        # self.previous_driving_vector: Optional[torch.Tensor] = None 
        # self.debug_star: Optional[pyglet_shapes.Star] = None # Pyglet specific
        logger.info(f"VectorDrivenVisualizer initialized (mandala_size: {self.mandala_size})")

    def _get_driving_vector(self, pattern: torch.Tensor, num_elements: int = 5) -> torch.Tensor:
        """Extracts or derives a driving vector from the pattern."""
        if not isinstance(pattern, torch.Tensor):
            # The skeleton had a TypeError check, which is good.
            raise TypeError(f"Input must be a PyTorch Tensor, got {type(pattern)}")

        if pattern.numel() == 0:
            logger.warning("_get_driving_vector received an empty tensor. Returning zeros.")
            return torch.zeros(num_elements, device=pattern.device, dtype=pattern.dtype)
        
        # Use tanh to bring values into a somewhat predictable range [-1, 1]
        # Then scale to [0,1] for easier use in parameter mapping
        pattern_flat = torch.tanh(pattern.flatten()) * 0.5 + 0.5 

        if pattern_flat.numel() >= num_elements:
            return pattern_flat[:num_elements]
        else:
            # Repeat and slice if the pattern is too short
            logger.warning(f"_get_driving_vector input pattern is shorter ({pattern_flat.numel()}) than num_elements ({num_elements}). Repeating.")
            return pattern_flat.repeat((num_elements // pattern_flat.numel()) + 1)[:num_elements]

    def create_vector_mandala(self, pattern: torch.Tensor) -> str:
        """Creates an ASCII mandala driven by the input pattern tensor."""
        # Ensure pattern is a tensor, consistent with _get_driving_vector
        if not isinstance(pattern, torch.Tensor):
            raise TypeError(f"Input pattern must be a PyTorch Tensor, got {type(pattern)}")

        # Using num_elements=5 as per the original ASCII mandala logic
        dv = self._get_driving_vector(pattern, num_elements=5)

        # Map driving vector components to mandala parameters
        # Ensure values are in reasonable ranges
        # .item() is used to get Python numbers from 0-dim tensors
        num_segments = max(3, int(dv[0].item() * 7) + 3)  # e.g., 3 to 10 segments
        layers = max(2, int(dv[1].item() * (self.mandala_size / 4)) + 1) # e.g., 2 to ~6 layers
        symbol_offset = int(dv[2].item() * (len(self.mandala_symbols) - 5)) # Start symbol index
        hue_rotation_factor = dv[3].item() * 360  # For potential color mapping later, now affects symbol choice
        density = dv[4].item() * 0.5 + 0.3  # Base density 0.3 to 0.8

        grid = [[' ' for _ in range(self.mandala_size)] for _ in range(self.mandala_size)]
        center_x, center_y = self.mandala_size // 2, self.mandala_size // 2

        for l_idx in range(1, layers + 1): # Corrected loop variable name from 'l' to 'l_idx'
            radius = l_idx * (center_x / layers) * 0.9 # Scale radius by number of layers
            points_in_layer = int(num_segments * l_idx * density) # More points in outer, denser layers
            if points_in_layer == 0: # Avoid division by zero if density makes it zero
                continue

            for i in range(points_in_layer):
                angle_rad = (2 * math.pi / points_in_layer) * i + (hue_rotation_factor * l_idx /180 * math.pi) # Rotate layers
                
                x_offset = radius * math.cos(angle_rad)
                y_offset = radius * math.sin(angle_rad)
                
                x = center_x + int(round(x_offset)) # round to nearest int for grid index
                y = center_y + int(round(y_offset)) # round to nearest int for grid index

                if 0 <= x < self.mandala_size and 0 <= y < self.mandala_size:
                    symbol_index = (l_idx + i + symbol_offset + int(hue_rotation_factor/36)) % len(self.mandala_symbols)
                    grid[y][x] = self.mandala_symbols[symbol_index]
        
        # Reflect to create symmetry
        for r in range(self.mandala_size):
            for c in range(center_x):
                grid[r][self.mandala_size - 1 - c] = grid[r][c]
        for c in range(self.mandala_size):
            for r in range(center_y):
                 grid[self.mandala_size - 1 - r][c] = grid[r][c]

        return "\n".join("".join(row) for row in grid)

    def get_mandala_elements(self, pattern: torch.Tensor) -> list[dict]:
        """
        Calculates mandala elements based on the input pattern tensor.
        Returns a list of dictionaries, each representing a point in the mandala
        with its coordinates, symbol, and layer information.
        """
        if not isinstance(pattern, torch.Tensor):
            raise TypeError(f"Input pattern must be a PyTorch Tensor, got {type(pattern)}")

        dv = self._get_driving_vector(pattern, num_elements=5)

        num_segments = max(3, int(dv[0].item() * 7) + 3)
        layers = max(2, int(dv[1].item() * (self.mandala_size / 4)) + 1)
        symbol_offset = int(dv[2].item() * (len(self.mandala_symbols) - 5))
        hue_rotation_factor = dv[3].item() * 360
        density = dv[4].item() * 0.5 + 0.3

        elements = []
        center_x, center_y = self.mandala_size // 2, self.mandala_size // 2

        raw_points = [] # Store (x, y, symbol_index, l_idx) before reflection

        for l_idx in range(1, layers + 1):
            radius = l_idx * (center_x / layers) * 0.9
            points_in_layer = int(num_segments * l_idx * density)
            if points_in_layer == 0:
                continue

            for i in range(points_in_layer):
                angle_rad = (2 * math.pi / points_in_layer) * i + (hue_rotation_factor * l_idx / 180 * math.pi)
                
                x_offset = radius * math.cos(angle_rad)
                y_offset = radius * math.sin(angle_rad)
                
                # Store floating point coordinates before rounding for potential higher precision rendering
                # For grid-based symbol assignment, we still use rounded coords for now.
                grid_x = center_x + int(round(x_offset))
                grid_y = center_y + int(round(y_offset))

                if 0 <= grid_x < self.mandala_size and 0 <= grid_y < self.mandala_size:
                    symbol_idx = (l_idx + i + symbol_offset + int(hue_rotation_factor/36)) % len(self.mandala_symbols)
                    # Store original calculated point
                    raw_points.append({
                        'x': grid_x, 
                        'y': grid_y, 
                        'symbol': self.mandala_symbols[symbol_idx], 
                        'layer': l_idx,
                        'angle_rad': angle_rad, # For potential use in Pyglet shape orientation
                        'radius': radius       # For potential use in Pyglet shape size scaling
                    })

        # Apply reflection to the collected raw_points
        # Current reflection logic in create_vector_mandala is grid-based.
        # For element-based reflection, we can iterate through raw_points and add their reflections.
        # This approach might lead to duplicate points if a point is on an axis of reflection.
        # A grid-based approach first, then converting to elements, is safer for visual consistency with ASCII.

        # Let's build a temporary grid to handle reflections correctly, then extract elements.
        temp_grid = {} # Using dict for sparse grid: (r, c) -> {symbol, layer, ...}

        for p in raw_points:
            # For now, use the grid_x, grid_y for simplicity in reflection, matching ASCII
            # Overwrite if multiple calculations map to the same grid cell (last one wins, similar to grid[y][x] = ...)
            temp_grid[(p['y'], p['x'])] = {'symbol': p['symbol'], 'layer': p['layer']}


        # Reflect across Y-axis (vertical reflection for each row)
        reflected_y_axis = {}
        for (r, c), data in temp_grid.items():
            reflected_y_axis[(r, c)] = data
            if c < center_x: # Reflect points from left to right
                 reflected_y_axis[(r, self.mandala_size - 1 - c)] = data
            elif c > center_x: # Reflect points from right to left if center not included
                 reflected_y_axis[(r, self.mandala_size - 1 - c)] = data
        
        # Reflect across X-axis (horizontal reflection for each column)
        final_elements_map = {} # Use a map to avoid duplicate elements from reflection
        for (r, c), data in reflected_y_axis.items():
            final_elements_map[(r,c)] = data
            if r < center_y: # Reflect points from top to bottom
                final_elements_map[(self.mandala_size - 1 - r, c)] = data
            elif r > center_y: # Reflect points from bottom to top
                final_elements_map[(self.mandala_size - 1 - r, c)] = data

        # Convert the final map of elements to the desired list of dictionaries
        for (r, c), data in final_elements_map.items():
            elements.append({
                'x': c,  # Grid column is x
                'y': r,  # Grid row is y
                'symbol': data['symbol'],
                'layer': data['layer']
                # We can add original float x_offset, y_offset if needed for smoother Pyglet rendering
            })
            
        return elements

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Testing VectorDrivenVisualizer from src/word_manifold/visualization/visualization_manager.py...")
    visualizer = VectorDrivenVisualizer(mandala_size=25) # Testing with a slightly larger size
    
    # Create a dummy tensor
    dummy_vector_1 = torch.rand(50) 
    print(f"\nCreating mandala for dummy vector 1 (shape: {dummy_vector_1.shape}):")
    mandala_1 = visualizer.create_vector_mandala(dummy_vector_1)
    print(mandala_1)

    # Test with a different vector (more varied values)
    dummy_vector_2 = torch.randn(10) * 10 
    print(f"\nCreating mandala for dummy vector 2 (shape: {dummy_vector_2.shape}):")
    mandala_2 = visualizer.create_vector_mandala(dummy_vector_2)
    print(mandala_2)

    # Test with short vector
    dummy_vector_3 = torch.tensor([0.1, 0.9, 0.5])
    print(f"\nCreating mandala for dummy vector 3 (short) (shape: {dummy_vector_3.shape}):")
    mandala_3 = visualizer.create_vector_mandala(dummy_vector_3)
    print(mandala_3)

    # Test with empty tensor
    # Note: _get_driving_vector now handles this by returning zeros, leading to a default mandala
    dummy_vector_empty = torch.empty(0)
    print(f"\nCreating mandala for empty dummy vector (shape: {dummy_vector_empty.shape}):")
    mandala_empty = visualizer.create_vector_mandala(dummy_vector_empty)
    print(mandala_empty)

    # Test with zero tensor
    dummy_vector_zeros = torch.zeros(10)
    print(f"\nCreating mandala for zero dummy vector (shape: {dummy_vector_zeros.shape}):")
    mandala_zeros = visualizer.create_vector_mandala(dummy_vector_zeros)
    print(mandala_zeros) 