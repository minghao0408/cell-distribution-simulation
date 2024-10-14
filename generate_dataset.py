import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def create_circular_region(size: int, radius: int) -> np.ndarray:
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    return dist_from_center <= radius

def simulate_cell_distribution(
    x_distribution,
    y_distribution,
    cell_count: int,
    allowed_region: np.ndarray,
    cell_interaction_radius: float,
    forbidden_region: np.ndarray = None,
    cell_radius: float = None,
    show_plots: bool = False
) -> Tuple[int, int, int, int, int, int]:
    grid_size = allowed_region.shape[0]
    cells = []
    
    # Generate all cells at once
    x_values = x_distribution.rvs(size=cell_count * 10)
    y_values = y_distribution.rvs(size=cell_count * 10)
    
    # Rescale values to [0, 1] range
    x_values = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
    y_values = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
    
    # Create cell placement mask
    cell_placement_mask = allowed_region.copy()
    if forbidden_region is not None:
        cell_placement_mask &= ~forbidden_region
    
    for i in range(len(x_values)):
        if len(cells) >= cell_count:
            break
        
        x, y = x_values[i], y_values[i]
        x_idx = min(int(x * grid_size), grid_size - 1)
        y_idx = min(int(y * grid_size), grid_size - 1)
        
        if cell_placement_mask[y_idx, x_idx]:
            if cell_radius is None or not any(np.hypot(x - cx, y - cy) < 2 * cell_radius for cx, cy in cells):
                cells.append((x, y))
                
                # Update cell_placement_mask if cell_radius is provided
                if cell_radius is not None:
                    cell_mask = create_circular_region(grid_size, int(2 * cell_radius * grid_size))
                    cell_center = (int(y * grid_size), int(x * grid_size))
                    y_start = max(0, cell_center[0] - cell_mask.shape[0]//2)
                    y_end = min(grid_size, cell_center[0] + cell_mask.shape[0]//2)
                    x_start = max(0, cell_center[1] - cell_mask.shape[1]//2)
                    x_end = min(grid_size, cell_center[1] + cell_mask.shape[1]//2)
                    cell_placement_mask[y_start:y_end, x_start:x_end] &= ~cell_mask[:y_end-y_start, :x_end-x_start]
    
    cells = np.array(cells)
    
    # Create interaction map
    interaction_map = np.zeros_like(allowed_region, dtype=int)
    y_indices, x_indices = np.ogrid[:grid_size, :grid_size]
    for cell in cells:
        distances = np.hypot(x_indices/grid_size - cell[0], y_indices/grid_size - cell[1])
        interaction_map += (distances <= cell_interaction_radius)
    
    # Apply allowed region mask
    if forbidden_region is not None:
        allowed_region = allowed_region & ~forbidden_region
    
    # Calculate statistics
    total_pixels = np.sum(allowed_region)
    
    if cell_radius is not None:
        # Create a mask of all cell areas
        cell_area_mask = np.zeros_like(allowed_region, dtype=bool)
        for cell in cells:
            cell_mask = create_circular_region(grid_size, int(cell_radius * grid_size))
            cell_center = (int(cell[1] * grid_size), int(cell[0] * grid_size))
            y_start = max(0, cell_center[0] - cell_mask.shape[0]//2)
            y_end = min(grid_size, cell_center[0] + cell_mask.shape[0]//2)
            x_start = max(0, cell_center[1] - cell_mask.shape[1]//2)
            x_end = min(grid_size, cell_center[1] + cell_mask.shape[1]//2)
            cell_area_mask[y_start:y_end, x_start:x_end] |= cell_mask[:y_end-y_start, :x_end-x_start]
        
        # Subtract cell areas from total pixels and apply to interaction map
        total_pixels -= np.sum(cell_area_mask & allowed_region)
        interaction_map[cell_area_mask] = 0
    
    # Calculate pixel coverages
    pixels_covered_0 = np.sum((interaction_map == 0) & allowed_region)
    pixels_covered_1 = np.sum((interaction_map == 1) & allowed_region)
    pixels_covered_2_plus = np.sum((interaction_map >= 2) & allowed_region)
    
    pixels_covered_2_plus_not_forbidden = pixels_covered_2_plus
    
    # Plotting
    if show_plots:
        plt.figure(figsize=(10, 10))
        plt.imshow(interaction_map, cmap='viridis', interpolation='nearest', extent=[0, 1, 0, 1])
        plt.colorbar(label='Interaction count')
        
        for cell in cells:
            plt.scatter(cell[0], cell[1], color='red', s=20)
            circle = Circle(cell, cell_interaction_radius, fill=False, color='r')
            plt.gca().add_artist(circle)
        
        if forbidden_region is not None:
            plt.imshow(forbidden_region, cmap='Reds', alpha=0.3, extent=[0, 1, 0, 1])
        
        plt.title('Cell Distribution Simulation')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    return (
        pixels_covered_0,
        pixels_covered_1,
        pixels_covered_2_plus,
        total_pixels,
        pixels_covered_2_plus_not_forbidden,
        total_pixels
    )

def simulate_single_distribution(
    size: int,
    allowed_shape: str,
    forbidden_shape: str,
    forbidden_size: float,
    cell_count: int,
    cell_radius: float,
    cell_interaction_radius: float,
    distribution_type: str,
    dist_params: Tuple[float, float],
    show_plots: bool = False
) -> dict:
    # Create allowed region
    if allowed_shape == 'square':
        allowed_region = np.ones((size, size), dtype=bool)
    else:  # circle
        allowed_region = create_circular_region(size, size // 2)
    
    # Create forbidden region
    if forbidden_shape == 'none':
        forbidden_region = None
    elif forbidden_shape == 'square':
        forbidden_size_pixels = int(size * forbidden_size)
        forbidden_region = np.zeros((size, size), dtype=bool)
        start = (size - forbidden_size_pixels) // 2
        forbidden_region[start:start+forbidden_size_pixels, start:start+forbidden_size_pixels] = True
    else:  # circle
        forbidden_region = create_circular_region(size, int(size * forbidden_size / 2))
    
    # Create distribution
    if distribution_type == 'uniform':
        x_dist = y_dist = stats.uniform(loc=dist_params[0], scale=dist_params[1])
    elif distribution_type == 'normal':
        x_dist = y_dist = stats.truncnorm(
            (0 - dist_params[0]) / dist_params[1], 
            (1 - dist_params[0]) / dist_params[1], 
            loc=dist_params[0], 
            scale=dist_params[1]
        )
    else:  # beta
        x_dist = y_dist = stats.beta(dist_params[0], dist_params[1])
    
    # Run simulation
    result = simulate_cell_distribution(
        x_distribution=x_dist,
        y_distribution=y_dist,
        cell_count=cell_count,
        allowed_region=allowed_region,
        cell_interaction_radius=cell_interaction_radius,
        forbidden_region=forbidden_region,
        cell_radius=cell_radius,
        show_plots=show_plots
    )
    
    return {
        'size': size,
        'allowed_shape': allowed_shape,
        'forbidden_shape': forbidden_shape,
        'forbidden_size': forbidden_size,
        'cell_count': cell_count,
        'cell_radius': cell_radius,
        'cell_interaction_radius': cell_interaction_radius,
        'distribution': f"{distribution_type}({dist_params[0]:.2f}, {dist_params[1]:.2f})",
        'pixels_covered_0': result[0],
        'pixels_covered_1': result[1],
        'pixels_covered_2_plus': result[2],
        'total_pixels': result[3],
        'pixels_covered_2_plus_not_forbidden': result[4],
        'total_pixels_not_forbidden': result[5]
    }

def generate_structured_dataset() -> pd.DataFrame:
    data = []
    
    # Define base parameters
    base_params = {
        'size': 100,
        'allowed_shape': 'square',
        'forbidden_shape': 'none',
        'forbidden_size': 0,
        'cell_count': 1000,  # Set to 1000 cells
        'cell_radius': 0.01,
        'cell_interaction_radius': 0.05,
        'distribution_type': 'normal',
        'dist_params': (0.5, 0.1)  # Mean of 0.5, std of 0.1
    }
    
    # Define parameter variations
    variations = {
        'size': [50, 100, 200],
        'allowed_shape': ['square', 'circle'],
        'forbidden_shape': ['none', 'square', 'circle'],
        'forbidden_size': [0, 0.1, 0.2, 0.3],
        'cell_count': [500, 1000, 2000],  # Adjust cell count variations
        'cell_radius': [0, 0.005, 0.01, 0.02],
        'cell_interaction_radius': [0.01, 0.03, 0.05, 0.07],
        'distribution_type': ['normal'],  # Keep only normal distribution
        'dist_params': [
            (0.5, 0.05), (0.5, 0.1), (0.5, 0.2)  # Vary standard deviation
        ]
    }
    
    # Generate data by varying one parameter at a time
    for param, values in variations.items():
        for value in values:
            current_params = base_params.copy()
            current_params[param] = value
            
            # Adjust cell_interaction_radius if cell_radius is changed
            if param == 'cell_radius':
                current_params['cell_interaction_radius'] = max(base_params['cell_interaction_radius'], value * 2)
            
            result = simulate_single_distribution(**current_params, show_plots=False)
            result['varied_parameter'] = param
            data.append(result)
    
    return pd.DataFrame(data)

# Generate the new dataset
df = generate_structured_dataset()

# Save to CSV
df.to_csv('structured_simulation_dataset_1000_cells.csv', index=False)

print("structured dataset generated and saved to 'structured_simulation_dataset_1000_cells.csv'")
print(f"Total number of simulations: {len(df)}")

# Display the first few rows of the dataset
print("\nFirst few rows of the corrected dataset:")
print(df.head())

# Check for any negative values in pixels_covered_0
if (df['pixels_covered_0'] < 0).any():
    print("\nWarning: Negative values found in pixels_covered_0")
else:
    print("\nNo negative values found in pixels_covered_0")
