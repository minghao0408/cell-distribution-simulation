# simulation.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from typing import Union, List, Tuple, Optional
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist

def create_binary_mask(mask_size: int, forbidden_region: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
    mask = np.ones((mask_size, mask_size), dtype=bool)
    (y1, y2), (x1, x2) = forbidden_region
    mask[y1:y2, x1:x2] = False
    return mask

def generate_cell_centers(
    N: Union[int, List[int]],
    x_distribution: Union[stats.rv_continuous, List[stats.rv_continuous]],
    y_distribution: Union[stats.rv_continuous, List[stats.rv_continuous]],
    binary_mask: np.ndarray,
    interaction_matrix: np.ndarray,
    min_distance: float = 0.0
) -> np.ndarray:
    if isinstance(N, int):
        N = [N]
    if not isinstance(x_distribution, list):
        x_distribution = [x_distribution] * len(N)
    if not isinstance(y_distribution, list):
        y_distribution = [y_distribution] * len(N)

    assert len(N) == len(x_distribution) == len(y_distribution), "N, x_distribution, and y_distribution must have the same length"
    assert interaction_matrix.shape == (len(N), len(N)), "Interaction matrix shape must match the number of cell types"

    cell_centers = []
    cell_types = []

    mask_y, mask_x = binary_mask.shape

    for cell_type in range(len(N)):
        cells_generated = 0
        attempts = 0
        max_attempts = N[cell_type] * 1000

        while cells_generated < N[cell_type] and attempts < max_attempts:
            attempts += 1

            x = x_distribution[cell_type].rvs(size=1)[0]
            y = y_distribution[cell_type].rvs(size=1)[0]

            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)

            mask_x_idx = int(x * (mask_x - 1))
            mask_y_idx = int(y * (mask_y - 1))

            if not binary_mask[mask_y_idx, mask_x_idx]:
                continue

            if cell_centers:
                distances = np.sqrt(np.sum((np.array(cell_centers) - [x, y])**2, axis=1))
                if np.min(distances) < min_distance:
                    continue

                for other_type in range(len(N)):
                    interaction = interaction_matrix[cell_type, other_type]
                    other_cells = np.array([c for c, t in zip(cell_centers, cell_types) if t == other_type])
                    if len(other_cells) > 0:
                        other_distances = np.sqrt(np.sum((other_cells - [x, y])**2, axis=1))
                        if interaction > 0:
                            if np.min(other_distances) > (1 - interaction) * 0.5:
                                continue
                        elif interaction < 0:
                            if np.min(other_distances) < -interaction * 0.5:
                                continue

            cell_centers.append([x, y])
            cell_types.append(cell_type)
            cells_generated += 1

        print(f"Cell type {cell_type}: generated {cells_generated} cells after {attempts} attempts")

    if not cell_centers:
        print("Warning: No cells were generated")
        return np.array([])

    result = np.column_stack((np.array(cell_centers), np.array(cell_types)))
    return result

def estimate_interaction(cell_centers: np.ndarray, cell_types: np.ndarray, original_interaction: np.ndarray) -> np.ndarray:
    num_types = len(np.unique(cell_types))
    interaction_matrix = np.zeros((num_types, num_types))

    distances = squareform(pdist(cell_centers))
    
    for i in range(num_types):
        for j in range(i, num_types):
            mask_i = cell_types == i
            mask_j = cell_types == j
            
            if i == j:
                same_type_distances = distances[mask_i][:, mask_i]
                np.fill_diagonal(same_type_distances, np.inf)
                mean_distance = np.mean(np.min(same_type_distances, axis=1)) if same_type_distances.size > 0 else 0
            else:
                cross_distances = distances[mask_i][:, mask_j]
                mean_distance = np.mean(np.min(cross_distances, axis=1)) if cross_distances.size > 0 else 0
            
            interaction = 1 / (1 + np.exp(5 * (mean_distance - 0.5)))
            
            adjusted_interaction = original_interaction[i, j] * interaction
            
            interaction_matrix[i, j] = interaction_matrix[j, i] = adjusted_interaction

    return interaction_matrix

def calculate_overlap(cell_centers: np.ndarray, cell_types: np.ndarray) -> np.ndarray:
    num_types = len(np.unique(cell_types))
    overlap_matrix = np.zeros((num_types, num_types))

    for i in range(num_types):
        for j in range(i, num_types):
            centers_i = cell_centers[cell_types == i]
            centers_j = cell_centers[cell_types == j]
            
            if i == j:
                overlap = 1.0
            else:
                from scipy.stats import gaussian_kde
                
                combined_centers = np.vstack([centers_i, centers_j])
                
                kde = gaussian_kde(combined_centers.T)
                
                x, y = np.mgrid[0:1:100j, 0:1:100j]
                positions = np.vstack([x.ravel(), y.ravel()])
                density = kde(positions).reshape(x.shape)
                
                kde_i = gaussian_kde(centers_i.T)
                kde_j = gaussian_kde(centers_j.T)
                density_i = kde_i(positions).reshape(x.shape)
                density_j = kde_j(positions).reshape(x.shape)
                
                overlap = np.sum(np.minimum(density_i, density_j)) / np.sum(density)
            
            overlap_matrix[i, j] = overlap_matrix[j, i] = overlap

    return overlap_matrix

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

def create_circular_region(size: int, radius: int) -> np.ndarray:
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    return dist_from_center <= radius

def run_simulation(config):
    allowed_region = create_binary_mask(config['mask_size'], config['forbidden_region'])
    forbidden_region = ~allowed_region

    for i, (count, x_dist, y_dist) in enumerate(zip(config['cell_counts'], config['x_distributions'], config['y_distributions'])):
        result = simulate_cell_distribution(
            x_distribution=x_dist,
            y_distribution=y_dist,
            cell_count=count,
            allowed_region=allowed_region,
            cell_interaction_radius=config['cell_interaction_radius'],
            forbidden_region=forbidden_region,
            cell_radius=config['min_distance'] / 2,
            show_plots=config['show_plots']
        )
        print(f"Results for cell type {i}:")
        print(f"Pixels covered by 0 cells: {result[0]}")
        print(f"Pixels covered by 1 cell: {result[1]}")
        print(f"Pixels covered by 2 or more cells: {result[2]}")
        print(f"Total pixels: {result[3]}")
        print(f"Pixels covered by 2 or more cells (not in forbidden area): {result[4]}")
        print(f"Total pixels (not in forbidden area): {result[5]}")
    
    cell_data = generate_cell_centers(
        config['cell_counts'],
        config['x_distributions'],
        config['y_distributions'],
        allowed_region,
        config['interaction_matrix'],
        config['min_distance']
    )

    if cell_data.size == 0:
        print("Error: No cells were generated. Simulation cannot continue.")
        return

    cell_centers = cell_data[:, :2]
    cell_types = cell_data[:, 2].astype(int)

    estimated_interaction = estimate_interaction(cell_centers, cell_types, config['interaction_matrix'])
    overlap_matrix = calculate_overlap(cell_centers, cell_types)

    print("Original interaction matrix:")
    print(config['interaction_matrix'])
    print("\nEstimated interaction matrix:")
    print(estimated_interaction)
    print("\nOverlap matrix:")
    print(overlap_matrix)

    plot_results(allowed_region, cell_centers, cell_types)

def plot_results(binary_mask: np.ndarray, cell_centers: np.ndarray, cell_types: np.ndarray):
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_mask, extent=[0, 1, 0, 1], alpha=0.3, cmap='gray_r')
    plt.scatter(cell_centers[cell_types == 0, 0], cell_centers[cell_types == 0, 1], alpha=0.6, label='Type 1')
    plt.scatter(cell_centers[cell_types == 1, 0], cell_centers[cell_types == 1, 1], alpha=0.6, label='Type 2')
    plt.title("Simulated Cell Distribution with Interactions")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.colorbar(label="Binary Mask")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from config import CONFIG
    run_simulation(CONFIG)

