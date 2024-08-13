# simulation.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from config import CONFIG
from typing import Union, List, Tuple

def create_binary_mask(mask_size: int, forbidden_region: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
    mask = np.ones((mask_size, mask_size), dtype=bool)
    (y1, y2), (x1, x2) = forbidden_region
    mask[y1:y2, x1:x2] = False
    return mask

def generate_cell_centers(
    N: Union[int, List[int]],
    x_distribution: Union[np.random.RandomState, List[np.random.RandomState]],
    y_distribution: Union[np.random.RandomState, List[np.random.RandomState]],
    binary_mask: np.ndarray,
    interaction_matrix: np.ndarray,
    min_distance: float = 0.0
) -> np.ndarray:
    print("Starting generate_cell_centers")
    print(f"N: {N}")
    print(f"binary_mask shape: {binary_mask.shape}")
    print(f"interaction_matrix shape: {interaction_matrix.shape}")
    print(f"min_distance: {min_distance}")
    
    if isinstance(N, int):
        N = [N]
    if not isinstance(x_distribution, list):
        x_distribution = [x_distribution] * len(N)
    if not isinstance(y_distribution, list):
        y_distribution = [y_distribution] * len(N)

    assert len(N) == len(x_distribution) == len(y_distribution), "N, x_distribution, and y_distribution must have the same length"
    assert interaction_matrix.shape == (len(N), len(N)), "Interaction matrix shape must match the number of cell types"

    total_cells = sum(N)
    cell_centers = []
    cell_types = []

    mask_y, mask_x = binary_mask.shape
    x_range = np.linspace(-3, 3, mask_x)
    y_range = np.linspace(-3, 3, mask_y)

    for cell_type in range(len(N)):
        cells_generated = 0
        attempts = 0
        max_attempts = N[cell_type] * 1000  # Set a maximum number of attempts
        while cells_generated < N[cell_type] and attempts < max_attempts:
            attempts += 1
            # Generate candidate coordinates
            x = x_distribution[cell_type].rvs(size=1)[0]
            y = y_distribution[cell_type].rvs(size=1)[0]

            # Map x and y to mask indices
            mask_x_idx = int(np.interp(x, [-3, 3], [0, mask_x-1]))
            mask_y_idx = int(np.interp(y, [-3, 3], [0, mask_y-1]))

            # Check binary mask
            if not binary_mask[mask_y_idx, mask_x_idx]:
                continue

            # Check minimum distance and apply interaction
            if cell_centers:
                distances = np.sqrt(np.sum((np.array(cell_centers) - [x, y])**2, axis=1))
                if np.min(distances) < min_distance:
                    continue

                # Apply interaction
                for other_type in range(len(N)):
                    interaction = interaction_matrix[cell_type, other_type]
                    other_cells = np.array([c for c, t in zip(cell_centers, cell_types) if t == other_type])
                    if len(other_cells) > 0:
                        other_distances = np.sqrt(np.sum((other_cells - [x, y])**2, axis=1))
                        if interaction > 0:
                            if np.min(other_distances) > (1 - interaction) * 3:
                                continue
                        elif interaction < 0:
                            if np.min(other_distances) < -interaction * 3:
                                continue

            cell_centers.append([x, y])
            cell_types.append(cell_type)
            cells_generated += 1

        print(f"Cell type {cell_type}: generated {cells_generated} cells after {attempts} attempts")

    if not cell_centers:
        print("Warning: No cells were generated")
        print(f"Final cell_centers: {cell_centers}")
        print(f"Final cell_types: {cell_types}")
        return np.array([])

    result = np.column_stack((np.array(cell_centers), np.array(cell_types)))
    print(f"Finished generate_cell_centers, returning array of shape {result.shape}")
    return result

def estimate_interaction(cell_centers: np.ndarray, cell_types: np.ndarray) -> np.ndarray:
    if cell_centers.size == 0 or cell_types.size == 0:
        print("Warning: No cells to estimate interaction")
        return np.array([[]])  # Return an empty 2D array instead of None

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
            
            interaction = 1 - (mean_distance / 3) if mean_distance > 0 else 0
            interaction_matrix[i, j] = interaction_matrix[j, i] = interaction

    return interaction_matrix

def run_simulation():
    binary_mask = create_binary_mask(CONFIG['mask_size'], CONFIG['forbidden_region'])
    
    cell_data = generate_cell_centers(
        CONFIG['cell_counts'],
        CONFIG['x_distributions'],
        CONFIG['y_distributions'],
        binary_mask,
        CONFIG['interaction_matrix'],
        CONFIG['min_distance']
    )

    if cell_data.size == 0:
        print("Error: No cells were generated. Simulation cannot continue.")
        return

    cell_centers = cell_data[:, :2]
    cell_types = cell_data[:, 2].astype(int)

    estimated_interaction = estimate_interaction(cell_centers, cell_types)

    print("Original interaction matrix:")
    print(CONFIG['interaction_matrix'])
    print("\nEstimated interaction matrix:")
    print(estimated_interaction)

    plot_results(binary_mask, cell_centers, cell_types)
    
def plot_results(binary_mask: np.ndarray, cell_centers: np.ndarray, cell_types: np.ndarray):
    plt.figure(figsize=CONFIG['plot_size'])
    plt.imshow(binary_mask, extent=CONFIG['plot_extent'], alpha=0.3, cmap='gray_r')
    plt.scatter(cell_centers[cell_types == 0, 0], cell_centers[cell_types == 0, 1], alpha=0.6, label='Type 1')
    plt.scatter(cell_centers[cell_types == 1, 0], cell_centers[cell_types == 1, 1], alpha=0.6, label='Type 2')
    plt.title("Simulated Cell Distribution with Interactions")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(CONFIG['plot_extent'][:2])
    plt.ylim(CONFIG['plot_extent'][2:])
    plt.grid(True)
    plt.colorbar(label="Binary Mask")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_simulation()

