# simulation.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

            # Generate candidate coordinates
            x = x_distribution[cell_type].rvs(size=1)[0]
            y = y_distribution[cell_type].rvs(size=1)[0]

            # Ensure x and y are within [0, 1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)

            # Map x and y to mask indices
            mask_x_idx = int(x * (mask_x - 1))
            mask_y_idx = int(y * (mask_y - 1))

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
                            # Positive interaction: cells should be closer
                            if np.min(other_distances) > (1 - interaction) * 0.5:
                                continue
                        elif interaction < 0:
                            # Negative interaction: cells should be further apart
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
            
            # Map the interaction strength to the range [0, 1] using the sigmoid function
            interaction = 1 / (1 + np.exp(5 * (mean_distance - 0.5)))
            
            # Adjust the estimate based on the original interaction matrix
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
                # Calculate the overlap using a kernel density estimation approach
                from scipy.stats import gaussian_kde
                
                # Combine the centers for both types
                combined_centers = np.vstack([centers_i, centers_j])
                
                # Create the KDE using the combined centers
                kde = gaussian_kde(combined_centers.T)
                
                # Evaluate the KDE on a grid
                x, y = np.mgrid[0:1:100j, 0:1:100j]
                positions = np.vstack([x.ravel(), y.ravel()])
                density = kde(positions).reshape(x.shape)
                
                # Calculate the overlap as the integral of the minimum of the two distributions
                kde_i = gaussian_kde(centers_i.T)
                kde_j = gaussian_kde(centers_j.T)
                density_i = kde_i(positions).reshape(x.shape)
                density_j = kde_j(positions).reshape(x.shape)
                
                overlap = np.sum(np.minimum(density_i, density_j)) / np.sum(density)
            
            overlap_matrix[i, j] = overlap_matrix[j, i] = overlap

    return overlap_matrix

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

    estimated_interaction = estimate_interaction(cell_centers, cell_types, CONFIG['interaction_matrix'])
    overlap_matrix = calculate_overlap(cell_centers, cell_types)

    print("Original interaction matrix:")
    print(CONFIG['interaction_matrix'])
    print("\nEstimated interaction matrix:")
    print(estimated_interaction)
    print("\nOverlap matrix:")
    print(overlap_matrix)

    plot_results(binary_mask, cell_centers, cell_types)
    
def plot_results(binary_mask: np.ndarray, cell_centers: np.ndarray, cell_types: np.ndarray):
    plt.figure(figsize=CONFIG['plot_size'])
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
    run_simulation()

