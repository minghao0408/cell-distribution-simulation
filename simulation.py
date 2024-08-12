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
    # Function implementation (as provided in the original code)
    ...

def estimate_interaction(cell_centers: np.ndarray, cell_types: np.ndarray) -> np.ndarray:
    # Function implementation (as provided in the original code)
    ...

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
