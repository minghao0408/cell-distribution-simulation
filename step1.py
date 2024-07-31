import numpy as np
from scipy.stats import norm, rv_continuous
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional

def generate_cell_centers(
    N: Union[int, List[int]],
    x_distribution: Union[rv_continuous, List[rv_continuous]],
    y_distribution: Union[rv_continuous, List[rv_continuous]],
    binary_mask: np.ndarray,
    min_distance: float = 0.0
) -> np.ndarray:
    """
    Generate cell centers based on given distributions, binary mask, and minimum distance.

    Args:
        N (int or List[int]): Number of cells to generate for each cell type.
        x_distribution (rv_continuous or List[rv_continuous]): Distribution(s) for x-coordinates.
        y_distribution (rv_continuous or List[rv_continuous]): Distribution(s) for y-coordinates.
        binary_mask (np.ndarray): 2D boolean array specifying allowed regions.
        min_distance (float): Minimum distance between cell centers.

    Returns:
        np.ndarray: Array of cell center coordinates.
    """
    if isinstance(N, int):
        N = [N]
    if not isinstance(x_distribution, list):
        x_distribution = [x_distribution] * len(N)
    if not isinstance(y_distribution, list):
        y_distribution = [y_distribution] * len(N)

    assert len(N) == len(x_distribution) == len(y_distribution), "N, x_distribution, and y_distribution must have the same length"

    total_cells = sum(N)
    cell_centers = []

    mask_y, mask_x = binary_mask.shape
    x_range = np.arange(mask_x)
    y_range = np.arange(mask_y)

    cell_type = 0
    cells_generated = 0

    while len(cell_centers) < total_cells:
        if cells_generated == N[cell_type]:
            cell_type += 1
            cells_generated = 0

        # Generate candidate coordinates
        x = x_distribution[cell_type].rvs(size=1)[0]
        y = y_distribution[cell_type].rvs(size=1)[0]

        # Map x and y to mask indices
        mask_x_idx = int(np.interp(x, [-3, 3], [0, mask_x-1]))
        mask_y_idx = int(np.interp(y, [-3, 3], [0, mask_y-1]))

        # Check binary mask
        if not binary_mask[mask_y_idx, mask_x_idx]:
            continue

        # Check minimum distance
        if min_distance > 0 and cell_centers:
            distances = np.sqrt(np.sum((np.array(cell_centers) - [x, y])**2, axis=1))
            if np.min(distances) < min_distance:
                continue

        cell_centers.append([x, y])
        cells_generated += 1

    return np.array(cell_centers)

# Example usage
mask_size = 100
binary_mask = np.ones((mask_size, mask_size), dtype=bool)
binary_mask[40:60, 40:60] = False  # Create a forbidden region in the center

# Generate cell centers for two cell types
cell_centers = generate_cell_centers(
    [50, 50],  # 50 cells of type 1, 50 cells of type 2
    [norm(loc=-1, scale=0.5), norm(loc=1, scale=0.5)],  # x distributions
    [norm(loc=-1, scale=0.5), norm(loc=1, scale=0.5)],  # y distributions
    binary_mask,
    min_distance=0.2  # minimum distance between cells
)

# Plot the generated cell centers
plt.figure(figsize=(10, 10))
plt.imshow(binary_mask, extent=[-3, 3, -3, 3], alpha=0.3, cmap='gray_r')
plt.scatter(cell_centers[:50, 0], cell_centers[:50, 1], alpha=0.6, label='Type 1')
plt.scatter(cell_centers[50:, 0], cell_centers[50:, 1], alpha=0.6, label='Type 2')
plt.title("Simulated Cell Distribution with Constraints")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.grid(True)
plt.colorbar(label="Binary Mask")
plt.legend()
plt.show()
