import numpy as np
import pytest
from src.cell_generator import generate_cell_centers
from scipy.stats import norm

def test_generate_cell_centers():
    mask_size = 100
    binary_mask = np.ones((mask_size, mask_size), dtype=bool)
    interaction_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])

    cell_data = generate_cell_centers(
        [10, 10],
        [norm(loc=0, scale=1), norm(loc=0, scale=1)],
        [norm(loc=0, scale=1), norm(loc=0, scale=1)],
        binary_mask,
        interaction_matrix,
        min_distance=0.2
    )

    assert cell_data.shape == (20, 3)
    assert np.all(cell_data[:, :2] >= -3) and np.all(cell_data[:, :2] <= 3)
    assert set(cell_data[:, 2]) == {0, 1}

# Add more tests as needed
