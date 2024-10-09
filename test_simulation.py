import unittest
import numpy as np
from scipy import stats
from simulation import create_binary_mask, simulate_cell_distribution, generate_cell_centers

class TestSimulation(unittest.TestCase):
    def test_create_binary_mask(self):
        mask = create_binary_mask(100, ((45, 55), (45, 55)))
        self.assertEqual(mask.shape, (100, 100))
        self.assertFalse(np.all(mask[45:55, 45:55]))
        self.assertTrue(np.all(mask[0:45, 0:45]))

    def test_simulate_cell_distribution(self):
        allowed_region = np.ones((100, 100), dtype=bool)
        x_dist = stats.uniform(loc=0, scale=1)
        y_dist = stats.uniform(loc=0, scale=1)
        
        result = simulate_cell_distribution(
            x_distribution=x_dist,
            y_distribution=y_dist,
            cell_count=50,
            allowed_region=allowed_region,
            cell_interaction_radius=0.05,
            cell_radius=0.01,
            show_plots=False
        )
        
        self.assertEqual(len(result), 6)
        self.assertGreater(result[0] + result[1] + result[2], 0)  # At least some pixels should be covered

    def test_generate_cell_centers(self):
        binary_mask = np.ones((100, 100), dtype=bool)
        x_dist = stats.uniform(loc=0, scale=1)
        y_dist = stats.uniform(loc=0, scale=1)
        interaction_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        cell_data = generate_cell_centers(
            N=[50, 50],
            x_distribution=[x_dist, x_dist],
            y_distribution=[y_dist, y_dist],
            binary_mask=binary_mask,
            interaction_matrix=interaction_matrix,
            min_distance=0.01
        )
        
        self.assertIsInstance(cell_data, np.ndarray)
        self.assertEqual(cell_data.shape[1], 3)  # x, y, cell_type
        self.assertEqual(len(np.unique(cell_data[:, 2])), 2)  # Two cell types

if __name__ == '__main__':
    unittest.main()
