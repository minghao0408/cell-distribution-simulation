# test_simulation.py

import unittest
import numpy as np
from simulation import create_binary_mask, generate_cell_centers, estimate_interaction
from config import CONFIG

class TestSimulation(unittest.TestCase):
    def test_create_binary_mask(self):
        mask = create_binary_mask(CONFIG['mask_size'], CONFIG['forbidden_region'])
        self.assertEqual(mask.shape, (CONFIG['mask_size'], CONFIG['mask_size']))
        self.assertFalse(np.all(mask[40:60, 40:60]))
        self.assertTrue(np.all(mask[0:40, 0:40]))

    def test_generate_cell_centers(self):
        binary_mask = create_binary_mask(CONFIG['mask_size'], CONFIG['forbidden_region'])
        cell_data = generate_cell_centers(
            CONFIG['cell_counts'],
            CONFIG['x_distributions'],
            CONFIG['y_distributions'],
            binary_mask,
            CONFIG['interaction_matrix'],
            CONFIG['min_distance']
        )
        self.assertEqual(cell_data.shape[0], sum(CONFIG['cell_counts']))
        self.assertEqual(cell_data.shape[1], 3)  # x, y, cell_type

    def test_estimate_interaction(self):
        # Generate some mock data
        cell_centers = np.random.rand(100, 2) * 6 - 3  # Range from -3 to 3
        cell_types = np.random.randint(0, 2, 100)
        
        estimated_interaction = estimate_interaction(cell_centers, cell_types)
        self.assertEqual(estimated_interaction.shape, (2, 2))
        self.assertTrue(np.allclose(estimated_interaction, estimated_interaction.T))  # Should be symmetric

if __name__ == '__main__':
    unittest.main()
