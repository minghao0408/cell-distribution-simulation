# config.py

import numpy as np
from scipy import stats

CONFIG = {
    'mask_size': 100,
    'forbidden_region': ((45, 55), (45, 55)),  # Adjust forbidden region
    'cell_counts': [200, 200],
    'x_distributions': [
        stats.truncnorm(a=-2, b=2, loc=0.5, scale=0.2),
        stats.truncnorm(a=-2, b=2, loc=0.5, scale=0.2)
    ],
    'y_distributions': [
        stats.truncnorm(a=-2, b=2, loc=0.5, scale=0.2),
        stats.truncnorm(a=-2, b=2, loc=0.5, scale=0.2)
    ],
    'interaction_matrix': np.array([[1.0, 0.5],  # Adjust interaction strength
                                    [0.5, 1.0]]),
    'min_distance': 0.02,  # Adjust minimum distance
    'plot_size': (10, 10),
    'plot_extent': [0, 1, 0, 1]
}
