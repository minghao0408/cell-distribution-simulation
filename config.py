# config.py

import numpy as np
from scipy.stats import norm

CONFIG = {
    'mask_size': 100,
    'forbidden_region': ((40, 60), (40, 60)),
    'cell_counts': [50, 50],
    'x_distributions': [norm(loc=-1, scale=0.5), norm(loc=1, scale=0.5)],
    'y_distributions': [norm(loc=-1, scale=0.5), norm(loc=1, scale=0.5)],
    'interaction_matrix': np.array([[1.0, 0.3],
                                    [0.3, 1.0]]),
    'min_distance': 0.2,
    'plot_size': (10, 10),
    'plot_extent': [-3, 3, -3, 3],
}
