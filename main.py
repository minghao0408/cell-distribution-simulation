from simulation import run_simulation
import numpy as np
from scipy import stats
import time

# Configuration for the simulation
CONFIG = {
    'mask_size': 100,
    'forbidden_region': ((45, 55), (45, 55)),
    'cell_counts': [50, 50],
    'x_distributions': [
        stats.truncnorm(a=-2, b=2, loc=0.5, scale=0.2),
        stats.truncnorm(a=-2, b=2, loc=0.5, scale=0.2)
    ],
    'y_distributions': [
        stats.truncnorm(a=-2, b=2, loc=0.5, scale=0.2),
        stats.truncnorm(a=-2, b=2, loc=0.5, scale=0.2)
    ],
    'interaction_matrix': np.array([[1.0, 0.8],
                                    [0.8, 1.0]]),
    'min_distance': 0.01,
    'cell_interaction_radius': 0.05,
    'show_plots': True
}

def main():
    print("Starting cell distribution simulation...")
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        if key not in ['x_distributions', 'y_distributions']:
            print(f"{key}: {value}")
    
    start_time = time.time()
    
    run_simulation(CONFIG)
    
    end_time = time.time()
    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
