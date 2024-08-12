import yaml
import numpy as np
from scipy.stats import norm
from src.cell_generator import generate_cell_centers
from src.interaction_estimator import estimate_interaction
from src.visualizer import plot_cell_distribution

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_simulation(config):
    # Create binary mask
    mask_size = config['simulation']['mask_size']
    binary_mask = np.ones((mask_size, mask_size), dtype=bool)
    binary_mask[mask_size//3:2*mask_size//3, mask_size//3:2*mask_size//3] = False

    # Set up distributions
    x_distributions = []
    y_distributions = []
    for i in range(1, config['simulation']['num_cell_types'] + 1):
        type_config = config['distributions'][f'type{i}']
        x_distributions.append(norm(loc=type_config['x']['loc'], scale=type_config['x']['scale']))
        y_distributions.append(norm(loc=type_config['y']['loc'], scale=type_config['y']['scale']))

    # Generate cells
    cell_data = generate_cell_centers(
        [config['simulation']['cells_per_type']] * config['simulation']['num_cell_types'],
        x_distributions,
        y_distributions,
        binary_mask,
        np.array(config['interaction_matrix']),
        config['simulation']['min_distance']
    )

    # Separate cell centers and types
    cell_centers = cell_data[:, :2]
    cell_types = cell_data[:, 2].astype(int)

    # Estimate interaction
    estimated_interaction = estimate_interaction(cell_centers, cell_types)

    # Visualize results
    plot_cell_distribution(cell_centers, cell_types, binary_mask)

    return cell_centers, cell_types, estimated_interaction

if __name__ == "__main__":
    config = load_config('config/default_config.yaml')
    cell_centers, cell_types, estimated_interaction = run_simulation(config)
    print("Estimated interaction matrix:")
    print(estimated_interaction)
