import numpy as np
import pandas as pd
from scipy import stats
from simulation import simulate_cell_distribution, simulate_two_populations, create_circular_region
from typing import Tuple, Optional

def simulate_single_distribution(
    size: int,
    allowed_shape: str,
    forbidden_shape: str,
    forbidden_size: float,
    cell_count: int,
    cell_radius: float,
    cell_interaction_radius: float,
    distribution_type: str,
    dist_params: Tuple[float, float],
    show_plots: bool = False
) -> dict:
    """Simulate single population distribution with given parameters."""
    # Create allowed region
    if allowed_shape == 'square':
        allowed_region = np.ones((size, size), dtype=bool)
    else:  # circle
        allowed_region = create_circular_region(size, size // 2)
    
    # Create forbidden region
    if forbidden_shape == 'none':
        forbidden_region = None
    elif forbidden_shape == 'square':
        forbidden_size_pixels = int(size * forbidden_size)
        forbidden_region = np.zeros((size, size), dtype=bool)
        start = (size - forbidden_size_pixels) // 2
        forbidden_region[start:start+forbidden_size_pixels, start:start+forbidden_size_pixels] = True
    else:  # circle
        forbidden_region = create_circular_region(size, int(size * forbidden_size / 2))
    
    # Create distribution
    if distribution_type == 'uniform':
        x_dist = y_dist = stats.uniform(loc=dist_params[0], scale=dist_params[1])
    elif distribution_type == 'normal':
        x_dist = y_dist = stats.truncnorm(
            (0 - dist_params[0]) / dist_params[1], 
            (1 - dist_params[0]) / dist_params[1], 
            loc=dist_params[0], 
            scale=dist_params[1]
        )
    else:  # beta
        x_dist = y_dist = stats.beta(dist_params[0], dist_params[1])
    
    # Run simulation
    result = simulate_cell_distribution(
        x_distribution=x_dist,
        y_distribution=y_dist,
        cell_count=cell_count,
        allowed_region=allowed_region,
        cell_interaction_radius=cell_interaction_radius,
        forbidden_region=forbidden_region,
        cell_radius=cell_radius,
        show_plots=show_plots
    )
    
    return {
        'size': size,
        'allowed_shape': allowed_shape,
        'forbidden_shape': forbidden_shape,
        'forbidden_size': forbidden_size,
        'cell_count': cell_count,
        'cell_radius': cell_radius,
        'cell_interaction_radius': cell_interaction_radius,
        'distribution': f"{distribution_type}({dist_params[0]:.2f}, {dist_params[1]:.2f})",
        'pixels_covered_0': result[0],
        'pixels_covered_1': result[1],
        'pixels_covered_2_plus': result[2],
        'total_pixels': result[3],
        'pixels_covered_2_plus_not_forbidden': result[4],
        'total_pixels_not_forbidden': result[5]
    }

def simulate_two_pop_distribution(
    size: int,
    allowed_shape: str,
    forbidden_shape: str,
    forbidden_size: float,
    cell_counts: Tuple[int, int],
    cell_radius: float,
    cell_interaction_radius: float,
    distribution_type: str,
    dist_params_pop1: Tuple[float, float],
    dist_params_pop2: Tuple[float, float],
    show_plots: bool = False
) -> dict:
    """Simulate two population distribution with given parameters."""
    # Create allowed region
    if allowed_shape == 'square':
        allowed_region = np.ones((size, size), dtype=bool)
    else:  # circle
        allowed_region = create_circular_region(size, size // 2)
    
    # Create forbidden region
    if forbidden_shape == 'none':
        forbidden_region = None
    elif forbidden_shape == 'square':
        forbidden_size_pixels = int(size * forbidden_size)
        forbidden_region = np.zeros((size, size), dtype=bool)
        start = (size - forbidden_size_pixels) // 2
        forbidden_region[start:start+forbidden_size_pixels, start:start+forbidden_size_pixels] = True
    else:  # circle
        forbidden_region = create_circular_region(size, int(size * forbidden_size / 2))
    
    # Create distributions
    if distribution_type == 'uniform':
        x_dist1 = y_dist1 = stats.uniform(loc=dist_params_pop1[0], scale=dist_params_pop1[1])
        x_dist2 = y_dist2 = stats.uniform(loc=dist_params_pop2[0], scale=dist_params_pop2[1])
    elif distribution_type == 'normal':
        x_dist1 = y_dist1 = stats.truncnorm(
            (0 - dist_params_pop1[0]) / dist_params_pop1[1], 
            (1 - dist_params_pop1[0]) / dist_params_pop1[1], 
            loc=dist_params_pop1[0], 
            scale=dist_params_pop1[1]
        )
        x_dist2 = y_dist2 = stats.truncnorm(
            (0 - dist_params_pop2[0]) / dist_params_pop2[1], 
            (1 - dist_params_pop2[0]) / dist_params_pop2[1], 
            loc=dist_params_pop2[0], 
            scale=dist_params_pop2[1]
        )
    else:  # beta
        x_dist1 = y_dist1 = stats.beta(dist_params_pop1[0], dist_params_pop1[1])
        x_dist2 = y_dist2 = stats.beta(dist_params_pop2[0], dist_params_pop2[1])
    
    # Run simulation
    result = simulate_two_populations(
        x_distribution1=x_dist1,
        y_distribution1=y_dist1,
        x_distribution2=x_dist2,
        y_distribution2=y_dist2,
        cell_counts=cell_counts,
        allowed_region=allowed_region,
        cell_interaction_radius=cell_interaction_radius,
        forbidden_region=forbidden_region,
        cell_radius=cell_radius,
        show_plots=show_plots
    )
    
    return {
        'size': size,
        'allowed_shape': allowed_shape,
        'forbidden_shape': forbidden_shape,
        'forbidden_size': forbidden_size,
        'cell_count_pop1': cell_counts[0],
        'cell_count_pop2': cell_counts[1],
        'cell_radius': cell_radius,
        'cell_interaction_radius': cell_interaction_radius,
        'distribution': distribution_type,
        'dist_params_pop1': f"({dist_params_pop1[0]:.2f}, {dist_params_pop1[1]:.2f})",
        'dist_params_pop2': f"({dist_params_pop2[0]:.2f}, {dist_params_pop2[1]:.2f})",
        'pixels_no_interaction': result[0],
        'pixels_pop1_only': result[1],
        'pixels_pop2_only': result[2],
        'pixels_both_populations': result[3],
        'total_pixels': result[4]
    }

def generate_datasets():
    """Generate both single and two population datasets."""
    # Parameters for single population
    single_pop_variations = {
        'size': [50, 100, 200],
        'allowed_shape': ['square', 'circle'],
        'forbidden_shape': ['none', 'square', 'circle'],
        'forbidden_size': [0, 0.1, 0.2, 0.3],
        'cell_count': [500, 1000, 2000],
        'cell_radius': [0, 0.005, 0.01, 0.02],
        'cell_interaction_radius': [0.01, 0.03, 0.05, 0.07],
        'distribution_type': ['normal'],
        'dist_params': [(0.5, 0.05), (0.5, 0.1), (0.5, 0.2)]
    }
    
    # Parameters for two populations
    two_pop_variations = {
        'size': [50, 100, 200],
        'allowed_shape': ['square', 'circle'],
        'forbidden_shape': ['none', 'square', 'circle'],
        'forbidden_size': [0, 0.1, 0.2, 0.3],
        'cell_counts': [(300, 700), (500, 500), (700, 300)],
        'cell_radius': [0, 0.005, 0.01, 0.02],
        'cell_interaction_radius': [0.01, 0.03, 0.05, 0.07],
        'distribution_type': ['normal'],
        'dist_params': [
            ((0.3, 0.1), (0.7, 0.1)),  # Different means
            ((0.5, 0.1), (0.5, 0.1)),  # Same parameters
            ((0.5, 0.05), (0.5, 0.2))   # Different spreads
        ]
    }
    
    # Generate single population dataset
    single_pop_data = []
    base_params_single = {
        'size': 100,
        'allowed_shape': 'square',
        'forbidden_shape': 'none',
        'forbidden_size': 0,
        'cell_count': 1000,
        'cell_radius': 0.01,
        'cell_interaction_radius': 0.05,
        'distribution_type': 'normal',
        'dist_params': (0.5, 0.1)
    }
    
    print("Generating single population dataset...")
    for param, values in single_pop_variations.items():
        for value in values:
            current_params = base_params_single.copy()
            current_params[param] = value
            
            # Adjust cell_interaction_radius if cell_radius is changed
            if param == 'cell_radius':
                current_params['cell_interaction_radius'] = max(base_params_single['cell_interaction_radius'], value * 2)
            
            result = simulate_single_distribution(**current_params, show_plots=False)
            result['varied_parameter'] = param
            single_pop_data.append(result)
    
    # Generate two population dataset
    two_pop_data = []
    base_params_two = {
        'size': 100,
        'allowed_shape': 'square',
        'forbidden_shape': 'none',
        'forbidden_size': 0,
        'cell_counts': (500, 500),
        'cell_radius': 0.01,
        'cell_interaction_radius': 0.05,
        'distribution_type': 'normal',
        'dist_params_pop1': (0.5, 0.1),
        'dist_params_pop2': (0.5, 0.1)
    }
    
    print("Generating two population dataset...")
    for param, values in two_pop_variations.items():
        for value in values:
            current_params = base_params_two.copy()
            
            if param == 'dist_params':
                current_params['dist_params_pop1'] = value[0]
                current_params['dist_params_pop2'] = value[1]
            elif param == 'cell_counts':
                current_params['cell_counts'] = value
            else:
                current_params[param] = value
            
            # Adjust cell_interaction_radius if cell_radius is changed
            if param == 'cell_radius':
                current_params['cell_interaction_radius'] = max(base_params_two['cell_interaction_radius'], value * 2)
            
            result = simulate_two_pop_distribution(**current_params, show_plots=False)
            result['varied_parameter'] = param
            two_pop_data.append(result)
    
    # Create DataFrames and save to CSV
    df_single = pd.DataFrame(single_pop_data)
    df_two = pd.DataFrame(two_pop_data)
    
    df_single.to_csv('simulation_dataset_1pop.csv', index=False)
    df_two.to_csv('simulation_dataset_2pop.csv', index=False)
    
    print(f"Generated {len(df_single)} simulations for single population")
    print(f"Generated {len(df_two)} simulations for two populations")
    
    return df_single, df_two

if __name__ == "__main__":
    df_single, df_two = generate_datasets()
