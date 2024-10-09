import numpy as np
import pandas as pd
from scipy import stats
from simulation import simulate_cell_distribution

def create_circular_region(size, radius):
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    return dist_from_center <= radius

def generate_dataset(num_iterations=100):
    data = []
    
    for i in range(num_iterations):
        # Randomly choose parameters
        size = np.random.choice([50, 100, 200])
        allowed_shape = np.random.choice(['square', 'circle'])
        forbidden_shape = np.random.choice(['none', 'square', 'circle'])
        cell_count = np.random.randint(10, 100)
        cell_radius = np.random.choice([0, np.random.uniform(0.5, 2.0)])
        cell_interaction_radius = np.random.uniform(max(cell_radius, 0.5), 5.0) / size
        
        # Create allowed region
        if allowed_shape == 'square':
            allowed_region = np.ones((size, size), dtype=bool)
        else:  # circle
            allowed_region = create_circular_region(size, size // 2)
        
        # Create forbidden region
        if forbidden_shape == 'none':
            forbidden_region = None
        elif forbidden_shape == 'square':
            forbidden_size = size // 4
            forbidden_region = np.zeros((size, size), dtype=bool)
            start = (size - forbidden_size) // 2
            forbidden_region[start:start+forbidden_size, start:start+forbidden_size] = True
        else:  # circle
            forbidden_region = create_circular_region(size, size // 8)
        
        # Randomly choose distribution type and parameters
        dist_type = np.random.choice(['uniform', 'normal', 'beta'])
        if dist_type == 'uniform':
            x_dist = stats.uniform(loc=0, scale=1)
            y_dist = stats.uniform(loc=0, scale=1)
            dist_params = f"uniform(0, 1)"
        elif dist_type == 'normal':
            mean, std = 0.5, 1/6
            x_dist = stats.truncnorm((0 - mean) / std, (1 - mean) / std, loc=mean, scale=std)
            y_dist = stats.truncnorm((0 - mean) / std, (1 - mean) / std, loc=mean, scale=std)
            dist_params = f"normal({mean:.2f}, {std:.2f})"
        else:  # beta
            a, b = np.random.uniform(0.5, 5, size=2)
            x_dist = stats.beta(a, b)
            y_dist = stats.beta(a, b)
            dist_params = f"beta({a:.2f}, {b:.2f})"
        
        # Run simulation
        result = simulate_cell_distribution(
            x_distribution=x_dist,
            y_distribution=y_dist,
            cell_count=cell_count,
            allowed_region=allowed_region,
            cell_interaction_radius=cell_interaction_radius,
            forbidden_region=forbidden_region,
            cell_radius=cell_radius / size if cell_radius > 0 else None,
            show_plots=False
        )
        
        # Collect data
        data.append({
            'iteration': i+1,
            'size': size,
            'allowed_shape': allowed_shape,
            'forbidden_shape': forbidden_shape,
            'cell_count': cell_count,
            'cell_radius': cell_radius,
            'cell_interaction_radius': cell_interaction_radius * size,
            'distribution': dist_params,
            'pixels_covered_0': result[0],
            'pixels_covered_1': result[1],
            'pixels_covered_2_plus': result[2],
            'total_pixels': result[3],
            'pixels_covered_2_plus_not_forbidden': result[4],
            'total_pixels_not_forbidden': result[5]
        })
    
    return pd.DataFrame(data)

# Generate the dataset
df = generate_dataset(num_iterations=100)

# Save to CSV
df.to_csv('simulation_dataset.csv', index=False)

print("Dataset generated and saved to 'simulation_dataset.csv'")