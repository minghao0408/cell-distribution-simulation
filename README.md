# Cell Distribution Simulation

A Python library for simulating and analyzing spatial distribution patterns of cell populations in defined regions.


## Overview

This project simulates how cells distribute spatially across specified regions, supporting both single and two-population scenarios. It's designed for researchers and students analyzing cell interaction patterns, spatial coverage, and distribution effects under various conditions.

## Features

- **Multiple Population Support**: Simulate single or dual cell population distributions
- **Flexible Configuration**:
  - Adjustable grid sizes and shapes (square or circular)
  - Customizable forbidden regions with variable sizes
  - Configurable cell counts, radii and interaction distances
  - Various distribution types (normal, uniform, beta)
- **Comprehensive Analysis**:
  - Detailed spatial coverage statistics
  - Interaction pattern analysis
  - Multi-simulation repeats for robust statistical validation
- **Visualization Tools**: Generate visual representations of simulation results

## Quick Start

```python
# Install dependencies
pip install numpy pandas scipy matplotlib

# Run a basic simulation
python generate_dataset.py

# Simulation results are saved in:
# - simulation_dataset_1pop.csv
# - simulation_dataset_2pop.csv
```

## Project Structure

```
.
├── simulation.py          # Core simulation functions
├── generate_dataset.py    # Dataset generation and analysis
├── test_simulation.py     # Test suite
```

## Detailed Usage

### Single Population Simulation

```python
from simulation import create_circular_region, simulate_cell_distribution
from generate_dataset import simulate_single_distribution
import numpy as np
from scipy import stats

# Using high-level function
result = simulate_single_distribution(
    size=100,                        # Grid size (pixels)
    allowed_shape='square',          # 'square' or 'circle'
    forbidden_shape='none',          # 'none', 'square', or 'circle'
    forbidden_size=0,                # Relative size (0-1)
    cell_count=1000,                 # Number of cells
    cell_radius=0.01,                # Cell radius (relative to grid)
    cell_interaction_radius=0.05,    # Interaction radius
    distribution_type='normal',      # 'normal', 'uniform', or 'beta'
    dist_params=(0.5, 0.1),          # Distribution parameters
    show_plots=True                  # Enable visualization
)

# Output metrics
print(f"Uncovered pixels: {result['pixels_covered_0']}")
print(f"Single-covered pixels: {result['pixels_covered_1']}")
print(f"Multi-covered pixels: {result['pixels_covered_2_plus']}")
print(f"Coverage percentage: {(result['pixels_covered_1'] + result['pixels_covered_2_plus']) / result['total_pixels'] * 100:.2f}%")
```

### Two Population Simulation

```python
from generate_dataset import simulate_two_pop_distribution

result = simulate_two_pop_distribution(
    size=100,
    allowed_shape='square',
    forbidden_shape='circle',        # Adding a circular forbidden region
    forbidden_size=0.2,              # 20% of grid size
    cell_counts=(500, 500),          # Cells in each population
    cell_radius=0.01,
    cell_interaction_radius=0.05,
    distribution_type='normal',
    dist_params_pop1=(0.3, 0.1),     # First population centered at 0.3
    dist_params_pop2=(0.7, 0.1),     # Second population centered at 0.7
    show_plots=True
)

# Calculate overlap statistics
overlap_percentage = result['pixels_both_populations'] / result['total_pixels'] * 100
print(f"Population overlap: {overlap_percentage:.2f}%")
```

## Parameter Reference

### Simulation Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| `size` | Grid size in pixels | Integer (e.g., 50, 100, 200) |
| `allowed_shape` | Shape of allowed region | 'square' or 'circle' |
| `forbidden_shape` | Shape of forbidden region | 'none', 'square', or 'circle' |
| `forbidden_size` | Relative size of forbidden region | Float between 0 and 1 |
| `cell_count` | Number of cells to simulate | Integer |
| `cell_radius` | Radius of individual cells | Float (relative to grid size) |
| `cell_interaction_radius` | Cell interaction distance | Float (relative to grid size) |

### Distribution Parameters

| Distribution Type | Parameters | Description |
|-------------------|------------|-------------|
| 'normal' | (mean, std) | Normal distribution with given mean and standard deviation |
| 'uniform' | (loc, scale) | Uniform distribution from loc to loc+scale |
| 'beta' | (alpha, beta) | Beta distribution with shape parameters alpha and beta |

## Output Metrics

### Single Population Output

| Metric | Description |
|--------|-------------|
| `pixels_covered_0` | Number of pixels with no cell coverage |
| `pixels_covered_1` | Number of pixels covered by exactly one cell |
| `pixels_covered_2_plus` | Number of pixels covered by multiple cells |
| `total_pixels` | Total number of available pixels |

### Two Population Output

| Metric | Description |
|--------|-------------|
| `pixels_no_interaction` | Pixels with no cell coverage |
| `pixels_pop1_only` | Pixels covered only by population 1 |
| `pixels_pop2_only` | Pixels covered only by population 2 |
| `pixels_both_populations` | Pixels covered by both populations |
| `total_pixels` | Total number of available pixels |

## Running Tests

```bash
# Run all tests
python test_simulation.py

# Run specific test categories
python test_simulation.py --test basic  # Basic functionality tests
python test_simulation.py --test edge   # Edge case tests
python test_simulation.py --test perf   # Performance tests
```

## Examples

### Example 1: Different Distribution Comparison

```python
import matplotlib.pyplot as plt
from simulation import simulate_cell_distribution
from scipy import stats
import numpy as np

grid_size = 100
allowed_region = np.ones((grid_size, grid_size), dtype=bool)

distributions = [
    ('Uniform', stats.uniform(loc=0, scale=1)),
    ('Normal', stats.norm(loc=0.5, scale=0.1)),
    ('Beta (α=2, β=2)', stats.beta(2, 2)),
    ('Beta (α=2, β=5)', stats.beta(2, 5))
]

plt.figure(figsize=(15, 10))
for i, (name, dist) in enumerate(distributions, 1):
    plt.subplot(2, 2, i)
    simulate_cell_distribution(
        x_distribution=dist,
        y_distribution=dist,
        cell_count=1000,
        allowed_region=allowed_region,
        cell_interaction_radius=0.05,
        show_plots=True
    )
    plt.title(f"Distribution: {name}")

plt.tight_layout()
plt.savefig("distribution_comparison.png")
plt.show()
```

### Example 2: Statistical Analysis

```python
from generate_dataset import generate_datasets
import pandas as pd
import matplotlib.pyplot as plt

# Generate datasets with multiple parameter variations
df_single, df_two = generate_datasets(n_repeats=5)

# Analyze how cell radius affects coverage
plt.figure(figsize=(10, 6))
coverage_by_radius = df_single[df_single['varied_parameter'] == 'cell_radius'].groupby('cell_radius').apply(
    lambda x: (x['pixels_covered_1'] + x['pixels_covered_2_plus']).mean() / x['total_pixels'].mean() * 100
)
plt.plot(coverage_by_radius.index, coverage_by_radius.values, 'o-')
plt.xlabel('Cell Radius')
plt.ylabel('Coverage (%)')
plt.title('Effect of Cell Radius on Coverage')
plt.grid(True)
plt.savefig("radius_coverage_analysis.png")
plt.show()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

To contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
