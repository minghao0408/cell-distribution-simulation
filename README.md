# Cell Distribution Simulation

This project simulates the distribution of different cell types based on spatial interactions and environmental constraints.

## Description

This simulation generates cell centers for multiple cell types, taking into account:
- Spatial distributions for each cell type
- Interactions between different cell types
- Environmental constraints (represented by a binary mask)
- Minimum distance between cells

The program also estimates the interaction matrix based on the generated cell distribution and visualizes the results.

## Features

- Generate cell distributions based on configurable parameters
- Estimate interaction matrices from cell distributions
- Visualize cell distributions and forbidden regions
- Configurable simulation parameters
- Unit tests to ensure code reliability

## Installation

1. Clone this repository: git clone https://github.com/minghao0408/cell-distribution-simulation.git   
2. Navigate to the project directory: cd cell-distribution-simulation
3. Install the required packages: pip install numpy scipy matplotlib

## Usage

To run the simulation: python main.py
This will run the unit tests and then execute the simulation, displaying the results.

## Configuration

You can modify the simulation parameters in the `config.py` file. Available parameters include:

- `mask_size`: Size of the simulation area
- `forbidden_region`: Coordinates of the forbidden region
- `cell_counts`: Number of cells for each type
- `x_distributions` and `y_distributions`: Spatial distributions for cell placement
- `interaction_matrix`: Matrix defining interactions between cell types
- `min_distance`: Minimum allowed distance between cells

## Project Structure

- `main.py`: Entry point of the program
- `config.py`: Configuration parameters
- `simulation.py`: Core simulation logic
- `test_simulation.py`: Unit tests

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was inspired by biological cell distribution patterns
- Thanks to the SciPy and Matplotlib communities for their excellent libraries
