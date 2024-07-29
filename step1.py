# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:39:41 2024

@author: HP
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from scipy.stats import rv_continuous

def generate_cell_centers(
    N: Union[int, List[int]],
    x_distribution: rv_continuous,
    y_distribution: rv_continuous,
    bounding_box: List[float] = None
) -> List[Tuple[float, float]]:
    # Convert N to a list if it is an integer
    if isinstance(N, int):
        N = [N]
    
    # Calculate the total number of cells
    total_cells = sum(N)
    
    # Generate x and y coordinates using the given distributions
    x_coords = x_distribution.rvs(size=total_cells)
    y_coords = y_distribution.rvs(size=total_cells)
    
    # If bounding box is provided, clip the coordinates to the bounds
    if bounding_box:
        x_lower, y_lower, x_upper, y_upper = bounding_box
        x_coords = np.clip(x_coords, x_lower, x_upper)
        y_coords = np.clip(y_coords, y_lower, y_upper)
    
    # Pack the coordinates into a list of tuples
    cell_centers = list(zip(x_coords, y_coords))
    
    return cell_centers

# Example usage
cell_centers = generate_cell_centers(
    100, 
    norm(loc=0, scale=1),  # x distribution (mean=0, std=1)
    norm(loc=0, scale=1),  # y distribution (mean=0, std=1)
    bounding_box=[-3, -3, 3, 3]  # bounding box
)

# Plot the generated cell centers
x_coords, y_coords = zip(*cell_centers)
plt.scatter(x_coords, y_coords, alpha=0.6)  # Set transparency to 0.6
plt.title("Simulated Cell Distribution with Normal Distribution (with Bounding Box)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.xlim([-4, 4])  # Set x-axis range
plt.ylim([-4, 4])  # Set y-axis range
plt.grid(True)
plt.show()
