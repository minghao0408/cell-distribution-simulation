import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Circle
from typing import Tuple, Optional, Union, List

def create_circular_region(size: int, radius: int) -> np.ndarray:
    """Create a circular boolean mask."""
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    return dist_from_center <= radius

def update_placement_mask(mask: np.ndarray, x: float, y: float, cell_radius: float, grid_size: int):
    """Update the placement mask after placing a cell."""
    cell_mask = create_circular_region(grid_size, int(2 * cell_radius * grid_size))
    cell_center = (int(y * grid_size), int(x * grid_size))
    y_start = max(0, cell_center[0] - cell_mask.shape[0]//2)
    y_end = min(grid_size, cell_center[0] + cell_mask.shape[0]//2)
    x_start = max(0, cell_center[1] - cell_mask.shape[1]//2)
    x_end = min(grid_size, cell_center[1] + cell_mask.shape[1]//2)
    mask[y_start:y_end, x_start:x_end] &= ~cell_mask[:y_end-y_start, :x_end-x_start]

def simulate_cell_distribution(
    x_distribution,
    y_distribution,
    cell_count: int,
    allowed_region: np.ndarray,
    cell_interaction_radius: float,
    forbidden_region: Optional[np.ndarray] = None,
    cell_radius: Optional[float] = None,
    show_plots: bool = False
) -> Tuple[int, int, int, int, int, int]:
    """
    Simulate cell distribution for a single population.
    """
    grid_size = allowed_region.shape[0]
    cells = []
    
    # Generate all cells at once
    x_values = x_distribution.rvs(size=cell_count * 10)
    y_values = y_distribution.rvs(size=cell_count * 10)
    
    # Rescale values to [0, 1] range
    x_values = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
    y_values = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
    
    # Create cell placement mask
    cell_placement_mask = allowed_region.copy()
    if forbidden_region is not None:
        cell_placement_mask &= ~forbidden_region
    
    for i in range(len(x_values)):
        if len(cells) >= cell_count:
            break
        
        x, y = x_values[i], y_values[i]
        x_idx = min(int(x * grid_size), grid_size - 1)
        y_idx = min(int(y * grid_size), grid_size - 1)
        
        if cell_placement_mask[y_idx, x_idx]:
            if cell_radius is None or not any(np.hypot(x - cx, y - cy) < 2 * cell_radius for cx, cy in cells):
                cells.append((x, y))
                if cell_radius is not None:
                    update_placement_mask(cell_placement_mask, x, y, cell_radius, grid_size)
    
    cells = np.array(cells)
    
    # Create interaction map
    interaction_map = np.zeros_like(allowed_region, dtype=int)
    y_indices, x_indices = np.ogrid[:grid_size, :grid_size]
    for cell in cells:
        distances = np.hypot(x_indices/grid_size - cell[0], y_indices/grid_size - cell[1])
        interaction_map += (distances <= cell_interaction_radius)
    
    # Apply allowed region mask and forbidden region
    if forbidden_region is not None:
        allowed_region = allowed_region & ~forbidden_region
    
    # Calculate statistics
    total_pixels = np.sum(allowed_region)
    
    if cell_radius is not None:
        # Create a mask of all cell areas
        cell_area_mask = np.zeros_like(allowed_region, dtype=bool)
        for cell in cells:
            cell_mask = create_circular_region(grid_size, int(cell_radius * grid_size))
            cell_center = (int(cell[1] * grid_size), int(cell[0] * grid_size))
            y_start = max(0, cell_center[0] - cell_mask.shape[0]//2)
            y_end = min(grid_size, cell_center[0] + cell_mask.shape[0]//2)
            x_start = max(0, cell_center[1] - cell_mask.shape[1]//2)
            x_end = min(grid_size, cell_center[1] + cell_mask.shape[1]//2)
            cell_area_mask[y_start:y_end, x_start:x_end] |= cell_mask[:y_end-y_start, :x_end-x_start]
        
        # Subtract cell areas from total pixels and apply to interaction map
        total_pixels -= np.sum(cell_area_mask & allowed_region)
        interaction_map[cell_area_mask] = 0
    
    # Calculate pixel coverages
    pixels_covered_0 = np.sum((interaction_map == 0) & allowed_region)
    pixels_covered_1 = np.sum((interaction_map == 1) & allowed_region)
    pixels_covered_2_plus = np.sum((interaction_map >= 2) & allowed_region)
    pixels_covered_2_plus_not_forbidden = pixels_covered_2_plus
    
    if show_plots:
        plt.figure(figsize=(10, 10))
        plt.imshow(interaction_map, cmap='viridis', interpolation='nearest', extent=[0, 1, 0, 1])
        plt.colorbar(label='Interaction count')
        
        for cell in cells:
            plt.scatter(cell[0], cell[1], color='red', s=20)
            circle = Circle(cell, cell_interaction_radius, fill=False, color='r')
            plt.gca().add_artist(circle)
        
        if forbidden_region is not None:
            plt.imshow(forbidden_region, cmap='Reds', alpha=0.3, extent=[0, 1, 0, 1])
        
        plt.title('Cell Distribution Simulation')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    return (
        pixels_covered_0,
        pixels_covered_1,
        pixels_covered_2_plus,
        total_pixels,
        pixels_covered_2_plus_not_forbidden,
        total_pixels
    )

def simulate_two_populations(
    x_distribution1, y_distribution1,
    x_distribution2, y_distribution2,
    cell_counts: Tuple[int, int],
    allowed_region: np.ndarray,
    cell_interaction_radius: float,
    forbidden_region: Optional[np.ndarray] = None,
    cell_radius: Optional[float] = None,
    show_plots: bool = False
) -> Tuple[int, int, int, int, int]:
    """
    Simulate cell distribution for two populations with fair placement.
    """
    grid_size = allowed_region.shape[0]
    cells1 = []
    cells2 = []
    
    # Generate all potential cell positions for both populations
    total_cells = cell_counts[0] + cell_counts[1]
    all_x_values = np.concatenate([
        x_distribution1.rvs(size=cell_counts[0] * 10),
        x_distribution2.rvs(size=cell_counts[1] * 10)
    ])
    all_y_values = np.concatenate([
        y_distribution1.rvs(size=cell_counts[0] * 10),
        y_distribution2.rvs(size=cell_counts[1] * 10)
    ])
    
    # Create population assignments (1 or 2)
    pop_assignments = np.concatenate([
        np.ones(cell_counts[0] * 10),
        np.ones(cell_counts[1] * 10) * 2
    ])
    
    # Shuffle all arrays together
    shuffle_idx = np.random.permutation(len(all_x_values))
    all_x_values = all_x_values[shuffle_idx]
    all_y_values = all_y_values[shuffle_idx]
    pop_assignments = pop_assignments[shuffle_idx]
    
    # Normalize coordinates
    all_x_values = (all_x_values - np.min(all_x_values)) / (np.max(all_x_values) - np.min(all_x_values))
    all_y_values = (all_y_values - np.min(all_y_values)) / (np.max(all_y_values) - np.min(all_y_values))
    
    # Create cell placement mask
    cell_placement_mask = allowed_region.copy()
    if forbidden_region is not None:
        cell_placement_mask &= ~forbidden_region
    
    # Place cells for both populations simultaneously
    for i in range(len(all_x_values)):
        if len(cells1) >= cell_counts[0] and len(cells2) >= cell_counts[1]:
            break
            
        x, y = all_x_values[i], all_y_values[i]
        x_idx = min(int(x * grid_size), grid_size - 1)
        y_idx = min(int(y * grid_size), grid_size - 1)
        
        if cell_placement_mask[y_idx, x_idx]:
            population = int(pop_assignments[i])
            if population == 1 and len(cells1) < cell_counts[0]:
                if cell_radius is None or not any(np.hypot(x - cx, y - cy) < 2 * cell_radius 
                                                for cx, cy in cells1 + cells2):
                    cells1.append((x, y))
                    if cell_radius is not None:
                        update_placement_mask(cell_placement_mask, x, y, cell_radius, grid_size)
                        
            elif population == 2 and len(cells2) < cell_counts[1]:
                if cell_radius is None or not any(np.hypot(x - cx, y - cy) < 2 * cell_radius 
                                                for cx, cy in cells1 + cells2):
                    cells2.append((x, y))
                    if cell_radius is not None:
                        update_placement_mask(cell_placement_mask, x, y, cell_radius, grid_size)
    
    cells1 = np.array(cells1)
    cells2 = np.array(cells2)
    
    # Create interaction maps for each population
    interaction_map1 = np.zeros_like(allowed_region, dtype=bool)
    interaction_map2 = np.zeros_like(allowed_region, dtype=bool)
    
    y_indices, x_indices = np.ogrid[:grid_size, :grid_size]
    for cell in cells1:
        distances = np.hypot(x_indices/grid_size - cell[0], y_indices/grid_size - cell[1])
        interaction_map1 |= (distances <= cell_interaction_radius)
    
    for cell in cells2:
        distances = np.hypot(x_indices/grid_size - cell[0], y_indices/grid_size - cell[1])
        interaction_map2 |= (distances <= cell_interaction_radius)
    
    # Apply allowed region mask
    if forbidden_region is not None:
        allowed_region = allowed_region & ~forbidden_region
        interaction_map1 &= allowed_region
        interaction_map2 &= allowed_region
    
    # Calculate statistics
    total_pixels = np.sum(allowed_region)
    pixels_pop1 = np.sum(interaction_map1)
    pixels_pop2 = np.sum(interaction_map2)
    pixels_overlap = np.sum(interaction_map1 & interaction_map2)
    pixels_no_interaction = np.sum(allowed_region & ~(interaction_map1 | interaction_map2))
    
    if show_plots:
        plt.figure(figsize=(10, 10))
        combined_map = np.zeros_like(allowed_region, dtype=int)
        combined_map[interaction_map1] = 1
        combined_map[interaction_map2] += 1
        plt.imshow(combined_map, cmap='viridis', interpolation='nearest', extent=[0, 1, 0, 1])
        plt.colorbar(label='Population overlap')
        
        plt.scatter(cells1[:, 0], cells1[:, 1], color='red', s=20, label='Population 1')
        plt.scatter(cells2[:, 0], cells2[:, 1], color='blue', s=20, label='Population 2')
        
        if forbidden_region is not None:
            plt.imshow(forbidden_region, cmap='Reds', alpha=0.3, extent=[0, 1, 0, 1])
        
        plt.title('Two Population Cell Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
    
    return (
        pixels_no_interaction,
        pixels_pop1 - pixels_overlap,
        pixels_pop2 - pixels_overlap,
        pixels_overlap,
        total_pixels
    )
