import matplotlib.pyplot as plt
import numpy as np

def plot_cell_distribution(cell_centers: np.ndarray, cell_types: np.ndarray, binary_mask: np.ndarray):
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_mask, extent=[-3, 3, -3, 3], alpha=0.3, cmap='gray_r')
    
    unique_types = np.unique(cell_types)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_types)))
    
    for i, cell_type in enumerate(unique_types):
        mask = cell_types == cell_type
        plt.scatter(cell_centers[mask, 0], cell_centers[mask, 1], 
                    color=colors[i], alpha=0.6, label=f'Type {cell_type}')

    plt.title("Simulated Cell Distribution with Interactions")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.grid(True)
    plt.colorbar(label="Binary Mask")
    plt.legend()
    plt.show()
