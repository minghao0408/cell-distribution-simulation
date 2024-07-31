This example demonstrates the generation and visualization of a constrained cell distribution model with two distinct cell types. The key features are:

1. Binary Mask: A 100x100 boolean array is created, with a central 20x20 region set as a "forbidden zone" where cells cannot be placed.

2. Cell Generation: Two types of cells are generated, 50 of each type. Each cell type follows a different spatial distribution:
   - Type 1: Normal distribution with mean -1 and standard deviation 0.5 for both x and y coordinates.
   - Type 2: Normal distribution with mean 1 and standard deviation 0.5 for both x and y coordinates.

3. Spatial Constraints: Cells are only placed in areas allowed by the binary mask, and a minimum distance of 0.2 is maintained between all cells.

4. Visualization: The results are plotted on a 10x10 inch figure. The binary mask is displayed as a background, with the forbidden zone visible. The two cell types are represented by scatter plots of different colors, clearly showing their distinct spatial distributions.
This model simulates a scenario where different cell types have varying spatial preferences and certain areas are restricted for cell placement. It could be used to study cell distribution patterns, spatial interactions, or as a basis for more complex biological simulations.
The example showcases the flexibility of the `generate_cell_centers` function, which can handle multiple cell types with different distributions while respecting spatial constraints defined by a binary mask.
