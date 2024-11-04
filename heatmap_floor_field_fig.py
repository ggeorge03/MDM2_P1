import numpy as np
import matplotlib.pyplot as plt

grid_size = (200, 200)  # Dimensions
exit_position = (0, 100)  # Exit at top middle

# Initialize floor field matrix to store distances
distance_field = np.zeros(grid_size)

# Calculate distance from the exit for each cell
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        # Calculate Euclidean distance from the exit cell
        distance = np.sqrt(
            (i - exit_position[0])**2 + (j - exit_position[1])**2)
        distance_field[i, j] = distance

# Normalize distances to range 0 to 100 for visualization
distance_field_normalized = distance_field / distance_field.max() * 100

plt.figure(figsize=(8, 6))
plt.imshow(distance_field_normalized, cmap='turbo', interpolation='nearest')
plt.colorbar(label='Distance from Exit')
plt.title('Distance-Based Floor Field Coloring')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.show()
