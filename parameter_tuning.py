import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from matplotlib import colors
from sklearn.metrics import confusion_matrix

rows, columns = 200, 200

# Define a 3-cell wide exit at the top middle of the grid (row 0, middle 3 columns)
exit_start = columns // 2 - 1  # The first column of the 3-cell wide exit
exit_location = [(0, exit_start), (0, exit_start + 1),
                 (0, exit_start + 2)]  # Exit is on row 0

move_prob = 1  # Probability of attempting to move
# exit_influence = 8  # Probability multiplier for moving towards the exit

# List to store the number of people remaining at each frame
people_remaining_over_time = []

# Initialize the grid with people randomly placed (1=person, 0=empty)


def initialize_grid(rows, columns, num_people=170):
    grid = np.zeros((rows, columns))
    positions = np.random.choice(rows * columns, num_people, replace=False)
    for pos in positions:
        grid[pos // columns, pos % columns] = 1
    return grid


def initialize_floor_field(rows, columns, exit_location=exit_location):
    floor_field = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            # Set the floor field value inversely proportional to distance to exit
            floor_field[i, j] = distance_to_exit(i, j, exit_location)
    # Invert values for visualization (higher values near exit)
    floor_field = np.max(floor_field) - floor_field
    return floor_field


def distance_to_exit(i, j, exit_location):
    '''Compute Manhattan distance to the nearest exit cell'''
    return min([abs(i - ex[0]) + abs(j - ex[1]) for ex in exit_location])

# Function to update the grid based on movement probabilities


def update(frameNum, img1, img2, grid, exit_location, floor_field, exit_influence, speed):
    new_grid = grid.copy()
    rows, columns = grid.shape
    floor_field = update_floor_field(floor_field)

    # Iterate over every cell in the grid
    for i in range(rows):
        for j in range(columns):
            if grid[i, j] == 1:  # If there's a person in the current cell
                neighbors = []
                probs = []
                # Check all neighboring cells (including diagonals)
                for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1),
                               (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]:
                    if 0 <= ni < rows and 0 <= nj < columns and grid[ni, nj] == 0:
                        neighbors.append((ni, nj))
                        dist = distance_to_exit(ni, nj, exit_location)
                        # Closer cells have higher probabilities
                        base_prob = move_prob / (dist + 1)
                        floor_field_influence = floor_field[ni, nj]
                        probs.append((base_prob * exit_influence) + floor_field_influence if dist <
                                     distance_to_exit(i, j, exit_location) else base_prob)

                # Normalize probabilities
                if neighbors:
                    probs = np.array(probs)
                    probs /= probs.sum()  # Normalize to sum to 1
                    # Move to a neighbor based on probability
                    if np.random.rand() < move_prob:  # Randomly decide if the person tries to move
                        new_location = neighbors[np.random.choice(
                            len(neighbors), p=probs)]
                        new_grid[i, j] = 0  # Current cell becomes empty
                        # Move person to the new cell
                        new_grid[new_location] = 1
                        # Increase floor field value at the new location
                        floor_field[new_location] += 1

    # Set the exit cells to 0 (people leave through any of the 3 exit cells)
    for ex in exit_location:
        new_grid[ex] = 0

    # Update the people grid in img1
    img1.set_data(new_grid)

    # Update the floor field display in img2 (apply transparency to overlay)
    img2.set_data(floor_field)
    img2.set_clim(vmin=0, vmax=np.max(floor_field))

    grid[:] = new_grid[:]  # Update the original grid

    # Count the number of people left
    num_people_remaining = np.sum(new_grid == 1)
    people_remaining_over_time.append(num_people_remaining)
    # Stop the simulation if no people are left
    if num_people_remaining == 0:
        plt.title("All people have exited. Close this window to exit.")
        ani.event_source.stop()  # Stop the animation

    return img1, img2  # Ensure both images are updated


def update_floor_field(floor_field, decay_rate=0.03):
    floor_field *= (1 - decay_rate)
  # Decay existing floor field values
    return floor_field

# Function to plot people remaining over time after simulation ends


def plot_people_remaining():
    plt.figure()
    plt.plot(people_remaining_over_time, label="People remaining")
    plt.xlabel("Frame")
    plt.ylabel("People Remaining")
    plt.title("People Remaining in Room Over Time")
    plt.legend()
    plt.show()

# Main function to run the cellular automaton


def run_egress_simulation(speed, exit_influence, floor_field_factor):
    grid = initialize_grid(rows, columns)
    floor_field = initialize_floor_field(rows, columns)
    floor_field *= floor_field_factor

    fig, ax = plt.subplots()
    cmap = colors.ListedColormap(['black', 'red'])
    bounds = [-0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap_floor = plt.cm.coolwarm

    img1 = ax.imshow(grid, interpolation='nearest', cmap=cmap, norm=norm)
    img2 = ax.imshow(floor_field, cmap=cmap_floor, alpha=0.6, vmin=0, vmax=1)

    global ani
    ani = animation.FuncAnimation(
        fig, update, fargs=(img1, img2, grid, exit_location,
                            floor_field, exit_influence, speed),
        frames=900, interval=80, save_count=50)
    plt.show()

    # After the animation stops, plot the remaining people over time
    return np.sum(np.array(people_remaining_over_time) > 0)


# Run the simulation
# run_egress_simulation()


def perform_grid_search():
    speed_range = np.arange(0.5, 2.1, 0.5)
    exit_influence_range = np.arange(1, 6, 1)
    floor_field_factor_range = np.arange(1, 11, 2)

    results = []

    actual_egress_time = 200  # Replace with actual data egress time

    for speed in speed_range:
        for exit_influence in exit_influence_range:
            for floor_field_factor in floor_field_factor_range:
                simulated_egress_time = run_egress_simulation(
                    speed, exit_influence, floor_field_factor)
                error = abs(simulated_egress_time - actual_egress_time)
                results.append({
                    'speed': speed,
                    'exit_influence': exit_influence,
                    'floor_field_factor': floor_field_factor,
                    'error': error
                })

    results_df = pd.DataFrame(results)
    print(results_df)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(results_df['speed'], results_df['exit_influence'], results_df['floor_field_factor'],
                    c=results_df['error'], cmap='viridis')
    ax.set_xlabel('Speed')
    ax.set_ylabel('Exit Influence')
    ax.set_zlabel('Floor Field Factor')
    plt.colorbar(sc, label="Error")
    plt.title("Error in Egress Time vs. Model Parameters")
    plt.show()
    plot_3d_with_conf_matrices(results_df)


def plot_3d_with_conf_matrices(results_df):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot confusion matrices on three random parameter slices
    slice_speeds = results_df['speed'].sample(1).values
    slice_exit_influences = results_df['exit_influence'].sample(1).values
    slice_floor_fields = results_df['floor_field_factor'].sample(1).values

    for speed in slice_speeds:
        df_slice = results_df[results_df['speed'] == speed]
        for _, row in df_slice.iterrows():
            conf_matrix = row['conf_matrix']
            plt_conf_matrix_in_3d(
                ax, conf_matrix, speed, row['exit_influence'], row['floor_field_factor'], axis='z')

    for exit_influence in slice_exit_influences:
        df_slice = results_df[results_df['exit_influence'] == exit_influence]
        for _, row in df_slice.iterrows():
            conf_matrix = row['conf_matrix']
            plt_conf_matrix_in_3d(
                ax, conf_matrix, row['speed'], exit_influence, row['floor_field_factor'], axis='y')

    for floor_field_factor in slice_floor_fields:
        df_slice = results_df[results_df['floor_field_factor']
                              == floor_field_factor]
        for _, row in df_slice.iterrows():
            conf_matrix = row['conf_matrix']
            plt_conf_matrix_in_3d(
                ax, conf_matrix, row['speed'], row['exit_influence'], floor_field_factor, axis='x')

    # Labels
    ax.set_xlabel('Speed')
    ax.set_ylabel('Exit Influence')
    ax.set_zlabel('Floor Field Factor')
    plt.title('3D Parameter Space with Confusion Matrix Slices')
    plt.show()


def plt_conf_matrix_in_3d(ax, conf_matrix, x, y, z, axis):
    fig, ax2 = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d",
                cmap="Blues", cbar=False, ax=ax2)
    ax2.set_title(
        f'Confusion Matrix at {axis}={x if axis=="x" else y if axis=="y" else z}')
    plt.close(fig)  # Close the extra plot

    # Render the confusion matrix as an image in 3D space
    img = fig.canvas.tostring_rgb()
    rows, cols = conf_matrix.shape
    x_offset, y_offset, z_offset = 0.1, 0.1, 0.1

    if axis == 'x':
        ax.plot_surface([[x - x_offset, x + x_offset], [x - x_offset, x + x_offset]],
                        [[y - y_offset, y - y_offset],
                            [y + y_offset, y + y_offset]],
                        [[z, z], [z, z]],
                        rstride=1, cstride=1, facecolors=img)
    elif axis == 'y':
        ax.plot_surface([[x - x_offset, x + x_offset], [x - x_offset, x + x_offset]],
                        [[y, y], [y, y]],
                        [[z - z_offset, z - z_offset],
                            [z + z_offset, z + z_offset]],
                        rstride=1, cstride=1, facecolors=img)
    elif axis == 'z':
        ax.plot_surface([[x, x], [x, x]],
                        [[y - y_offset, y + y_offset],
                            [y - y_offset, y + y_offset]],
                        [[z - z_offset, z + z_offset],
                            [z - z_offset, z + z_offset]],
                        rstride=1, cstride=1, facecolors=img)


# Run grid search
perform_grid_search()
