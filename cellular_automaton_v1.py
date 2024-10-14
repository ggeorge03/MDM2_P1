import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors

rows, columns = 200, 200
exit_location = (rows - 1, columns // 2)

move_prob = 1  # Probability of attempting to move
exit_influence = 800  # Probability multiplier for moving towards the exit

# List to store the number of people remaining at each frame
people_remaining_over_time = []

# Initialize the grid with people randomly placed (1=person, 0=empty)


def initialize_grid(rows, columns, num_people=150):
    grid = np.zeros((rows, columns))
    positions = np.random.choice(rows * columns, num_people, replace=False)
    for pos in positions:
        grid[pos // columns, pos % columns] = 1
    return grid


def distance_to_exit(i, j, exit_location):
    '''Compute Manhattan distance to the exit'''
    return abs(i - exit_location[0]) + abs(j - exit_location[1])

# Function to update the grid based on movement probabilities


def update(frameNum, img, grid, exit_location, num_frames):
    new_grid = grid.copy()
    rows, columns = grid.shape

    # Iterate over every cell in the grid
    for i in range(rows):
        for j in range(columns):
            if grid[i, j] == 1:  # If there's a person in the current cell
                neighbors = []
                probs = []

                # Check all four neighboring cells (up, down, left, right) +diagonals
                for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1),(i-1,j-1),(i+1,j-1),(i-1,j+1),(i+1,j+1)]:
                    if 0 <= ni < rows and 0 <= nj < columns and grid[ni, nj] == 0:
                        neighbors.append((ni, nj))
                        dist = distance_to_exit(ni, nj, exit_location)
                        # Closer cells have higher probabilities
                        base_prob = move_prob / (dist + 1)
                        probs.append(base_prob * exit_influence if dist <
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

    # Set the exit cell to 0 (people leave)
    new_grid[exit_location] = 0

    # Update data for the plot
    img.set_data(new_grid)
    grid[:] = new_grid[:]  # Update the original grid

    # Count the number of people left
    num_people_remaining = np.sum(new_grid == 1)
    print(f"Frame {frameNum}: {num_people_remaining} people remaining")

    # Store the number of remaining people for plotting later
    people_remaining_over_time.append(num_people_remaining)

    # Stop the simulation if no people are left
    if num_people_remaining == 0:
        print(f"All people have evacuated in {frameNum} frames.")
        ani.event_source.stop()  # Stop the animation

    return img,

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


def run_egress_simulation():
    grid = initialize_grid(rows, columns)

    # Define a colormap: empty cells = white, people = black, exit = red
    cmap = colors.ListedColormap(['white', 'black', 'red'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Set up the plot
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest', cmap=cmap, norm=norm)

    # Run the animation (slower animation with interval = 100 ms)
    global ani
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, exit_location, 0),
                                  frames=900, interval=80, save_count=50)
    plt.show()

    # After the animation stops, plot the remaining people over time
    plot_people_remaining()


# Run the simulation
run_egress_simulation()
