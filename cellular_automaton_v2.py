import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from matplotlib import colors
from sklearn.metrics import confusion_matrix

max_agents_total = 176  # Total agents to be processed in the simulation
current_agents = 40     # Initial agents on the grid
agents_processed = 40   # Initial count includes the starting agents


agent_ids = {}      # Dictionary to store agent positions and their IDs
exited_agents = set()  # Set to track agents that have exited

next_agent_id = 0      # Variable to assign unique IDs

agents_left=0 #to count number of agents having left

rows, columns = 13, 10
rect_height, rect_width = 8, 10  # Height and width of the main area
path_length, path_width = 5,2 # Path width and length leading to exit

# Calculate start of exit area
exit_start = int(columns // 2 - path_width // 2)
exit_end = int(columns // 2 + path_width // 2)    # Calculate end of exit area

# Create a list of coordinates for the exit cells
exit_location = [(0, col) for col in range(exit_start, exit_end)]

# recording_start=columns // 2 - 1
# recording_location = [(path_length, recording_start), (path_length, exit_start + 1),
#                  (path_length, exit_start + 2)]

move_prob = 1  # Probability of attempting to move
# exit_influence = 16  # Probability multiplier for moving towards the exit




# Initialize the grid with people randomly placed (1=person, 0=empty)


def initialize_grid(rows, columns, initial_agents=30):  # Start with 40 agents
    global next_agent_id
    next_agent_id=0
    grid = np.full((rows, columns), -1)
    global rect_start_row
    rect_start_row = rows - rect_height
    rect_start_col = (columns - rect_width) // 2
    grid[rect_start_row:rect_start_row + rect_height, rect_start_col:rect_start_col + rect_width] = 0
    global path_start_row, path_start_col
    path_start_row = rect_start_row - path_length
    path_start_col = (columns - path_width) // 2
    grid[path_start_row:path_start_row + path_length, path_start_col:path_start_col + path_width] = 0

    # Define free positions within the main area and outside the pathway
    free_positions = np.argwhere(
    (grid == 0) &  # Empty cells only
    (
        (np.arange(rows)[:, None] >= rows - 4) &  # Limit to the last 4 rows only
        (
            (np.arange(columns) < path_start_col) | (np.arange(columns) >= path_start_col + path_width)  # Exclude pathway columns within these rows
        )
    )
)
    positions = free_positions[np.random.choice(len(free_positions), initial_agents, replace=False)]
    for pos in positions:
        grid[pos[0], pos[1]] = 1
        agent_ids[(pos[0], pos[1])] = next_agent_id  # Assign a unique ID to each agent
        next_agent_id += 1

    return grid




def initialize_floor_field(rows, columns, exit_location=exit_location, alpha=0.0001):
    floor_field = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            # Set the floor field value inversely proportional to distance to exit
            distance = distance_to_exit(i, j, exit_location)
            floor_field[i, j] = np.exp(-alpha * distance)
    # Invert values for visualization (higher values near exit)
    floor_field = np.max(floor_field) - floor_field
    return floor_field


def distance_to_exit(i, j, exit_location):
    '''Compute Manhattan distance to the nearest exit cell'''
    return min([abs(i - ex[0]) + abs(j - ex[1]) for ex in exit_location])

# Function to update the grid based on movement probabilities


# Set a slower spawn rate and specify back rows for spawning
  # Frames between each spawn attempt
  # Only spawn agents in the last 3 rows of the main area
# Increase spawn rate and batch size for each spawn event
# spawn_rate = 10   # Spawn every 10 frames
# batch_size = 10   # Number of agents to spawn per batch

def update(frameNum, img1, img2, agent_scatter, grid, exit_location, floor_field, exit_influence, speed):
    global exited_agents, next_agent_id, agents_processed
    new_grid = grid.copy()
    rows, columns = grid.shape
    floor_field = update_floor_field(floor_field)

    # Movement logic remains mostly the same
    for i in range(rows):
        for j in range(columns):
            if grid[i, j] == 1:
                agent_id = agent_ids.get((i, j))  # Get the agent's unique ID
                neighbors = []
                probs = []
                
                # Check all neighboring cells
                for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]:
                    if 0 <= ni < rows and 0 <= nj < columns and grid[ni, nj] == 0 and new_grid[ni, nj] != 1:
                        neighbors.append((ni, nj))
                        dist = distance_to_exit(ni, nj, exit_location)
                        base_prob = move_prob / (dist + 1)
                        floor_field_influence = floor_field[ni, nj]
                        probs.append((base_prob * exit_influence) + floor_field_influence if dist < distance_to_exit(i, j, exit_location) else base_prob)
                
                # Move the agent based on calculated probabilities
                if neighbors:
                    probs = np.array(probs)
                    probs /= probs.sum()
                    new_location = neighbors[np.random.choice(len(neighbors), p=probs)]
                    new_row, new_col = new_location

                    # Move agent to the new location
                    new_grid[i, j] = 0
                    new_grid[new_location] = 1
                    floor_field[new_location] += 1
                    agent_ids[new_location] = agent_id  # Update agent's new position in agent_ids
                    del agent_ids[(i, j)]  # Remove old position

                    # Check if the agent has exited (i.e., reached the top rows in the pathway area)
                    if new_row < path_length and agent_id not in exited_agents:
                        exited_agents.add(agent_id)  # Mark this agent as exited
                        print(f"Agent {agent_id} exited. Total exits: {len(exited_agents)}")

    # Clear exit cells to ensure they remain empty
    for ex in exit_location:
        new_grid[ex] = 0

    # Spawn new agents at a slower rate in the back rows
    spawn_rate = 20   # Spawn every 10 frames
    batch_size = 10   # Number of agents to spawn per batch
    if frameNum % spawn_rate == 0 and agents_processed < max_agents_total:
        free_positions = np.argwhere(
            (new_grid == 0) &  # Empty cells only
            (np.arange(rows)[:, None] >= rows - 4) &  # Limit to last 4 rows
            ((np.arange(columns) < path_start_col) | (np.arange(columns) >= path_start_col + path_width))  # Exclude pathway columns
        )  
        
        # Set the number of agents to spawn based on `batch_size`
        num_new_agents = min(batch_size, max_agents_total - agents_processed)
        if len(free_positions) > 0 and num_new_agents > 0:
            # Filter free positions to last 4 rows only
            free_positions = free_positions[free_positions[:, 0] >= rect_start_row + rect_height - 4]

            # Adjust `num_new_agents` if fewer free positions than needed
            num_new_agents = min(len(free_positions), num_new_agents)

            # Now sample positions safely
            if num_new_agents > 0:  # Proceed only if there are positions to sample
                new_agent_positions = free_positions[np.random.choice(len(free_positions), num_new_agents, replace=False)]

                for pos in new_agent_positions:
                    new_grid[pos[0], pos[1]] = 1
                    agent_ids[(pos[0], pos[1])] = next_agent_id  # Assign a unique ID to each new agent
                    next_agent_id += 1
                agents_processed += num_new_agents
                print(f"Spawned {num_new_agents} new agents. Total processed: {agents_processed}")

    # Update scatter plot with new agent positions
    agent_positions = np.argwhere(new_grid == 1)
    agent_scatter.set_offsets(agent_positions[:, ::-1])  # Scatter expects [x, y] format

    # Track the number of people remaining on the grid
    num_people_remaining = max_agents_total - len(exited_agents)
    print(f"Remaining agents: {num_people_remaining}")  # Debug print statement to observe remaining agents
    print(f"Agents processed: {agents_processed}")
    people_remaining_over_time.append(num_people_remaining)

    # Update the grid and floor field visuals
    img1.set_data(new_grid)
    img2.set_data(floor_field)
    img2.set_clim(vmin=0, vmax=np.percentile(floor_field, 95))

    # Stop the animation if all agents have exited
    if num_people_remaining == 0:
        plt.title("All people have exited. Close this window to exit.")
        ani.event_source.stop()  # Stop the animation

    # Sync the updated grid state back to `grid`
    grid[:] = new_grid[:]

    # Return updated visuals for FuncAnimation
    return img1, img2, agent_scatter


def update_floor_field(floor_field, decay_rate=0.001):
    floor_field *= (1 - decay_rate)  # delta
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
    global people_remaining_over_time, exited_agents, agents_processed
    exited_agents=set()
    people_remaining_over_time = []
    agents_processed=30
    grid = initialize_grid(rows, columns)
    floor_field = initialize_floor_field(rows, columns)
    floor_field *= floor_field_factor

    # Define a colormap: empty cells = white, people = red, obstacles = black
    cmap = colors.ListedColormap(['white', 'black'])  
    bounds = [-1, 0, 2]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap_floor = plt.cm.plasma

    # Set up the plot
    fig, ax = plt.subplots()
    img1 = ax.imshow(grid, interpolation='nearest', cmap=cmap, norm=norm)
    img2 = ax.imshow(floor_field, cmap=cmap_floor, alpha=0.6, vmin=0, vmax=1)

    # Initialize the scatter plot for agents (red dots)
    agent_positions = np.argwhere(grid == 1)  # Initial positions of agents
    agent_scatter = ax.scatter(agent_positions[:, 1], agent_positions[:, 0], c='red', s=60, marker='o')

    ax.invert_yaxis()

    # Run the animation
    global ani
    ani = animation.FuncAnimation(fig, update, fargs=(img1, img2, agent_scatter, grid, exit_location, floor_field, exit_influence, speed),
                                  frames=900, interval=80, save_count=50)
    plt.show()

    # After the animation stops, plot the remaining people over time
    plot_people_remaining()
    return np.sum(np.array(people_remaining_over_time) > 0)




def perform_grid_search():
    speed_range = np.arange(0.8, 1, 0.2)
    exit_influence_range = np.arange(1, 6, 11)
    floor_field_factor_range = np.arange(1, 11, 16)

    results = []

    actual_egress_time = 303  # frames from conversion seconds to frames

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


# Run the simulation
# run_egress_simulation()
perform_grid_search()
