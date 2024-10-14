import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib.animation import FuncAnimation

# Room dimensions
room_size = (100, 100)

# Number of people (points)
num_people = 20

# Door position (at the bottom center of the room)
door_position = np.array([room_size[0] // 2, 0])

# Random initial positions for people
people_positions = np.random.rand(num_people, 2) * room_size

# Add a small jitter to avoid degenerate configurations
def add_jitter(positions, jitter_amount=1e-3):
    return positions + np.random.randn(*positions.shape) * jitter_amount

# Compute Voronoi diagram for the initial positions of people
def generate_voronoi(positions):
    # Add jitter to avoid degenerate Voronoi issues
    jittered_positions = add_jitter(positions)
    return Voronoi(jittered_positions)

# Function to move people towards the door
def move_towards_door(people_positions, door_position, speed=1):
    new_positions = []
    for pos in people_positions:
        direction = door_position - pos
        distance = np.linalg.norm(direction)
        if distance > speed:  # Only move if not at the door
            direction = direction / distance  # Normalize direction
            new_pos = pos + direction * speed
        else:
            new_pos = door_position  # If close enough, snap to door
        new_positions.append(new_pos)
    return np.array(new_positions)

# Initialize plot
fig, ax = plt.subplots()
ax.set_xlim(0, room_size[0])
ax.set_ylim(0, room_size[1])
ax.set_aspect('equal')

# Plot the door as a red point
door_scatter = ax.scatter(door_position[0], door_position[1], color='red', s=100, label='Door')

# People plot (initial positions)
people_scatter = ax.scatter(people_positions[:, 0], people_positions[:, 1], color='blue', s=50, label='People')

# Initialize an empty list for storing the Voronoi polygons
voronoi_polygons = []

def draw_voronoi(vor):
    global voronoi_polygons
    # Remove previous polygons
    for poly in voronoi_polygons:
        poly.remove()  # Correctly remove each polygon
    voronoi_polygons = []
    
    # Plot new Voronoi regions
    for region in vor.regions:
        if not -1 in region and len(region) > 0:  # Filter out incomplete regions
            polygon = [vor.vertices[i] for i in region]
            voronoi_polygons.append(ax.fill(*zip(*polygon), alpha=0.2)[0])  # Store the actual Polygon object

# Update function for the animation
def update(frame):
    global people_positions
    people_positions = move_towards_door(people_positions, door_position, speed=0.5)  # Move people
    people_scatter.set_offsets(people_positions)  # Update positions
    
    # Update Voronoi diagram
    try:
        vor = generate_voronoi(people_positions)
        draw_voronoi(vor)
    except Exception as e:
        print(f"Voronoi failed at frame {frame}: {e}")
        # Skip this frame in case of failure to keep the animation running

    return people_scatter,

# Animation function
ani = FuncAnimation(fig, update, frames=200, interval=100)

plt.legend()
plt.show()
print("code finished ")