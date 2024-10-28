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
    jittered_positions = add_jitter(positions)
    return Voronoi(jittered_positions)

# Initialize random speeds for each person
people_speeds = np.random.uniform(0.5, 1.5, size=num_people)

# Initialize egress times to infinity for people still in the room
egress_times = np.full(num_people, np.inf)

# Total egress time variable
total_egress_time = None

# Animation interval (100 ms = 0.1 seconds per frame)
frame_interval_ms = 100
frame_interval_s = frame_interval_ms / 1000

# Function to move people towards the door
def move_towards_door(people_positions, door_position, people_speeds, frame):
    new_positions = []
    for i, pos in enumerate(people_positions):
        if np.isfinite(egress_times[i]):
            new_positions.append(pos)
            continue

        direction = door_position - pos
        distance = np.linalg.norm(direction)

        speed = people_speeds[i]
        
        if distance < 10:
            nearby_people = np.linalg.norm(people_positions - door_position, axis=1) < 15
            congestion_factor = 1.0 / np.sum(nearby_people)
            speed = max(0.3, speed * congestion_factor)
        
        randomness = np.random.uniform(-0.1, 0.1, size=2)
        if distance > speed:
            direction = direction / distance
            new_pos = pos + (direction + randomness) * speed
        else:
            new_pos = door_position
            egress_times[i] = frame
        new_positions.append(new_pos)

    return np.array(new_positions)

# Initialize plot
fig, ax = plt.subplots()
ax.set_xlim(0, room_size[0])
ax.set_ylim(0, room_size[1])
ax.set_aspect('equal')

# Set background color for better contrast
ax.set_facecolor('lightgray')  # Change background color to light gray

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
        poly.remove()
    voronoi_polygons = []
    
    # Plot new Voronoi regions with filled colors only
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            color = np.random.rand(3,)  # Random color for each polygon
            poly = ax.fill(*zip(*polygon), color=color, alpha=0.5)  # Fill with transparency
            voronoi_polygons.append(poly[0])  # Store the actual Polygon object

# Update function for the animation
def update(frame):
    global people_positions, total_egress_time

    people_positions = move_towards_door(people_positions, door_position, people_speeds, frame)
    people_scatter.set_offsets(people_positions)
    
    # Update Voronoi diagram
    try:
        vor = generate_voronoi(people_positions)
        draw_voronoi(vor)
    except Exception as e:
        print(f"Voronoi failed at frame {frame}: {e}")

    # Check if all people have exited the room
    if np.all(np.isfinite(egress_times)) and total_egress_time is None:
        total_egress_time = frame

    # Stop the animation when all people have exited
    if total_egress_time is not None:
        total_time_seconds = total_egress_time * frame_interval_s
        total_time_minutes = total_time_seconds / 60
        print(f"Total egress time: {total_egress_time} frames ({total_time_seconds:.2f} seconds or {total_time_minutes:.2f} minutes)")
        ani.event_source.stop()

    return people_scatter,

# Animation function
ani = FuncAnimation(fig, update, frames=200, interval=frame_interval_ms)

plt.legend()
plt.show()

# Once the animation ends, display the total egress time
if total_egress_time is not None:
    total_time_seconds = total_egress_time * frame_interval_s
    total_time_minutes = total_time_seconds / 60
    print(f"All people exited in {total_egress_time} frames, which is {total_time_seconds:.2f} seconds or {total_time_minutes:.2f} minutes.")
else:
    print("Some people did not exit within the animation frames.")

print("code finished ")