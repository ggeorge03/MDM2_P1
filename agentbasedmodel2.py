import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Agent:
    def __init__(self, position, speed, goal, behavior='normal'):
        self.position = np.array(position)
        self.speed = speed
        self.original_speed = speed
        self.goal = np.array(goal)
        self.behavior = behavior  # 'normal', 'disoriented', 'hysterical', 'injured'
        self.acceleration = 0.0
        self.angle = np.random.uniform(0, 2 * np.pi)  # Random initial direction for disoriented movement

    def move(self, agents, time_step, room):
        # Behavior-specific adjustments
        if self.behavior == 'disoriented':
            self.angle += np.random.uniform(-0.5, 0.5)  # Randomly change angle
            direction = np.array([np.cos(self.angle), np.sin(self.angle)])
            self.speed = max(0.5, self.original_speed * 0.8)  # Reduce speed due to disorientation
        elif self.behavior == 'hysterical':
            self.acceleration = np.random.uniform(0.1, 0.3)  # Add acceleration
            self.speed += self.acceleration * time_step  # Increase speed over time
            direction = self.goal - self.position
        elif self.behavior == 'injured':
            self.speed = max(0.3, self.original_speed * 0.5)  # Slow down significantly
            direction = self.goal - self.position
        else:
            # Normal behavior
            direction = self.goal - self.position

        # Normalize direction
        direction = direction / np.linalg.norm(direction)

        # Move towards the goal or in the adjusted direction
        self.position += self.speed * direction * time_step

        # Avoid collisions with other agents
        for other in agents:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if distance < 0.5:  # Minimum distance to avoid collision
                    avoid_direction = self.position - other.position
                    avoid_direction = avoid_direction / np.linalg.norm(avoid_direction)
                    self.position += avoid_direction * self.speed * time_step * 0.5

        # Check for accidents (randomly trip an agent with some probability)
        if np.random.rand() < 0.01 and self.behavior != 'injured':
            self.behavior = 'injured'
            self.speed *= 0.5  # Reduce speed significantly if injured

    def has_exited(self):
        # Check if the agent has reached the exit
        return np.linalg.norm(self.position - self.goal) < 0.5


class Room:
    def __init__(self, width, height, exit_position):
        self.width = width
        self.height = height
        self.exit_position = np.array(exit_position)
        self.agents = []
        self.obstacles = []  # Environmental factors such as obstacles
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_obstacle(self, position, radius):
        self.obstacles.append({'position': np.array(position), 'radius': radius})

    def simulate(self, time_step=0.1, max_time=60):
        times_to_exit = []
        ani = FuncAnimation(self.fig, self.update_plot, frames=np.arange(0, max_time, time_step),
                            fargs=(time_step, times_to_exit), repeat=False)
        plt.show()
        return times_to_exit

    def update_plot(self, current_time, time_step, times_to_exit):
        for agent in self.agents[:]:  # Iterate over a copy of the agents list
            agent.move(self.agents, time_step, self)
            if agent.has_exited():
                times_to_exit.append(current_time)
                self.agents.remove(agent)

        self.ax.clear()  # Clear the previous plot
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.scatter(*self.exit_position, color='red', s=100, label='Exit')
        agent_positions = np.array([agent.position for agent in self.agents])
        if len(agent_positions) > 0:
            self.ax.scatter(agent_positions[:, 0], agent_positions[:, 1], color='blue', s=50)

        # Plot obstacles
        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle['position'], obstacle['radius'], color='grey', alpha=0.5)
            self.ax.add_patch(circle)

        self.ax.set_title(f'Egress Simulation at Time = {current_time:.1f} seconds')
        self.ax.set_xlabel('Room Width')
        self.ax.set_ylabel('Room Height')
        self.ax.legend()


# Setting up the simulation
room_width = 10
room_height = 10
exit_position = [room_width - 1, room_height / 2]

# Create the room
room = Room(room_width, room_height, exit_position)

# Add some obstacles
room.add_obstacle([5, 5], 1)  # Add an obstacle at the center

# Add agents with different behaviors
num_agents = 20
behaviors = ['normal', 'disoriented', 'hysterical', 'injured']
for _ in range(num_agents):
    start_position = [np.random.uniform(1, room_width - 2), np.random.uniform(1, room_height - 2)]
    agent_speed = np.random.uniform(1, 1.5)
    behavior = np.random.choice(behaviors, p=[0.7, 0.1, 0.1, 0.1])  # Higher probability for 'normal' behavior
    agent = Agent(start_position, agent_speed, exit_position, behavior=behavior)
    room.add_agent(agent)

# Run the simulation and collect egress times
egress_times = room.simulate(time_step=0.1, max_time=60)

# Analyze the results
average_egress_time = np.mean(egress_times)
print(f'Average Egress Time: {average_egress_time:.2f} seconds')


