import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, position, speed, goal):
        self.position = np.array(position)
        self.speed = speed
        self.goal = np.array(goal)

    def move(self, agents, time_step):
        # Compute the direction towards the goal
        direction = self.goal - self.position
        direction = direction / np.linalg.norm(direction)

        # Move towards the goal
        self.position += self.speed * direction * time_step

        # Avoid collisions with other agents
        for other in agents:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if distance < 0.5:  # Minimum distance to avoid collision
                    # Move away from the other agent
                    avoid_direction = self.position - other.position
                    avoid_direction = avoid_direction / np.linalg.norm(avoid_direction)
                    self.position += avoid_direction * self.speed * time_step * 0.5  # Adjust speed away from other agents

    def has_exited(self):
        # Check if the agent has reached the exit (goal)
        return np.linalg.norm(self.position - self.goal) < 0.5


class Room:
    def __init__(self, width, height, exit_position):
        self.width = width
        self.height = height
        self.exit_position = np.array(exit_position)
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def simulate(self, time_step=0.1, max_time=60):
        times_to_exit = []
        for t in np.arange(0, max_time, time_step):
            for agent in self.agents[:]:  # Iterate over a copy of the agents list
                agent.move(self.agents, time_step)
                if agent.has_exited():
                    times_to_exit.append(t)
                    self.agents.remove(agent)  # Remove agent once it has exited

            # Visualization for debugging or demonstration
            self.plot_room(t)

            # If all agents have exited, end simulation
            if len(self.agents) == 0:
                break

        return times_to_exit

    def plot_room(self, current_time):
        plt.figure(figsize=(8, 6))
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.scatter(*self.exit_position, color='red', s=100, label='Exit')
        for agent in self.agents:
            plt.scatter(*agent.position, color='blue', s=50)
        plt.title(f'Egress Simulation at Time = {current_time:.1f} seconds')
        plt.xlabel('Room Width')
        plt.ylabel('Room Height')
        plt.legend()
        plt.show()


# Setting up the simulation
room_width = 10
room_height = 10
exit_position = [room_width - 1, room_height / 2]

# Create the room
room = Room(room_width, room_height, exit_position)

# Add agents with random starting positions and a fixed speed
num_agents = 20
for _ in range(num_agents):
    start_position = [np.random.uniform(1, room_width - 2), np.random.uniform(1, room_height - 2)]
    agent_speed = np.random.uniform(1, 1.5)  # Random speed between 1 and 1.5 units per second
    agent = Agent(start_position, agent_speed, exit_position)
    room.add_agent(agent)

# Run the simulation and collect egress times
egress_times = room.simulate(time_step=0.1, max_time=60)

# Analyze the results
average_egress_time = np.mean(egress_times)
print(f'Average Egress Time: {average_egress_time:.2f} seconds')
