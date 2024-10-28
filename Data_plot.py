import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=np.genfromtxt('data.txt', dtype=float) #convert text file to numpy array
user_ids = data[:, 0]  # takes column of user
unique_user_ids = list(set(user_ids)) #count unique IDS
print(len(unique_user_ids))
cmap = plt.get_cmap('tab10') #adding more colors
cmap = plt.get_cmap('tab10')
colors = [cmap(i % 10) for i in range(len(unique_user_ids))]
for i in range(0,len(unique_user_ids),20): #takes every 20 user avoids having too many users
    user_data = data[data[:, 0] == unique_user_ids[i]]  # Filter rows for the current user
    time = user_data[:, 1]  # Time column
    x_positions = user_data[:, 2]  # X Position
    y_positions = user_data[:, 3]  # Y Position

    # Plot X and Y positions
    plt.plot(time, x_positions, label=f'X Position of {int(unique_user_ids[i])}', marker='o', color=colors[i//20])
    plt.plot(time, y_positions, label=f'Y Position of {int(unique_user_ids[i])}', marker='x', color=colors[i//20])
    # exit()
    # Adding labels and title
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('X and Y Position as a Function of Time for Random Users')
plt.legend()
plt.grid()

    # Show the plot for the current user

plt.show()


