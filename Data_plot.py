import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=np.genfromtxt('data.txt', dtype=float)
user_ids=data[:,0]
user_ids1=list(set(user_ids))
print(len(user_ids1))
cmap = plt.get_cmap('tab10')
plt.figure(figsize=(10, 6))
for i in range(0,len(user_ids1),20):
    # print([record[1] for record in data if record[0] == user_ids1[i]])
    plt.plot([record[1] for record in data if record[0] == user_ids1[i]], [record[2] for record in data if record[0] == user_ids1[i]], label='X Position of %d' % user_ids1[i], marker='o', color=cmap(i//20))
    plt.plot([record[1] for record in data if record[0] == user_ids1[i]], [record[3] for record in data if record[0] == user_ids1[i]], label='Y Position of %d' % user_ids1[i], marker='x', color=cmap(i//20))
    # exit()
    # Adding labels and title
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('X and Y Position as a Function of Time for Random Users')
plt.legend()
plt.grid()

    # Show the plot for the current user

plt.show()


