import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Read in the data.
df = pd.read_csv('data.txt', sep=' ', header=None,
                 names=['userid', 'timestamp', 'x', 'y'])


def plot_trajectories(num_users=25):  # Specify amount of trajectories to plot.
    '''
    Function that plots the x-y trajectories of a
    specified number of people from the dataset.
    '''
    user_ids = df['userid'].unique()[:num_users]
    for user_id in user_ids:
        user_data = df[df['userid'] == user_id]
        plt.scatter(user_data['x'], user_data['y'],
                    label=f'User {user_id}', s=10)

    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title(f'Trajectories of {num_users} Users')
    plt.legend()
    plt.show()
    # plt.savefig('trajectories.png')


def plot_density_heatmap():
    '''
    Function that plots a heatmap showing the density
    of footfall in an area.
    '''
    plt.figure(figsize=(8, 6))
    plt.hist2d(df['x'], df['y'], bins=200, cmap='Reds')  # Faster execution than seaborn's kdeplot
    plt.colorbar(label='Density')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Density Heatmap of Movement')
    plt.show()
    # plt.savefig('heatmap.png')


def plot_users_left_over_time():
    '''
    Function that plots the cumulative number of users
    who have left the room at each timestamp.
    '''
    # Find the last timestamp for each user
    last_timestamps = df.groupby('userid')['timestamp'].max()

    # Get unique timestamps and sort them
    unique_timestamps = sorted(df['timestamp'].unique())

    # Initialize list to store the cumulative count of users left at each timestamp
    left_counts = []

    # Track cumulative number of people who have left
    cumulative_left_count = 0

    # Iterate through each unique timestamp in ascending order
    for timestamp in unique_timestamps:
        # Count users whose last recorded timestamp matches the current timestamp
        num_leaving_now = (last_timestamps == timestamp).sum()

        # Update the cumulative count with users leaving at this timestamp
        cumulative_left_count += num_leaving_now
        left_counts.append(cumulative_left_count)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(unique_timestamps, left_counts, color='red', lw=2)
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Users Who Left')
    plt.title('Cumulative Number of Users Who Left Over Time')
    plt.grid()
    plt.show()
    # plt.savefig('users_left_over_time.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot user movement data.")
    parser.add_argument('--plot', choices=['trajectories', 'heatmap', 'left'], default='trajectories',
                        help="Specify which plot to create: 'trajectories', 'heatmap', or 'left' (default: 'trajectories')")
    parser.add_argument('--num_users', type=int, default=25,
                        help="Number of users to plot trajectories for (default: 25)")

    args = parser.parse_args()

    if args.plot == 'trajectories':
        plot_trajectories(args.num_users)
    elif args.plot == 'heatmap':
        plot_density_heatmap()
    elif args.plot == 'left':
        plot_users_left_over_time()






