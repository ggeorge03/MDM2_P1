import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Read in the data.
df = pd.read_csv('data.txt', sep=' ', header=None,
                 names=['userid', 'timestamp', 'x', 'y'])

# Check the total unique users in the dataset
unique_user_ids = df['userid'].nunique()
print(f"Total unique users in dataset: {unique_user_ids}")


def plot_trajectories(num_users=25):
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
    of footfall in an area, flipped vertically.
    '''
    plt.figure(figsize=(8, 6))
    plt.hist2d(df['x'], -df['y'], bins=200, cmap='Reds')  # Flip y-axis by using -df['y']
    plt.colorbar(label='Density')
    plt.xlabel('X position')
    plt.ylabel('Y position (flipped)')
    plt.title('Density Heatmap of Movement (Flipped Vertically)')
    plt.show()
    # plt.savefig('heatmap_flipped.png')


def plot_users_remaining_over_time():
    '''
    Function that plots the number of users remaining
    in the room at each timestamp, starting from timestamp = 0.
    '''
    # Find the last timestamp for each user (exit time)
    last_timestamps = df.groupby('userid')['timestamp'].max()

    # Define the range of timestamps, starting from 0
    unique_timestamps = sorted(df['timestamp'].unique())
    min_timestamp = 0
    max_timestamp = unique_timestamps[-1]
    all_timestamps = range(min_timestamp, max_timestamp + 1)

    # Total number of unique users in the dataset
    total_users = last_timestamps.size  # This counts unique users correctly

    # Print to check total users
    print(f"Total users at the beginning: {total_users}")

    # Initialize a list to store the count of users left in the room at each timestamp
    users_remaining = []

    # Track cumulative count of users who have left
    cumulative_left_count = 0

    # Calculate the number of people left in the room at each timestamp
    for timestamp in all_timestamps:
        # Count users whose last timestamp (exit time) is at or before the current timestamp
        num_leaving_now = (last_timestamps <= timestamp).sum()

        # Print debug information
        print(f"Timestamp: {timestamp}, Users leaving: {num_leaving_now}, Cumulative left: {cumulative_left_count}")

        # Update the cumulative count of users who have left
        cumulative_left_count = num_leaving_now  # Update to count total leaving up to this timestamp

        # Calculate the number of users remaining in the room
        users_in_room = total_users - cumulative_left_count
        users_remaining.append(users_in_room)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(all_timestamps, users_remaining, color='blue', lw=2)
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Users Remaining')
    plt.title('Number of Users Remaining in Room Over Time (Starting at Timestamp 0)')
    plt.grid()
    plt.show()
    # plt.savefig('users_remaining_over_time.png')


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
        plot_users_remaining_over_time()







