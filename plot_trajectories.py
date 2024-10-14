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
    specified number of people from from dataset.
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
    # sns.kdeplot(x=df['x'], y=df['y'], cmap='Reds', fill=True, bw_adjust=0.5)
    plt.hist2d(df['x'], df['y'], bins=200, cmap='Reds') #Quicker way to execute then seaborn takes a second compared to a few minutes 
    plt.colorbar(label='Density')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Density Heatmap of Movement')
    plt.show()
    # plt.savefig('heatmap.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot user movement data.")
    parser.add_argument('--plot', choices=['trajectories', 'heatmap'], required=True,
                        help="Specify which plot to create: 'trajectories' or 'heatmap'")
    parser.add_argument('--num_users', type=int, default=25,
                        help="Number of users to plot trajectories for (default: 25)")

    args = parser.parse_args()

    if args.plot == 'trajectories':
        plot_trajectories(args.num_users)
    elif args.plot == 'heatmap':
        plot_density_heatmap()
