import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Import CSV files and combine them
file_1 = 'results.csv'
file_2 = 'results2.csv'
file_3 = 'results3.csv'
file_4 = 'results4.csv'
file_5 = 'results5.csv'

df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)
df3 = pd.read_csv(file_3)
df4 = pd.read_csv(file_4)
df5 = pd.read_csv(file_5)

combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# Step 2: Compute the average error for each combination of parameters
average_errors = (
    combined_df.groupby(['speed', 'exit_influence', 'floor_field_factor'])
    .agg({'error': 'mean'})
    .reset_index()
)

# Step 3: Ensure unique combinations for pivoting


def plot_confusion_matrix(data, x, y, value, title, cmap='coolwarm'):
    """
    Plots a heatmap for the specified x, y, and value columns.
    """
    # Aggregate the mean error to ensure unique combinations
    aggregated_data = (
        data.groupby([x, y])
        .agg({value: 'mean'})
        .reset_index()
    )

    # Pivot table for heatmap
    pivot_table = aggregated_data.pivot(index=y, columns=x, values=value)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap=cmap,
                cbar_kws={'label': 'Average Error'})
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


# Step 4: Plot confusion matrices for each parameter pair
plot_confusion_matrix(
    average_errors,
    x='speed',
    y='exit_influence',
    value='error',
    title='Average Error by Speed and Exit Influence'
)

plot_confusion_matrix(
    average_errors,
    x='speed',
    y='floor_field_factor',
    value='error',
    title='Average Error by Speed and Floor Field Factor'
)

plot_confusion_matrix(
    average_errors,
    x='exit_influence',
    y='floor_field_factor',
    value='error',
    title='Average Error by Exit Influence and Floor Field Factor'
)
