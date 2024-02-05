import pandas as pd
import matplotlib.pyplot as plt

# Properties to plot
properties = ['alpha', 'mu', 'homo', 'lumo', 'gap', 'Cv']

# Create a 3x2 subplot figure for the combined plot
fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Adjust the figure size as needed
axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier indexing

for i, property in enumerate(properties):
    # Read the data
    df = pd.read_csv(f'experiments/pareto/{property}.csv')
    df = df.drop_duplicates(subset=['guidance_weight'], keep='last')
    df.sort_values(by=['guidance_weight'], inplace=True)
    
    # Plotting individual plots
    plt.figure()  # Create a new figure for the individual plot
    plt.scatter(df['mae'], df['mol_stability'])
    plt.xlabel('mae')
    plt.ylabel('mol_stability')
    plt.title(f'{property}: mae vs mol_stability')
    for j, txt in enumerate(df['guidance_weight']):
        plt.annotate(txt, (df['mae'].iloc[j], df['mol_stability'].iloc[j]))
    plt.savefig(f'{property}_pareto.png')  # Save the individual plot
    plt.close()  # Close the current figure to free memory
    
    # Plotting in the combined subplot
    ax = axs[i]
    ax.scatter(df['mae'], df['mol_stability'])
    ax.set_xlabel('mae')
    ax.set_ylabel('mol_stability')
    ax.set_title(f'{property}: mae vs mol_stability')
    for j, txt in enumerate(df['guidance_weight']):
        ax.annotate(txt, (df['mae'].iloc[j], df['mol_stability'].iloc[j]))

# Adjust layout to prevent overlapping in the combined subplot
plt.tight_layout()

# Save the combined subplot
plt.savefig('combined_pareto_plots.png')
