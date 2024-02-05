import pandas as pd
import matplotlib.pyplot as plt

# Read the data
for property in ['alpha', 'mu', 'homo', 'lumo', 'gap', 'Cv']:
    df = pd.read_csv(f'experiments/pareto/{property}.csv')
    print(df)
    df = df.drop_duplicates(subset=['guidance_weight'], keep='last')
    print(df)

    cols = ['guidance_weight','mol_stability', 'mae']

    df.sort_values(by=['guidance_weight'], inplace=True)
    print(df)
    # Plot the data
    plt.scatter(df['mae'], df['mol_stability'])
    plt.xlabel('mae')
    plt.ylabel('mol_stability')
    plt.title('mae vs mol_stability')

    # Add the labels
    for i, txt in enumerate(df['guidance_weight']):
        plt.annotate(txt, (df['mae'].iloc[i], df['mol_stability'].iloc[i]))

    # Save the plot
    plt.savefig(f'{property}_pareto.png')