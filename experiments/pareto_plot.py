import pandas as pd
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('experiments/pareto.csv')
print(df)
df = df.drop_duplicates(subset=['Guidance weight'], keep='last')
print(df)

cols = ['Guidance weight','Molecular Stability', 'MAE']

df.sort_values(by=['Guidance weight'], inplace=True)
print(df)
# Plot the data
plt.scatter(df['MAE'], df['Molecular Stability'])
plt.xlabel('MAE')
plt.ylabel('Molecular Stability')
plt.title('MAE vs Molecular Stability')

# Add the labels
for i, txt in enumerate(df['Guidance weight']):
    plt.annotate(txt, (df['MAE'].iloc[i], df['Molecular Stability'].iloc[i]))

# Save the plot
plt.savefig('pareto.png')