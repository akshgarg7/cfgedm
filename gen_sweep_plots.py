import os 
import pandas as pd
import matplotlib.pyplot as plt

# properties = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv']
properties = ['homo']
for property in properties:
    base_folder = f'outputs/single_cfg_{property}_resume/analysis/guidance'
    sub_runs = os.listdir(base_folder)
    # match all runs which have a number in their name
    valid_runs = [run for run in sub_runs if run is not 'run']
    results_csv = os.path.join(base_folder, 'run/gen_properties.csv')
    df = pd.read_csv(results_csv)
    ans = df.iloc[0]
    true = ans['True']
    predicted = ans['Predicted']
    print(ans)
    print(true)
    print(predicted)

    for run in [valid_runs[0]]:
        run_folder = os.path.join(base_folder, run)
        # find all .png files in the run_folder 

        pngs = [file for file in os.listdir(run_folder) if '.png' in file]
        png_idx = [int(file[-7:-4]) for file in pngs]

        pngs_sorted_by_indx = [x[1] for x in sorted(zip(png_idx, pngs))]

        
        ncols = 3  # Number of columns in the plot grid
        nrows = (len(pngs_sorted_by_indx) + ncols - 1) // ncols  # Calculate rows needed
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        # Plot each PNG in the sorted list
        for idx, png in enumerate(pngs_sorted_by_indx):
            img_path = os.path.join(run_folder, png)  # Construct the full image path
            img = plt.imread(img_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')  # Hide axes
            axes[idx].set_title(f'Image {idx + 1}')  # Optional: Set a title for each subplot

        # Hide any unused axes if the number of PNGs is not a perfect multiple of ncols
        for ax in axes[len(pngs_sorted_by_indx):]:
            ax.axis('off')

        plt.tight_layout()

        # Save the figure
        # Define the save path (you can customize the filename as needed)
        save_path = os.path.join(run_folder, 'combined_images.png')
        plt.savefig(save_path)

        plt.show()  # Display the plot after saving
        print(f"Figure saved to {save_path}")