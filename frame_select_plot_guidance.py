import os 
import pandas as pd
import matplotlib.pyplot as plt
from lds import longest_decreasing_subsequence_ids

# def generate_figs(exp_name, epoch, frames, property):
#     base_folder = f"outputs/analysis/{exp_name}/guidance/run{epoch}"
#     csv_path = f"{base_folder}/gen_properties.csv"
#     df = pd.read_csv(csv_path)

#     maes = df['mae'].to_list()
#     _, ids_with_decreasing_mae, _ = longest_decreasing_subsequence_ids(maes)
#     ws = df['w'].to_list()

#     pngs = [file for file in os.listdir(base_folder) if '.png' in file]
#     png_idx = [int(file[-7:-4]) for file in pngs]
#     pngs_sorted_by_indx = [x[1] for x in sorted(zip(png_idx, pngs))]

#     mae_approved = [png for i, png in enumerate(pngs_sorted_by_indx) if i in ids_with_decreasing_mae]
#     ws = [ws[i] for i in range(len(ws)) if i in ids_with_decreasing_mae]
#     maes = [maes[i] for i in range(len(maes)) if i in ids_with_decreasing_mae]


#     new_ws = []
#     new_maes = []
#     filtered_pngs = []
#     for i, img in enumerate(mae_approved):
#         if i in frames:
#             filtered_pngs.append(img)
#             new_ws.append(ws[i])
#             new_maes.append(maes[i])

#     ws = new_ws
#     maes = new_maes
    
#     print(filtered_pngs)

#     # Plot the filtered PNG files
#     ncols = 5  # Adjust based on how many images per row you want
#     nrows = (len(filtered_pngs) + ncols - 1) // ncols
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, nrows * (5+1)))
#     axes = axes.flatten()

#     for idx, png in enumerate(filtered_pngs):
#         img_path = os.path.join(base_folder, png)
#         img = plt.imread(img_path)
#         axes[idx].imshow(img)
#         axes[idx].axis('off')  # Hide axes
#         title = f'$w$={ws[idx]}, MAE={maes[idx]:.2f}'
#         axes[idx].set_title(title, fontsize=26) 

#     # Hide unused subplots if any
#     for ax in axes[len(filtered_pngs):]:
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()

#     save_path = f"final_figures2/guidance/{property}_{epoch}_frame_select.png"
#     plt.savefig(save_path)

#     plt.show()  # Display the plot after saving
#     print(f"Figure saved to {save_path}")

import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_figs(exp_name, epoch, frames, property):
    base_folder = f"outputs/analysis/{exp_name}/guidance/run{epoch}"
    csv_path = f"{base_folder}/gen_properties.csv"
    df = pd.read_csv(csv_path)

    # Assuming longest_decreasing_subsequence_ids is a function defined elsewhere
    maes = df['mae'].to_list()
    _, ids_with_decreasing_mae, _ = longest_decreasing_subsequence_ids(maes)
    ws = df['w'].to_list()

    pngs = [file for file in os.listdir(base_folder) if '.png' in file]
    png_idx = [int(file[-7:-4]) for file in pngs]
    pngs_sorted_by_indx = [x for _, x in sorted(zip(png_idx, pngs))]

    mae_approved = [png for i, png in enumerate(pngs_sorted_by_indx) if i in ids_with_decreasing_mae]
    ws = [ws[i] for i in ids_with_decreasing_mae]
    maes = [maes[i] for i in ids_with_decreasing_mae]

    new_ws = []
    new_maes = []
    filtered_pngs = []
    for i, img in enumerate(mae_approved):
        if i in frames:
            filtered_pngs.append(img)
            new_ws.append(ws[i])
            new_maes.append(maes[i])

    # Save individual images
    for idx, png in enumerate(filtered_pngs):
        img_path = os.path.join(base_folder, png)
        img = plt.imread(img_path)
        plt.figure(figsize=(5, 5))  # Adjust the figure size as needed
        plt.imshow(img)
        plt.axis('off')  # Hide axes
        title = f'$w$={new_ws[idx]}, MAE={new_maes[idx]:.2f}'
        plt.title(title, fontsize=30)
        save_path = f"final_figures2/guidance/{property}_{epoch}_frame_{frames[idx]}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the plot to free memory
        print(f"Image saved to {save_path}")

# You would call your function here with the correct parameters as before


property = 'alpha'
exp_name = f'single_cfg_{property}_resume'
base_folder = f"outputs/prop_sweep/{exp_name}/"
epoch = 0
frames = [0,1,2,3,4]
# frames = [0,2,4,6,8]
generate_figs(exp_name, epoch, frames, property)