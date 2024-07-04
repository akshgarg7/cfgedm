# import os 
# import pandas as pd
# import matplotlib.pyplot as plt

# def gen_prop_sweep_figs(exp_name, epoch, context, frames, property):
#     base_folder = f"outputs/prop_sweep/{exp_name}/run{epoch}"
#     pngs = [file for file in os.listdir(base_folder) if '.png' in file]
#     png_idx = [int(file[-7:-4]) for file in pngs]
#     pngs_sorted_by_indx = [x[1] for x in sorted(zip(png_idx, pngs))]

#     filtered_pngs = []
#     for i, img in enumerate(pngs_sorted_by_indx):
#         if i in frames:
#             filtered_pngs.append(img)


#     # Filter the PNG files based on the frames
#     # filtered_pngs = [pngs_sorted_by_indx[i] for i in frames]
#     # filtered_pngs = [pngs_sorted_by_indx[i] for i in frames]

#     # Plot the filtered PNG files
#     ncols = 5  # Adjust based on how many images per row you want
#     nrows = (len(filtered_pngs) + ncols - 1) // ncols
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, nrows * (5+2)))
#     axes = axes.flatten()

#     for idx, png in enumerate(filtered_pngs):
#         img_path = os.path.join(base_folder, png)
#         img = plt.imread(img_path)
#         axes[idx].imshow(img)
#         axes[idx].axis('off')  # Hide axes
#         title = f'{context[frames[idx]]:.2f}'
#         axes[idx].set_title(title, fontsize=60) 

#     # Hide unused subplots if any
#     for ax in axes[len(filtered_pngs):]:
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()

#     save_path = f"final_figures2/{property}_{epoch}_frame_select.png"
#     plt.savefig(save_path)

#     plt.show()  # Display the plot after saving
#     print(f"Figure saved to {save_path}")


import os
import pandas as pd
import matplotlib.pyplot as plt

def gen_prop_sweep_figs(exp_name, epoch, context, frames, property):
    base_folder = f"outputs/prop_sweep/{exp_name}/run{epoch}"
    pngs = [file for file in os.listdir(base_folder) if '.png' in file]
    png_idx = [int(file[-7:-4]) for file in pngs]
    pngs_sorted_by_indx = [x[1] for x in sorted(zip(png_idx, pngs))]
    
    filtered_pngs = [pngs_sorted_by_indx[i] for i in frames]

    for idx, png in enumerate(filtered_pngs):
        img_path = os.path.join(base_folder, png)
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.axis('off')  # Hide axes
        title = f'{context[frames[idx]]:.2f}'
        plt.title(title, fontsize=40)
        plt.tight_layout()
        
        # Save each figure with the title as the filename
        save_path = f"final_figures2/{property}_{epoch}_frame_{frames[idx]}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the plot to free memory
        
        print(f"Image {title} saved to {save_path}")

property = 'alpha'
exp_name = f'single_cfg_{property}_resume'
base_folder = f"outputs/prop_sweep/{exp_name}/"
context_csv = os.path.join(base_folder, 'context.csv')
context = pd.read_csv(context_csv, header=None).values.flatten()
epoch = 3
# frames = [0,1,2,3,4]
frames = [0,2,4,6,8]
gen_prop_sweep_figs(exp_name, epoch, context, frames, property)


