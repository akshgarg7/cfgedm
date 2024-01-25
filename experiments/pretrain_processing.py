import os
import shutil
import glob

# Path to the main directory
main_dir = '/atlas/u/akshgarg/cfgedm/pretrained'

# Iterate over each subdirectory in the main directory
for subdir in next(os.walk(main_dir))[1]:
    subdir_path = os.path.join(main_dir, subdir)

    # Find the args and generative_model_ema files
    args_files = glob.glob(os.path.join(subdir_path, 'args_*.pickle'))
    generative_model_files = glob.glob(os.path.join(subdir_path, 'generative_model_ema_*.npy'))

    # Copy the files if found
    if args_files:
        shutil.copy(args_files[0], os.path.join(subdir_path, 'args.pickle'))

    if generative_model_files:
        shutil.copy(generative_model_files[0], os.path.join(subdir_path, 'generative_model_ema.npy'))

print("Files copied successfully.")
