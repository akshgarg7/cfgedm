#!/bin/bash
#SBATCH --job-name=sin_PROPERTY_PLACEHOLDER
#SBATCH --output=./logs/icml_sin_PROPERTY_PLACEHOLDER_%j.out
#SBATCH --error=./logs/icml_sin_PROPERTY_PLACEHOLDER%j.err
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --partition=atlas
#SBATCH --account=atlas
#SBATCH --gres=gpu:4  # Adjust if you need GPU resources

mkdir -p ./logs

source /sailhome/akshgarg/.bashrc
cd /atlas/u/akshgarg/cfgedm/

# Activate the Conda environment
conda activate torch3.7

# Your job's commands go here
python main_qm9.py --exp_name single_cfg_PROPERTY_PLACEHOLDER \
                   --model MODEL_PLACEHOLDER \
                   --lr 2e-4 \
                   --nf 192 \
                   --n_layers 9 \
                   --save_model True \
                   --diffusion_steps 1000 \
                   --sin_embedding False \
                   --n_epochs EPOCHS_PLACEHOLDER \
                   --n_stability_samples STABILITY_SAMPLES_PLACEHOLDER \
                   --diffusion_noise_schedule polynomial_2 \
                   --diffusion_noise_precision 1e-5 \
                   --dequantization deterministic \
                   --include_charges False \
                   --diffusion_loss_type l2 \
                   --batch_size BATCH_SIZE_PLACEHOLDER \
                   --conditioning PROPERTY_PLACEHOLDER \
                   --dataset qm9_second_half \
                   --classifier_free_guidance \
                   --guidance_weight GUIDANCE_WEIGHT_PLACEHOLDER \
                   --test_epochs EPOCH_REPLACE_PLACEHOLDER \
                   --class_drop_prob DROP_PROB_PLACEHOLDER \
                   --normalize_factors [1,8,1] \
