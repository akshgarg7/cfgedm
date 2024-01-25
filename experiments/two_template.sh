#!/bin/bash
#SBATCH --job-name=JOB_NAME_PLACEHOLDER
#SBATCH --output=./logs/JOB_NAME_PLACEHOLDER_%j.out
#SBATCH --error=./logs/JOB_NAME_PLACEHOLDER_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=atlas
#SBATCH --account=atlas
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1  # Adjust if you need GPU resources

mkdir -p ./logs

cd /atlas/u/akshgarg/cfgdm/GeoLDM

# Activate the Conda environment
conda activate torch1

# Your job's commands go here
echo python main_qm9.py --exp_name single_cfg_PROPERTY1_PLACEHOLDER_PROPERTY2_PLACEHOLDER \
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
                   --conditioning PROPERTY1_PLACEHOLDER PROPERTY2_PLACEHOLDER \
                   --dataset qm9_second_half \
                   --train_diffusion \
                   --trainable_ae \
                   --latent_nf 1 \
                   --classifier_free_guidance \
                   --guidance_weight GUIDANCE_WEIGHT_PLACEHOLDER \
                   --test_epochs 20 \
                   --class_drop_prob DROP_PROB_PLACEHOLDER \
                   --dataset_portion DATASET_PORTION_PLACEHOLDER \
                   --normalize_factors [1,8,1] \
