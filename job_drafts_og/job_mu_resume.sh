#!/bin/bash
#SBATCH --job-name=job_mu
#SBATCH --output=./logs/job_mu_%j.out
#SBATCH --error=./logs/job_mu_%j.err
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
python main_qm9.py --exp_name icml_sin_mu \
                   --model egnn_dynamics \
                   --lr 2e-4 \
                   --nf 192 \
                   --n_layers 9 \
                   --save_model True \
                   --diffusion_steps 1000 \
                   --sin_embedding False \
                   --n_epochs 6000 \
                   --n_stability_samples 500 \
                   --diffusion_noise_schedule polynomial_2 \
                   --diffusion_noise_precision 1e-5 \
                   --dequantization deterministic \
                   --include_charges False \
                   --diffusion_loss_type l2 \
                   --batch_size 256 \
                   --conditioning mu \
                   --dataset qm9_second_half \
                   --classifier_free_guidance \
                   --resume pretrained/cEDM_mu \
                   --guidance_weight 0.25 \
                   --test_epochs 100 \
                   --class_drop_prob 0.1 \
                   --normalize_factors [1,8,1] \
