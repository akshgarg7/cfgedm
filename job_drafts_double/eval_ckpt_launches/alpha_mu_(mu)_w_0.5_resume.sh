#!/bin/bash
#SBATCH --job-name=eval_alpha_mu_mu_resume_0.5_
#SBATCH --output=./logs/eval_alpha_mu_mu_resume_0.5_%j.out
#SBATCH --error=./logs/eval_alpha_mu_mu_resume_0.5_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=16G
#SBATCH --partition=atlas
#SBATCH --account=atlas
#SBATCH --gres=gpu:1  # Adjust if you need GPU resources

mkdir -p ./logs

source /sailhome/akshgarg/.bashrc
cd /atlas/u/akshgarg/cfgedm/

# Activate the Conda environment
conda activate torch3.7

python eval_conditional_qm9.py --generators_path outputs/single_cfg_alpha_mu_resume \
                               --classifiers_path pretrained/evaluate_mu \
                               --property mu  \
                               --iterations 10  \
                               --batch_size 4 \
                               --task edm \
                               --use_wandb \
                               --exp_name full_scale_eval_alpha_mu_mu_w_0.5_ckpt_0 \
                               --override_guidance 0.5 \
                               --ckpt 0
