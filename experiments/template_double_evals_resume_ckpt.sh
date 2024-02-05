#!/bin/bash
#SBATCH --job-name=eval_PROPERTY1_PROPERTY2_MAIN_PROPERTY_resume_GUIDANCE_WEIGHT_
#SBATCH --output=./logs/eval_PROPERTY1_PROPERTY2_MAIN_PROPERTY_resume_GUIDANCE_WEIGHT_%j.out
#SBATCH --error=./logs/eval_PROPERTY1_PROPERTY2_MAIN_PROPERTY_resume_GUIDANCE_WEIGHT_%j.err
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

python eval_conditional_qm9.py --generators_path outputs/single_cfg_PROPERTY1_PROPERTY2_resume \
                               --classifiers_path pretrained/evaluate_MAIN_PROPERTY \
                               --property MAIN_PROPERTY  \
                               --iterations 10  \
                               --batch_size BATCH_SIZE \
                               --task edm \
                               --use_wandb \
                               --exp_name EXP_NAME \
                               --override_guidance GUIDANCE_WEIGHT \
                               --ckpt CKPT
