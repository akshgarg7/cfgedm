#!/bin/bash
JOBNAME="jobname"
#source your_virtual_env_dir
cd /sailhome/kellyyhe/atlasdemo/
GPUS=0
echo "Number of GPUs: ${GPUS}"
WRAP="python main.py"
LOG_FOLDER="./logs/"
echo ${WRAP}
echo "Log Folder:${LOG_FOLDER}"
mkdir -p ${LOG_FOLDER}
sbatch --output=${LOG_FOLDER}%j.out --error=${LOG_FOLDER}%j.err \
       --exclude=atlas1,atlas2,atlas3,atlas4,atlas5,atlas6 \
       --nodes=1 --ntasks-per-node=1 --time=1-00:00:00 --mem=32G \
       --partition=atlas --cpus-per-task=20 \
       --gres=gpu:titianxp:${GPUS} --job-name=${JOBNAME} --wrap="${WRAP}"
