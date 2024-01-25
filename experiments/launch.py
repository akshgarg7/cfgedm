import subprocess
import argparse

# Command to get GPU information
command = "nvidia-smi --query-gpu=name,memory.total --format=csv"
# Variables
MODEL = "egnn_dynamics"
DROP_PROB = "0.1"
DATASET_PORTION = "0.5"
EPOCHS = "3000"
STABILITY_SAMPLES = "500"
GUIDANCE_WEIGHT = "0.25"  # Replace 'some_value' with the actual value
TEST_EPOCHS = "100"  # Define TEST_EPOCHS
BATCH_SIZE = None

parser = argparse.ArgumentParser(description='Launcher')
parser.add_argument('--resume', action='store_true', default=False)
args = parser.parse_args()

# Run the command and capture the output
try:
    result = subprocess.check_output(command, shell=True, text=True)
    # The result is a string containing the GPU information
    print("GPU Information:")
    print(result)

    # You can process the result string further as per your requirements
    # For example, splitting it into lines and parsing each line
    gpu_info = [line.split(', ') for line in result.strip().split('\n')[1:]]
    print("Processed GPU Information:")
    print(gpu_info)

    total_memory = 0
    for gpu in gpu_info:
        total_memory += int(gpu[1].split()[0])

    scale = 16376 * 4
    batch_size_at_16GB = 256
    batch_size = int(batch_size_at_16GB * total_memory / scale)
    batch_size = batch_size - batch_size % 32
    BATCH_SIZE = str(batch_size)

except subprocess.CalledProcessError as e:
    print("An error occurred while trying to retrieve GPU information.")
    print(e.output)

import os

def create_job_file(property):
    job_name = f"job_{property}"
    input_file = 'experiments/template_single.sh'
    output_file = f'job_drafts/job_{property}.sh'
    if args.resume:
        input_file = 'experiments/template_single_resume.sh'
        output_file = f'job_drafts/job_{property}_resume.sh'

    with open(input_file, 'r') as file:
        content = file.read()
    
    content = content.replace('JOB_NAME_PLACEHOLDER', job_name)
    content = content.replace('MODEL_PLACEHOLDER', MODEL)
    content = content.replace('EPOCHS_PLACEHOLDER', EPOCHS)
    content = content.replace('STABILITY_SAMPLES_PLACEHOLDER', STABILITY_SAMPLES)
    content = content.replace('BATCH_SIZE_PLACEHOLDER', BATCH_SIZE)
    content = content.replace('GUIDANCE_WEIGHT_PLACEHOLDER', GUIDANCE_WEIGHT)
    content = content.replace('DROP_PROB_PLACEHOLDER', DROP_PROB)
    content = content.replace('PROPERTY_PLACEHOLDER', property)
    content = content.replace('EPOCH_REPLACE_PLACEHOLDER', TEST_EPOCHS)

    with open(output_file, 'w') as file:
        file.write(content)

    # Optional: Submit the job and/or remove the file
    # os.system(f'sbatch experiments/temp_job_{property}.sh')
    # os.remove(f'experiments/temp_job_{property}.sh')

properties = ['alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv']

for property in properties:
    create_job_file(property)
