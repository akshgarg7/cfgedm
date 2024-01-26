import subprocess
import argparse

# Command to get GPU information
command = "nvidia-smi --query-gpu=name,memory.total --format=csv"

parser = argparse.ArgumentParser(description='Launcher')
parser.add_argument('--launch', action='store_true', default=False)
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

def create_job_file(property, guidance=0.25):
    job_name = f"eval_{property}"
    input_file = 'experiments/template_evals.sh'
    output_file = f'job_drafts/evals_{property}_w_{guidance}.sh'

    with open(input_file, 'r') as file:
        content = file.read()
    
    exp_name = f"eval_{property}_w_{guidance}"
    content = content.replace("PROPERTY", property)
    content = content.replace("BATCH_SIZE", BATCH_SIZE)
    content = content.replace("GUIDANCE_WEIGHT", guidance)
    content = content.replace("EXP_NAME", exp_name)

    with open(output_file, 'w') as file:
        file.write(content)

    # Optional: Submit the job and/or remove the file
    # os.system(f'sbatch experiments/temp_job_{property}.sh')
    # os.remove(f'experiments/temp_job_{property}.sh')

properties = ['alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv']
guidance_weights = [0.1, 0.25, 0.5, 1]

for property in properties:
    for guidance in guidance_weights:
        create_job_file(property, str(guidance))

        if property == 'alpha' and args.launch:
            os.system(f'sbatch job_drafts/evals_{property}_w_{guidance}.sh')
