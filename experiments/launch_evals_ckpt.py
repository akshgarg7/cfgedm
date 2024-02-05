import subprocess
import argparse
from best_ckpts import gen_file

# Command to get GPU information
command = "nvidia-smi --query-gpu=name,memory.total --format=csv"

parser = argparse.ArgumentParser(description='Launcher')
parser.add_argument('--launch', action='store_true', default=False)
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
    BATCH_SIZE = str(int(batch_size))

except subprocess.CalledProcessError as e:
    print("An error occurred while trying to retrieve GPU information.")
    print(e.output)

import os

def create_job_file(property, input_file, output_file, epoch, guidance=0.25):
    with open(input_file, 'r') as file:
        content = file.read()
    
    exp_name = f"full_scale_eval_{property}_w_{guidance}_ckpt_{epoch}"
    content = content.replace("PROPERTY", property)
    content = content.replace("BATCH_SIZE", BATCH_SIZE)
    content = content.replace("GUIDANCE_WEIGHT", str(guidance))
    content = content.replace("EXP_NAME", exp_name)
    content = content.replace("CKPT", str(epoch))

    with open(output_file, 'w') as file:
        file.write(content)

    # Optional: Submit the job and/or remove the file
    # os.system(f'sbatch experiments/temp_job_{property}.sh')
    # os.remove(f'experiments/temp_job_{property}.sh')

# properties = ['mu', 'homo', 'lumo', 'gap', 'Cv']
properties = ['Cv']
# epochs = [13, 15, 17, 20, 21, 26]
# epochs = epochs[::-1]
# epochs = [x * 100 for x in epochs]

# guidance_weights = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0]
guidance_weights = [0.5, 1, 1.5, 2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# guidance_weights = [5.5, 6, 6.5, 7, 7.5]
# guidance_weights = [x-0.25 for x in guidance_weights] + guidance_weights

for property in properties:
    epochs = gen_file(property)[-5:]
    for epoch in epochs:
        for guidance in guidance_weights:
            output_file = f'job_drafts/ckpt_launches/evals_{property}_w_{guidance}.sh'
            input_file = 'experiments/template_evals_ckpt.sh'
            if args.resume:
                output_file = f'job_drafts/ckpt_launches/evals_{property}_w_{guidance}_{epoch}_resume.sh'
                input_file = 'experiments/template_evals_resume_ckpt.sh'

            create_job_file(property, input_file, output_file, epoch, str(guidance))

            if args.launch:
                os.system(f'sbatch {output_file}')
