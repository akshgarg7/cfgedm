import subprocess
import re

def gen_file(property1, property2=None, resume=True):
    resume_str = '_resume'
    if not resume:
        resume_str = ''

    # The path you want to list
    path = f'/atlas/u/akshgarg/cfgedm/outputs/icml_sin_{property1}{resume_str}/'
    if property2:
        path = f'/atlas/u/akshgarg/cfgedm/outputs/icml_{property1}_{property2}{resume_str}/'

    # Regular expression to match the pattern 'generative_model_{ckpt}.npy'
    pattern = re.compile(r'generative_model_(\d+)\.npy')

    # List to hold extracted checkpoint numbers
    checkpoint_numbers = []

    # Run the ls command and capture the output
    result = subprocess.run(['ls', path], capture_output=True, text=True)
    # Check if the command was successful
    if result.returncode == 0:
        # Split the output into lines
        lines = result.stdout.split('\n')
        # Process each line
        for line in lines:
            # Find all matches in the current line
            matches = pattern.findall(line)
            # Add the found checkpoint numbers to the list
            checkpoint_numbers.extend(matches)
    else:
        print(f"Error running ls on {path}: {result.stderr}")
        exit(1)

    # Convert string numbers to integers and sort them
    checkpoint_numbers = sorted(map(int, checkpoint_numbers))
    # checkpoint_numbers = [int(x/100) for x in checkpoint_numbers]
    return checkpoint_numbers

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Launcher')
    parser.add_argument('--property', type=str, required=True)

    args = parser.parse_args()
    ckpt_numbers = gen_file(args.property)
    print(ckpt_numbers)

