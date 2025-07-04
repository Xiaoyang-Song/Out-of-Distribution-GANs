import os
import numpy as np


# GL
# ACCOUNT = 'jhjin1'
ACCOUNT = 'sunwbgt0'


TIME = "2:00:00"
# Configuration
EXP_DSET = 'Param-SA'

n=64
beta_ood_list = [10, 1, 0.1, 0.01, 0.001]
beta_z_list = [10, 1, 0.1, 0.01, 0.001]

config_dir = os.path.join('config', 'GAN', 'SA')

CMD_ONLY = False

print("Generating job files...")
# Create logging directory
log_path = os.path.join('checkpoint', 'log', EXP_DSET)
os.makedirs(log_path, exist_ok=True)

for beta_ood in beta_ood_list:
    for beta_z in beta_z_list:
        print(f"sbatch jobs/{EXP_DSET}/{beta_ood}-{beta_z}.sh")
        # Create job directory
        job_path = os.path.join('jobs', EXP_DSET)
        os.makedirs(job_path, exist_ok=True)
        # Declare job name
        filename = os.path.join('jobs', EXP_DSET, f"{beta_ood}-{beta_z}.sh")
        # Write files
        f = open(filename, 'w')
        f.write("#!/bin/bash\n\n")
        f.write(f"#SBATCH --account={ACCOUNT}\n")
        f.write(f"#SBATCH --job-name=J{beta_ood}-{beta_z}\n")
        f.write("#SBATCH --mail-user=xysong@umich.edu\n")
        f.write("#SBATCH --mail-type=BEGIN,END,FAIL\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --partition=gpu\n")
        f.write("#SBATCH --gpus=1\n")
        f.write("#SBATCH --mem-per-gpu=16GB\n")
        f.write(f"#SBATCH --time={TIME}\n")
        f.write(f"#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/{EXP_DSET}-{beta_ood}-{beta_z}.log\n\n")

        f.write(f"python3 main/main_ood.py --config=config/GAN/SA/OOD-GAN-FashionMNIST-{beta_ood}-{beta_z}.yaml --n_ood={n} > checkpoint/log/{EXP_DSET}/log-{beta_ood}-{beta_z}.txt\n")
        f.close()
