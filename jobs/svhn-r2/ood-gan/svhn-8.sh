#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=GANSV8
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=14GB
#SBATCH --time=168:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/GANSV8-R2.log

module purge
conda init bash
conda activate OoD

python3 main/main_ood.py --config=config/GAN/OOD-GAN-SVHN-R2.yaml --n_ood=8 > checkpoint/log/SVHN-R2/OOD-GAN/log-8.txt