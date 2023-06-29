#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=GANFM1024
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=12GB
#SBATCH --time=120:00:00
#SBATCH --output=/home/xysong/slurm-jobs/GANFM1024.log

module purge
conda init bash
conda activate OoD

python3 main/main_ood.py --config=config/GAN/OOD-GAN-FashionMNIST.yaml --n_ood=1024 > checkpoint/log/FashionMNIST/OOD-GAN/log-1024.txt