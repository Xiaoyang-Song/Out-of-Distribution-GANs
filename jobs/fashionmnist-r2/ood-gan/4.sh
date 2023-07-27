#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=GANFM4
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=120:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/GANFM4-R2.log

module purge
conda init bash
conda activate OoD

python3 main/main_ood.py --config=config/GAN/OOD-GAN-FashionMNIST-R2.yaml --n_ood=4 > checkpoint/log/FashionMNIST-R2/OOD-GAN/log-4.txt