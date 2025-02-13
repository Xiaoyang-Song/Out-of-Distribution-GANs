#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=CSGAN16
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=12GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/CSGAN16.log

python3 main/main_ood.py --config=config/GAN/OOD-GAN-CIFAR10-SVHN.yaml --n_ood=16 > checkpoint/log/CIFAR10-SVHN/OOD-GAN/log-16.txt