#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=CP256
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/CP64.log

python3 main/main_ood.py --config=config/GAN/CIFAR100-Places365.yaml --n_ood=256 > checkpoint/log/CIFAR100-Places365-32/log-256.txt
