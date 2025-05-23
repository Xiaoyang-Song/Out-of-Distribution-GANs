#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=WDCS1024
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/CSWD1024.log

python3 main/main_wood.py --config=config/WOOD/WOOD-CIFAR10-SVHN.yaml --n_ood=1024 > checkpoint/log/CIFAR10-SVHN/WOOD/log-1024.txt