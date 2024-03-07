#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=j1024
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/CIFAR100-Texture-1024.log

python3 main/main_ood.py --config=config/GAN/CIFAR100-Texture.yaml --n_ood=1024 > checkpoint/log/CIFAR100-Texture/log-1024.txt
