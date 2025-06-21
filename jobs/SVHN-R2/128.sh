#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=j128
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/SVHN-R2-128.log

python3 main/main_ood.py --config=config/GAN/OOD-GAN-SVHN-R2.yaml --n_ood=128 > checkpoint/log/SVHN-R2/log-128.txt
