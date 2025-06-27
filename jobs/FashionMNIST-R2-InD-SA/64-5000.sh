#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=j64
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/FashionMNIST-R2-InD-SA-64-5000.log

python3 main/main_ood.py --config=config/GAN/OOD-GAN-FashionMNIST-R2.yaml --n_ood=64 --n_ind=5000 > checkpoint/log/FashionMNIST-R2-InD-SA/log-64-5000.txt
