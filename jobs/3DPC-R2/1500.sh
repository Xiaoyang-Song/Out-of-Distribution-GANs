#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=j1500
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/3DPC-R2-1500.log

python3 main/main_ood.py --config=config/GAN/OOD-GAN-3DPC-R2.yaml --n_ood=1500 > checkpoint/log/3DPC-R2/log-1500.txt
