#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=GANSV1024
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/SV-I-1024.log

python3 main/main_ood.py --config=config/GAN/OOD-GAN-SVHN.yaml --n_ood=1024 > checkpoint/log/SVHN/OOD-GAN/log-1024.txt