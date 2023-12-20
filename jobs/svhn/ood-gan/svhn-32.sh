#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=GANSV32
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=14GB
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/SV-I-32.log

python3 main/main_ood.py --config=config/GAN/OOD-GAN-SVHN.yaml --n_ood=32 > checkpoint/log/SVHN/OOD-GAN/log-32.txt