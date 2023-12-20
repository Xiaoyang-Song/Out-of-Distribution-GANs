#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=CTGAN32
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/CTGAN32.log

module purge
conda init bash
conda activate OoD

python3 main/main_ood.py --config=config/GAN/OOD-GAN-CIFAR10-Texture.yaml --n_ood=32 > checkpoint/log/CIFAR10-Texture/log-32.txt