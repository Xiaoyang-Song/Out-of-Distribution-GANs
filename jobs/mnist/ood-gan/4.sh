#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=GANM4
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=8GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/m-4.log

python3 main/main_ood.py --config=config/GAN/OOD-GAN-MNIST.yaml --n_ood=4 > checkpoint/log/MNIST/OOD-GAN/log-4.txt