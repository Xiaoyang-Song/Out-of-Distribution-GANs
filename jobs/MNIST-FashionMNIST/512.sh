#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=j512
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/MNIST-FashionMNIST-512.log

python3 main/main_ood.py --config=config/GAN/OOD-GAN-MNIST-FashionMNIST.yaml --n_ood=512 > checkpoint/log/MNIST-FashionMNIST/log-512.txt
