#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=J0.001-0.1
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/Param-SA-0.001-0.1.log

python3 main/main_ood.py --config=config/GAN/SA/OOD-GAN-FashionMNIST-0.001-0.1.yaml --n_ood=64 > checkpoint/log/Param-SA/log-0.001-0.1.txt
