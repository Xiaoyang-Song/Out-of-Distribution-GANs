#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=jall2
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/3DPC-R2-all.log

python3 main/main_ood.py --config=config/GAN/OOD-GAN-3DPC-R2.yaml --n_ood=100 > checkpoint/log/3DPC-R2/log-100.txt

python3 main/main_ood.py --config=config/GAN/OOD-GAN-3DPC-R2.yaml --n_ood=200 > checkpoint/log/3DPC-R2/log-200.txt

python3 main/main_ood.py --config=config/GAN/OOD-GAN-3DPC-R2.yaml --n_ood=500 > checkpoint/log/3DPC-R2/log-500.txt

python3 main/main_ood.py --config=config/GAN/OOD-GAN-3DPC-R2.yaml --n_ood=1000 > checkpoint/log/3DPC-R2/log-1000.txt

python3 main/main_ood.py --config=config/GAN/OOD-GAN-3DPC-R2.yaml --n_ood=1500 > checkpoint/log/3DPC-R2/log-1500.txt

python3 main/main_ood.py --config=config/GAN/OOD-GAN-3DPC-R2.yaml --n_ood=2000 > checkpoint/log/3DPC-R2/log-2000.txt
