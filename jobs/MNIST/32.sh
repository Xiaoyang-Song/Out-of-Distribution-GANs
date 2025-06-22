#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=j32
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/MNIST-32.log

python3 main/main_ood.py --config=config/GAN/OOD-GAN-MNIST.yaml --n_ood=32 > checkpoint/log/MNIST/log-32.txt
