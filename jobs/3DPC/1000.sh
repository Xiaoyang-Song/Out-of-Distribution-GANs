#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=j1000
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/3DPC-1000.log

python3 main/main_ood.py --config=config/GAN/3DPC.yaml --n_ood=1000 > checkpoint/log/3DPC/log-1000.txt
