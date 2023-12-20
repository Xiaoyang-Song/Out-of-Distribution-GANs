#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=AUROC-CT
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/AUROC-CT.log



python3 cal_metric.py --dataset CIFAR10-Texture --regime Balanced