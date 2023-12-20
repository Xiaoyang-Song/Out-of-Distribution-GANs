#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=AUROC-CS
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/AUROC-CS.log

module purge
conda init bash
conda activate OoD

python3 cal_metric.py --dataset CIFAR10-SVHN --regime Balanced