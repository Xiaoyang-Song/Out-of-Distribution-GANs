#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=OoD-training
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=12GB
#SBATCH --time=96:00:00

module purge
conda init bash
conda activate OoD
python3 main/main_ood.py --config=config/GAN/OOD-GAN-CIFAR10-SVHN.yaml --n_ood=16 > checkpoint/log/CIFAR10-SVHN/OOD-GAN/log-16.txt