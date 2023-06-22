#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=OoD-training
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=144:00:00

module purge
conda init bash
conda activate OoD

python3 main/main_wood.py --config=config/WOOD/WOOD-CIFAR10-SVHN.yaml --n_ood=4 > checkpoint/log/CIFAR10-SVHN/WOOD/log-4.txt