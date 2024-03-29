#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=OoD-training
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=14GB
#SBATCH --time=108:00:00

module purge
conda init bash
conda activate OoD

python3 main/main_wood.py --config=config/WOOD/WOOD-FashionMNIST.yaml --n_ood=128 > checkpoint/log/FashionMNIST/WOOD/log-128.txt