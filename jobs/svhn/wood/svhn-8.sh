#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=OoD-training
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=14GB
#SBATCH --time=96:00:00

module purge
conda init bash
conda activate OoD

python3 main/main_wood.py --config=config/WOOD/WOOD-SVHN.yaml --n_ood=8 > checkpoint/log/SVHN/WOOD/log-8.txt