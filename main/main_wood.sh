#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=OoD-training
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=24:00:00

module purge
conda init bash
conda activate OoD

python3 main_wood.py --mc=5 --num_epochs=2 --balanced=imbalance --n_ood=8