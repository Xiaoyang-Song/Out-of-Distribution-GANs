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

python3 main_ood.py --mc=3 --num_epochs=1 --balanced=True --n_ood=32