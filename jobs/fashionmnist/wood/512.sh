#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=WDFM512
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=14GB
#SBATCH --time=144:00:00
#SBATCH --output=/home/xysong/slurm-jobs/WDFM512.log

module purge
conda init bash
conda activate OoD

python3 main/main_wood.py --config=config/WOOD/WOOD-FashionMNIST.yaml --n_ood=512 > checkpoint/log/FashionMNIST/WOOD/log-512.txt