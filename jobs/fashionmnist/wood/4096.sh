#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=WDFM4096-New
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=144:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/WDFM4096-50-10.log

module purge
conda init bash
conda activate OoD

python3 main/main_wood.py --config=config/WOOD/WOOD-FashionMNIST.yaml --n_ood=4096 > checkpoint/log/FashionMNIST/WOOD/log-4096-50-10.txt