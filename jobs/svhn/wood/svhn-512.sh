#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=WDSV512
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=14GB
#SBATCH --time=144:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/WDSV512.log

module purge
conda init bash
conda activate OoD

python3 main/main_wood.py --config=config/WOOD/WOOD-SVHN.yaml --n_ood=512 > checkpoint/log/SVHN/WOOD/log-512.txt