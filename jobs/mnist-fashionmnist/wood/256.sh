#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=WDM256
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=144:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/WDM256.log


module purge
conda init bash
conda activate OoD

python3 main/main_wood.py --config=config/WOOD/WOOD-MNIST.yaml --n_ood=256 > checkpoint/log/MNIST/WOOD/log-256.txt