#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=GANFM4096
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=8GB
#SBATCH --time=144:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/GANFM4096.log

module purge
conda init bash
conda activate OoD

python3 main/main_ood.py --config=config/GAN/OOD-GAN-FashionMNIST.yaml --n_ood=4096 > checkpoint/log/FashionMNIST/OOD-GAN/log-4096.txt