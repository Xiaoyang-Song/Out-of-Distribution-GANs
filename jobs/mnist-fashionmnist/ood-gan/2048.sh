#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=GANMFM2048
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=32
#SBATCH --time=288:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/GANMFM2048.log

module purge
conda init bash
conda activate OoD

python3 main/main_ood.py --config=config/GAN/OOD-GAN-MNIST-FashionMNIST.yaml --n_ood=2048 > checkpoint/log/MNIST-FashionMNIST/OOD-GAN/log-2048.txt