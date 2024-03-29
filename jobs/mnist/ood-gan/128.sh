#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=GANM128
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=32
#SBATCH --time=240:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/GANM128.log

module purge
conda init bash
conda activate OoD

python3 main/main_ood.py --config=config/GAN/OOD-GAN-MNIST.yaml --n_ood=128 > checkpoint/log/MNIST/OOD-GAN/log-128.txt