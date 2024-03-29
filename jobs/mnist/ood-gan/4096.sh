#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=GANM4096
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=32
#SBATCH --time=144:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/GANM4096.log

module purge
conda init bash
conda activate OoD

python3 main/main_ood.py --config=config/GAN/OOD-GAN-MNIST.yaml --n_ood=4096 > checkpoint/log/MNIST/OOD-GAN/log-4096.txt