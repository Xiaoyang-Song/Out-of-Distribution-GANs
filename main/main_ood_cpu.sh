#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=OoD-training
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00

module purge
conda init bash
conda activate OoD

# python3 main_ood.py --config=../config/GAN/OOD-GAN-FashionMNIST.yaml
# python3 main_ood.py --config=../config/GAN/OOD-GAN-FashionMNIST-MNIST.yaml
python3 main_ood.py --config=../config/GAN/OOD-GAN-CIFAR10-SVHN.yaml