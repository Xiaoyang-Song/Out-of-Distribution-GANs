#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=OoD-training
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=14GB
#SBATCH --time=36:00:00

module purge
conda init bash
conda activate OoD

# python3 main_wood.py --config=../config/WOOD/WOOD-FashionMNIST.yaml
# python3 main_wood.py --config=../config/WOOD/WOOD-FashionMNIST-MNIST.yaml
python3 main_wood.py --config=../config/WOOD/WOOD-CIFAR10-SVHN.yaml
# python3 main_wood.py --config=../config/WOOD/WOOD-SVHN.yaml > SVHN/WOOD/4-100.txt
# python3 main_wood.py --config=../config/WOOD/WOOD-MNIST.yaml