#!/bin/bash

#SBATCH --account=jhjin1
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

# Run Sampling Script

# SVHN
# python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=4
# python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=8
# python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=16
# python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=32
# python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=64
# python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=128
# python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=256
# python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=512
# python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=1024
# python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=2048
# python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=4096

#
# python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=4
# python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=8
# python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=16
# python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=32
# python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=64
# python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=128
# python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=256
# python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=512
# python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=1024
# python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=2048
# python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=4096

# CIFAR10-SVHN
python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=4
python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=8
python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=16
python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=32
python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=64
python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=128
python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=256
python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=512
python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=1024
python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=2048
python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=4096