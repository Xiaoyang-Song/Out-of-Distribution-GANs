#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=OoD-Sampling
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/sample_ood.log

module purge
conda init bash
conda activate OoD

# Run Sampling Script

# SVHN
python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=4
python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=8
python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=16
python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=32
python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=64
python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=128
python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=256
python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=512
python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=1024
python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=2048
python3 sample_ood.py --config=../config/sampling/sample-svhn.yaml --n_ood=4096

# FashionMNIST
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
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=4
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=8
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=16
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=32
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=64
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=128
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=256
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=512
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=1024
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=2048
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=4096

# MNIST
# python3 sample_ood.py --config=../config/sampling/sample-mnist.yaml --n_ood=4
# python3 sample_ood.py --config=../config/sampling/sample-mnist.yaml --n_ood=8
# python3 sample_ood.py --config=../config/sampling/sample-mnist.yaml --n_ood=16
# python3 sample_ood.py --config=../config/sampling/sample-mnist.yaml --n_ood=32
# python3 sample_ood.py --config=../config/sampling/sample-mnist.yaml --n_ood=64
# python3 sample_ood.py --config=../config/sampling/sample-mnist.yaml --n_ood=128
# python3 sample_ood.py --config=../config/sampling/sample-mnist.yaml --n_ood=256
# python3 sample_ood.py --config=../config/sampling/sample-mnist.yaml --n_ood=512
# python3 sample_ood.py --config=../config/sampling/sample-mnist.yaml --n_ood=1024
# python3 sample_ood.py --config=../config/sampling/sample-mnist.yaml --n_ood=2048
# python3 sample_ood.py --config=../config/sampling/sample-mnist.yaml --n_ood=4096

# MNIST-FashionMNIST
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=1
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=2
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=3
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=4
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=8
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=16
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=32
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=64
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=128
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=256
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=512
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=1024
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=2048
# python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=4096