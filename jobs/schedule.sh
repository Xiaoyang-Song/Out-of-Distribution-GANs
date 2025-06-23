#!/bin/bash

# Path configuration
# conda activate OoD

# Environment Configuration
export PYTHONPATH=$PYTHONPATH$:`pwd`

# Sample (if necessary)
# ...
# scheduling parallel jobs

# sbatch jobs/MNIST/8.sh
# sbatch jobs/MNIST/16.sh
# sbatch jobs/MNIST/32.sh
# sbatch jobs/MNIST/64.sh
# sbatch jobs/MNIST/128.sh
# sbatch jobs/MNIST/256.sh
# sbatch jobs/MNIST/512.sh
# sbatch jobs/MNIST/1024.sh
# sbatch jobs/MNIST/2048.sh
# sbatch jobs/MNIST/4096.sh

# sbatch jobs/MNIST-FashionMNIST/8.sh
# sbatch jobs/MNIST-FashionMNIST/16.sh
# sbatch jobs/MNIST-FashionMNIST/32.sh
# sbatch jobs/MNIST-FashionMNIST/64.sh
# sbatch jobs/MNIST-FashionMNIST/128.sh
# sbatch jobs/MNIST-FashionMNIST/256.sh
# sbatch jobs/MNIST-FashionMNIST/512.sh
# sbatch jobs/MNIST-FashionMNIST/1024.sh
# sbatch jobs/MNIST-FashionMNIST/2048.sh
# sbatch jobs/MNIST-FashionMNIST/4096.sh

# sbatch jobs/FashionMNIST/8.sh
# sbatch jobs/FashionMNIST/16.sh
# sbatch jobs/FashionMNIST/32.sh
# sbatch jobs/FashionMNIST/64.sh
# sbatch jobs/FashionMNIST/128.sh
# sbatch jobs/FashionMNIST/256.sh
# sbatch jobs/FashionMNIST/512.sh
# sbatch jobs/FashionMNIST/1024.sh
# sbatch jobs/FashionMNIST/2048.sh
# sbatch jobs/FashionMNIST/4096.sh

# Newly generated jobs for FashionMNIST-R2
# sbatch jobs/FashionMNIST-R2/4.sh
# sbatch jobs/FashionMNIST-R2/8.sh
# sbatch jobs/FashionMNIST-R2/16.sh
# sbatch jobs/FashionMNIST-R2/32.sh
# sbatch jobs/FashionMNIST-R2/64.sh
# sbatch jobs/FashionMNIST-R2/128.sh
# sbatch jobs/FashionMNIST-R2/256.sh
# sbatch jobs/FashionMNIST-R2/512.sh
# sbatch jobs/FashionMNIST-R2/1024.sh
# sbatch jobs/FashionMNIST-R2/2048.sh
# sbatch jobs/FashionMNIST-R2/4096.sh

# Newly generated jobs for CIFAR10-SVHN
# sbatch jobs/CIFAR10-SVHN/8.sh
# sbatch jobs/CIFAR10-SVHN/16.sh
# sbatch jobs/CIFAR10-SVHN/32.sh
# sbatch jobs/CIFAR10-SVHN/64.sh
# sbatch jobs/CIFAR10-SVHN/128.sh
# sbatch jobs/CIFAR10-SVHN/256.sh
# sbatch jobs/CIFAR10-SVHN/512.sh
# sbatch jobs/CIFAR10-SVHN/1024.sh
# sbatch jobs/CIFAR10-SVHN/2048.sh
# sbatch jobs/CIFAR10-SVHN/4096.sh


# sbatch jobs/SVHN/8.sh
# sbatch jobs/SVHN/16.sh
# sbatch jobs/SVHN/32.sh
# sbatch jobs/SVHN/64.sh
# sbatch jobs/SVHN/128.sh
# sbatch jobs/SVHN/256.sh
# sbatch jobs/SVHN/512.sh
# sbatch jobs/SVHN/1024.sh
# sbatch jobs/SVHN/2048.sh
# sbatch jobs/SVHN/4096.sh

# Newly generated jobs for SVHN-R2
# sbatch jobs/SVHN-R2/8.sh
# sbatch jobs/SVHN-R2/16.sh
# sbatch jobs/SVHN-R2/32.sh
# sbatch jobs/SVHN-R2/64.sh
# sbatch jobs/SVHN-R2/128.sh
# sbatch jobs/SVHN-R2/256.sh
# sbatch jobs/SVHN-R2/512.sh
# sbatch jobs/SVHN-R2/1024.sh
# sbatch jobs/SVHN-R2/2048.sh
# sbatch jobs/SVHN-R2/4096.sh

# Sensitivity analysis for FashionMNIST experiments
# sample_sizes = [1000, 2000, 5000, 10000, 20000, 40000, 60000]
# sbatch jobs/FashionMNIST-InD-SA/64-1000.sh
# sbatch jobs/FashionMNIST-InD-SA/64-2000.sh
# sbatch jobs/FashionMNIST-InD-SA/64-5000.sh
# sbatch jobs/FashionMNIST-InD-SA/64-10000.sh
# sbatch jobs/FashionMNIST-InD-SA/64-20000.sh
# sbatch jobs/FashionMNIST-InD-SA/64-40000.sh
# sbatch jobs/FashionMNIST-InD-SA/64-60000.sh # (not necessary, since 60000 is the full dataset)
sbatch jobs/FashionMNIST-InD-SA/64-all.sh

# Check memory on disk
# home-quota

# Check current job lists
# squeue -u xysong