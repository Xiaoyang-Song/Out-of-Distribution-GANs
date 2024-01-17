#!/bin/bash

# Path configuration
# conda activate OoD

# Environment Configuration
export PYTHONPATH=$PYTHONPATH$:`pwd`

# Sample (if necessary)
# ...
# scheduling parallel jobs

# SVHN WOOD
# sbatch jobs/svhn/wood/svhn-4.sh
# sbatch jobs/svhn/wood/svhn-8.sh
# sbatch jobs/svhn/wood/svhn-16.sh
# sbatch jobs/svhn/wood/svhn-32.sh
# sbatch jobs/svhn/wood/svhn-64.sh
# sbatch jobs/svhn/wood/svhn-128.sh
# sbatch jobs/svhn/wood/svhn-256.sh
# sbatch jobs/svhn/wood/svhn-512.sh
# sbatch jobs/svhn/wood/svhn-1024.sh
# sbatch jobs/svhn/wood/svhn-2048.sh
# sbatch jobs/svhn/wood/svhn-4096.sh

# FashionMNIST OoD GAN
# sbatch jobs/fashionmnist/ood-gan/4.sh
# sbatch jobs/fashionmnist/ood-gan/8.sh
# sbatch jobs/fashionmnist/ood-gan/16.sh
# sbatch jobs/fashionmnist/ood-gan/32.sh
# sbatch jobs/fashionmnist/ood-gan/64.sh
# sbatch jobs/fashionmnist/ood-gan/128.sh
# sbatch jobs/fashionmnist/ood-gan/256.sh
# sbatch jobs/fashionmnist/ood-gan/512.sh
# sbatch jobs/fashionmnist/ood-gan/1024.sh
# sbatch jobs/fashionmnist/ood-gan/2048.sh
# sbatch jobs/fashionmnist/ood-gan/4096.sh

# MNIST OoD GAN
# sbatch jobs/mnist/ood-gan/4.sh
# sbatch jobs/mnist/ood-gan/8.sh
# sbatch jobs/mnist/ood-gan/16.sh
# sbatch jobs/mnist/ood-gan/32.sh
# sbatch jobs/mnist/ood-gan/64.sh
# sbatch jobs/mnist/ood-gan/128.sh
# sbatch jobs/mnist/ood-gan/256.sh
# sbatch jobs/mnist/ood-gan/512.sh
# sbatch jobs/mnist/ood-gan/1024.sh
# sbatch jobs/mnist/ood-gan/2048.sh
# sbatch jobs/mnist/ood-gan/4096.sh

# MNIST-FashionMNIST OoD GAN
# sbatch jobs/mnist-fashionmnist/ood-gan/4.sh
# sbatch jobs/mnist-fashionmnist/ood-gan/8.sh
# sbatch jobs/mnist-fashionmnist/ood-gan/16.sh
# sbatch jobs/mnist-fashionmnist/ood-gan/32.sh
# sbatch jobs/mnist-fashionmnist/ood-gan/64.sh
# sbatch jobs/mnist-fashionmnist/ood-gan/128.sh
# sbatch jobs/mnist-fashionmnist/ood-gan/256.sh
# sbatch jobs/mnist-fashionmnist/ood-gan/512.sh
# sbatch jobs/mnist-fashionmnist/ood-gan/1024.sh
# sbatch jobs/mnist-fashionmnist/ood-gan/2048.sh
# sbatch jobs/mnist-fashionmnist/ood-gan/4096.sh

# CIFAR10-SVHN OoD GAN
# sbatch jobs/cifar10-svhn/ood-gan/4.sh
# sbatch jobs/cifar10-svhn/ood-gan/8.sh
# sbatch jobs/cifar10-svhn/ood-gan/16.sh
# sbatch jobs/cifar10-svhn/ood-gan/32.sh
# sbatch jobs/cifar10-svhn/ood-gan/64.sh
# sbatch jobs/cifar10-svhn/ood-gan/128.sh
# sbatch jobs/cifar10-svhn/ood-gan/256.sh
# sbatch jobs/cifar10-svhn/ood-gan/512.sh
# sbatch jobs/cifar10-svhn/ood-gan/1024.sh
# sbatch jobs/cifar10-svhn/ood-gan/2048.sh
# sbatch jobs/cifar10-svhn/ood-gan/4096.sh


# SVHN OoD GAN
sbatch jobs/svhn/ood-gan/svhn-4.sh
# sbatch jobs/svhn/ood-gan/svhn-8.sh
# sbatch jobs/svhn/ood-gan/svhn-16.sh
# sbatch jobs/svhn/ood-gan/svhn-32.sh
# sbatch jobs/svhn/ood-gan/svhn-64.sh
# sbatch jobs/svhn/ood-gan/svhn-128.sh
# sbatch jobs/svhn/ood-gan/svhn-256.sh
# sbatch jobs/svhn/ood-gan/svhn-512.sh
# sbatch jobs/svhn/ood-gan/svhn-1024.sh
# sbatch jobs/svhn/ood-gan/svhn-2048.sh
# sbatch jobs/svhn/ood-gan/svhn-4096.sh

# FashionMNIST OoD GAN - Regime II
# sbatch jobs/fashionmnist-r2/ood-gan/4.sh
# sbatch jobs/fashionmnist-r2/ood-gan/8.sh
# sbatch jobs/fashionmnist-r2/ood-gan/16.sh
# sbatch jobs/fashionmnist-r2/ood-gan/32.sh
# sbatch jobs/fashionmnist-r2/ood-gan/64.sh
# sbatch jobs/fashionmnist-r2/ood-gan/128.sh
# sbatch jobs/fashionmnist-r2/ood-gan/256.sh
# sbatch jobs/fashionmnist-r2/ood-gan/512.sh
# sbatch jobs/fashionmnist-r2/ood-gan/1024.sh
# sbatch jobs/fashionmnist-r2/ood-gan/2048.sh
# sbatch jobs/fashionmnist-r2/ood-gan/4096.sh

# CIFAR10-SVHN OoD GAN - Regime II
# sbatch jobs/cifar10-svhn-r2/ood-gan/4.sh
# sbatch jobs/cifar10-svhn-r2/ood-gan/8.sh
# sbatch jobs/cifar10-svhn-r2/ood-gan/16.sh
# sbatch jobs/cifar10-svhn-r2/ood-gan/32.sh
# sbatch jobs/cifar10-svhn-r2/ood-gan/64.sh
# sbatch jobs/cifar10-svhn-r2/ood-gan/128.sh
# sbatch jobs/cifar10-svhn-r2/ood-gan/256.sh
# sbatch jobs/cifar10-svhn-r2/ood-gan/512.sh
# sbatch jobs/cifar10-svhn-r2/ood-gan/1024.sh
# sbatch jobs/cifar10-svhn-r2/ood-gan/2048.sh
# sbatch jobs/cifar10-svhn-r2/ood-gan/4096.sh


# SVHN OoD GAN - Regime II
# sbatch jobs/svhn-r2/ood-gan/svhn-4.sh
# sbatch jobs/svhn-r2/ood-gan/svhn-8.sh
# sbatch jobs/svhn-r2/ood-gan/svhn-16.sh
# sbatch jobs/svhn-r2/ood-gan/svhn-32.sh
# sbatch jobs/svhn-r2/ood-gan/svhn-64.sh
# sbatch jobs/svhn-r2/ood-gan/svhn-128.sh
# sbatch jobs/svhn-r2/ood-gan/svhn-256.sh
# sbatch jobs/svhn-r2/ood-gan/svhn-512.sh
# sbatch jobs/svhn-r2/ood-gan/svhn-1024.sh
# sbatch jobs/svhn-r2/ood-gan/svhn-2048.sh
# sbatch jobs/svhn-r2/ood-gan/svhn-4096.sh

# REBUTTAL
# sbatch jobs/cifar10-texture/4.sh
# sbatch jobs/cifar10-texture/8.sh
# sbatch jobs/cifar10-texture/16.sh
# sbatch jobs/cifar10-texture/32.sh
# sbatch jobs/cifar10-texture/64.sh
# sbatch jobs/cifar10-texture/128.sh
# sbatch jobs/cifar10-texture/256.sh
# sbatch jobs/cifar10-texture/512.sh
# sbatch jobs/cifar10-texture/1024.sh
# sbatch jobs/cifar10-texture/2048.sh

# sbatch jobs/cifar100-texture/16.sh
# sbatch jobs/cifar100-texture/32.sh 
# sbatch jobs/cifar100-texture/64.sh
# sbatch jobs/cifar100-texture/128.sh
# sbatch jobs/cifar100-texture/256.sh

# METRICS
# sbatch jobs/metrics/cifar10-svhn.sh
# sbatch jobs/metrics/fashionmnist.sh
# sbatch jobs/metrics/fashionmnist-r2.sh
# sbatch jobs/metrics/svhn.sh
# sbatch jobs/metrics/svhn-r2.sh
# sbatch jobs/metrics/cifar10-texture.sh

# Check memory on disk
# home-quota

# Check current job lists
# squeue -u xysong