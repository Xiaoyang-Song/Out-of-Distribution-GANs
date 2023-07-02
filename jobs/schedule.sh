#!/bin/bash

# scheduling

# FashionMNIST WOOD
sbatch jobs/fashionmnist/wood/4.sh
sbatch jobs/fashionmnist/wood/8.sh
sbatch jobs/fashionmnist/wood/16.sh
sbatch jobs/fashionmnist/wood/32.sh
sbatch jobs/fashionmnist/wood/64.sh
sbatch jobs/fashionmnist/wood/128.sh
sbatch jobs/fashionmnist/wood/256.sh
sbatch jobs/fashionmnist/wood/512.sh
sbatch jobs/fashionmnist/wood/1024.sh
sbatch jobs/fashionmnist/wood/2048.sh
sbatch jobs/fashionmnist/wood/4096.sh

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