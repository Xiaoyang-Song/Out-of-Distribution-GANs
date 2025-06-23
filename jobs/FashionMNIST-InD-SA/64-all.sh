#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=j64-SA
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/FashionMNIST-InD-SA-64-all.log

python3 main/main_ood.py --config=config/GAN/OOD-GAN-FashionMNIST.yaml --n_ood=64 --n_ind=1000 > checkpoint/log/FashionMNIST-InD-SA/log-64-1000.txt

python3 main/main_ood.py --config=config/GAN/OOD-GAN-FashionMNIST.yaml --n_ood=64 --n_ind=2000 > checkpoint/log/FashionMNIST-InD-SA/log-64-2000.txt

python3 main/main_ood.py --config=config/GAN/OOD-GAN-FashionMNIST.yaml --n_ood=64 --n_ind=5000 > checkpoint/log/FashionMNIST-InD-SA/log-64-5000.txt

python3 main/main_ood.py --config=config/GAN/OOD-GAN-FashionMNIST.yaml --n_ood=64 --n_ind=10000 > checkpoint/log/FashionMNIST-InD-SA/log-64-10000.txt

python3 main/main_ood.py --config=config/GAN/OOD-GAN-FashionMNIST.yaml --n_ood=64 --n_ind=20000 > checkpoint/log/FashionMNIST-InD-SA/log-64-20000.txt

python3 main/main_ood.py --config=config/GAN/OOD-GAN-FashionMNIST.yaml --n_ood=64 --n_ind=40000 > checkpoint/log/FashionMNIST-InD-SA/log-64-40000.txt

python3 main/main_ood.py --config=config/GAN/OOD-GAN-FashionMNIST.yaml --n_ood=64 --n_ind=60000 > checkpoint/log/FashionMNIST-InD-SA/log-64-60000.txt