#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=OoD-Sampling
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/sample_ood.log


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
python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=8
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
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=32
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=64
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=128
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=256
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=512
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=1024
# python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=2048
python3 sample_ood.py --config=../config/sampling/sample-cifar10-svhn.yaml --n_ood=4096

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
python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=4
python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=8
python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=16
python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=32
python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=64
python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=128
python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=256
python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=512
python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=1024
python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=2048
python3 sample_ood.py --config=../config/sampling/sample-mnist-fashionmnist.yaml --n_ood=4096




# python3 sample_ood.py --config=../config/sampling/sample-iNaturalist.yaml --n_ood=4
# python3 sample_ood.py --config=../config/sampling/sample-iNaturalist.yaml --n_ood=8
# python3 sample_ood.py --config=../config/sampling/sample-iNaturalist.yaml --n_ood=16
# python3 sample_ood.py --config=../config/sampling/sample-iNaturalist.yaml --n_ood=32
# python3 sample_ood.py --config=../config/sampling/sample-iNaturalist.yaml --n_ood=64
# python3 sample_ood.py --config=../config/sampling/sample-iNaturalist.yaml --n_ood=128
# python3 sample_ood.py --config=../config/sampling/sample-iNaturalist.yaml --n_ood=256
# python3 sample_ood.py --config=../config/sampling/sample-iNaturalist.yaml --n_ood=512
# python3 sample_ood.py --config=../config/sampling/sample-iNaturalist.yaml --n_ood=1024
# python3 sample_ood.py --config=../config/sampling/sample-iNaturalist.yaml --n_ood=2048



# python3 sample_ood.py --config=../config/sampling/sample-Places.yaml --n_ood=4
# python3 sample_ood.py --config=../config/sampling/sample-Places.yaml --n_ood=8
# python3 sample_ood.py --config=../config/sampling/sample-Places.yaml --n_ood=16
# python3 sample_ood.py --config=../config/sampling/sample-Places.yaml --n_ood=32
# python3 sample_ood.py --config=../config/sampling/sample-Places.yaml --n_ood=64
# python3 sample_ood.py --config=../config/sampling/sample-Places.yaml --n_ood=128
# python3 sample_ood.py --config=../config/sampling/sample-Places.yaml --n_ood=256
# python3 sample_ood.py --config=../config/sampling/sample-Places.yaml --n_ood=512
# python3 sample_ood.py --config=../config/sampling/sample-Places.yaml --n_ood=1024
# python3 sample_ood.py --config=../config/sampling/sample-Places.yaml --n_ood=2048