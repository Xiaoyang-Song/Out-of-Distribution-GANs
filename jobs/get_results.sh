

# SEE-OOD results only
# python get_results.py --name=MNIST --regime=I
# python get_results.py --name=FashionMNIST --regime=I
# python get_results.py --name=MNIST-FashionMNIST --regime=I
# python get_results.py --name=SVHN --regime=I
# python get_results.py --name=CIFAR10-SVHN --regime=I

# python get_results.py --name=FashionMNIST-R2 --regime=II
# python get_results.py --name=SVHN-R2 --regime=II

# Baselines - full baselines
python get_results.py --name=MNIST --regime=I --type=full
python get_results.py --name=FashionMNIST --regime=I --type=full
python get_results.py --name=MNIST-FashionMNIST --regime=I --type=full
python get_results.py --name=SVHN --regime=I --type=full
python get_results.py --name=CIFAR10-SVHN --regime=I --type=full

python get_results.py --name=FashionMNIST-R2 --regime=II --type=full
python get_results.py --name=SVHN-R2 --regime=II --type=full

# Baselines - only include good performance ones
# python get_results.py --name=MNIST --regime=I --type=short
# python get_results.py --name=FashionMNIST --regime=I --type=short
# python get_results.py --name=MNIST-FashionMNIST --regime=I --type=short
# python get_results.py --name=SVHN --regime=I --type=short
# python get_results.py --name=CIFAR10-SVHN --regime=I --type=short

# python get_results.py --name=FashionMNIST-R2 --regime=II --type=short
# python get_results.py --name=SVHN-R2 --regime=II --type=short

# Case study
# python get_results.py --case --name=3DPC-R1
# python get_results.py --case --name=3DPC-R2

# Simulation
# python get_results.py --sim --setting=I
# python get_results.py --sim --setting=II

# Sensitivity analysis (InD)
# python get_results.py --ind_sa --name=FashionMNIST
# python get_results.py --ind_sa --name=FashionMNIST-R2