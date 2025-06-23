

# bash jobs/summarize.sh > checkpoint/log/summary.txt
python summarize.py --name FashionMNIST
python summarize.py --name CIFAR10-SVHN
python summarize.py --name SVHN
python summarize.py --name MNIST
python summarize.py --name MNIST-FashionMNIST


python summarize.py --name FashionMNIST-R2
python summarize.py --name SVHN-R2

python summarize.py --name 3DPC-R1
python summarize.py --name 3DPC-R2


