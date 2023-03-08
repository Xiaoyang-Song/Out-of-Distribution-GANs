#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=test
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000m
#SBATCH --time=1:00:00
#SBATCH --partition=standard

module purge
pip3 install --user icecream
pip3 install --user torch

python3 gl.py