#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=test
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=1000m
#SBATCH --time=1:00:00
#SBATCH --cpus-per-gpu=1

module purge
pip3 install torch
pip3 install icecream

python3 gl.py