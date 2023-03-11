#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=test
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=1000m
#SBATCH --time=1:00:00
#SBATCH --partition=gpu

python3 gl.py
