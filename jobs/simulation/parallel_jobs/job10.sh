#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=J10
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

module purge
conda init bash
conda activate OoD

# Run Simulation experiments
# Step 1: Generate Settings
# python3 simulation.py --mode=G --config=config/simulation/setting_1_config.yaml

# Step 2: Run Experiments with different hyperparameters
python3 simulation.py --mode=R --config=config/simulation/run_config.yaml --n_ood=2 --h=128 --beta=1 --w_ce=1 --w_ood=0.01 --w_z=1 --wood_lr=0.001 --gan_lr=0.0001 --bsz_tri=256 --bsz_val=256 --bsz_ood=2 --n_d=2 --n_g=1




