#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=J19
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/j19.log

module purge
conda init bash
conda activate OoD

# Run Simulation experiments
# Step 1: Generate Settings
# python3 simulation.py --mode=G --config=config/simulation/setting_1_config.yaml

# Step 2: Run Experiments with different hyperparameters
python3 simulation.py --mode=R --config=config/simulation/run_config.yaml --n_ood=2 --h=128 --beta=1 --w_ce=1 --w_ood=1 --w_z=100 --wood_lr=0.001 --d_lr=0.0001 --g_lr=0.0001 --bsz_tri=256 --bsz_val=256 --bsz_ood=2 --n_d=1 --n_g=1




