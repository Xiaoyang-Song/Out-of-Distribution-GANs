#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=GSetting
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/home/xysong/Out-of-Distribution-GANs/slurm-jobs/generate_setting.log

module purge
conda init bash
conda activate OoD

# Run Simulation experiments
# Step 1: Generate Settings
python3 simulation.py --mode=G --config=config/simulation/setting_1_config.yaml


