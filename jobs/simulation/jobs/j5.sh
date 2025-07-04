#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=sim2-5
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/out/sim-2-5.log

python3 simulation.py --mode=R --config=config/simulation/run_config.yaml --JID=28 --n_ood=2 --h=128 --beta=1 --w_ce=1 --w_ood=1 --w_z=100 \
    --wood_lr=0.001 --d_lr=0.0001 --g_lr=0.001 --bsz_tri=256 --bsz_val=256 --bsz_ood=2 --n_d=1 --n_g=3 --seed=99



