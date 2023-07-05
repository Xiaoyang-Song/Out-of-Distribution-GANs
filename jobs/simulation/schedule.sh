#!/bin/bash

# Path configuration
conda activate OoD

# Environment Configuration
export PYTHONPATH=$PYTHONPATH$:`pwd`

# scheduling simulation study on GL
# Generate Settings
# sbatch jobs/simulation/generate_setting.sh

# Jobs
# sbatch jobs/simulation/parallel_jobs/job1.sh
# sbatch jobs/simulation/parallel_jobs/job2.sh
# sbatch jobs/simulation/parallel_jobs/job3.sh
# sbatch jobs/simulation/parallel_jobs/job4.sh
# sbatch jobs/simulation/parallel_jobs/job5.sh
# sbatch jobs/simulation/parallel_jobs/job6.sh
# sbatch jobs/simulation/parallel_jobs/job7.sh
# sbatch jobs/simulation/parallel_jobs/job8.sh
# sbatch jobs/simulation/parallel_jobs/job9.sh
# sbatch jobs/simulation/parallel_jobs/job10.sh
# sbatch jobs/simulation/parallel_jobs/job11.sh
# sbatch jobs/simulation/parallel_jobs/job12.sh
# sbatch jobs/simulation/parallel_jobs/job13.sh
# sbatch jobs/simulation/parallel_jobs/job14.sh
# sbatch jobs/simulation/parallel_jobs/job15.sh
# sbatch jobs/simulation/parallel_jobs/job16.sh
# sbatch jobs/simulation/parallel_jobs/job17.sh
# sbatch jobs/simulation/parallel_jobs/job18.sh
# sbatch jobs/simulation/parallel_jobs/job19.sh
# sbatch jobs/simulation/parallel_jobs/job20.sh
# sbatch jobs/simulation/parallel_jobs/job21.sh
# sbatch jobs/simulation/parallel_jobs/job22.sh
# sbatch jobs/simulation/parallel_jobs/job23.sh
# sbatch jobs/simulation/parallel_jobs/job24.sh
# sbatch jobs/simulation/parallel_jobs/job25.sh
# sbatch jobs/simulation/parallel_jobs/job26.sh

# n_g > n_d experiments (jobs)
# sbatch jobs/simulation/parallel_jobs/job27.sh
# sbatch jobs/simulation/parallel_jobs/job28.sh
# sbatch jobs/simulation/parallel_jobs/job29.sh
# sbatch jobs/simulation/parallel_jobs/job30.sh
# sbatch jobs/simulation/parallel_jobs/job31.sh
# sbatch jobs/simulation/parallel_jobs/job32.sh
# sbatch jobs/simulation/parallel_jobs/job33.sh
# sbatch jobs/simulation/parallel_jobs/job34.sh
# sbatch jobs/simulation/parallel_jobs/job35.sh
sbatch jobs/simulation/parallel_jobs/job36.sh
sbatch jobs/simulation/parallel_jobs/job37.sh
sbatch jobs/simulation/parallel_jobs/job38.sh
sbatch jobs/simulation/parallel_jobs/job39.sh