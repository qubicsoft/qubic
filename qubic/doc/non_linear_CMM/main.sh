#!/bin/bash

#SBATCH --job-name=CMM32

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=hpc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=1      # n
#SBATCH --cpus-per-task=4        # N
#SBATCH --mem=10G
#SBATCH --time=1-10:00:00
#SBATCH --output=mulitple_jobs_%j.log

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python pipeline_non_linear.py $1 $2 $3 $4 $5 $6