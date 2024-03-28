#!/bin/bash

#SBATCH --job-name=BBPip

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=bigmem
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=2      # n
#SBATCH --cpus-per-task=1        # N
#SBATCH --mem=30G
#SBATCH --time=0-10:00:00
#SBATCH --output=mulitple_jobs_%j.log
#SBATCH --array=1-50

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mpirun -np $SLURM_NTASKS python main.py