#!/bin/bash

#SBATCH --job-name=ATM

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=htc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=1      # n
#SBATCH --cpus-per-task=10        # N
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=mulitple_jobs_%j.log
#SBATCH --array=1-1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load mpich

mpirun -np $SLURM_NTASKS python Polychromatic_MM_cc.py
