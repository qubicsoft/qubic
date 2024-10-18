#!/bin/bash

#SBATCH --job-name=FMM

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=htc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=2      # n
#SBATCH --cpus-per-task=2        # N
#SBATCH --mem=50G
#SBATCH --time=0-05:00:00
#SBATCH --output=mulitple_jobs_%j.log
#SBATCH --array=1-1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load mpich

mpirun -np $SLURM_NTASKS python run_fmm.py $1 $2
