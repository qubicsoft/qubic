#!/bin/bash

#SBATCH --job-name=spec

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=hpc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=1      # n
#SBATCH --cpus-per-task=4        # N
#SBATCH --mem=40G
#SBATCH --time=0-03:00:00
#SBATCH --output=mulitple_jobs_%j.log

PYTHON_SCRIPT=spectrum.py

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mpirun -np ${SLURM_NTASKS} python ${PYTHON_SCRIPT} $1

