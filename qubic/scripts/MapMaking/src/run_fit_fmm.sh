#!/bin/bash

#SBATCH --job-name=Fit-FMM

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=htc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=1      # n
#SBATCH --cpus-per-task=10        # N
#SBATCH --mem=10G
#SBATCH --time=0-02:00:00
#SBATCH --output=FMM/slurm_logs/multiple_jobs_%A_%a.log
###SBATCH --array=1-1

mkdir -p FMM/slurm_logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load mpich

mpirun -np $SLURM_NTASKS python run_fit_fmm.py $1

