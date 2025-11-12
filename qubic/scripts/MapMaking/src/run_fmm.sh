#!/bin/bash

#SBATCH --job-name=FMM

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=htc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=1      # n
#SBATCH --cpus-per-task=4        # N
#SBATCH --mem=50G
#SBATCH --time=0-05:00:00
#SBATCH --output=mulitple_jobs_%j.log
#SBATCH --array=1-10

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load mpich

# Unique seed base for this job submission
# (current epoch time in seconds mod 100000)
SEED_BASE=$(( $(date +%s) % 100000 ))

# Combine job array ID and base offset
SEED=$(( SEED_BASE + SLURM_ARRAY_TASK_ID ))

mpirun -np $SLURM_NTASKS python run_fmm.py $1 $2 --seed "$SEED"
