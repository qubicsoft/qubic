#!/bin/bash
#SBATCH --job-name=CMM
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=1-10:00:00
#SBATCH --output=CMM/slurm_logs/multiple_jobs_%A_%a.log
#SBATCH --array=1-300

mkdir -p CMM/slurm_logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module load mpich

# Deterministic unique seed for each array element
SEED=$((SLURM_JOB_ID * 10 + SLURM_ARRAY_TASK_ID))

mpirun -np $SLURM_NTASKS python run_cmm.py "$1" "$2" --seed "$SEED"
