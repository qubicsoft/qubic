#!/bin/bash
#SBATCH --job-name=FMM
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=0-05:00:00
#SBATCH --output=FMM/slurm_logs/multiple_jobs_%A_%a.log
#SBATCH --array=1-300

mkdir -p FMM/slurm_logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module load mpich

# Deterministic unique seed for each array element
# SEED=$((SLURM_JOB_ID * 10 + SLURM_ARRAY_TASK_ID))
SEED=$((SLURM_ARRAY_TASK_ID))

mpirun -np $SLURM_NTASKS python run_fmm.py "$1" "$2" --seed "$SEED"
