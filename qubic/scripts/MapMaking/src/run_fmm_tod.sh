#!/bin/bash
#SBATCH --job-name=FMM-TOD
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=0-01:00:00
#SBATCH --array=0-9
#SBATCH --output=FMM/slurm_logs/fmm_tod_%A_%a.log

N_SIMS=10
BASE_SEED=100

mkdir -p FMM/slurm_logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module load mpich

mpirun -np $SLURM_NTASKS python run_fmm_tod.py "$1" \
    --n_sims "$N_SIMS" \
    --base_seed "$BASE_SEED" \
    --job_id "$SLURM_ARRAY_TASK_ID"
