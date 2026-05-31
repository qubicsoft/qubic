#!/bin/bash
#SBATCH --job-name=FMM-REC
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=0-02:00:00
#SBATCH --array=0-99
#SBATCH --output=FMM/slurm_logs/fmm_rec_%A_%a.log

# $1 = params.yaml
# $2 = path to the noiseless TOD file (tod_combined_0000.h5)
# Array range must match N_REAL in submit_all.sh

BASE_SEED=100

mkdir -p FMM/slurm_logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module load mpich

mpirun -np $SLURM_NTASKS python run_fmm_reconstruction.py "$1" \
    --noiseless_tod "$2" \
    --real_id       "$SLURM_ARRAY_TASK_ID" \
    --base_seed     "$BASE_SEED"
