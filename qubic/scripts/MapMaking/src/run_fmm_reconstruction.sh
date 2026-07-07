#!/bin/bash
#SBATCH --job-name=FMM-REC
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=0-10:00:00
#SBATCH --output=FMM/slurm_logs/fmm_rec_%A_%a.log

# $1 = params.yaml
# $2 = path to the noiseless TOD file (tod_combined_0000.h5)
# Array range is injected by submit_all.sh: --array=0-$((N_REAL - 1))

BASE_SEED=100

mkdir -p FMM/slurm_logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module load mpich

mpirun -np $SLURM_NTASKS python run_fmm_reconstruction.py "$1" \
    --noiseless_tod "$2" \
    --real_id       "$SLURM_ARRAY_TASK_ID" \
    --base_seed     "$BASE_SEED"
