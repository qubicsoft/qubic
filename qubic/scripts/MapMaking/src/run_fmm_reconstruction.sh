#!/bin/bash
#SBATCH --job-name=FMM-REC
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=0-02:00:00
#SBATCH --array=0-9
#SBATCH --output=FMM/slurm_logs/fmm_rec_%A_%a.log

# $1 = params.yaml
# $2 = directory containing the TOD files  (default: FMM/test/Dict)
# $3 = noise job_id prefix used in run_fmm_noise_real.sh (default: 0000)
#
# Usage: sbatch run_fmm_reconstruction.sh params.yaml FMM/test/Dict 0000
#   Array range (0-9) must match --n_real used during noise generation.

TOD_DIR="${2:-FMM/test/Dict}"
NOISE_JOB_ID="${3:-0000}"

mkdir -p FMM/slurm_logs

printf -v TASK_ID '%04d' "$SLURM_ARRAY_TASK_ID"
TOD_FILE="${TOD_DIR}/tod_real_${NOISE_JOB_ID}_${TASK_ID}.h5"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module load mpich

mpirun -np $SLURM_NTASKS python run_fmm_reconstruction.py "$1" \
    --tod_file "$TOD_FILE"
