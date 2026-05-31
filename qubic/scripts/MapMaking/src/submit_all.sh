#!/bin/bash
# Submit the full FMM pipeline: noiseless TOD generation + N reconstructions.
#
# Usage: bash submit_all.sh params.yaml N_REAL
#
#   params.yaml  : QUBIC parameters file
#   N_REAL       : number of noise realizations to reconstruct

PARAMS=${1:?Usage: bash submit_all.sh params.yaml N_REAL}
N_REAL=${2:?Usage: bash submit_all.sh params.yaml N_REAL}

FOLDERNAME=$(python -c "import yaml; print(yaml.safe_load(open('${PARAMS}'))['foldername'])")
NOISELESS_TOD="FMM/${FOLDERNAME}/Dict/tod_combined_0000.h5"

echo "=== Step 1: noiseless TOD generation ==="
echo "  Output: ${NOISELESS_TOD}"
JOB1=$(sbatch --parsable run_fmm_tod.sh "${PARAMS}")
echo "  Submitted job ${JOB1}"

echo ""
echo "=== Step 2: ${N_REAL} reconstructions (starts after job ${JOB1}) ==="
JOB2=$(sbatch --parsable \
       --dependency=afterok:${JOB1} \
       --array=0-$((N_REAL - 1)) \
       run_fmm_reconstruction.sh "${PARAMS}" "${NOISELESS_TOD}")
echo "  Submitted array job ${JOB2} (tasks 0–$((N_REAL - 1)))"

echo ""
echo "Done. Monitor with: squeue -j ${JOB1},${JOB2}"
