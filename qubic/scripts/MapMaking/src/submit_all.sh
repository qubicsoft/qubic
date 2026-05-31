#!/bin/bash
# Submit the full FMM pipeline: TOD generation + N reconstructions [+ spectrum fit].
#
# Usage: bash submit_all.sh params.yaml N_REAL [fit_params.yaml]
#
#   params.yaml      : QUBIC parameters file
#   N_REAL           : number of noise realizations to reconstruct
#   fit_params.yaml  : (optional) fit parameters file; required when
#                      Pipeline.spectrum=True in params.yaml

PARAMS=${1:?Usage: bash submit_all.sh params.yaml N_REAL [fit_params.yaml]}
N_REAL=${2:?Usage: bash submit_all.sh params.yaml N_REAL [fit_params.yaml]}
FIT_PARAMS=${3:-}

FOLDERNAME=$(python -c "import yaml; print(yaml.safe_load(open('${PARAMS}'))['foldername'])")
DO_SPECTRUM=$(python -c "import yaml; p=yaml.safe_load(open('${PARAMS}')); print(p.get('Pipeline', {}).get('spectrum', False))")
NOISELESS_TOD="FMM/${FOLDERNAME}/TOD/tod_combined_0000.h5"

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

if [ "${DO_SPECTRUM}" = "True" ]; then
    echo ""
    if [ -z "${FIT_PARAMS}" ]; then
        echo "WARNING: Pipeline.spectrum=True but no fit_params.yaml given (arg 3). Skipping step 3."
        echo "Done. Monitor with: squeue -j ${JOB1},${JOB2}"
    else
        echo "=== Step 3: spectrum fit (starts after all reconstructions) ==="
        JOB3=$(sbatch --parsable \
               --dependency=afterok:${JOB2} \
               run_fit_fmm.sh "${FIT_PARAMS}")
        echo "  Submitted job ${JOB3}"
        echo ""
        echo "Done. Monitor with: squeue -j ${JOB1},${JOB2},${JOB3}"
    fi
else
    echo ""
    echo "Done. Monitor with: squeue -j ${JOB1},${JOB2}"
fi
