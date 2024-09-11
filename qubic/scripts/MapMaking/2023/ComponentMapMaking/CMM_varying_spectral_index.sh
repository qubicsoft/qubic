#!/bin/bash

export NUMBA_NUM_THREADS=5
export MKL_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5
export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export VECLIB_MAXIMUM_THREADS=5
export PYOPERATORS_NO_MPI=5

eval "$(/soft/anaconda3/bin/conda shell.bash hook)"
conda activate myqubic

python CMM_varying_spectral_index.py
