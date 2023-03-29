#!/bin/bash -l

export NUMBA_NUM_THREADS=5
export MKL_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5
export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export VECLIB_MAXIMUM_THREADS=5
export PYOPERATORS_NO_MPI=5

export NUM_CPUS=20

eval "$(/soft/anaconda3/bin/conda shell.bash hook)"
conda activate myqubic

python CMM_constant_spectral_index.py $1 $2 $3 $4
