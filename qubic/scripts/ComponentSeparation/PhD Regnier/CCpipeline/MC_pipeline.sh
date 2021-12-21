#!/bin/bash
#$ -o /pbs/home/m/mregnier/sps1/QUBIC+/results
#$ -N pipeline
#$ -q mc_huge

export mydir=/pbs/home/m/mregnier/sps1/QUBIC+

export NUMBA_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export VECLIB_MAXIMUM_THREADS=12
export PYOPERATORS_NO_MPI=12


export QUBIC_DATADIR=/pbs/home/m/mregnier/Libs/qubic/qubic/
export QUBIC_DICT=$QUBIC_DATADIR/dicts

python /pbs/home/m/mregnier/sps1/QUBIC+/MC_pipeline.py $1
