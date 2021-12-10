#!/bin/bash
#$ -o /pbs/home/m/mregnier/sps1/QUBIC+/results
#$ -N compsep
#$ -q huge

export mydir=/pbs/home/m/mregnier/sps1/QUBIC+

export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYOPERATORS_NO_MPI=1


export QUBIC_DATADIR=/pbs/home/m/mregnier/Libs/qubic/qubic/
export QUBIC_DICT=$QUBIC_DATADIR/dicts

python /pbs/home/m/mregnier/sps1/QUBIC+/MC_compsep.py $1 $2 $3 $4
