#!/bin/bash
#$ -o /pbs/home/m/mregnier/sps1/QUBIC+/d0/cls/results
#$ -N betas
#$ -q mc_long

export mydir=/pbs/home/m/mregnier/sps1/QUBIC+

export NUMBA_NUM_THREADS=6
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=6
export PYOPERATORS_NO_MPI=6


export QUBIC_DATADIR=/pbs/home/m/mregnier/Libs/qubic/qubic/
export QUBIC_DICT=$QUBIC_DATADIR/dicts

python /pbs/home/m/mregnier/sps1/QUBIC+/d0/cls/MC_cls.py $1 $2 $3 $4 $5 $6
