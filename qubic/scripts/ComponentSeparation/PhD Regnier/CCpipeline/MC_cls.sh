#!/bin/bash
#$ -o /pbs/home/m/mregnier/sps1/QUBIC+/d0/cls/results
#$ -N nss_fixsync
#$ -q mc_huge

export mydir=/pbs/home/m/mregnier/sps1/QUBIC+

export NUMBA_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export PYOPERATORS_NO_MPI=8


export QUBIC_DATADIR=/pbs/home/m/mregnier/Libs/qubic/qubic/
export QUBIC_DICT=$QUBIC_DATADIR/dicts

####### Arguments ########
    # 1/ Number of iterations
    # 2/ Name of iterations
    # 3/ nubreak
    # 4/ Value of r
    # 5/ Bandpass integration
    # 6/ Fix synchrotron
    # 7/ # of sub-bands

#python /pbs/home/m/mregnier/sps1/QUBIC+/d0/cls/MC_cls_new.py $1 $2 $3 $4 $5 $6 $7
python /pbs/home/m/mregnier/sps1/QUBIC+/d0/cls/get_r.py $1 $2 $3 $4 $5 $6 $7
