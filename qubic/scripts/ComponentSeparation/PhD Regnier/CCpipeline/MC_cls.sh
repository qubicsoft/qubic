#!/bin/bash

#SBATCH --output=output_100_10.log

export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYOPERATORS_NO_MPI=1


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
    # 8/ Model that you want to fit for



python /pbs/home/m/mregnier/sps1/QUBIC+/d0/cls/MC_cls_new.py $1 $2 $3 $4 $5 $6 $7 $8 $9
#python /pbs/home/m/mregnier/sps1/QUBIC+/d0/cls/get_r.py $1 $2 $3 $4 $5 $6 $7 $8 $9
