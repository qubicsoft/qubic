#!/bin/bash

#SBATCH --output=output_100_10.log
#SBATCH --mem=20gb                    # Job memory request
#SBATCH --time=10:00:00
#SBATCH --job-name=d0bis    # Job name

export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYOPERATORS_NO_MPI=1


export QUBIC_DATADIR=/pbs/home/m/mregnier/Libs/qubic/qubic/
export QUBIC_DICT=$QUBIC_DATADIR/dicts


python /pbs/home/m/mregnier/sps1/scalingnoise/scalingnoise_qubic.py $1 $2 $3
