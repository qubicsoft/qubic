#!/bin/bash

#SBATCH --job-name=MC_cls_test        # Job name                                                                                                                               

#SBATCH --mem=18gb                    # Job memory request                                                                                                                    

#SBATCH --time=24:00:00               # Time limit hrs:min:sec                                                                                                                

#SBATCH --output=output_single.log   # Standard output and error log

#export NUMBA_NUM_THREADS=6
#export MKL_NUM_THREADS=6
#export NUMEXPR_NUM_THREADS=6
#export OMP_NUM_THREADS=6
#export OPENBLAS_NUM_THREADS=6
#export VECLIB_MAXIMUM_THREADS=6
#export PYOPERATORS_NO_MPI=6


export QUBIC_DATADIR=/sps/qubic/Users/emanzan/libraries/qubic/qubic/
export QUBIC_DICT=$QUBIC_DATADIR/dicts

source ~/.bashrc
conda activate qubic

python -u /sps/qubic/Users/emanzan/work-dir/CCpipeline/MC_cls.py $1 $2 $3 $4 $5 $6
