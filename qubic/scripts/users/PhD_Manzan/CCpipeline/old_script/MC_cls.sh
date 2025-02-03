#!/bin/bash

#SBATCH --job-name=MC_cls        # Job name                                                                                                                                  

#SBATCH --ntasks=1                    # Run on a single CPU                                                                                                                  

#SBATCH --mem=96gb                    # Job memory request                                                                                                                    

#SBATCH --time=20:00:00               # Time limit hrs:min:sec                                                                                                                

#SBATCH --output=output_100iter.log   # Standard output and error log

export NUMBA_NUM_THREADS=6
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=6
export PYOPERATORS_NO_MPI=6


export QUBIC_DATADIR=/sps/qubic/Users/emanzan/libraries/qubic/qubic/
export QUBIC_DICT=$QUBIC_DATADIR/dicts

source ~/.bashrc
conda activate qubic

python -u /sps/qubic/Users/emanzan/work-dir/CCpipeline/MC_cls.py $1 $2 $3 $4 $5 $6
