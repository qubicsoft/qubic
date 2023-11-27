#!/bin/sh
# Template for running batch scripts on cca.
# From command line run : sbatch -t 01:00:00 -n 1 --gres=gpu:v100:1 --mem 32G batch_script.sh
   
# SBATCH --job-name=test        # Job name  
                                                                                          
export NUMBA_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export VECLIB_MAXIMUM_THREADS=12
export PYOPERATORS_NO_MPI=12

export QUBIC_DATADIR=/sps/qubic/Users/emanzan/libraries/qubic/qubic/
export QUBIC_DICT=$QUBIC_DATADIR/dicts

source ~/.bashrc
conda activate qubic

python -u ./Template_script_diagnostic.py
