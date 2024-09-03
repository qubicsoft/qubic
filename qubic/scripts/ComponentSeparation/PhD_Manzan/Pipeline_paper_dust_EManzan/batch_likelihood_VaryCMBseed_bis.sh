#!/bin/sh
##!/bin/bash

##SBATCH --array=0-9

# SBATCH --time=16:00:00               # Time limit hrs:min:sec    

# SBATCH --job-name=Test_map_Cls        # Job name  

# SBATCH --ntasks=1                    # Run a single task   

# SBATCH --mem=6G                   # Job memory request                                                                                            
##SBATCH --output=test.log

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

python -u ./Dls_and_Posterior_likelihood_universalcode_bis.py $1
