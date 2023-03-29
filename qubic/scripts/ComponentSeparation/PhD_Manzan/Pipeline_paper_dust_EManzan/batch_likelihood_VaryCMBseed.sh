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

#python -u /sps/qubic/Users/emanzan/work-dir/Pipeline_multi_dust/Map_generation_and_MCcompsep.py $1 $2 $3 $4 $5 $6
#python -u ./Cls_Likelihood_visualization_VaryingCMBseed.py $1 $2 $3 $4 $5 $6 $7 $8 $9
python -u ./Dls_and_Posterior_likelihood.py

# python -u /sps/qubic/Users/emanzan/work-dir/Pipeline_multi_dust/Test_map_generation.py $1 #$2 $3 $4 $5 $6 $7
# python -u /sps/qubic/Users/emanzan/work-dir/CCpipeline/MC_compsep_to_cls_bandint.py $1 ${SLURM_ARRAY_TASK_ID} $2 $3 $4 $5 $6 $7
