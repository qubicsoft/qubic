#!/bin/bash
#$ -o /pbs/home/m/mregnier/sps1/QUBIC+/results
#$ -N maps
#$ -q mc_highmem

export mydir=/pbs/home/m/mregnier/sps1/QUBIC+

export NUMBA_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export PYOPERATORS_NO_MPI=2


export QUBIC_DATADIR=/pbs/home/m/mregnier/Libs/qubic/qubic/
export QUBIC_DICT=$QUBIC_DATADIR/dicts

python /pbs/home/m/mregnier/sps1/QUBIC+/MC_generatemaps.py $1 $2 $3 $4 $5 $6 $7
python /pbs/home/m/mregnier/sps1/QUBIC+/MC_compsep.py $1 $2 $3 $4 $5 $6 $7
#rm -r /pbs/home/m/mregnier/sps1/QUBIC+/results/onereals_maps_fwhm*
