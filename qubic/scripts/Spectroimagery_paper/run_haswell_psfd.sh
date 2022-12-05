#!/bin/bash
#SBATCH -N 4
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J Nfsb40DiffNrec_5
#SBATCH --mail-user=mgamboa@fcaglp.unlp.edu.ar
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH -o %j.out

# using qubicsoft common dependences
source /project/projectdirs/qubic/Software/cori/cori-python-env.sh
# OpenMP settings:
export OMP_NUM_THREADS=1
# enabling nodes communications
unset PYOPERATORS_NO_MPI
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Write the JobID in the output file
echo 'JobID : ' ${SLURM_JOBID}

# Directory containing the python script to run
SCRIPT_DIR=/global/homes/m/mmgamboa/qubicsoft/qubic/scripts/Spectroimagery_paper

# Create a directory called JobID where all output files will be sent
OUT_DIR=/project/projectdirs/qubic/mmgamboa/${SLURM_JOBID}
mkdir ${OUT_DIR}

# Arguments for the python script
SIMU_NAME=PointSource
AMP=1e5
DICT=$DIRDICT/spectroimaging_article_psf.dict
NFSUB=40
NFRECON=[5,]
POINTINGS=8500
TOL=1e-2
NREALS=1	
NU0=150
SHIFTNU=-17
#sbcast --compress=lz4 ${SCRIPT_DIR}/spectroimaging_pointsource_psf_dif_MPI.py /tmp/SpIm_ps_dif_MPI.py
# Run the scrip
srun -n 64 -c 4 --cpu_bind=cores python ${SCRIPT_DIR}/spectroimaging_pointsource_psf_dif_MPI.py ${OUT_DIR} ${SIMU_NAME} ${AMP} ${DICT} ${SLURM_JOBID} ${NFSUB} ${NFRECON} ${POINTINGS} ${TOL} ${NREALS} ${NU0} ${SHIFTNU}

# Copy the output file in the output directory
mv ${SLURM_JOBID}.out ${OUT_DIR}/.
mv ${SLURM_JOBID}.log ${OUT_DIR}/.
