#!/bin/bash

#SBATCH --job-name=fitting

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=hpc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=10      # n
#SBATCH --cpus-per-task=2        # N
#SBATCH --mem=15G
#SBATCH --time=0-02:00:00
#SBATCH --output=UWB_cmbdust_nrec2_HFI_%j.log

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load mpich
mpirun -np $SLURM_NTASKS python fit_SO.py