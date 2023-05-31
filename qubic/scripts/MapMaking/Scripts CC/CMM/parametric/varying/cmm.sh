#!/bin/bash

#SBATCH --job-name=CMM

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=quiet
#SBATCH --nodes=2                # c
#SBATCH --ntasks-per-node=4      # n
#SBATCH --cpus-per-task=1        # N
#SBATCH --mem=50G
#SBATCH --time=0-01:00:00
#SBATCH --output=mulitple_jobs_%j.log

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo $(hostname)

if [[ $(hostname) == "node01" || $(hostname) == "node02" || $(hostname) == "node03" || $(hostname) == "node04" || $(hostname) == "node05" 
                              || $(hostname) == "node06" || $(hostname) == "node07" || $(hostname) == "node08" || $(hostname) == "node09" 
                              || $(hostname) == "node10" || $(hostname) == "node11" || $(hostname) == "node12" || $(hostname) == "node13" 
                              || $(hostname) == "node14" || $(hostname) == "node15" || $(hostname) == "node16" ]]; then
    
    interface="--mca btl_tcp_if_include enp24s0f0np0"
else
    interface="-mca btl_tcp_if_include enp24s0f1"
fi

echo $interface

eval "$(/soft/anaconda3/bin/conda shell.bash hook)"
conda activate myqubic

mpirun $interface -np $SLURM_NTASKS python cmm_mpi.py $1 $2
