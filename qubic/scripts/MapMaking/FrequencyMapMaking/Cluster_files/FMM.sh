#!/bin/bash -l

#SBATCH --job-name=FMM

# we ask for n MPI tasks with N cores each on c nodes
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=4      # n
#SBATCH --cpus-per-task=4        # N
#SBATCH --mem-per-cpu=1500MB
#SBATCH --partition=bigmem

# run for five minutes d-hh:mm:ss
#SBATCH --time=0-01:00:00

### nodes [1-16] -> --mca btl_tcp_if_include enp24s0f0np0
### nodes [17-23] -> --mca btl_tcp_if_include enp24s0f1

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

mpirun $interface -np $SLURM_NTASKS python FMM.py $1 $2 $3 $4 $5 $6 $7
