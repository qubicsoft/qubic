#!/usr/bin/env bash

#SBATCH --job-name=FMM

# we ask for n MPI tasks with N cores each on c nodes
#SBATCH --partition=hpc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=6      # n
#SBATCH --cpus-per-task=5        # N
#SBATCH --mem=60G
#SBATCH --time=0-01:00:00

python3 FMM.py $1 $2 $3 $4 $5 $6 