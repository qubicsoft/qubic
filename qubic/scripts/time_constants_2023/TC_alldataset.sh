#!/bin/bash
#SBATCH -t 00-12:00
#SBATCH -m 16GB
#SBATCH -N 1 # 1 nodes
#SBATCH -n 32 # 32 tasks
#SBATCH -c 1 # 1 core per task

conda activate myqubic

srun -N 1 -n 1 python ime_constants_computation_alldatasets.py 0 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 1 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 2 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 3 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 4 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 5 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 6 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 7 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 8 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 9 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 10 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 11 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 12 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 13 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 14 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 15 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 16 &

wait
