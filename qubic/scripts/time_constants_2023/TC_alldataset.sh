#!/bin/bash

conda activate myqubic

srun -N 1 -n 1 python time_constants_computation_alldatasets.py 0 2 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 2 4 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 4 6 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 6 8 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 8 10 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 10 12 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 12 14 &
srun -N 1 -n 1 python time_constants_computation_alldatasets.py 14 17&

wait
