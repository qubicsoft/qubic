#!/bin/bash -l

#eval "$(/soft/anaconda3/bin/conda shell.bash hook)"
conda activate myqubic

python time_constants_computation_alldatasets.py
