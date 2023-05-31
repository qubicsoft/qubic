import os
import time
import numpy as np
import sys

n_min = int(sys.argv[1])
n_max = int(sys.argv[2])

config = ["wide", "two"]

nside_fgb=0
for i in range(n_min, n_max):
    for j in range(len(config)):
        os.system(f'sbatch -t 1-00:00 -n 1 --mem 50G mapmaking.sh {i} {config[j]}')

