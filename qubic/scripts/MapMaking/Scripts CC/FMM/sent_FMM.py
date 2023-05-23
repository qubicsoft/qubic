import os
import time
import numpy as np


ite = 20
seed = np.arange(1, 2, 1)
pointings = [12000]

for p in pointings:
    for iseed in seed:
        for i in range(1, ite + 1):
            print(iseed, i)
            os.system(f'sbatch FMM.sh {iseed} {i} 220 1 0 0 {p}')