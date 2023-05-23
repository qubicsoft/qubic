import os
import time
import numpy as np


ite = 40
seed = np.arange(1, 2, 1)

for iseed in seed:
    for i in range(1, ite + 1):
        print(iseed, i)
        os.system(f'sbatch FMM.sh {iseed} {i} wide 1 0 0')