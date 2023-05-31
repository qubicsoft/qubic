import os
import time
import numpy as np


ite = [1]
seeds = np.arange(1, 2, 1)
    
for s in seeds:
    for i in ite:
        os.system(f'sbatch cmm.sh {s} {i} 1 1 1')
