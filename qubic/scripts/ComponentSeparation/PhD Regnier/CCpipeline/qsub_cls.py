import os
import time
import numpy as np

truenub=[100, 125, 150, 175, 200, 225, 250]
nb_iteration=1

for i in range(1, 101):
    print(i)

    for j in range(len(truenub)):
        os.system(f'qsub -pe multicores 6 MC_cls.sh {nb_iteration} {i} {truenub[j]} 1 0')
