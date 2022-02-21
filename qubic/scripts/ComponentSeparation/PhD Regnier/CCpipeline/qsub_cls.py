import os
import time
import numpy as np

truenub=[100, 120, 140, 150, 160, 180, 200, 225, 250]#[100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 230, 250, 270]
nb_iteration=3

for i in range(91, 151):
    print(i)

    for j in range(len(truenub)):

        # Fix temp and no iib
        os.system(f'qsub -pe multicores 6 MC_cls.sh {nb_iteration} {i} {truenub[j]} 1 1 1')
        os.system(f'qsub -pe multicores 6 MC_cls.sh {nb_iteration} {i} {truenub[j]} 1 0 1')

        # Fix temp and no iib
        os.system(f'qsub -pe multicores 6 MC_cls.sh {nb_iteration} {i} {truenub[j]} 1 1 5')
        os.system(f'qsub -pe multicores 6 MC_cls.sh {nb_iteration} {i} {truenub[j]} 1 0 5')

        # Fix temp and no iib
        os.system(f'qsub -pe multicores 6 MC_cls.sh {nb_iteration} {i} {truenub[j]} 1 1 10')
        os.system(f'qsub -pe multicores 6 MC_cls.sh {nb_iteration} {i} {truenub[j]} 1 0 10')
