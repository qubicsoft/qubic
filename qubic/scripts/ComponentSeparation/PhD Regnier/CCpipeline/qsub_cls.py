import os
import time
import numpy as np

nubreak=[150]#[100, 120, 150, 170, 200, 220, 250]#[100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 230, 250, 270]
r=[0]
nb_iteration=10

NB_CORES=8
get_r_apply=1

if get_r_apply == 0:
    for i in range(1, 201):
        print(i)
        for ir in r :
            for inubreak in nubreak :
                os.system(f'qsub -pe multicores {NB_CORES} MC_cls.sh {nb_iteration} {i} {inubreak} {ir}')
        #os.system(f'qsub -pe multicores {NB_CORES} MC_cls.sh {nb_iteration} {i} {model_to_fit} 0.000 0 10')

else:
    for ir in r :
        for inubreak in nubreak :
            iterations=[1, 2, 3, 5, 10, 20, 30, 40, 50, 100, 150, 200]
            for ite in iterations:
                os.system(f'qsub -pe multicores {NB_CORES} MC_cls.sh {nb_iteration} {ite} {inubreak} {ir}')
