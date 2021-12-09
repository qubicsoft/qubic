import os
import time
import numpy as np

for i in range(1, 6):
    print("ite : {}".format(i))
    #os.system(f'qsub MC_fraction.sh 2 0.5 1.24 1.84 265 145 40 {i} 0')
    os.system(f'qsub -pe multicores 12 MC_fraction.sh 1 0.5 1.54 1.54 265 145 40 {i} 1')
    time.sleep(0.1)
