import os
import time


for i in range(1, 201):
    print(i)
    os.system(f'qsub MC_separe.sh 42 10 {i}')
    #os.system(f'qsub MC_separe.sh {i} 50')
    #os.system(f'qsub MC_separe.sh {i} 100')
