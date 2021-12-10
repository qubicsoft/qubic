import os
import time
import numpy as np

for i in range(1, 21):
    print("ite : {}".format(i))

    # CMBS4 (one beta and 2 betas at infinite resolution)
    os.system(f'qsub MC_generatemaps.sh 0 1.44 1.64 268 {i} 0 1')
    os.system(f'qsub MC_generatemaps.sh 0 1.44 1.64 268 {i} 0 0')
    time.sleep(0.1)
