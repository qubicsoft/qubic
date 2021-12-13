import os
import time
import numpy as np

for i in range(1, 201):
    print("ite : {}".format(i))

    # CMBS4 (one beta and 2 betas at infinite resolution)
    os.system(f'qsub MC_fullpipeline.sh 0 {i} S4 0 1.44 1.64 260')
    os.system(f'qsub MC_fullpipeline.sh 0 {i} BI 0 1.44 1.64 260')
    time.sleep(0.1)
