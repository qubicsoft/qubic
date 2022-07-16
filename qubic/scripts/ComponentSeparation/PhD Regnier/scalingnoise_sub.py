import os
import time
import numpy as np

N=200


os.system(f'sbatch -p gpu --gres=gpu:1 scalingnoise_qubic.sh {N} BKPL 1')
os.system(f'sbatch -p gpu --gres=gpu:1 scalingnoise_qubic.sh {N} S4 1')
os.system(f'sbatch -p gpu --gres=gpu:1 scalingnoise_qubic.sh {N} SO 1')

bands=np.arange(1, 11, 1)
for i in bands:
    os.system(f'sbatch -p gpu --gres=gpu:1 scalingnoise_qubic.sh {N} QUBIC {i}')
