import os
import time
import numpy as np

N=100
nside=0
r=0.000
model='d0'

keyword='ability'


if keyword == 'test':
    # BK
    os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 0 {r} 2 1 d0 BKPL 0.0')
    os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 0 {r} 2 1 d0 BKPL 1.0')
    os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 0 {r} 2 1 d0 SO 0.5')
    os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 0 {r} 2 1 d0 SO 1.0')
    os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 0 {r} 2 1 d0 S4 0.1')
    os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 0 {r} 2 1 d0 S4 1.0')

if keyword == 'lensing':
    for i in np.arange(0, 1.1, 0.1):
        os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} {nside} {r} 2 1 {model} S4 0.03 {i}')
        os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} {nside} {r} 2 1 {model} SO 0.1 {i}')
        os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} {nside} {r} 2 1 {model} BKPL 0.02 {i}')

if keyword == 'finddeco':
    for i in np.arange(1, 11, 1):
        os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 8 {r} {i} 1 d6 BKPL 0.02 1.0')

if keyword == 'BI':
    for i in np.arange(2, 9, 1):
        os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 8 {r} 6 {i} d6 BKPL 0.02 1.0')

if keyword == 'comparison':
    for i in [9, 10]:
        os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 8 {r} {i} 1 d6 S4 0.1')
        os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 8 {r} {i} 5 d6 S4 0.1')
        os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 8 {r} {i} 1 d6 SO 0.5')
        os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 8 {r} {i} 5 d6 SO 0.5')
        os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 8 {r} {i} 1 d6 BKPL 1.0')
        os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 8 {r} {i} 5 d6 BKPL 1.0')


if keyword == 'ability':
    os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 8 0.000 7 5 d6 S4 0.1')
    os.system(f'sbatch -p gpu --gres=gpu:1 xFC.sh {N} 8 0.005 7 5 d6 S4 0.1')
