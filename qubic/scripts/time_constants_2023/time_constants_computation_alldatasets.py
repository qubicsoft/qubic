# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import glob
import numpy as np

from qubic import time_constants_tools as tct

base_save_path = '/sps/qubic/Users/nahuelmg'
save_path = base_save_path+'/time_constants_results/alldatasets_to_may_2023/'
if os.path.exists(save_path):
    print('Save path:',save_path)
else:
    print('Insert save path.')

base_dir = '/sps/qubic/Data/Calib-TD/'

days = ['2019-06-27','2019-07-03','2019-11-12','2019-11-14','2020-07-24','2020-07-25',
        '2020-07-27','2020-10-16','2020-11-12','2022-04-06','2022-08-18','2022-08-24',
        '2023-03-02','2023-03-07','2023-03-31','2023-04-03','2023-04-17']

keywords = ['*TimeCst*','*Fibers*','*New*'+'*TimeCstScript*','*TimeCstScript*','*TimeCstScript*',
            '*TimeCstScript*','*TimeCstScript*','*TimeCstScript*','*carbonfibre*',
            '*NoiseMeasurement_FileDuration_CalSourceON180*','*timeconstant_TimePerPos*','*TimeCstScript*',
            '*calsource','*DomeOpen-Amp*','*CF-0*','*CFiber_square*','*carbon-fiber_0.2*']

fmods = [[0.25],[0.25],[0.6],[0.6],[0.6],[0.6],[0.6],[0.6],[0.8],[0.2],[None],[0.3],[0.2],
         [0.2, 0.7],[0.6, 0.6, 0.2, 0.2],[0.2],[0.2]]

dcs = [[30],[33.33333],[33.33333],[33.33333],[33.33333],[33.33333],[33.33333],[33.33333],
       [33],[60],[None],[30],[66],[66, 66],[33, 66, 66, 66],[33],[33]]

for i,day in enumerate(days):
    data_dir = base_dir + day + '/'
    thedirs = np.sort(glob.glob(data_dir+keywords[i]))

    print('Available files to analyze time constants through square modulation corresponding to {}'.format(day))
    
    for j,thedir in enumerate(thedirs):
        print(j,thedir)
        
    if len(fmods[i])==1:

        fmod = fmods[i][0]
        dc = dcs[i][0]

        for thedatadir in thedirs:
            
            d = tct.compute_tc_squaremod(thedatadir, fmod = fmod, dutycycle = dc, save_path = save_path)
        
    elif len(fmods[i])>1 and len(fmods[i])==len(thedirs):
        
        for k in range(len(fmods[i])):
            
            fmod = fmods[i][k]
            dc = dcs[i][k]
            thedatadir = thedirs[k]
            
            d = tct.compute_tc_squaremod(thedatadir, fmod = fmod, dutycycle = dc, save_path = save_path)
