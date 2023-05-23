import sys
import os
import glob
import gc
import numpy as np
from pathlib import Path
from importlib import reload
from matplotlib.pyplot import *
from qubic import time_constants_tools as tct

# +
base_dir = '/sps/qubic/Data/Calib-TD/' # where to read the datasets
base_save_path = '/sps/qubic/Users/nahuelmg'

# base_dir = '/media/nahue/files_hdd/heavy-data/'
# base_save_path = '/media/nahue/files_hdd/heavy-data'

specific_save_folder = '/time_constants_results/all_TC_datasets_up_to_may'

save_path = base_save_path+specific_save_folder # where to store the results

Path(save_path).mkdir( parents=True, exist_ok=True )
    
# print(save_path)

# +
days = [['2019-06-27'],['2019-07-03'],['2019-11-12'],['2019-11-14'],['2020-07-24'],['2020-07-25'],
	['2020-07-27'],['2020-10-16'],['2020-11-12'],['2022-04-06'],['2022-08-18'],['2022-08-24'],
	['2023-03-02'],['2023-03-07'],['2023-03-31'],['2023-04-03'],['2023-04-17']]

keywords = [['*TimeCst*'],['*Fibers*'],['*New*'+'*TimeCstScript*'],['*TimeCstScript*'],['*TimeCstScript*'],
	    ['*TimeCstScript*'],['*TimeCstScript*'],['*TimeCstScript*'],['*carbonfibre*'],
	    ['*NoiseMeasurement_FileDuration_CalSourceON180*'],['*timeconstant_TimePerPos*'],['*TimeCstScript*'],
	    ['*calsource'],['*DomeOpen-Amp*'],['*CF-0*'],['*CFiber_square*'],['*carbon-fiber_0.2*']]

fmods = [[0.25],[0.25],[0.6],[0.6],[0.6],[0.6],[0.6],[0.6],[0.8],[0.2],[None],[0.3],[0.2],
	 [0.2, 0.7],[0.6, 0.6, 0.2, 0.2],[0.2],[0.2]]

dcs = [[30],[33.33333],[33.33333],[33.33333],[33.33333],[33.33333],[33.33333],[33.33333],
       [33],[60],[None],[30],[66],[66, 66],[33, 66, 66, 66],[33],[33]]

# +
index_ini = sys.argv[0]
index_fin = sys.argv[1]
# index_ini = 0
# index_fin = 1

days = days[index_ini:index_fin]
keywords = keywords[index_ini:index_fin]
fmods = fmods[index_ini:index_fin]
dcs = dcs[index_ini:index_fin]

# +
# reload(tct)

doplot = None
save_dict = True
saveplot = ['focal_plane', 'folded_data']
force_sync = True

for i in range(len(days)):#enumerate(zip(keywords,days)):  #, (keyword, day)
    
    day = days[i][0]
    keyword = keywords[i][0]
    
    data_dir = base_dir + day + '/'
    thedirs = np.sort(glob.glob(data_dir+keyword))

    print('Available files to analyze time constants through square modulation corresponding to {}'.format(day))
    
    if len(thedirs) == 0 : # empty list
        
        print('None. No datasets were found.')

    else:

        for j,thedir in enumerate(thedirs):
            print('{}/{}'.format(j+1,len(thedirs)),thedir)
        
        
    if len(fmods[i])==1:

        fmod = fmods[i][0]
        dc = dcs[i][0]

        for j,thedatadir in enumerate(thedirs):
                        
#             try:
                
            print('{}/{}'.format(j+1,len(thedirs)),'TC computation started for the dataset {}'.format(str.split(thedatadir,'/')[-1]))

            d = tct.compute_tc_squaremod(thedatadir, save_dict = save_dict, fmod = fmod, dutycycle = dc, save_path = save_path, doplot = doplot, saveplot = saveplot, force_sync = force_sync)

            print('{}/{}'.format(j+1,len(thedirs)),'TC computation finished for the dataset {}'.format(str.split(thedatadir,'/')[-1]))
                
#             except:
                
#                 print('{}/{}'.format(j+1,len(thedirs)),'tct.compute_tc_squaremod did not work for the dataset {}'.format(str.split(thedatadir,'/')[-1]))
            
    elif len(fmods[i])>1 and len(fmods[i])==len(thedirs):
        
        for k in range(len(fmods[i])):
            
            fmod = fmods[i][k]
            dc = dcs[i][k]
            thedatadir = thedirs[k]
            
#             try:
                
            print('{}/{}'.format(j+1,len(thedirs)),'TC computation started for the dataset {}'.format(str.split(thedatadir,'/')[-1]))

            d = tct.compute_tc_squaremod(thedatadir, save_dict = save_dict, fmod = fmod, dutycycle = dc, save_path = save_path, doplot = doplot, saveplot = saveplot, force_sync = force_sync)

            print('{}/{}'.format(j+1,len(thedirs)),'TC computation finished for the dataset {}'.format(str.split(thedatadir,'/')[-1]))

#             except:
                
#                 print('{}/{}'.format(j+1,len(thedirs)),'tct.compute_tc_squaremod did not work for the dataset {}'.format(str.split(thedatadir,'/')[-1]))
# -





