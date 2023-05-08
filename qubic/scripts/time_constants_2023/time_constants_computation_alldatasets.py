import sys
import os
import glob
import numpy as np

from qubic import time_constants_tools as tct

index_ini = int(sys.argv[1])
index_fin = int(sys.argv[2])

base_save_path = '/sps/qubic/Users/nahuelmg'
save_path = base_save_path+'/time_constants_results/alldatasets_to_may_2023/'
if os.path.exists(save_path):
    print('Save path:',save_path)
else:
    raise Exception('Insert save path.')

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
       
days = days[index_ini:index_fin]
keywords = keywords[index_ini:index_fin]
fmods = fmods[index_ini:index_fin]
dcs = dcs[index_ini:index_fin]

for i, (keyword, day) in enumerate(zip(keywords,days)):
    data_dir = base_dir + day + '/'
    thedirs = np.sort(glob.glob(data_dir+keyword))

    print('Available files to analyze time constants through square modulation corresponding to {}'.format(day))
    
    for j,thedir in enumerate(thedirs):
        print(j,thedir)
        
    if len(fmods[i])==1:

        fmod = fmods[i][0]
        dc = dcs[i][0]

        for thedatadir in thedirs:
            
            try:
            
            	d = tct.compute_tc_squaremod(thedatadir, fmod = fmod, dutycycle = dc, save_path = save_path)
            
            except:
            
                print('tct.compute_tc_squaremod did not work for {}'.format(thedatadir))
            	
    elif len(fmods[i])>1 and len(fmods[i])==len(thedirs):
        
        for k in range(len(fmods[i])):
            
            fmod = fmods[i][k]
            dc = dcs[i][k]
            thedatadir = thedirs[k]
            
            try:
            
            	d = tct.compute_tc_squaremod(thedatadir, fmod = fmod, dutycycle = dc, save_path = save_path)
            
            except:
            
                print('tct.compute_tc_squaremod did not work for {}'.format(thedatadir))
