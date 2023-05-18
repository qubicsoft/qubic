import sys
import os
import glob
import gc
import numpy as np
from pathlib import Path
from matplotlib.pyplot import *

from qubic import time_constants_tools as tct

# +
# base_dir = '/sps/qubic/Data/Calib-TD/'
base_dir = '/media/nahue/files_hdd/heavy-data/'
base_save_path = '/media/nahue/files_hdd/heavy-data'
specific_save_folder = '/time_constants_results/all_TC_datasets_my_computer_2'
save_path = base_save_path+specific_save_folder

Path(save_path).mkdir( parents=True, exist_ok=True )
    
# print(save_path)
# -

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

# +
numberday = -1
day = days[numberday]
keyword = keywords[numberday]

data_dir = base_dir + day + '/'
thedirs = np.sort(glob.glob(data_dir+keyword))

for j,thedir in enumerate(thedirs):
    print(j,thedir)

numfile = 0
thedatadir = thedirs[numfile]

from qubicpack.qubicfp import qubicfp

a = qubicfp()
a.assign_verbosity(0)
a.read_qubicstudio_dataset(thedatadir)

tt, alltod = a.tod()
calsource_dict = a.calsource_info()
print(calsource_dict)
caltime, calsourcedata = a.calsource()
# -

figure()
plot(caltime, calsourcedata)
