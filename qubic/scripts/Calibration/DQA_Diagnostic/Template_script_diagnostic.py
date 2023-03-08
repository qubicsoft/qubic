##############################################################################
#Authors: E. Manzan
#Date: 2023-03-08 
#Template script showing how to use the DQA_Diagnosticlib.py library

import qubic
import qubicpack
from qubicpack.qubicfp import qubicfp
from qubic import demodulation_lib as dl

import DQA_Diagnosticlib

import numpy as np
from pylab import *
import glob
import pickle

from matplotlib import rc
rc('figure',figsize=(5,5))
rc('font',size=16)


#specify where to find data based on the date
data_path = '/sps/qubic/Data/Calib-TD/'
data_date = '2023-03-02' #'2022-04-27' #'2023-03-03' #'2023-03-02'
data_dir = data_path+data_date+'/'

#save and print all files corresponding to desired data: 
keyword = '*Fixed-Dome*' #'*bandpass-measurement*' #'*Fixed-Dome*'
dirs = np.sort(glob.glob(data_dir+keyword))
num_meas = len(dirs) #[:3]
print(dirs)
print('There are {} datasets'.format(num_meas))

#save full dataset name (data + type of acquisition)
if num_meas == 1:
    data_date_full = dirs[0].split('/')[-1]
else:
    date = []
    for i in range(num_meas):
        date.append(dirs[i].split('/')[-1])
    data_date_full = np.array((date))


#load qubicfp object
if num_meas == 1: #if there's only one dataset, just load it
    a=qubicfp()
    a.read_qubicstudio_dataset(dirs[0])    
else: #if there's many dataset, create a list of qubicfp
    a = []
    for i in range(num_meas):
        a.append(qubicfp())
    for i in range(num_meas):
        print()
        a[i].read_qubicstudio_dataset(dirs[i])

#call the diagnostic library. The user can set a saturation time threshold and a saturation ADU signal. Defaults are: sat_thr=0.01; upper_satval = 4.19*1e6     
myclass = DQA_Diagnosticlib.Diagnostic(a, data_date_full, sat_thr=0.01, upper_satval = 4.19*1e6)

#Call the following function to perform a global analysis
myclass.analysis(do_plot=True)

#Call each individual functions if you are interested in only some aspect of the analysis
#myclass.plot_raw_focal_plane()
#myclass.tes_saturation_state()
