# #############################################################################
# Authors: E. Manzan, M.Regnier, B.Costanza
# Date: 2023-03-08 
# Template script showing how to use the DQA_Diagnosticlib.py library

import qubic
import qubicpack
from qubicpack.qubicfp import qubicfp
from qubic import demodulation_lib as dl

import DQA_Diagnosticlib

import numpy as np
from pylab import *
import glob
import pickle

# To use the DQA_Diagnosticlib.py library in a plug-and-play fashion, the user has to: 
# 1. Select a dataset
# 2. Define a string with the dataset name
# 3. Load the dataset in a qubicfp object
# In this template I will show you how to do that

# PART 1 : specify where to find data based on the date
data_path = '/sps/qubic/Data/Calib-TD/' #Use this path if you're running on cca. Otherwise, substitute this with the path where you saved the data on your pc
data_date = '2023-03-02' #Specify the date of the dataset. Others could be : '2022-04-27' , '2023-03-03' , '2023-03-07'
data_dir = data_path+data_date+'/'

#save and print all the files corresponding to desired data: 
keyword = '*Fixed-Dome*' #Choose a keyword to easily select the dataset you are interested in. Others could be : '*bandpass-measurement*' , '*Fixed-Dome*2000*'
dirs = np.sort(glob.glob(data_dir+keyword)) #get all the dataset corresponding to the keyword
num_meas = len(dirs) 
print(dirs)
print('There are {} datasets'.format(num_meas))

# PART 2 : save full dataset name (data + type of acquisition). You can run the following lines as they are, or you can define your own dataset name.
if num_meas == 1:
    data_date_full = dirs[0].split('/')[-1]
else:
    date = []
    for i in range(num_meas):
        date.append(dirs[i].split('/')[-1])
    data_date_full = np.array((date))

# PART 3 : load the qubicfp object. You can run the following lines as they are.
if num_meas == 1: #if there's only one dataset, just loads it
    a=qubicfp()
    a.read_qubicstudio_dataset(dirs[0])    
else: #if there's many dataset, create a list of qubicfp
    a = []
    for i in range(num_meas):
        a.append(qubicfp())
    for i in range(num_meas):
        print()
        a[i].read_qubicstudio_dataset(dirs[i])

# Now you can call the diagnostic library ---------------------------------------

# Option 1 : do an housekeeping diagnostic 
hk_diagnostic_class = DQA_Diagnosticlib.HK_Diagnostic(a, data_date_full)
# plot all monitors (plots will be save in the current directory)
hk_diagnostic_class.analysis()


#Option 2: do flux jump detection on a subset of detectors. A dedicated subdirectory will be created to store the plots: FluxJumps_date_full
FJclass = DQA_Diagnosticlib.FluxJumps(data_date_full, use_verbose = True, doplot = True)

Tesnum = 14
testimeline = a.timeline(TES=Tesnum)
time = a.timeaxis(asic=1, datatype = 'sci')
print('Doing TES :', Tesnum)
number_of_jumps, start_jumps, end_jumps = FJclass.fluxjumps_analysis(time, testimeline, Tesnum)

# Option 3: do a scientific diagnostic
sci_diagnostic_class = DQA_Diagnosticlib.Diagnostic(a, data_date_full) #The user can set a saturation time threshold and a saturation ADU signal. Defaults are: sat_thr=0.01; upper_satval = 4.19*1e6  

# Call the following function to perform a global analysis (detect saturation, flux jumps and plot focal plane; compute PSD and plot focal plane)
sci_diagnostic_class.analysis() #A dedicated subdirectory will be created to store the plots: FocalPlane_data_date_full

# Or, call each individual functions if you are interested in only some aspect of the analysis
sci_diagnostic_class.plot_raw_focal_plane() #plot focal plane with raw tod
#sci_diagnostic_class.tes_saturation_state() #detect saturated tods and plot focal plane
#sci_diagnostic_class.tes_fluxjump_state() #detect saturated tods, flux jumps and plot focal plane
#sci_diagnostic_class.tes_noise_estimate_and_spectrum() #compute power spectral density, estimate white noise level and plot focal plane



