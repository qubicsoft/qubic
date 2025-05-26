# #############################################################################
# Authors: E. Manzan, M. Haun
# Date: 2023-07-20
# Template script showing how to use the DQA_Diagnosticlib.py + DataFlagging.py libraries

import qubic
import qubicpack
from qubicpack.qubicfp import qubicfp

import DQA_Diagnosticlib
import DataFlagging_Margaret

import numpy as np
from pylab import *
import glob
import pickle
from datetime import datetime

# PART 1 : specify where to find data based on the date
data_path = '/sps/qubic/Data/Calib-TD/' #Use this path if you're running on cca. Otherwise, substitute this with the path where you saved the data on your pc
data_date = '2023-03-02' #Specify the date of the dataset. Others could be : '2022-04-27' , '2023-03-03' , '2023-03-07', '2023-03-16', '2023-03-02'
data_dir = data_path+data_date+'/'

#save and print all the files corresponding to desired data: 
keyword = '*Fixed-Dome*' #Choose a keyword to easily select the dataset you are interested in. Others could be : '*bandpass-measurement*' , '*Fixed-Dome*2000*', '*17.06.31*' , '*14.34.02*', '*15.14*'
dirs = np.sort(glob.glob(data_dir+keyword)) #get all the datasets corresponding to the keyword
num_meas = 1 #len(dirs)  
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
    print('Start date : ', a.obsdate)
    print('End date : ', a.endobs)
else: #if there's many dataset, create a list of qubicfp
    a = []
    for i in range(num_meas):
        a.append(qubicfp())
    for i in range(num_meas):
        print()
        a[i].read_qubicstudio_dataset(dirs[i])

#PART 4 : the analysis ---------------------------------------
# Call the diagnostic library 
sci_diagnostic_class = DQA_Diagnosticlib.Diagnostic(a, data_date_full, sat_thr=0.0) 
#sci_diagnostic_class.get_plot_tod_and_temp_monitor()

# Apply flagging+masking
#turn on what you want to flag: from False to True
userflag = {'saturation': True, 'cosmic ray': False, 
              'uncorrelated flux jumps': False, 'end of scan': False,
              'Tbath above 330mK': False, 'Tbath above 340mK': False, 
              'Tbath above 350mK': False, 'Tbath rising': False,
              '1K above 1.1K': False, '1K above 1.2K': True, 
              '1K above 1.3K': True, '1K rising': True,
              'correlated flux jumps': False}

# get all the flags
print('-------> Start data flagging...')
starting_time = datetime.now()
flag_class = DataFlagging_Margaret.FlagData(a)

sat_flag_array = flag_class.flag_sat()['flag'] #saturation: THIS IS A 2D ARRAY!
temp_flag_array = flag_class.flag_1Ktemp()['flag'] #1K temperature thresholds: THIS IS A 1D ARRAY!
temprise_flag_array = flag_class.flag_1Ktemp_rise_beta()['flag'] #1K temperature rising: THIS IS A 1D ARRAY!
print('-------> ...flags are in. Duration = ', datetime.now() - starting_time)

# sum all the flags
if userflag['saturation']:
    flag_array_tot = np.zeros_like(sci_diagnostic_class.tod, dtype=np.int64)
    flag_array_tot += sat_flag_array
    #temperature flags are 1D: convert them into 2D
    flag_array_tot += np.tile(temp_flag_array, (sci_diagnostic_class.num_det,1))
    flag_array_tot += np.tile(temprise_flag_array, (sci_diagnostic_class.num_det,1))
    print('-------> ...total flags are in. Duration = ', datetime.now() - starting_time)
else:
    flag_array_tot = np.zeros_like(sci_diagnostic_class.timeaxis, dtype=np.int64)
    flag_array_tot += temp_flag_array
    flag_array_tot += temprise_flag_array
    print('-------> ...total flags are in. Duration = ', datetime.now() - starting_time)

#convert into mask
mask_class = DataFlagging_Margaret.FlagToMask_beta()
mask2d = mask_class(flag_array_tot, userflag) #binary mask: 0 = ok ; 1 = flagged

if userflag['saturation']:
    mask_array = ma.make_mask(mask2d) # bool mask: False = ok; True = flagged. WE NEED THE OPPOSITE OF THIS MASK!!!
else:
    mask2d_def = np.tile(mask2d, (sci_diagnostic_class.num_det,1))
    mask_array = ma.make_mask(mask2d_def)

print('-------> ...mask is in. Duration = ', datetime.now() - starting_time)

#apply (opposite) mask to tod and time
masked_tod = np.ma.MaskedArray(sci_diagnostic_class.tod, ~mask_array) 
masked_time = np.ma.MaskedArray(sci_diagnostic_class.time_fp, ~mask_array) 
print('original array shapes: ', sci_diagnostic_class.time_fp.shape, sci_diagnostic_class.tod.shape)
print('masked array shapes: ', masked_time.shape, masked_tod.shape)
print('-------> ...Data flagging ended. Duration = ', datetime.now() - starting_time)

#run saturation analysis on whole FP
sci_diagnostic_class.tod = masked_tod.copy()
sci_diagnostic_class.time_fp = masked_time.copy()
print('New time axis shape :', sci_diagnostic_class.timeaxis.shape)
print('New tod shape :', sci_diagnostic_class.tod.shape)

sci_diagnostic_class.tes_saturation_state() #detect saturated tods and plot focal plane