# +
import sys
import os
import glob
import gc
import numpy as np
from pathlib import Path
from matplotlib.pyplot import *
# %matplotlib notebook

from qubicpack.qubicfp import qubicfp
from qubic import fibtools as ft
# from qubic import time_constants_tools as tct

# +
# base_dir = '/sps/qubic/Data/Calib-TD/'
base_dir = '/media/nahue/files_hdd/heavy-data/'
# base_save_path = '/media/nahue/files_hdd/heavy-data'
# specific_save_folder = '/time_constants_results/all_TC_datasets_my_computer_2'
# save_path = base_save_path+specific_save_folder

# Path(save_path).mkdir( parents=True, exist_ok=True )
    
# print(save_path)

# +
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

print('\n')
print('We will analyze: {}'.format(thedatadir))
print('\n')

dataset_info = str.split(thedatadir,'/')[-1]

a = qubicfp()
a.assign_verbosity(0)
a.read_qubicstudio_dataset(thedatadir)

# calsource_dict = a.calsource_info()
# print(calsource_dict)
# caltime, calsourcedata = a.calsource()

# +
tt, alltod = a.tod(axistype='computertime')

tt1 = a.timeaxis(asic=1)
tt2 = a.timeaxis(asic=2)
tod1 = a.timeline_array(asic=1)
tod2 = a.timeline_array(asic=2)

tt1_ct = a.timeaxis(asic=1,axistype='computertime')
tt2_ct = a.timeaxis(asic=2,axistype='computertime')
tod1_ct = a.timeline_array(asic=1)
tod2_ct = a.timeline_array(asic=2)

fmod = 0.2
period = 1/fmod
nbins = 100

folded, t_fold, folded_nonorm, newdata = ft.fold_data(tt, alltod, period, nbins, median=True, rebin=False, verbose=False)

folded1, t_fold1, folded_nonorm1, newdata1 = ft.fold_data(tt1, tod1, period, nbins, median=True, rebin=False, verbose=False)

folded2, t_fold2, folded_nonorm2, newdata2 = ft.fold_data(tt2, tod2, period, nbins, median=True, rebin=False, verbose=False)

folded1_ct, t_fold1_ct, folded_nonorm1_ct, newdata1_ct = ft.fold_data(tt1_ct, tod1_ct, period, nbins, median=True, rebin=False, verbose=False)

folded2_ct, t_fold2_ct, folded_nonorm2_ct, newdata2_ct = ft.fold_data(tt2_ct, tod2_ct, period, nbins, median=True, rebin=False, verbose=False)

# +
# figure()
# plot(caltime, calsourcedata)

# +
# highcut = 10 #Hz

# alltod_f = ft.filter_data(tt,alltod,highcut=highcut)
# tod1_f = ft.filter_data(tt1,tod1,highcut=highcut)
# tod2_f = ft.filter_data(tt2,tod2,highcut=highcut)
# tod1_ct_f = ft.filter_data(tt1_ct,tod1_ct,highcut=highcut)
# tod2_ct_f = ft.filter_data(tt2_ct,tod2_ct,highcut=highcut)

# +
## asic 1

figure(figsize=(8,4))

tesnum = 21

if tesnum > 128:
    asic = 2
    tod_asic = tod2
    tt_asic = tt2
    tod_asic_ct = tod2_ct
    tt_asic_ct = tt2_ct
    
else:
    asic = 1
    tod_asic = tod1
    tt_asic = tt1
    tod_asic_ct = tod1_ct
    tt_asic_ct = tt1_ct

# title('TES #{} (ASIC = {}) \n Dataset: {}'.format(tesnum,asic,dataset_info))
# xlabel('Time [s]')
# ylabel('ADU voltage')
plot(tt,alltod[tesnum-1],'.',label='method a.tod() TES #{} (ASIC = {})'.format(tesnum,asic))
plot(tt_asic,tod_asic[tesnum-1 - (asic-1) * 128],label='a.timeaxis() TES #{} (ASIC = {})'.format(tesnum,asic))
plot(tt_asic_ct,tod_asic_ct[tesnum-1 - (asic-1) * 128],label='a.timeaxis(\'computertime\') TES #{} (ASIC = {})'.format(tesnum,asic))
# grid()
# legend()
# tight_layout

# figure()

# plot(tt,alltod_f[tesnum-1],'.')
# plot(tt1,tod1_f[tesnum-1])
# plot(tt1_ct,tod1_ct_f[tesnum-1])

## asic 2

# figure(figsize = (8,4))

tesnum = 205

if tesnum > 128:
    asic = 2
    tod_asic = tod2
    tt_asic = tt2
    tod_asic_ct = tod2_ct
    tt_asic_ct = tt2_ct
    
else:
    asic = 1
    tod_asic = tod1
    tt_asic = tt1
    tod_asic_ct = tod1_ct
    tt_asic_ct = tt1_ct


title('Dataset: {}'.format(dataset_info))
xlabel('Time [s]')
ylabel('ADU voltage')
plot(tt,alltod[tesnum-1]+200000,'.',label='a.tod() TES #{} (ASIC = {})'.format(tesnum,asic))
plot(tt_asic,tod_asic[tesnum-1 - (asic-1) * 128]+200000,label='a.timeaxis() TES #{} (ASIC = {})'.format(tesnum,asic))
plot(tt_asic_ct,tod_asic_ct[tesnum-1 - (asic-1) * 128]+200000,label='a.timeaxis(\'computertime\') TES #{} (ASIC = {})'.format(tesnum,asic))
grid()
legend()
tight_layout

# figure()

# plot(tt,alltod_f[tesnum-1+128],'.')
# plot(tt2,tod2_f[tesnum-1])
# plot(tt2_ct,tod2_ct_f[tesnum-1])

# +
figure(figsize=(8,4))

title('Method: a.tod() (latest qubicpack)\n Dataset: {}'.format(dataset_info))
ylabel('Normalized folded data')
xlabel('Time [s]')
for i in range(128):
    plot(t_fold, folded[i,:], 'k-',alpha=0.1)
    plot(t_fold, folded[i+128,:], 'b-',alpha=0.1)
ylim(-2.5,2.5)
grid()

# +
figure(figsize=(8,4))

title('Method: a.timeaxis() (latest qubicpack)\n Dataset: {}'.format(dataset_info))
ylabel('Normalized folded data')
xlabel('Time [s]')
for i in range(128):
    plot(t_fold1, folded1[i,:], 'k-',alpha=0.1)
    plot(t_fold2, folded2[i,:], 'b-',alpha=0.1)
ylim(-2.5,2.5)
grid()
# +
figure(figsize=(8,4))

title('Method: a.timeaxis(axistype=\'computertime\') (latest qubicpack)\n Dataset: {}'.format(dataset_info))
ylabel('Normalized folded data')
xlabel('Time [s]')
for i in range(128):
    plot(t_fold1_ct, folded1_ct[i,:], 'k-',alpha=0.1)
    plot(t_fold2_ct, folded2_ct[i,:], 'b-',alpha=0.1)
ylim(-2.5,2.5)
grid()
# -





