import matplotlib.pyplot as plt
import numpy as np

# plt.rc('figure',figsize=(20,12))
plt.rc('figure',figsize=(10,7))
plt.rc('font',size=16)

import pickle
from astropy.coordinates import EarthLocation
import astropy.units as u

from qubicpack.utilities import Qubic_DataDir
from qubic.lib import Qdictionary

import SimulateMoonTOD as simumoon
import pipeline_moon_functions as pmf


### Data directories on CC
mydatadirin = '/sps/qubic/Data/Calib-TD/'
mydatadirout = '/sps/qubic/Users/ahuchet/simu/'

### We read the dataset
ObsDate = "2022-07-14"
# ObsDate = "2026-03-11"

# ObsDate = '2026-03-13' # good map
# ObsDate = "2026-05-03"
# ObsDate = "2026-06-24" #23, 24, 25, 26 # none of these data have the Moon in it (or well-hidden)
# ObsDate = "2026-07-27" # 23, 24, 25, 26, 27, 28, 29 of July
# ObsDate = "2026-07-26" # 2026-22_rise fridge cycle (?) in the middle, 2026-23_set very noisy, 2026-24_set cst azel
scanning = "rise" # "rise" or "set"
ObsName = "{}_{}".format(ObsDate, scanning)
# "2026-07-26_rise" not as good as the others for TES 218
# both "2026-07-28_rise" and "2026-07-29_set" too noisy, the Moon is (barely) visible only in a small number of TES
year_data = ObsDate[:4]
datadirs = pmf.observation_dirs_moon(ObsDate, scanning, mydatadirin)
print("{} folders, from {} to {}".format(len(datadirs), datadirs[0], datadirs[-1]))

dir_plots = "/Users/huchet/qubic/qubic/scripts/MoonProject/figures/"

compute_tshift = False # do we want to compute a scan by scan tshift
new_method_clean = True # do we use the known det l.o.s.


###
Salta_CNEA = {'lat':-24.731358*u.deg,
              'lon':-65.409535*u.deg,
              'height':1152*u.m,
              'UTC_Offset':-3*u.hour}
LaPuna_QUBIC = {'lat':-24.186583*u.deg,
                'lon':-66.478*u.deg,
                'height':4869*u.m,
                'UTC_Offset':-3*u.hour}

if year_data == "2022":
    ObsSite = Salta_CNEA
    azqubic = 116.4
elif year_data == "2026":
    ObsSite = LaPuna_QUBIC
    azqubic = 0

Obs_Site = EarthLocation(lat=ObsSite["lat"], lon=ObsSite["lon"], height=ObsSite["height"])

###
tshift = 0

data = pmf.format_data(az_qubic=azqubic, ObsSite=ObsSite,
                        speedmin=0.1, datadir=datadirs, tshift=tshift,
                        year_data=ObsDate[:4], simu=True)

tt, tinit, alltod, QPidx, azt, elt, newazt, newelt, scantype, Tbath, T1K, ifile, _, _, _, boresight_angle = data


### Let's simulate TOD from this
# Repository for dictionary and input maps
global_dir = Qubic_DataDir(datafile='instrument.py', datadir='../')
dictfilename = global_dir + '/dicts/pipeline_MoonSalta.dict'
d = Qdictionary.qubicDict()
d.read_from_file(dictfilename)
d['nside'] = 2048 #512 #256
d['config'] = 'TD' # no calfile for TD, or even for FI at 150 GHz
d['kind'] = 'I'
d['nf_sub'] = 12 # monoband not implemented on this branch
d['nf_recon'] = 2
d['MultiBand'] = True
d['sweeping_pointing_deadtime'] = False
d['noiseless'] = True
d["use_synthbeam_fits_file"] = False
d["n_nu_fit"] = int(d['nf_sub']/2) # monoband not implemented on this branch
d['detector_tau'] = 0.01

### TOD output file name
TOD_file = "simulated_TOD_Burgdorf_{}_nside{}".format(ObsName, d["nside"])

###
size_batch = 100000 # pointings at a time
TOD_simu = []
pixel_power = []
for i_batch in range(len(tt)//size_batch + 1):
    print("\nBatch number {}".format(i_batch))
    range_min = i_batch*size_batch
    range_max = (i_batch + 1)*size_batch
    TOD_i, pixel_power_nu = simumoon.create_moon_tod(d, azt[range_min:range_max], elt[range_min:range_max], tt[range_min:range_max] + tinit, Obs_Site)
    TOD_simu.append(TOD_i)
    pixel_power.append(pixel_power_nu)
TOD_simu = np.concatenate(TOD_simu, axis=1)
pixel_power = np.array(pixel_power) # although it is almost the same for every batches
pickle.dump( [TOD_simu, pixel_power], open( mydatadirout + TOD_file, "wb" ) )
print("Saved! {}".format(mydatadirout + TOD_file))