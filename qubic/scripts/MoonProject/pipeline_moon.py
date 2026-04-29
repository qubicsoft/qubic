# This file is the concatenated version of pipeline_moon_filter_TOD_create_maps_2026.ipynb
# Its purpose is to:
# 0. - read TOD
# 1. - change the coordinates to ones where the Moon position in the sky is the zenith at all times
#      (thus keeping the real distances between detector apparent Moon position and real position)
#    - clean it and build maps in coordinates where the telescope/mount los sees the Moon at zenith
#    - measure the position, in this configuration, between the los of the telescope and the one of each TES
# 2. - change the coordinates to ones where the Moon position in the sky is the zenith at all times for each detector this time
#    - clean it and build maps in coordinates where the telescope/mount los sees the Moon at zenith for each detector
#    - we now have maps where angles and distances between order 0 and 1 should be
# 3. - use these maps to fit the synthbeam of each detector on its associated Moon map
#    - this should give us the spectrum of the Moon (before any atmosphere mitigation!)


#### General imports
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pickle
import fitsio
from scipy.interpolate import LinearNDInterpolator
import astropy.units as u

#### QUBIC IMPORT
from qubic.lib import Qdictionary 
from qubic.lib.Instrument import Qinstrument
import qubic.lib.Calibration.Qfiber as ft
from qubic.lib.Qdictionary import qubicDict
from qubic.lib.Instrument.Qinstrument import QubicMultibandInstrument
from qubic.lib.Qscene import QubicScene

#### Specific libraries for the Moon
import fitting as fit
import time_domain_tools as tdt
import pipeline_moon_plotting as pmp
import pipeline_moon_functions as pmf

# Save map at first stage (with Moon for telescope at zenith)
do_part_1 = False
# Save map at second stage (with Moon for each detector at zenith)
do_part_2 = False
# Save pos

#### These are directories used in the analysis, not really clean
dirtemplibs = ["/Users/huchet/qubic/qubic/scripts/MoonProject/", "/Users/huchet/Documents/code/scripts/", "/Users/huchet/Documents/code/data/"] #[os.environ['QUBIC_DATADIR']+'scripts/MoonProject/']
for rep in dirtemplibs:     
    if rep not in sys.path:
        sys.path.append(rep)

mydatadir = '/Users/huchet/Documents/code/data/ComissioningTD/'
mydatadir2 = "/Users/huchet/Documents/code/scripts/MoonProject/"
mydatadir3 = "/Users/huchet/Documents/code/data/"

# ObsDate = '2022-07-14'
# ObsDate = '2026-03-11'
ObsDate = '2026-03-13'
year_data = ObsDate[:4]
ObsSession = 0
if year_data == "2022":
    dirs = glob.glob(mydatadir + ObsDate + '/*')
else:
    # dirs = glob.glob(mydatadir + ObsDate + '/*')
    # dirs = glob.glob(mydatadir + ObsDate + '/*Moon_instrument_not_quite_ready')
    # dirs = glob.glob(mydatadir + ObsDate + '/2026-03-11_06.45.41__Moon_instrument_not_quite_ready')
    # dirs = glob.glob(mydatadir + ObsDate + '/2026-03-11_06.46.21__Moon_instrument_not_quite_ready')
    # dirs = glob.glob(mydatadir + ObsDate + '/2026-03-11_15.39.59__Moon_el30')
    # dirs = glob.glob(mydatadir + ObsDate + "/*Moon_el60")
    # dirs = glob.glob(mydatadir + ObsDate + "/2026-03-11_07.58.25__Moon_el60")
    dirs = glob.glob(mydatadir + ObsDate + "/*Moon")
    # dirs = glob.glob(mydatadir + ObsDate + "/*07.54.24__Moon")
# print(dirs)
i_file = 1 # only 1 works for 2026-03-13
datadir = dirs[i_file]
print(datadir)
dir_plots = "/Users/huchet/qubic/qubic/scripts/MoonProject/figures/"


#### Observing sites
Salta_CNEA = {'lat':-24.731358*u.deg,
              'lon':-65.409535*u.deg,
              'height':1152*u.m,
              'UTC_Offset':-3*u.hour}
LaPuna_QUBIC = {'lat':-24.186583*u.deg,
                'lon':-66.478*u.deg,
                'height':4869*u.m,
                'UTC_Offset':-3*u.hour}
if year_data == "2022":
    Obs_Site = Salta_CNEA
elif year_data == "2026":
    Obs_Site = LaPuna_QUBIC


# Creation of list of TES numbers
nb_TES = 256
allTESNum = np.arange(nb_TES) + 1

# Same for all maps
nside = 256 # 256, 340
if year_data == "2022":
    azqubic = 116.4
else:
    azqubic = 0

if year_data == "2022":
    tshift = 0.21
    start_tt = 10000
    speedmin = 0.1
else:
    if ObsDate == "2026-03-11":
        tshift = -1.35
        start_tt = 1000
        speedmin = 0.
    elif ObsDate == "2026-03-13": # different shift in time due to a clock inversion problem between ASIC 1 and 2, fixed since
        # also, there is a clock drift happening during the data acquition, so the maps are not the best they could be
        tshift_ASIC2 = -0.4 #-0.6 # TES 218
        tshift_ASIC1 = -1.35 # TES 96
        tshift = np.array([tshift_ASIC2, tshift_ASIC1])
        start_tt = 50000
        speedmin = 0.1

if do_part_1:
    # 1. First run, we only know the position of the Moon in the sky and the direction the instrument points at
    allmaps, data_TOD, center, newazt, newelt, scantype = pmf.make_coadded_maps(datadir, Obs_Site, allTESNum, data=None, 
                                        nside=nside, doplot=False, az_qubic=azqubic, parallel=True, ObsDate=ObsDate, tshift=tshift,
                                        det_pos=None, theo_sb=None)
    
    pickle.dump( [allTESNum, allmaps], open( mydatadir2 + "202603-allmaps-13032026_zenith_tshift{}.pkl".format(tshift), "wb" ) )

elif do_part_2:
    # We read the maps from a previous run
    allTESNum, allmaps = pickle.load( open( mydatadir2 + "202603-allmaps-13032026_zenith_tshift-1.35.pkl", "rb" ) )
    allTESNum, allmaps_ = pickle.load( open( mydatadir2 + "202603-allmaps-13032026_zenith_tshift-0.4.pkl", "rb" ) ) # shift is different between ASIC 1 and 2
    allmaps[128:] = allmaps_[128:]


if do_part_2:

    # 2026-03-13, zenith, all TES with the Moon shape
    # This visual inspection that discards the bad TES is so far needed because of crashes in Minuit, to be improved
    visibly_ok_arr = [False, False, False, False, False, False, False, False,  True, False, False, False,
    False, False, False,  True, False,  True, False, False, False, False, False, False,
    False,  True,  True, False, False, False,  True, False,  True,  True, False, False,
    False, False, False, False, False, False,  True, False, False,  True,  True, False,
    False,  True,  True, False, False, False, False, False,  True,  True,  True, False,
    True,  True,  True,  True, False,  True,  True, False,  True,  True,  True,  True,
    True, False, False,  True,  True,  True,  True,  True,  True,  True, False,  True,
    True, False, False,  True, False, False, False, False,  True,  True,  True,  True,
    False,  True,  True, False, False,  True, False, False, False, False,  True, False,
    True,  True,  True,  True, False, False, False, False,  True, False, False, False,
    False,  True, False, False, False, False, False, False,  True,  True,  True, False,
    True, False,  True,  True,  True,  True,  True,  True, False, False, False, False,
    True,  True, False,  True,  True,  True,  True, False,  True, False, False, False,
    False,  True,  True,  True,  True,  True,  True, False, False, False,  True, False,
    False,  True,  True,  True, False, False,  True, False,  True,  True,  True, False,
    True,  True, False,  True,  True,  True,  True,  True,  True, False, False, False,
    False, False,  True, False, False, False,  True,  True, False, False, False, False,
    True, False,  True, False,  True,  True,  True,  True,  True, False, False,  True,
    False,  True, False,  True,  True, False,  True,  True, False, False, False, False,
    False, False,  True, False, False, False, False,  True, False, False,  True,  True,
    True,  True, False, False,  True,  True, False,  True, False, False, False, False,
    True, False, False, False,]

    visibly_ok_arr[241 - 1] = False

    # We want to fit the Moon position for each TES
    myrotinit = np.array([0, 90, 0]) # the Moon should be around zenith

    reso = 6 #4 #5
    xs = 201
    allamp = np.zeros(len(allTESNum))
    allerramp = np.zeros(len(allTESNum))
    allFWHM = np.zeros(len(allTESNum)) * np.nan
    allerrFWHM = np.zeros(len(allTESNum))
    allxy = np.zeros((len(allTESNum), 2)) * np.nan
    allerrxy = np.zeros((len(allTESNum), 2))

    moon_fit = []

    for i in range(len(allTESNum)):
        # print(i)
        # idx = np.where(np.array(allTESNum-1)==i)[0][0]
        if not visibly_ok_arr[i]:
            moon_fit.append([(np.nan, np.nan), np.nan, np.nan])
            continue
        print(allTESNum[i])
        m = pmf.fit_one_tes(allmaps[i,:], xs, reso, rot=myrotinit, verbose=False, renorm=True, doplot=False)
        allFWHM[i] = m.values[3] * pmf.conv_reso_fwhm
        allerrFWHM[i] = m.errors[3] * pmf.conv_reso_fwhm
        allamp[i] = m.values[0] * pmf.conv_reso_fwhm
        allerramp[i] = m.errors[0] * pmf.conv_reso_fwhm
        # allxy[i, :] = m.values[1:3]
        # allerrxy[i, :] = m.errors[1:3]
        allxy[i, :] = [m.values[2], m.values[1]] # m.values contains amplitude, elevation, azimuth, sigma
        allerrxy[i, :] = [m.errors[2], m.errors[1]]
        print('TES#{0}: FWHM = {1:5.2f}'.format(i + 1, m.values[3] * pmf.conv_reso_fwhm))
        moon_fit.append([allxy[i, :], allFWHM[i], allamp[i]])


    not_fitted = ~DBscan_ok
    np.shape(allxy_rot[not_fitted])
    newvec_full = np.reshape(function_fit(np.ravel(xycreidhe), m.values), np.shape(xycreidhe))
    allxy_rot[not_fitted] = newvec_full[not_fitted]

    # let's go back to zenith
    new_det_pos_rot = pmf.spherical2cartesian(1, allxy_rot[:, 0], allxy_rot[:, 1], coord="horizontal", axis="first")
    rot_mat_zen = pmf.get_simple_rotation_matrix(axis="y", angle=np.radians(90)) # we go back to zenith
    new_det_pos_zen = np.einsum("il,ik->kl", new_det_pos_rot, rot_mat_zen)
    full_det_pos_zen = pmf.cartesian2spherical(new_det_pos_zen[0], new_det_pos_zen[1], new_det_pos_zen[2], coord="horizontal", axis="last")[:, 1:]

    full_det_pos_zen = np.load('full_det_pos_zen.npy')

    pickle.dump( [allTESNum, allmaps], open( mydatadir2 + "202603-allmaps-13032026_zenith_det.pkl", "wb" ) )





