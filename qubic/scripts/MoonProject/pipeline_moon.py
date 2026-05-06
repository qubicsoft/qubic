# This file is the concatenated version of pipeline_moon_filter_TOD_create_maps_2026.ipynb
# Its purpose is to:
# 0. - read TOD
# 1. - change the coordinates to ones where the Moon position in the sky is the zenith at all times
#      (thus keeping the real distances between detector apparent Moon position and real position)
#    - clean it and build maps in coordinates where the telescope/mount los sees the Moon at zenith
# 2. - measure the position, in this configuration, between the los of the telescope and the one of each TES
# 3. - change the coordinates to ones where the Moon position in the sky is the zenith at all times for each detector this time
#    - clean it and build maps in coordinates where the telescope/mount los sees the Moon at zenith for each detector
#    - we now have maps where angles and distances between order 0 and 1 should be
# 4. - use these maps to fit the synthbeam of each detector on its associated Moon map
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
from scipy.optimize import least_squares

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

# Save maps at first stage (with Moon for telescope at zenith)
do_part_1 = False
# Save positions of order 0
do_part_2 = True
# Save maps at second stage (with Moon for each detector at zenith)
do_part_3 = False
# Save the spectrum of the Moon for each detector
do_part_4 = False


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
        # tshift = np.array([tshift_ASIC2, tshift_ASIC1])
        tshift = tshift_ASIC1
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

    ### We load offsets measured from Maynooth simulations

    #### Offsets from Créidhe
    quadrant = 3 # for TD, quadrant = 3; formerly "pointing_offsets_fixed_hole.pickle"
    # import pickle
    offsets = pickle.load( open( mydatadir3 + 'pointing_offsets_Q{}.pickle'.format(quadrant), 'rb') )
    print(np.shape(offsets))

    mytesn = np.array(allTESNum)
    azel_maynooth = offsets[mytesn -1 , :]
    print(np.shape(azel_maynooth))

    ######## Here we apply the inversion betwwen Az and El ##############
    invert_azel = True # was due to an inversion in the numbering of TES, or was it? It doesn't fit well with the inversion
    if invert_azel is True:
        azel_maynooth = np.flip(azel_maynooth, axis=1)
    # print(azel_maynooth[0])

    invert_az = True # might be an error in the data taking (orientation east/west)?
    if invert_az is True:
        azel_maynooth[:, 0] = -azel_maynooth[:, 0]
    # print(azel_maynooth[0])

    invert_el = True # might be because we do el - elmoon in our data?
    if invert_el is True:
        azel_maynooth[:, 1] = -azel_maynooth[:, 1]

    ### We rotate Maynooth to zenith (note that it is in general not the same as fitting at zenith from the start)
    ### this is in case we want to use this data as guess for the order 0 fit
    maynooth_cart = pmf.spherical2cartesian(1, azel_maynooth[:, 0], azel_maynooth[:, 1], coord="horizontal", axis="last")
    rot_mat_zen = pmf.get_simple_rotation_matrix(axis="y", angle=np.radians(90)) # we go back to zenith
    maynooth_zen_cart = np.einsum("li,ik->lk", maynooth_cart, rot_mat_zen)
    maynooth_zen = pmf.cartesian2spherical(maynooth_zen_cart[:, 0], maynooth_zen_cart[:, 1], maynooth_zen_cart[:, 2], coord="horizontal", axis="last")[:, 1:]

    reso = 6 #4 #5
    xs = 301
    allamp = np.zeros(len(allTESNum))
    allerramp = np.zeros(len(allTESNum))
    allFWHM = np.zeros(len(allTESNum)) * np.nan
    allerrFWHM = np.zeros(len(allTESNum))
    pos_zen = np.zeros((len(allTESNum), 2)) * np.nan
    pos_zen_err = np.zeros((len(allTESNum), 2))

    moon_fit = []

    for i in range(len(allTESNum)):
        # print(i)
        # idx = np.where(np.array(allTESNum-1)==i)[0][0]
        if not visibly_ok_arr[i]:
            moon_fit.append([(np.nan, np.nan), np.nan, np.nan])
            continue
        print(allTESNum[i])
        # now in pixel space from maynooth guess
        m, ijres, ijerr = pmf.fit_one_tes(allmaps[i,:], xs, reso, rot=myrotinit, verbose=False, renorm=True, doplot=False, distok=50)
        # m, ijres, ijerr = pmf.fit_one_tes(allmaps[i,:], xs, reso, rot=myrotinit, xycreid_corr=maynooth_zen[i], verbose=False, renorm=True, doplot=False, distok=50)
        # m, ijres, ijerr = pmf.fit_one_tes(allmaps[i,:], xs, reso, rot=myrotinit, xycreid_corr=fitted_maynooth[i], verbose=False, renorm=True, doplot=False, distok=25)
        allFWHM[i] = m.values[3] * pmf.conv_reso_fwhm
        allerrFWHM[i] = m.errors[3] * pmf.conv_reso_fwhm
        allamp[i] = m.values[0] * pmf.conv_reso_fwhm
        allerramp[i] = m.errors[0] * pmf.conv_reso_fwhm
        pos_zen[i, :] = [ijres[1], ijres[0]] # azt, eltc
        pos_zen_err[i, :] = [ijerr[1], ijerr[0]]
        print('TES#{0}: FWHM = {1:5.2f}'.format(i + 1, m.values[3] * pmf.conv_reso_fwhm))
        moon_fit.append([pos_zen[i, :], allFWHM[i], allamp[i]])

    ### Here we move to cartesian coordinates in order to compare with Maynooth at zenith
    all_det_pos_cart = pmf.spherical2cartesian(1, pos_zen[:, 0], pos_zen[:, 1], coord="horizontal", axis="first")

    ### The DBscan allows us to detect the cluster of TES that have the same offset with theory (as expected if theory is right with an offset)
    ### Works well, a bit because the z-axis points to the Moon for the telescope
    DBscan_ok = pmf.get_DBscan_res_cart(all_det_pos_cart[:, 0], all_det_pos_cart[:, 1], all_det_pos_cart[:, 2], maynooth_zen_cart[:, 0], maynooth_zen_cart[:, 1], maynooth_zen_cart[:, 2], visibly_ok_arr, doplot=True, eps=0.01, min_samples=10)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='polar')
    ax.plot(np.radians(maynooth_zen[:,  0]), 90 - maynooth_zen[:,  1], 'ro', alpha=0.2)
    ax.plot(np.radians(maynooth_zen[DBscan_ok,  0]), 90 - maynooth_zen[DBscan_ok,  1], 'ro', label="Simulated positions")
    ax.plot(np.radians(pos_zen[:, 0]), 90 - pos_zen[:, 1], 'ko', label="Observed positions")
    # plt.plot(pos_zen_rot[:,0], pos_zen_rot[:,1], 'ko', label="Fitted positions")
    ax.set_xlabel('$el^{Moon}$')
    ax.set_ylabel('$az^{Moon}$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(dir_plots + "simulated_vs_observed_moonpos_zen.pdf", dpi=600)
    plt.show()

    ### We want to fit the rotations necessary to move the Maynooth simulations to the real positions
    fake_pos = pos_zen - pos_zen_err
    fake_pos_cart = pmf.spherical2cartesian(1, fake_pos[:, 0], fake_pos[:, 1], coord="horizontal", axis="last")
    all_det_err_cart = np.abs(all_det_pos_cart - fake_pos_cart) * 1

    initvec = maynooth_zen_cart[DBscan_ok]
    outvec = all_det_pos_cart[DBscan_ok, :]
    initvec_sph = pmf.cartesian2spherical(initvec[:, 0], initvec[:, 1], initvec[:, 2], coord="horizontal", axis="last")[:, 1:]
    outvec_sph = pmf.cartesian2spherical(outvec[:, 0], outvec[:, 1], outvec[:, 2], coord="horizontal", axis="last")[:, 1:]

    ### Plotting the new positions w.r.t. the old ones and the real ones
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='polar')
    ax.set_aspect(1)
    ax.plot(np.radians(initvec_sph[:, 0]), 90 - initvec_sph[:, 1], 'bo', label = 'Maynooth')
    ax.plot(np.radians(outvec_sph[:, 0]), 90 - outvec_sph[:, 1], 'go', label='Moon VI')
    theta_0 = np.array([0, 0, 0])
    fit_res = least_squares(pmf.fun_minimise, theta_0, args=(initvec.flatten(), outvec.flatten()))
    theta_fit = fit_res.x
    newvec = np.reshape(pmf.rotate_2d_zen_pts(np.ravel(initvec), theta_fit), np.shape(initvec))
    newvec_sph = pmf.cartesian2spherical(newvec[:, 0], newvec[:, 1], newvec[:, 2], coord="horizontal", axis="last")[:, 1:]
    mylabel = 'Fit to match Moon: \n'+ r'Rot$_x$={0:3.2f}+/-{1:3.2f} deg'.format(theta_fit[0])
    mylabel += '\n' + r'Rot$_y$={0:3.2f}+/-{1:3.2f} deg'.format(theta_fit[1])
    mylabel += '\n' + r'Rot$_z$={0:3.2f}+/-{1:3.2f} deg'.format(theta_fit[2])
    ax.plot(np.radians(newvec_sph[:, 0]), 90 - newvec_sph[:, 1], 'b+', ms=13, label=mylabel)
    ax.legend()
    ax.set_xlabel('$\Delta_{az}$ [deg.]')
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position+10), ax.get_rmax()/2., '$\Delta_{el}$ [deg.]', # the r label is put manually
        rotation=label_position, ha='center', va='center')
    for pointi in range(len(initvec[:, 0])):
        plt.plot(np.radians([outvec_sph[pointi, 0], newvec_sph[pointi, 0]]), 90 - np.array([outvec_sph[pointi, 1], newvec_sph[pointi, 1]]), c="k")
    plt.tight_layout()
    plt.savefig(dir_plots + "fitted_maynooth_pos.pdf", dpi=600)
    plt.show()

    pos_zen_full = pos_zen.copy()
    not_fitted = ~DBscan_ok
    newvec_full = np.reshape(pmf.rotate_2d_zen_pts(np.ravel(maynooth_zen_cart), theta_fit), np.shape(azel_maynooth))
    pos_zen_full[not_fitted] = newvec_full[not_fitted]

    fitted_maynooth = pmf.cartesian2spherical(newvec_full[:, 0], newvec_full[:, 1], newvec_full[:, 2], coord="horizontal", axis="last")[:, 1:]
    # fitted_maynooth is the result from Maynooth simulation fitted as a whole (all detectors at once) to the measured Moon positions for the detectors
    # use this to fit the order 0 in a second round

    np.save('pos_zen_full.npy', pos_zen_full)
    np.save('fitted_maynooth.npy', fitted_maynooth)
    print("saved")

if do_part_3:
    pos_zen_full = np.load('pos_zen_full.npy')

    allmaps_zendet, data_TOD, center, newazt, newelt, scantype = pmf.make_coadded_maps(datadir, Obs_Site, allTESNum, data=None, 
                                        nside=nside, doplot=False, az_qubic=azqubic, parallel=True, ObsDate=ObsDate, tshift=tshift,
                                        det_pos=pos_zen_full, theo_sb=None)

    pickle.dump( [allmaps_zendet, allmaps], open( mydatadir2 + "202603-allmaps-13032026_zenith_det.pkl", "wb" ) )#


if do_part_4:

    well_fitted = []

    # Let's first test it in the notebook


