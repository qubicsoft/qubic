from __future__ import division, print_function

import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import pickle as pk
from scipy.spatial.distance import cdist
from scipy.stats import sigmaclip
import pandas as pd
from sklearn.cluster import DBSCAN

from qubicpack.utilities import Qubic_DataDir
import qubic
import qubic.sb_fitting as sbfit

from qubicpack.pixel_translation import tes2index


# ============= Functions =================
def get_all_fit(rep):
    c50 = np.cos(np.radians(50))
    azmin = -15. / c50
    azmax = 15. / c50

    tes_fit = []
    tes_newxxyy = []
    for tes in range(256):
        themap, az, el, fitmap, newxxyy = sbfit.get_flatmap(tes + 1, rep, fitted_directory=rep + '/FitSB',
                                                            azmin=azmin, azmax=azmax)
        tes_newxxyy.append(newxxyy)
        tes_fit.append(fitmap / np.sum(fitmap))

    return np.array(tes_newxxyy), np.array(tes_fit)


def get_alpha(az, el, deg=False):
    if deg == True:
        az = np.deg2rad(az)
        el = np.deg2rad(el)
    # Unit vector in spherical coordinates
    unit_vector = np.array([np.sin(np.pi / 2 - el) * np.cos(az),
                            np.sin(np.pi / 2 - el) * np.sin(az),
                            np.cos(np.pi / 2 - el)])

    # Scalar product to get alpha
    ntes, npeaks = np.shape(az)
    cosalpha = np.empty((npeaks, ntes, ntes))
    for peak in range(npeaks):
        cosalpha[peak] = np.dot(unit_vector[:, :, peak].T, unit_vector[:, :, peak])
        # print(cosalpha)
    alpha = np.arccos(cosalpha)
    # print(alpha)
    return alpha


def get_centersquare_azel(rep, tes):
    thefile = open(rep + '/FitSB/fit-TES{}.pk'.format(tes), 'rb')
    fitpars = pk.load(thefile, encoding='latin1')
    return fitpars[:2]


def get_tes_xycoords_radial_dist(q):
    tes_xy = np.zeros((256, 2))
    tes_radial_dist = np.zeros(256)
    for i in range(256):
        if i < 128:
            tes = i + 1
            asic = 1
        else:
            tes = i - 128 + 1
            asic = 2
        index = tes2index(tes, asic)

        # None are the thermometers
        if index is not None:
            index_place = np.where(q.detector.index == index)[0][0]
            x = q.detector.center[index_place, 0]
            y = q.detector.center[index_place, 1]
            tes_radial_dist[i] = np.sqrt(x ** 2 + y ** 2)
            print(tes, index, tes_radial_dist[i])
            tes_xy[i, :] = ([x, y])

    return tes_xy, tes_radial_dist


def DBSCAN_cut(results, doplot=False):
    clustering = DBSCAN(eps=0.45, min_samples=10).fit(results)
    labels = clustering.labels_
    ok = (labels == 0)
    if doplot:
        plt.figure()
        plt.scatter(results[:, 0], results[:, 1])
        plt.scatter(results[:, 0][ok], results[:, 1][ok])
        plt.show()
    return ok


def normalize(x):
    return (x - np.nanmean(x)) / np.nanstd(x)




def measure_focal_lengthnew(tes_xy, rdist, alpha, npeaks=9, ntes=256, nsig=3, goodtes=None):
    """

    Parameters
    ----------
    tes_xy
    rdist
    alpha
    npeaks
    ntes
    nsig
    goodtes: list
        List containing FP indices of good TES.

    Returns
    -------

    """

    tes_dist = cdist(tes_xy, tes_xy, 'euclidean')
    tanalpha = np.tan(alpha)

    if goodtes is not None:
        for i in range(ntes):
            if i < 128:
                tes = i + 1
                asic = 1
            else:
                tes = i - 128 + 1
                asic = 2
            index = tes2index(tes, asic)
            if index not in goodtes:
                print(i, index)
                tes_dist[i, :] = np.nan
                tes_dist[:, i] = np.nan

    fl_mean = np.zeros((npeaks, ntes))
    fl_std = np.zeros((npeaks, ntes))
    for peak in range(npeaks):
        print('Peak ', peak, '\n')
        # alpha = np.rad2deg().deg2rad(cdist(peak_azel[:, :, peak], peak_azel[:, :, peak], 'euclidean'))

        # Compute k = Drcos(phi)
        for tes in range(ntes):
            print('tes', tes)
            k = (tes_xy[:, 0] - tes_xy[tes, 0]) * tes_xy[:, 0] \
                + (tes_xy[:, 1] - tes_xy[tes, 1]) * tes_xy[:, 1]

            D = tes_dist[tes]
            tg = tanalpha[peak, tes, :]
            Delta = D ** 4 - 4 * tg ** 2 * k ** 2 * (1 + D ** 2 / k)
            Xplus = (-2 * k * tg ** 2 + D ** 2 + np.sqrt(Delta)) / (2 * tg ** 2)

            fl = np.sqrt(Xplus - rdist[tes] ** 2)
            print('fl', fl)
            fl = fl[~np.isnan(fl)]
            fl_clip, mini, maxi = sigmaclip(fl, low=nsig, high=nsig)
            print(mini, maxi)
            print('fl_clip', fl_clip.shape)

            # Mean and STD for each TES
            fl_mean[peak, tes] = np.mean(fl_clip)
            fl_std[peak, tes] = np.std(fl_clip) / np.sqrt(len(fl_clip))

    return fl_mean, fl_std


def plot_flonfp(fl_mean, fl_std, xy, radial_dist, npeaks=9):
    # Compute the mean and the global std over TES
    ntes = np.shape(fl_std)[1]
    final_mean = np.nanmean(fl_mean, axis=1)
    final_std2 = np.nansum(fl_std ** 2, axis=1)
    final_std = np.sqrt(final_std2 / ntes)

    # Plot
    for peak in range(npeaks):
        print('Peak {}: {} +- {}'.format(peak, final_mean[peak], final_std[peak]))

        plt.figure(figsize=(15, 4))
        plt.suptitle('Peak {}'.format(peak))

        plt.subplot(131)
        plt.errorbar(radial_dist, fl_mean[peak], yerr=fl_std[peak],
                     marker='o', linestyle='none',
                     label='$FL = {:.5f} \pm {:.5f}$'.format(final_mean[peak], final_std[peak]))
        plt.xlabel('Radial TES distance (m)')
        plt.ylabel('TES focal length (m)')
        plt.legend()


        plt.subplot(132)
        plt.title('Mean focal length on the FP')
        plt.scatter(xy[:, 0], xy[:, 1], marker='s', s=150, c=fl_mean[peak],
                    vmin=0.2, vmax=0.45)
        plt.xlim((-0.055, 0.))
        plt.ylim((-0.055, 0.))
        plt.gca().set_aspect('equal')
        plt.colorbar()

        plt.subplot(133)
        plt.title('STD focal length on the FP')
        plt.scatter(xy[:, 0], xy[:, 1], marker='s', s=150, c=fl_std[peak],
                    vmin=0.01, vmax=0.02)
        plt.xlim((-0.055, 0.))
        plt.ylim((-0.055, 0.))
        plt.gca().set_aspect('equal')
        plt.colorbar()

        plt.show()

    return final_mean, final_std


def measure_global_focal_length(tes_xy, peak_azel, npeaks=9, nsig=3):
    tes_dist = cdist(tes_xy, tes_xy, 'euclidean')
    tes_dist = np.triu(tes_dist)
    print(np.max(tes_dist))
    plt.figure()
    plt.imshow(tes_dist)

    allfl_clip, alltes_dist_cut, alltanalpha_cut = [], [], []
    for peak in range(npeaks):
        # Get all angular distances between peak position
        azel = peak_azel[:, :, peak]
        alpha = np.deg2rad(cdist(azel, azel, 'euclidean'))
        alpha = np.triu(alpha)
        tanalpha = np.tan(alpha)

        focal_length = tes_dist / tanalpha
        fl = focal_length[~np.isnan(focal_length)]
        fl_clip, mini, maxi = sigmaclip(fl, low=nsig, high=nsig)
        print(mini, maxi)
        print(fl_clip.shape)

        tes_dist_cut = tes_dist[(focal_length > mini) & (focal_length < maxi)]
        tanalpha_cut = tanalpha[(focal_length > mini) & (focal_length < maxi)]
        print(tes_dist_cut.shape, tanalpha_cut.shape)

        allfl_clip.append(fl_clip)
        alltes_dist_cut.append(tes_dist_cut)
        alltanalpha_cut.append(tanalpha_cut)

    fl_mean = [np.mean(fl) for fl in allfl_clip]
    fl_std = [np.std(fl) / np.sqrt(len(fl)) for fl in allfl_clip]

    return allfl_clip, fl_mean, fl_std


# =========== Radial TES distances on the FP ================
# Get a dictionary
basedir = Qubic_DataDir(datafile='instrument.py', )
print('basedir : ', basedir)
dictfilename = basedir + '/dicts/global_source_oneDet.dict'

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)
print(d['detarray'])

d['config'] = 'FI'
q = qubic.QubicInstrument(d)

tes_xy, rdist = get_tes_xycoords_radial_dist(q)

# Remove thermometers
r = rdist[rdist != 0.]
x = tes_xy[:, 0]
x = x[x != 0]

y = tes_xy[:, 1]
y = y[y != 0]

# Check we have the right radial distance by ploting in on the FP
plt.figure()
plt.scatter(x, y, marker='s', s=150, c=r, vmin=0., vmax=0.06)
plt.title('Radial distances')
plt.xlabel('m')
plt.ylabel('m')
plt.xlim(-0.06, 0.06)
plt.ylim(-0.06, 0.06)
plt.colorbar()
plt.show()

# Using a qubic function
# r = np.zeros((128, 2))
# r[:, 0] = rdist[:128]
# r[:, 1] = rdist[128:]
#
# dist_on_fp = sc.tes_signal2image_fp(r, [1, 2])
# plt.figure()
# plt.imshow(dist_on_fp)
# plt.title('Radial distances')
# plt.colorbar()

# ============= Bad TES ==================
calfile = fits.open(basedir + 'calfiles/CalQubic_DetArray_P87_TD.fits')
fp_image = calfile['removed'].data
plt.imshow(fp_image)
plt.show()

number_goodtes = len(np.where( fp_image == 0))

goodtes = []
nside = 34
for i in range(nside):
    for j in range(nside):
        FP_index = j % (nside) + i * (nside)

        if fp_image[i, j] == 0:
            goodtes.append(FP_index)
            print(FP_index)

# ============= Focal length with the 9 peaks ==================

# Get the fit of the synthesized beams
freq_source = 150
rep = Qubic_DataDir(datafile='allFitSB_{}.pdf'.format(freq_source), datadir='/home/lmousset/QUBIC/Qubic_work')
print(rep)

tes_newxxyy, tesfit = get_all_fit(rep)

az = np.deg2rad(tes_newxxyy[:, 0, :]) / np.cos(np.deg2rad(50.))
el = np.deg2rad(tes_newxxyy[:, 1, :])
alpha = get_alpha(az, el)

plt.figure()
for i in range(9):
    plt.plot(tes_newxxyy[93, 0, i], tes_newxxyy[93, 1, i], 'o')
    plt.pause(1)

plt.figure()
for peak in range(9):
    az_coord = tes_newxxyy[:, 0, peak] #- tes_newxxyy[:, 0, 4]
    el_coord = tes_newxxyy[:, 1, peak] #- tes_newxxyy[:, 1, 4]
    plt.plot(az_coord, el_coord, '.', label='peak {}'.format(peak))
plt.xlabel('Az (°)')
plt.ylabel('el (°)')
plt.legend()
plt.show()

# Get all distances between TES
azel = tes_newxxyy[:, 0:2, :]
allfl_clip, fl_mean, fl_std = measure_global_focal_length(tes_xy, azel, npeaks=9, nsig=3)

finalmean = np.mean(fl_mean)
allstd_fl2 = [std ** 2 for std in fl_std]
finalstd = np.sqrt(np.sum(allstd_fl2) / 9)

nsig = 3
print(finalmean, finalstd)
plt.figure('FL_hist_{}GHz'.format(freq_source), figsize=(15, 12))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
for peak in range(9):
    plt.subplot(3, 3, peak + 1)
    plt.hist(allfl_clip[peak], bins=100,
             label='$FL = {:.5f} \pm {:.5f}$'.format(fl_mean[peak], fl_std[peak]))
    plt.title('Peak {}'.format(peak))
    plt.legend(fontsize=8)
    plt.xlim(0, 0.9)
plt.suptitle('Focal length histogram cut at {} sigma, {} GHz \n'
             '$f = {:.5f} \pm {:.5f}$'.format(nsig, freq_source, finalmean, finalstd))
plt.show()

# =============== Get one FL for each TES with the 9 peaks =====================
xy = tes_xy[rdist != 0.]

fl_mean, fl_std = measure_focal_length(tes_xy, alpha, goodtes=goodtes)
# Remove thermometers (they have a radial distance = 0)
fl_mean = fl_mean[:, rdist != 0.]
fl_std = fl_std[:, rdist != 0.]

final_mean, final_std = plot_flonfp(fl_mean, fl_std, xy, r, npeaks=1)

# Adding the small correction
fl_meancorr, fl_stdcorr = measure_focal_lengthnew(tes_xy, rdist, alpha, goodtes=goodtes)
# Remove thermometers (they have a radial distance = 0)
fl_meancorr = fl_meancorr[:, rdist != 0.]
fl_stdcorr = fl_stdcorr[:, rdist != 0.]

final_meancorr, final_stdcorr = plot_flonfp(fl_meancorr, fl_stdcorr, xy, r, npeaks=1)

# ============= Focal length with the center of the square ===================
center_square = np.zeros((256, 2, 1))
for tes in range(1, 257):
    center_square[tes - 1, :, 0] = get_centersquare_azel(rep, tes)

az_center = np.deg2rad(center_square[:, 0, :]) / np.cos(np.deg2rad(50.))
el_center = np.deg2rad(center_square[:, 1, :])
alpha_cs = get_alpha(az_center, el_center)
fl_mean_cs, fl_std_cs = measure_focal_length(tes_xy, alpha_cs, npeaks=1)

fl_mean_cs = fl_mean_cs[:, rdist != 0.]
fl_std_cs = fl_std_cs[:, rdist != 0.]

final_mean_cs, final_std_cs = plot_flonfp(fl_mean_cs, fl_std_cs, xy, r, npeaks=1)

# Same but remove outliers (not working, should be updated)
results = np.array([normalize(center_square[:, 0, 0]),
                    normalize(center_square[:, 1, 0])]).T

ok = DBSCAN_cut(results, doplot=True)
ntes = np.sum(ok)

fl_mean, fl_std = measure_focal_length(tes_xy[ok], alpha_cs[:, ok, ok], npeaks=1, ntes=ntes)

fl_mean = fl_mean[:, rdist[ok] != 0.]
fl_std = fl_std[:, rdist[ok] != 0.]

r_ok = rdist[ok][rdist[ok] != 0.]

final_mean, final_std = plot_flonfp(fl_mean, fl_std, xy[ok], r_ok, npeaks=1)

# ============= Compare with David synthetic beam simulations ===================
freq_source = 170
rep_david = '/home/lmousset/QUBIC/Qubic_work/Calibration/focal_length_measurement/David_simu/'
# Get the fit of the synthesized beams
rep = Qubic_DataDir(datafile='allFitSB_{}.pdf'.format(freq_source), datadir='/home/lmousset/QUBIC/Qubic_work')
print(rep)
tes_newxxyy, tesfit = get_all_fit(rep)

fig, axs = plt.subplots(2, 3, figsize=(10, 8))
axs = np.ravel(axs)
plt.suptitle('Frequency: {} GHz'.format(freq_source))

azel_tes = np.zeros((6, 2, 9))
for t, tes in enumerate([6, 37, 50, 58, 76, 93]):

    # Get the file from David
    file = glob.glob(rep_david + '*{}*/*_{}_*/*'.format(freq_source, tes))
    data = pd.read_csv(file[0], sep='\t', skiprows=0)

    x = data['x position (deg)'].iloc[::10]
    y = data['y position (deg)'].iloc[::10]

    nn = int(np.sqrt(len(x)))
    print(nn)
    a = data['amplitude'].iloc[::10]
    # amp = np.reshape(np.asarray(a), (nn, nn))

    dx = (x.iloc[1] - x.iloc[0]) / 2.
    dy = (y.iloc[1] - y.iloc[0]) / 2.
    extent = [x.iloc[0] - dx, x.iloc[-1] + dx, y.iloc[0] - dy, y.iloc[-1] + dy]

    azel_tes[t] = tes_newxxyy[tes-1, 0:2, :]
    azel_tes[t, 1, :] -= 50 # substract 50 to the elevation
    print(azel_tes[t])

    # Plot the fit measurement on the David simulation
    ax = axs[t]
    # ax.imshow(np.rot90(amp), extent=extent)
    ax.scatter(x, y, c=a, marker='s')
    ax.scatter(azel_tes[t, 0, :], azel_tes[t, 1, :], color='r', s=10)
    ax.set_title('TES {}'.format(tes))
    ax.axis('equal')

#     with open(rep_david + 'fit_peak_positions{}.txt'.format(freq_source), 'a') as f:
#         np.savetxt(f, azel_tes[t], fmt='%06.8f', header='TES {}'.format(tes))
# f.close()
plt.show()

# ================== Test with simulated beams from Qubic soft ==================
import healpy as hp
from scipy import ndimage
d['config'] = 'TD'
d['nside'] = 512
q = qubic.QubicInstrument(d)
s = qubic.QubicScene(d)

sb = q.get_synthbeam(s, idet=None, external_A=None, hwp_position=0)
print(sb.shape)

from pyoperators import Cartesian2SphericalOperator
delta = 14
sb_mask = sb.copy()
lonlat_center = np.zeros((248, 2))
vec_center = np.zeros((248, 3))
for tes in range(248):
    position = q.detector.center[tes]
    uvec = position / np.sqrt(np.sum(position ** 2, axis=-1))[..., None]
    lon, lat = Cartesian2SphericalOperator('azimuth,elevation', degrees=True)(uvec)

    lonlat_center[tes, :] = lon, lat + 180
    vec_center[tes, :] = hp.ang2vec(lon, lat, lonlat=True)

# Using the projected map
azel_peak = np.zeros((248, 2))
for m in range(248):
    print('\n m =', m)
    if m == 0:
        map_proj = hp.gnomview(sb[m, :], rot=(0, 90), reso=10, return_projected_map=True, no_plot=False)
        plt.show()
    else:
        map_proj = hp.gnomview(sb[m, :], rot=(0, 90), reso=10, return_projected_map=True, no_plot=True)

    # labeled_image, number_of_objects = ndimage.label(map_proj / np.mean(map_proj))
    # print(number_of_objects)
    # maxi = ndimage.measurements.center_of_mass(map_proj, labeled_image, np.arange(1, number_of_objects + 1))

    # maxi = ndimage.measurements.maximum_position(map_proj)
    maxi = np.where(map_proj==np.max(map_proj))
    maxi = np.array(maxi)
    if m == 0:
        plt.figure()
        plt.imshow(map_proj)
        plt.scatter(maxi[1], maxi[0], color='r')
        plt.show()

    print(maxi)
    azel_peak[m, 0] = np.mean(maxi[0])
    azel_peak[m, 1] = np.mean(maxi[1])

    print(azel_peak[m, :])

azel_peak[:, 0] = (azel_peak[:, 0] - 100)  * 10 / 60
azel_peak[:, 1] = (azel_peak[:, 1] - 100)  * 10 / 60 + 50

az_simu = np.expand_dims(azel_peak[:, 0], axis=1)
el_simu = np.expand_dims(azel_peak[:, 1], axis=1)

alpha_simu = get_alpha(az_simu, el_simu, deg=True)


# Using the healpix map
vec_peak = np.zeros((248, 3))
for m in range(248):
    print('\n m =', m)
    maxi_bis = np.where(sb[m, :] == np.max(sb[m, :]))[0]
    print('Maxi bis:', maxi_bis)
    # sb[m, maxi_bis[0]] = hp.UNSEEN
    # hp.mollview(sb[m, :], rot=(0, 90), cbar='hist')
    # plt.show()


    vec_peak[m, :] = hp.pix2vec(d['nside'], maxi_bis[0])
    print(vec_peak[m, :])

alpha_simu = np.arccos(np.dot(vec_center, vec_center.T))
alpha_simu = np.expand_dims(alpha_simu, axis=0)

# --------------------------------------------------
tes_xy_qubicsoft = q.detector.center[:, :2]

rdist_qubicsoft = np.sqrt(tes_xy_qubicsoft[:, 0]**2 + tes_xy_qubicsoft[:, 1]**2)

fl_mean, fl_std = measure_focal_length(tes_xy_qubicsoft, alpha_simu, npeaks=1, ntes=248,
                                       nsig=1, goodtes=None)
fl_meannew, fl_stdnew = measure_focal_lengthnew(tes_xy_qubicsoft, rdist_qubicsoft, alpha_simu,
                                                npeaks=1, ntes=248, nsig=2, goodtes=None)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(tes_xy_qubicsoft[:, 0], tes_xy_qubicsoft[:, 1], c=fl_meannew[0], marker='s', s=150)
plt.xlim((-0.055, 0.))
plt.ylim((-0.055, 0.))
plt.gca().set_aspect('equal')
plt.colorbar()

plt.subplot(122)
plt.scatter(tes_xy_qubicsoft[:, 0], tes_xy_qubicsoft[:, 1], c=fl_stdnew[0], marker='s', s=150)
plt.xlim((-0.055, 0.))
plt.ylim((-0.055, 0.))
plt.gca().set_aspect('equal')
plt.colorbar()

plt.show()


def measure_focal_length(tes_xy, alpha, npeaks=9, ntes=256, nsig=3, goodtes=None):
    tes_dist = cdist(tes_xy, tes_xy, 'euclidean')
    tanalpha = np.tan(alpha)

    if goodtes is not None:
        for i in range(ntes):
            if i < 128:
                tes = i + 1
                asic = 1
            else:
                tes = i - 128 + 1
                asic = 2
            index = tes2index(tes, asic)
            if index not in goodtes:
                print(i, index)
                tes_dist[i, :] = np.nan
                tes_dist[:, i] = np.nan

    fl_mean = np.zeros((npeaks, ntes))
    fl_std = np.zeros((npeaks, ntes))
    for peak in range(npeaks):
        print('Peak ', peak, '\n')
        focal_length = tes_dist / tanalpha[peak]
        print(focal_length)
        print(())
        print(focal_length.shape)
        for tes in range(ntes):
            print(tes)
            fl = focal_length[tes]
            fl = fl[~np.isnan(fl)]
            print('fl', fl.shape)
            fl_clip, mini, maxi = sigmaclip(fl, low=nsig, high=nsig)
            print(mini, maxi)
            print('fl_clip', fl_clip.shape)

            # Mean and STD for each TES
            fl_mean[peak, tes] = np.mean(fl_clip)
            fl_std[peak, tes] = np.std(fl_clip) / np.sqrt(len(fl_clip))

    return fl_mean, fl_std
