from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from qubicpack.utilities import Qubic_DataDir
import qubic
import qubic.sb_fitting as sbfit
import qubic.selfcal_lib as sc
import qubic.fibtools as ft

from qubicpack.pixel_translation import tes2index


def get_all_fit(rep):
    c50 = np.cos(np.radians(50))
    azmin = -15. / c50
    azmax = 15. / c50

    tes_fit = []
    tes_newxxyy = []
    for tes in range(256):
        themap, az, el, fitmap, newxxyy = sbfit.get_flatmap(tes+1, rep, fitted_directory=rep + '/FitSB',
                                                            azmin=azmin, azmax=azmax)
        tes_newxxyy.append(newxxyy)
        tes_fit.append(fitmap / np.sum(fitmap))

    return np.array(tes_newxxyy), np.array(tes_fit)


# Get the data
freq_source = 170
rep = Qubic_DataDir(datafile='allFitSB_{}.pdf'.format(freq_source))
print(rep)

tes_newxxyy, tesfit = get_all_fit(rep)

plt.figure()
for i in range(9):
    plt.plot(tes_newxxyy[93, 0, i], tes_newxxyy[93, 1, i], 'o')
    plt.pause(0.4)

plt.figure()
for peak in range(9):
    az_coord = tes_newxxyy[:, 0, peak]
    el_coord = tes_newxxyy[:, 1, peak]

    plt.plot(az_coord, el_coord, '.', label='peak {}'.format(peak))
plt.xlabel('Az (°)')
plt.ylabel('el (°)')
plt.legend()

# Get the radial distance for each TES on the FP
# Get a dictionary
basedir = Qubic_DataDir(datafile='instrument.py', )
print('basedir : ', basedir)
dictfilename = basedir + '/dicts/global_source_oneDet.dict'

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)
print(d['detarray'])

d['config'] = 'FI'
q = qubic.QubicInstrument(d)

tes_coords = np.zeros((256, 5))
for i in range(256):
    if i < 128:
        tes = i + 1
        asic = 1
    else:
        tes = i - 128 + 1
        asic = 2
    index = tes2index(tes, asic)

    if index is not None:
        index_place = np.where(q.detector.index == index)[0][0]
        x = q.detector.center[index_place, 0]
        y = q.detector.center[index_place, 1]
        r = np.sqrt(x ** 2 + y ** 2)
        print(tes, index, r)
        tes_coords[i, :] = ([tes, index, x, y, r])

# Check we have the right distance
tes_dist_radial = np.zeros((128, 2))
tes_dist_radial[:, 0] = tes_coords[:128, 4]
tes_dist_radial[:, 1] = tes_coords[128:, 4]

dist_on_fp = sc.tes_signal2image_fp(tes_dist_radial, [1, 2])
plt.figure()
plt.imshow(dist_on_fp)
plt.colorbar()

# Focal length
#
# dist_testes = np.zeros((256, 256))
# angular_sep = np.zeros((256, 256))
# peak = 2
# for tes1 in range(256):
#     x1 = tes_coords[tes1, 2]
#     y1 = tes_coords[tes1, 3]
#     az1 = tes_newxxyy[tes1, 0, peak]
#     el1 = tes_newxxyy[tes1, 1, peak]
#     for tes2 in range(256):
#         x2 = tes_coords[tes2, 2]
#         y2 = tes_coords[tes2, 3]
#         dist_testes[tes1, tes2] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#
#         az2 = tes_newxxyy[tes2, 0, peak]
#         el2 = tes_newxxyy[tes2, 1, peak]
#         alpha = np.sqrt((az2 - az1) ** 2 + (el2 - el1) ** 2)
#         angular_sep[tes1, tes2] = np.deg2rad(alpha)



tes_xy = tes_coords[:, 2:4]
tes_dist = cdist(tes_xy, tes_xy, 'euclidean')
plt.figure()
plt.imshow(tes_dist)

all_focal_length = []
all_mean = []
all_std = []
for peak in range(9):
    azel = tes_newxxyy[:, 0:2, peak]
    alpha = np.deg2rad(cdist(azel, azel, 'euclidean'))

    focal_length = np.ravel(tes_dist / np.tan(alpha))

    # if focal_length is not np.nan and focal_length <1.:
    fl = focal_length[~np.isnan(focal_length)]
    fl = fl[fl<2.]
    # plt.figure()
    # plt.hist(fl, bins=100)
    print(np.max(fl))
    all_focal_length.append(fl)
    print(fl.shape)
    mean_fl, std_fl = ft.meancut(fl, 3, disp=False)
    all_mean.append(mean_fl)
    all_std.append(std_fl*np.sqrt(2))

# all_focal_length = np.array(all_focal_length)
# print(all_focal_length.shape)

plt.figure()
for peak in range(9):
    plt.subplot(3, 3, peak+1)
    plt.hist(all_focal_length[peak], bins=100,
             label='Focal length = ${:.5f} \pm {:.5f}$'.format(all_mean[peak], all_std[peak]))
    plt.legend()
    plt.xlim(0, 2)


# mean_fl = np.nanmean(focal_length, axis=0)
# std_fl = np.nanstd(focal_length, axis=0)

r = tes_coords[:, 4]
plt.figure()
plt.hist(fl, bins=30,
         label='Focal length = ${} \pm {}$'.format(mean_fl, std_fl))
plt.legend()

plt.figure()
tanalpha = np.tan(alpha)
plt.plot(tanalpha, tes_dist, 'o')
plt.plot(tanalpha, tanalpha * 0.3)
#
