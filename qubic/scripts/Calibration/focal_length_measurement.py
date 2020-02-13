from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import sigmaclip

from qubicpack.utilities import Qubic_DataDir
import qubic
import qubic.sb_fitting as sbfit
import qubic.selfcal_lib as sc

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

# Get the coordinates and the radial distances of each TES on the FP
# Get a dictionary
basedir = Qubic_DataDir(datafile='instrument.py', )
print('basedir : ', basedir)
dictfilename = basedir + '/dicts/global_source_oneDet.dict'

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)
print(d['detarray'])

d['config'] = 'FI'
q = qubic.QubicInstrument(d)

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

    if index is not None:
        index_place = np.where(q.detector.index == index)[0][0]
        x = q.detector.center[index_place, 0]
        y = q.detector.center[index_place, 1]
        tes_radial_dist[i] = np.sqrt(x ** 2 + y ** 2)
        print(tes, index, tes_radial_dist[i])
        tes_xy[i, :] = ([x, y])

# Check we have the right radial distance by plooting in on the FP
r = np.zeros((128, 2))
r[:, 0] = tes_radial_dist[:128]
r[:, 1] = tes_radial_dist[128:]

dist_on_fp = sc.tes_signal2image_fp(r, [1, 2])
plt.figure()
plt.imshow(dist_on_fp)
plt.title('Radial distances')
plt.colorbar()

# =============Focal length==================
# Get all distances between TES
tes_dist = cdist(tes_xy, tes_xy, 'euclidean')
tes_dist = np.triu(tes_dist)
print(np.max(tes_dist))
plt.figure()
plt.imshow(tes_dist)

allfl_clip, alltes_dist_cut, alltanalpha_cut = [], [], []
nsig = 3
for peak in range(9):
    # Get all angular distances between peak position
    azel = tes_newxxyy[:, 0:2, peak]
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

allmean_fl = [np.mean(fl) for fl in allfl_clip]
allstd_fl = [np.std(fl) / np.sqrt(len(fl)) for fl in allfl_clip]

plt.figure()
for peak in range(9):
    plt.subplot(3, 3, peak+1)
    plt.hist(allfl_clip[peak], bins=100,
             label='$FL = {:.5f} \pm {:.5f}$'.format(allmean_fl[peak], allstd_fl[peak]))
    plt.title('Peak {}'.format(peak))
    plt.legend()
    plt.xlim(0, 0.9)
plt.suptitle('Focal length hist cut at {} sigma'.format((nsig)))


plt.figure()
for peak in range(9):
    plt.subplot(3, 3, peak+1)
    plt.plot(alltanalpha_cut[peak], alltes_dist_cut[peak], 'o')
    plt.plot(alltanalpha_cut[peak], alltanalpha_cut[peak] * 0.3)

finalmean = np.mean(allmean_fl)
allstd_fl2 = [std**2 for std in allstd_fl]
finalstd = np.sqrt(np.sum(allstd_fl2) / 9)

print(finalmean, finalstd)
