from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from sklearn.cluster import DBSCAN

from qubicpack.utilities import Qubic_DataDir
import qubic.fibtools as ft
import qubic.sb_fitting as sbfit
import qubic.selfcal_lib as sc


def get_all_fit(rep):
    c50 = np.cos(np.radians(50))
    azmin = -15. / c50
    azmax = 15. / c50

    tes_fit = []
    for tes in range(1, 257):
        themap, az, el, fitmap, newxxyy = sbfit.get_flatmap(tes, rep, fitted_directory=rep + '/FitSB',
                                                            azmin=azmin, azmax=azmax)
        tes_fit.append(fitmap / np.sum(fitmap))

    return np.array(tes_fit)

def get_one_scan_azimuth(tes_fit, elev, quadrant=3):

    (ntes, nel, naz) = np.shape(tes_fit)
    print('# azimuth = ', naz)

    tes_fit_elev = tes_fit[:, elev, :]
    # Separate the 2 asics
    signal = np.empty((128, 2, naz))
    signal[:, 0, :] = tes_fit_elev[:128, :]
    signal[:, 1, :] = tes_fit_elev[128:, :]

    all_quartfp = []
    for i in range(naz):
        image_fp = sc.tes_signal2image_fp(signal[:, :, i], [1, 2])
        _, quart_fp = sc.get_real_fp(image_fp, quadrant=quadrant)
        all_quartfp.append(quart_fp)

    return all_quartfp


def measure_peak_dist(fp_images, det_size=3e-3):
    allbright = []
    distances = []
    centroids = []
    for i, fp in enumerate(fp_images):

        # Define a threshold
        x = fp[~np.isnan(fp)]  # Remove NAN values
        mean, std = ft.meancut(x, nsig=7)
        #     std = np.nanstd(quart_fp)
        threshold = 3 * std
        print('threshold: ', threshold)

        # Keep only pixels above threshold
        bright = fp > threshold
        brightpixels = np.column_stack(np.where(bright.T))

        # Apply DBSCAN
        clustering = DBSCAN(eps=1, min_samples=3).fit(brightpixels)
        labels = clustering.labels_

        nfound = len(np.unique(np.sort(labels)))
        print('nfound: ', nfound)

        if nfound == 3:
            allbright.append(bright)

            centers = np.zeros((nfound, 2))

            for k in range(nfound):
                ok = labels == k
                centers[k, :] = np.mean(brightpixels[ok, :], axis=0)
            centroids.append(centers)

            # Distance calculation
            a = np.sqrt((centers[0, 0] - centers[1, 0]) ** 2 + (centers[0, 1] - centers[1, 1]) ** 2) * det_size
            distances.append(a)

    return allbright, np.array(centroids), distances


def make_hist(distances, threshold):
    dist = np.array(distances)

    dist = dist[dist > threshold]
    print('Sample size = ', dist.shape)

    mean = np.mean(dist)
    std = np.std(dist)
    plt.figure()
    plt.hist(dist, bins=8)
    plt.xlabel('Distance a (m)')
    plt.title('Source freq = {}'.format(freq_source))

    plt.axvline(mean, color='r', linestyle='dashed', linewidth=2)

    min_ylim, max_ylim = plt.ylim()
    plt.text(mean * 1.02, max_ylim * 0.9, 'Mean: {:.3f} m \n Std: {:.3f} m'.format(mean, std))

    return mean, std

# Get the data
freq_source = 170
rep = Qubic_DataDir(datafile='allFitSB_{}.pdf'.format(freq_source))
print(rep)

tes_fit = get_all_fit(rep)

# Project one scan on the fp
all_quartfp = get_one_scan_azimuth(tes_fit, elev=75)

plt.figure()
for i, fp in enumerate(all_quartfp):
    plt.imshow(fp, vmin=0, vmax=2e-3)
    plt.pause(0.2)
    plt.title('Azimuth {}'.format(i))

allbright, centroids, distances = measure_peak_dist(all_quartfp)


plt.figure('bright')
for i, fp in enumerate(allbright):
    plt.imshow(fp)
    plt.plot(centroids[i][:, 0], centroids[i][:, 1], 'ro')
    plt.title('Azimuth {}'.format(i))
    plt.pause(0.2)
    plt.clf()

mean, std = make_hist(distances, threshold=0.02)
print(mean, std)

