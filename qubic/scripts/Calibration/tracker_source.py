#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits
import qubic

basedir = '/home/louisemousset/QUBIC/MyGitQUBIC'
dictfilename = basedir + '/qubic/qubic/scripts/global_source.dict'

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

# Get sweeping pointing in a fix frame
p = qubic.get_pointing(d)
nptg = len(p.index)  # Pointing number

el_angles = np.unique(p.elevation)  # all different elevation angles
az_angles = np.unique(p.azimuth)  # all different azimuth angles
print('el_angles = ' + str(el_angles))
print('az_angles = ' + str(az_angles))

plt.figure('pointing')
plt.subplot(3, 1, 1)
plt.plot(p.index, p.azimuth, '.')
plt.xlabel('index pointing')
plt.ylabel('azimuth angle')

plt.subplot(3, 1, 2)
plt.plot(p.index, p.elevation, '.')
plt.xlabel('index pointing')
plt.ylabel('elevation angle')

plt.subplot(3, 1, 3)
plt.plot(p.azimuth, p.elevation, '.')
plt.xlabel('azimuth angle')
plt.ylabel('elevation angle')

# ============= Point source ==========
az, el = d['fix_azimuth']['az'], d['fix_azimuth']['el']
print('source_coords = {0},{1}'.format(az, el))

npix = 12 * d['nside'] ** 2  # pixel number
sky = np.zeros(npix)
source_pix = hp.pixelfunc.ang2pix(d['nside'], el, az, lonlat=True)
sky[source_pix] = 1.
arcToRad = np.pi / (180 * 60.)  # conversion factor
sky = hp.sphtfunc.smoothing(sky, fwhm=30 * arcToRad)

hp.gnomview(sky, rot=(az, el))

## Sky x0 with only I

# For a Multiband instrument
x0 = np.zeros((d['nf_sub'], npix, 3))
x0[:, :, 0] = sky

# For a PolyInstrument
# x0 = np.zeros((npix, 3))
# x0[:, 0] = sky


# =========== Get TOD for the FI =========
def get_tod(d, p, x0, closed_horns=None):
    """

    Parameters
    ----------
    d : dictionnary
    p : pointing
    x0 : sky
    closed_horns : list
        index of closed horns

    Returns
    -------
    Returns an instrument with closed horns and TOD
    """
    # Polychromatic instrument model
    q = qubic.QubicMultibandInstrument(d)
    # q = qubic.QubicInstrument(d)
    if closed_horns is not None:
        for i in xrange(d['nf_sub']):
            for h in closed_horns:
                q[i].horn.open[h]=False

    # Scene
    s = qubic.QubicScene(d)

    # Number of sub frequencies to build the TOD
    _, nus_edge_in, _, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9,
                                                    d['filter_relative_bandwidth'], d['nf_sub'])

    # Multi-band acquisition operator
    a = qubic.QubicMultibandAcquisition(q, p, s, d, nus_edge_in)
    # a = qubic.QubicPolyAcquisition(q, p, s, d)

    tod = a.get_observation(x0, convolution=False, noiseless=True)

    return q, tod


q, tod_fi = get_tod(d, p, x0, closed_horns=None)
q[0].horn.plot()


# ========= Get indices of TES on the focal plane ============
detarray = fits.open(basedir + '/qubic/qubic/calfiles/CalQubic_DetArray_FI.fits')

tes_index = detarray['index'].data
plt.clf()
plt.imshow(tes_index, interpolation='nearest', origin='lower')

# size of one side of the focal plane
size_fi = tes_index.shape[0]

# Position of the TES in the array
tes_position = detarray['removed'].data
plt.clf()
plt.imshow(tes_position, interpolation='nearest', origin='lower')  # 0 on focal plane, 1 elsewhere

plt.clf()
tes_position_rev = tes_position * (-1) + 1
plt.imshow(tes_position_rev, interpolation='nearest', origin='lower')  # reverse

# Indices of TES on the focal plane, 0 outside
focal_plane = tes_index * tes_position_rev
plt.clf()
plt.imshow(focal_plane, interpolation='nearest', origin='lower')

# ========= Project TOD on the focal plane for each pointing ============
def plot_focalplane(size, nptg, focal_plane, tod):
    plt.figure('focalplane')
    tod_focal_plane = np.zeros((size, size, nptg))
    for ptg in xrange(nptg):
        j = 0
        for i in xrange(1156):
            pos = np.where(focal_plane == i)
            # print(pos[0].size)

            #  For the TES 0
            if pos[0].size == 165:
                tod_focal_plane[0, 17, ptg] = tod[j, ptg]
                j += 1

            # For other TES
            elif pos[0].size == 1:
                # print(j)
                tod_focal_plane[pos[0][0], pos[1][0], ptg] = tod[j, ptg]
                j += 1
                # print(pos)

        plt.imshow(tod_focal_plane[:, :, ptg], interpolation='nearest', origin='lower')
        plt.title('ptg' + str(ptg) + ' el=' + str(p.elevation[ptg]) + ' az=' + str(p.azimuth[ptg]))
        plt.pause(0.5)
    plt.colorbar()
    return

plot_focalplane(size_fi, nptg, focal_plane, tod_fi)

# ========= Same thing with closed horns to see the franges ============
a = 40
b = 100
q_a, tod_fi_a = get_tod(d, p, x0, closed_horns=[a])
q_a[0].horn.plot()
plt.pause(0.5)

q_b, tod_fi_b = get_tod(d, p, x0, closed_horns=[b])
q_b[0].horn.plot()
plt.pause(0.5)

q_ab, tod_fi_ab = get_tod(d, p, x0, closed_horns=[a, b])
q_ab[0].horn.plot()

#  Plot the interference pattern on the focal plane for each pointing
tod_frange = tod_fi - tod_fi_a - tod_fi_b + tod_fi_ab
plot_focalplane(size_fi, nptg, focal_plane, tod_frange)
