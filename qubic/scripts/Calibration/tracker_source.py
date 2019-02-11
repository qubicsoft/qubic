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

x0 = np.zeros((d['nf_sub'], npix, 3))
x0[:, :, 0] = sky


# =========== Get TOD for the FI =========
def get_tod(d, p, x0):
    # Polychromatic instrument model
    q = qubic.QubicMultibandInstrument(d)

    # Scene
    s = qubic.QubicScene(d)

    # Number of sub frequencies to build the TOD
    _, nus_edge_in, _, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9,
                                                    d['filter_relative_bandwidth'], d['nf_sub'])

    # Multi-band acquisition operator
    a = qubic.QubicMultibandAcquisition(q, p, s, d, nus_edge_in)

    tod, _ = a.get_observation(x0, noiseless=True)

    return q, tod


q, tod_fi = get_tod(d, p, x0)

# ========= Project TOD on the focal plane ============
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

#  Plot the signal on the focal plane for each pointing
plt.figure('tracker')
tod_focal_plane = np.zeros((size_fi, size_fi, nptg))
for ptg in xrange(nptg):
    j = 0
    for i in xrange(1156):
        pos = np.where(focal_plane == i)
        # print(pos[0].size)

        #  For the TES 0
        if pos[0].size == 165:
            tod_focal_plane[0, 17, ptg] = tod_fi[j, ptg]
            j += 1

        # For other TES
        elif pos[0].size == 1:
            # print(j)
            tod_focal_plane[pos[0][0], pos[1][0], ptg] = tod_fi[j, ptg]
            j += 1
            # print(pos)

    plt.imshow(tod_focal_plane[:, :, ptg], interpolation='nearest', origin='lower')
    plt.title('ptg' + str(ptg) + ' el=' + str(p.elevation[ptg]) + ' az=' + str(p.azimuth[ptg]))
    plt.pause(0.5)
plt.colorbar()
