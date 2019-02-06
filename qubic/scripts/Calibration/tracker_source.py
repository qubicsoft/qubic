#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import qubic
import fibtools as ft

dictfilename = '/home/louisemousset/QUBIC/MyGitQUBIC/qubic/qubic/scripts/global_source.dict'
# dictfilename = '/home/james/moussetqubic/qubic/qubic/scripts/global_source.dict'
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

# Get sweeping pointing in a fix frame
p = qubic.get_pointing(d)

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

# Point source
az = 50.
el = 50.
source_coords = (az, el)
print('source_coords = {}'.format(source_coords))

npix = 12 * d['nside'] ** 2  # pixel number
sky = np.zeros(npix)
source_pix = hp.pixelfunc.ang2pix(d['nside'], el, az, lonlat=True)
sky[source_pix] = 1.
arcToRad = np.pi / (180 * 60.)  # conversion factor
sky = hp.sphtfunc.smoothing(sky, fwhm=30 * arcToRad)

hp.gnomview(sky, rot=source_coords)

x0 = np.zeros((d['nf_sub'], npix, 3))
x0[:, :, 0] = sky


# Get TOD
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

# Take tod of the TD instrument
nTES = tod_fi.shape[0] / 4
npointings = tod_fi.shape[1]
print('nTES = {0} npointings = {1}'.format(nTES, npointings))
tes_simu = q[0].detector.index[q[0].detector.quadrant == 0]

tod_td = np.take(tod_fi, tes_simu, axis=0)

plt.figure('TOD')
for i in xrange(248):
    plt.plot(tod_td[i, :])

tod_mean = np.mean(tod_td)

# Let's insert the 8 thermometers among the TES
thermos = np.array([3, 35, 67, 99, 131, 163, 195, 227])  # Indices of thermometers
tod_thermo = np.zeros((nTES + 8, npointings))
for ptg in xrange(npointings):
    tod_thermo[:, ptg] = np.insert(tod_td[:, ptg], thermos - np.arange(0, 8), 0., axis=None)

# Check thermometers have been inserted at good places
for i in xrange(nTES + 8):
    if tod_thermo[i, 0] == tod_mean:
        if i in thermos:
            print(str(i) + ' ok')
        else:
            print(str(i) + ' bug')


# Use the experimental TES positions
def mix_tod(tod, thermos):

    tes_exp = ft.image_asics(all1=np.arange(0, 256))
    tes_exp = np.reshape(tes_exp, 17**2)
    tes_exp = tes_exp[~np.isnan(tes_exp)]
    #tes_exp = np.delete(tes_exp, thermos)


    tod_mix = np.copy(tod)
    for i in xrange(256):
        if i not in thermos:
            print(i)
            exp = np.int(tes_exp[i])
            print(exp)
            tod_mix[i, :] = tod[exp, :]
    return tod_mix

tod_thermo_mix = mix_tod(tod_thermo, thermos)


plt.figure('focal plane')
for ptg in xrange(0, 35):
    img = ft.image_asics(all1=tod_thermo_mix[:, ptg])
    plt.imshow(img, interpolation='nearest')
    plt.title('ptg' + str(ptg) + ' el=' + str(p.elevation[ptg]) + ' az=' + str(p.azimuth[ptg]))
    plt.pause(1)

plt.colorbar()

# Find the TES number on the  focal plane
test = np.arange(0, 256)
img = ft.image_asics(all1=test)
plt.figure('TES position')
plt.imshow(img, interpolation='nearest')
plt.colorbar()
