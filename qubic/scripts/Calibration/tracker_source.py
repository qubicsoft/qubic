#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 21:35:22 2019

@author: Louise
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import qubic

dictfilename = '/home/louisemousset/QUBIC/MyGitQUBIC/qubic/qubic/scripts/global_source.dict' 
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

# Get sweeping pointing in a fix frame
p = qubic.get_pointing(d)
el_angles = np.unique(p.elevation) # all different elevation angles 
az_angles = np.unique(p.azimuth) # all different azimuth angles 
print('el_angles = ' + str(el_angles))
print('az_angles = ' + str(az_angles))

plt.figure('pointing')
plt.subplot(3,1,1)
plt.plot(p.index, p.azimuth)
plt.xlabel('index pointing')
plt.ylabel('azimuth angle')

plt.subplot(3,1,2)
plt.plot(p.index, p.elevation)
plt.xlabel('index pointing')
plt.ylabel('elevation angle')

plt.subplot(3,1,3)
plt.plot(p.azimuth, p.elevation, '.')
plt.xlabel('azimuth angle')
plt.ylabel('elevation angle')

# Point source
az = 50.
el = 50.
source_coords = (az, el)

npix = 12*d['nside']**2 # pixel number
sky = np.zeros(npix)
source = hp.pixelfunc.ang2pix(d['nside'], el, az, lonlat=True)
sky[source] = 1.
arcToRad = np.pi/(180*60.) #conversion factor
sky = hp.sphtfunc.smoothing(sky, fwhm=30*arcToRad)

hp.gnomview(sky, rot=source_coords)

x0 = np.zeros((d['nf_sub'], npix, 3))
x0[:,:,0] = sky


# Polychromatic instrument model
q = qubic.QubicMultibandInstrument(d)
q[0].detector.plot()

# Scene
s = qubic.QubicScene(d)

# Number of sub frequencies to build the TOD
_, nus_edge_in, _, _, _, _ = qubic.compute_freq(d['filter_nu']/1e9, 
                                                d['filter_relative_bandwidth'], d['nf_sub'])

# Multi-band acquisition operator
a = qubic.QubicMultibandAcquisition(q, p, s, d, nus_edge_in)

# Get TOD
TOD, _ = a.get_observation(x0, noiseless=True)

# We have to find a way to have the position of the TES in the focal plane from its number 
# and then we can plot TOD on the focal plane as a function of the pointing.
plt.figure()
for i in xrange(10):
    plt.plot(TOD[:, i], '.', label='time = '+str(i))
plt.legend()