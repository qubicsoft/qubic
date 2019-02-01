#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import qubic
import fibtools as ft

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
plt.plot(p.index, p.azimuth, '.')
plt.xlabel('index pointing')
plt.ylabel('azimuth angle')

plt.subplot(3,1,2)
plt.plot(p.index, p.elevation, '.')
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

npix = 12 * d['nside']**2 # pixel number
sky = np.zeros(npix)
source = hp.pixelfunc.ang2pix(d['nside'], el, az, lonlat=True)
sky[source] = 1.
arcToRad = np.pi/(180*60.) #conversion factor
sky = hp.sphtfunc.smoothing(sky, fwhm=30*arcToRad)

hp.gnomview(sky, rot=source_coords)

x0 = np.zeros((d['nf_sub'], npix, 3))
x0[:,:,0] = sky

# Get TOD
def get_TOD(d, p, x0):
    # Polychromatic instrument model
    q = qubic.QubicMultibandInstrument(d)

    # Scene
    s = qubic.QubicScene(d)

    # Number of sub frequencies to build the TOD
    _, nus_edge_in, _, _, _, _ = qubic.compute_freq(d['filter_nu']/1e9, 
                                                    d['filter_relative_bandwidth'], d['nf_sub'])

    # Multi-band acquisition operator
    a = qubic.QubicMultibandAcquisition(q, p, s, d, nus_edge_in)

    TOD, _ = a.get_observation(x0, noiseless=True)
    return TOD

TOD = get_TOD(d, p, x0)
print('TOD.shape = ' + str(TOD.shape))
nTES = TOD.shape[0] 
npointings = TOD.shape[1]

TOD_mean = np.mean(TOD)

# Let's insert the 8 thermometers among the TES
thermos = np.array([3, 35, 67, 99, 131, 163, 195, 227]) # Indices of thermometers
TOD_thermo = np.zeros((nTES+8,npointings))
for ptg in xrange(npointings):
    TOD_thermo[:,ptg] = np.insert(TOD[:,ptg], thermos-np.arange(0,8), 0., axis=None)


# Check thermometers have been inserted at good places
for i in xrange(nTES+8):
    if TOD_thermo[i,0]==TOD_mean:
        if i in thermos:
            print(str(i) + ' ok')
        else:
            print(str(i) + ' bug') 

# Normalization 
TOD_thermo_mean = TOD_thermo #/ TOD_mean
#TOD_thermo_mean = np.abs(TOD_thermo - TOD_mean) / TOD_mean

#bug = np.concatenate((TOD_thermo_mean[128:,0], TOD_thermo_mean[0:128, 0]))

plt.figure('focal plane')
for ptg in xrange(40,75):
    img = ft.image_asics(all1=TOD_thermo_mean[:,ptg])
    plt.imshow(np.log(img), interpolation='nearest')
    plt.title('ptg' + str(ptg) + ' el=' + str(p.elevation[ptg]) + ' az=' + str(p.azimuth[ptg])))
    plt.pause(1)

plt.colorbar()

# Find the TES number on the  focal plane
test = np.arange(0, 256)
img = ft.image_asics(all1=test)
plt.figure('TES position')
plt.imshow(img, interpolation='nearest')
plt.colorbar()

#Test
truc = np.concatenate((test[128:], test[0:128]))
img = ft.image_asics(all1=truc)
plt.figure('TES position')
plt.imshow(img, interpolation='nearest')
plt.colorbar()

plt.figure('focal plane')
for ptg in xrange(40,75):
    bug = np.concatenate((TOD_thermo_mean[128:,ptg], TOD_thermo_mean[0:128, ptg]))
    img = ft.image_asics(all1=bug)
    plt.imshow(np.log(img), interpolation='nearest')
    plt.title('ptg' + str(ptg) + ' el=' + str(p.elevation[ptg]) + ' az=' + str(p.azimuth[ptg]))
    plt.pause(1)

