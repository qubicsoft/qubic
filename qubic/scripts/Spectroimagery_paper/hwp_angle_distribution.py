#!/bin/env python
# coding: utf-8
from __future__ import division
import sys
import os
import time

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D

import qubic
from pysimulators import FitsArray

import SpectroImLib as si

t0 = time.time()

### Instrument ###
d = qubic.qubicdict.qubicDict()
d.read_from_file("parameters.dict")

### Sky ###
skypars = {'dust_coeff':1.39e-2, 'r':0}
x0 = si.create_input_sky(d, skypars) #shape is (num of sub-bands, npix, IQU)

### Frame ###
p = qubic.get_pointing(d)
q = qubic.QubicMultibandInstrument(d)                        
s = qubic.QubicScene(d)

### Hit maps ###
rot_beams = si.get_hitmap(q, s, p)

hwp_angles = np.unique(p.angle_hwp)
hwp_angles_distribution = np.zeros((len(hwp_angles), s.npixel))
for i, angle in enumerate(hwp_angles):
    hwp_angles_distribution[i] = np.sum(
        rot_beams[np.where(p.angle_hwp == angle)[0]], axis=0)

longest_idx = np.sum(hwp_angles_distribution != 0, axis=1).argmin()
nonzero_idxs = np.argwhere(hwp_angles_distribution[longest_idx, :] != 0)
relevant_values = hwp_angles_distribution[:, nonzero_idxs.ravel()]

hwp_angles_distribution[:, 0] = np.full(len(hwp_angles), np.int(np.max(
    hwp_angles_distribution)))
hwp_angles_distribution[hwp_angles_distribution == 0] = np.NaN

### Plot maps ###
fig, axes = mp.subplots(4, 2, figsize=(12, 15))
fig.tight_layout()
k = 0
j = 0
for i, alpha in enumerate(hwp_angles):
    mp.axes(axes[k, j])
    hp.mollview(hwp_angles_distribution[i, :], coord=['C','G'], hold=True,
                title="HWP angle = {} deg".format(alpha), unit='hits', min=0,
                norm=None)
    k += 1
    if i == 3:
        k = 0
        j = 1
mp.savefig("/home/federico/Desktop/hwp_angular_distrib_{}pointings.png".format(
    d['npointings']))

### Plot 3D Hystogram ###
fig = mp.figure()
ax = fig.add_subplot(111, projection='3d')
x_data, y_data = np.meshgrid(np.arange(relevant_values.shape[1]),
                             np.arange(relevant_values.shape[0]))
x_data = x_data.flatten()
y_data = y_data.flatten()
z_data = relevant_values.flatten()

ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data)

mp.savefig("/home/federico/Desktop/hwp_angular_hystogram_{}pointings.png".format(
    d['npointings']))

# mp.show()
