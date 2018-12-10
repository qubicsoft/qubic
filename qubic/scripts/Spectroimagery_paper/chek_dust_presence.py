#!/bin/env python
from __future__ import division
import sys
import os
import time

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp

import qubic
from pysimulators import FitsArray

import SpectroImLib as si

t0 = time.time()

### Instrument ###
d = qubic.qubicdict.qubicDict()
d.read_from_file("parameters.dict")

dust_coeffs = [0, 1.39e-2, 1.39e-1, 1.39, 1.39e1, 1.39e2]

mean_res = np.zeros(len(dust_coeffs))
mean_res_max = np.zeros(len(dust_coeffs))

for j, dust in enumerate(dust_coeffs):
    ### Sky ###
    skypars = {'dust_coeff':dust, 'r':0} # 1.39e-2
    x0 = si.create_input_sky(d, skypars) #shape is (num of sub-bands, npix, IQU)
    
    ### QUBIC TOD ###
    p = qubic.get_pointing(d)
    TODq = si.create_TOD(d, p, x0)
    
    ### Put Q and U to zero ###
    x0[..., 1:3] = 0
    
    ### Planck TOD ###
    xav = np.mean(x0, axis=0)
    TODp = si.create_TOD(d, p, np.repeat(xav[None, ...], d['nf_sub'], axis=0))
    
    ### Create difference TOD ###
    TOD = TODq - TODp
    
    ### QUBIC TOD with I=0 ###
    x01 = si.create_input_sky(d, skypars) #shape is (num of sub-bands, npix, IQU)
    x01[..., 0] = 0
    TOD0 = si.create_TOD(d, p, x01)

    for i in range(TOD.shape[0]):                             
        mp.figure()
        mp.plot((TOD[i, :] - TOD0[i, :]) / TOD[i, :])
        mp.savefig("/home/federico/Desktop/TOD_relative_diff_dust={}.png".format(dust))

    mean_res[j] = np.mean(np.abs(TOD - TOD0))
    mean_res_max[j] = np.abs(TOD.max() - TOD0.max())
    
mp.figure()
mp.plot(mean_res, 'x')
mp.xticks(np.arange(len(dust_coeffs)), dust_coeffs)
mp.savefig("/home/federico/Desktop/TOD_coeffVSres.png")

mp.figure()
mp.plot(mean_res_max, 'x')
mp.xticks(np.arange(len(dust_coeffs)), dust_coeffs)
mp.savefig("/home/federico/Desktop/TOD_coeffVSres_max.png")
