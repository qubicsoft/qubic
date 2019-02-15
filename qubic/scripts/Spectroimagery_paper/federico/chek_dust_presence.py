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

def get_maps_recon(TOD, d, p, nf_sub_rec, x0):
    maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(
        TOD, d, p, nf_sub_rec, x0=x0)
    if nf_sub_rec==1:
        maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
    cov = np.sum(cov, axis=0)
    maxcov = np.max(cov)
    unseen = cov < maxcov*0.1
    #diffmap = maps_convolved - maps_recon
    maps_convolved[:,unseen,:] = hp.UNSEEN
    maps_recon[:,unseen,:] = hp.UNSEEN
    #maps_recon += np.repeat(xav[None, ...], nf_sub_rec, axis=0)
    return maps_recon

t0 = time.time()

### Instrument ###
d = qubic.qubicdict.qubicDict()
d.read_from_file("parameters.dict")
p = qubic.get_pointing(d)

dust_coeffs = [0, 1.39e-2, 1.39e-1, 1.39, 1.39e1, 1.39e2]

map_res = []

nf_sub_rec = 4

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

    ##### Mapmaking #####
    maps_recon = get_maps_recon(TOD, d, p, nf_sub_rec, x0)
    maps_recon0 = get_maps_recon(TOD0, d, p, nf_sub_rec, x01)

    map_res.append(maps_recon - maps_recon0)
    
t1 = time.time()
print('************************** All Done in {} minutes'.format((t1-t0)/60))

mean_res_val = np.zeros(len(dust_coeffs))
 
fig, axes = mp.subplots(len(dust_coeffs), 4, figsize=(12, 15))
fig.tight_layout()
for i, d in enumerate(dust_coeffs):
    mean_res_val[i] = np.mean(map_res[i][..., 0][map_res[i][..., 0] > 0])
    for l in range(4):
        mp.axes(axes[i, l])
        hp.gnomview(map_res[i][l][:, 0],
                    rot=[-45, -60],
                    xsize=2000,
                    hold=True,
                    title="{}".format(d),
                    unit='K')

mp.show()
