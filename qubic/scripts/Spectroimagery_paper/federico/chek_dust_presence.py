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


def make_plots(maprec, dust_coeffs):
    fig, axes = mp.subplots(len(dust_coeffs), 4, figsize=(12, 15))
    fig.tight_layout()
    for i, d in enumerate(dust_coeffs):
        for l in range(4):
            mp.axes(axes[i, l])
            hp.gnomview(maprec[i][l][:, 0],
                        rot=[-45, -60],
                        xsize=2000,
                        hold=True,
                        title="{}".format(d))

    mp.show(block=False)

    
t0 = time.time()

### Instrument ###
d = qubic.qubicdict.qubicDict()
d.read_from_file("parameters.dict")
p = qubic.get_pointing(d)

dust_coeffs = [0, 1.39e-2, 1.39e-1, 1.39, 1.39e1, 1.39e2]

map_res = []
map_recon = []
map_recon0 = []

nf_sub_rec = 4

for j, dust in enumerate(dust_coeffs):
    ### Sky ###
    sky_config = {
        'synchrotron': models('s1', d['nside']),
        'dust': models('d1', d['nside']),
        'freefree': models('f1', d['nside']), #not polarized
        'cmb': models('c1', d['nside']),
        'ame': models('a1', d['nside']),  #not polarized
    }

    skypars = {'dust_coeff':dust, 'r':0} # 1.39e-2
    x0 = si.create_input_sky(d, skypars) #shape is (num of sub-bands, npix, IQU)
    
    ### QUBIC TOD ###
    p = qubic.get_pointing(d)
    TODq = si.create_TOD(d, p, x0)
        
    ### Planck TOD ###
    ### Put Q and U to zero ###
    x0[..., 1:3] = 0
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

    map_recon.append(maps_recon)
    map_recon0.append(maps_recon0)
    map_res.append(maps_recon - maps_recon0)
    
t1 = time.time()
print('************************** All Done in {} minutes'.format((t1-t0)/60))

make_plots(map_recon, dust_coeffs)
make_plots(map_recon0, dust_coeffs)
make_plots(map_res, dust_coeffs)

mean_res_val = np.zeros(len(dust_coeffs))
mean_res_val0 = np.zeros(len(dust_coeffs))
for i in range(len(dust_coeffs)):
    mean_res_val[i] = np.std(maprec[i][..., 0][maprec[i][..., 0] > 0])
    mean_res_val0[i] = np.std(maprec0[i][..., 0][maprec0[i][..., 0] > 0])

mp.figure()
mp.plot(mean_res_val, "x")
mp.plot(mean_res_val0, "x")
mp.show(block=False)
