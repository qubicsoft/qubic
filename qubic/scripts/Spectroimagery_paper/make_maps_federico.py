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

### Sky ###
skypars = {'dust_coeff':1.39e-2, 'r':0}
x0 = si.create_input_sky(d, skypars) #shape is (num of sub-bands, npix, IQU)
#x0[..., 1:3] = np.zeros_like(x0[..., 1:3])

### QUBIC TOD ###
p = qubic.get_pointing(d)
TODq = si.create_TOD(d, p, x0)

### Planck TOD ###
xav = np.mean(x0, axis=0)
TODp = si.create_TOD(d, p, np.repeat(xav[None, ...], d['nf_sub'], axis=0))

### Create difference TOD ###
TOD = TODq - TODp

##### Mapmaking #####

#Numbers of subbands for spectroimaging
noutmin = 1
noutmax = 4
path = 'bpmaps'
for nf_sub_rec in np.arange(noutmin, noutmax+1):
    print('-------------------------- Map-Making on {} sub-map(s)'.format(nf_sub_rec))
    maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p, nf_sub_rec, x0=x0)
    if nf_sub_rec==1:
        maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
    cov = np.sum(cov, axis=0)
    maxcov = np.max(cov)
    unseen = cov < maxcov*0.1
    #diffmap = maps_convolved - maps_recon
    maps_convolved[:,unseen,:] = hp.UNSEEN
    maps_recon[:,unseen,:] = hp.UNSEEN
    maps_recon += np.repeat(xav[None, ...], nf_sub_rec, axis=0)
    #diffmap[:,unseen,:] = hp.UNSEEN
    #therms = np.std(diffmap[:,~unseen,:], axis = 1)
    
    print('************************** Map-Making on {} sub-map(s)Done'.format(nf_sub_rec))

    FitsArray(nus).save(path+'_nf{0}'.format(nf_sub_rec)+'_nus.fits')
    FitsArray(nus_edge).save(path+'_nf{0}'.format(nf_sub_rec)+'_nus_edges.fits')
    FitsArray(maps_convolved).save(path+'_nf{0}'.format(nf_sub_rec)+'_maps_convolved.fits')
    FitsArray(maps_recon).save(path+'_nf{0}'.format(nf_sub_rec)+'_maps_recon.fits')
    
    t1 = time.time()
    print('************************** All Done in {} minutes'.format((t1-t0)/60))



