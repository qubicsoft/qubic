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


dictfilename = 'test_spectroim.dict' 

name = 'simu_QU'

#Numbers of subbands for spectroimaging
noutmin = 1
noutmax = 4

## Input sky parameters
skypars = {'dust_coeff':1.39e-2, 'r':0}

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

#Print dictionary and others parameters
#Save a file with al parameters
tem = sys.stdout
sys.stdout = f = open('simu_QU.txt','wt')

print('Simulation General Name: ' + name)
print('Dictionnary File: ' + dictfilename)
for k in d.keys(): 
    print(k, d[k])

print('Minimum Number of Sub Frequencies: {}'.format(noutmin))
print('Maximum Number of Sub Frequencies: {}'.format(noutmax))
print(skypars)

sys.stdout = tem
f.close()

####################################################################################################


##### Sky Creation #####

t0 = time.time()
x0 = si.create_input_sky(d, skypars)
t1 = time.time()
print('********************* Input Sky done in {} seconds'.format(t1-t0))

##### Pointing strategy #####
p = qubic.get_pointing(d)
t2 = time.time()
print('************************** Pointing done in {} seconds'.format(t2-t1))


##### TOD making #####
TOD = si.create_TOD(d, p, x0)

##### Mapmaking #####
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
    #diffmap[:,unseen,:] = hp.UNSEEN
    #therms = np.std(diffmap[:,~unseen,:], axis = 1)
    
    print('************************** Map-Making on {} sub-map(s)Done'.format(nf_sub_rec))

    FitsArray(nus).save(name+'_nf{0}'.format(nf_sub_rec)+'_nus.fits')
    FitsArray(nus_edge).save(name+'_nf{0}'.format(nf_sub_rec)+'_nus_edges.fits')
    FitsArray(maps_convolved).save(name+'_nf{0}'.format(nf_sub_rec)+'_maps_convolved.fits')
    FitsArray(maps_recon).save(name+'_nf{0}'.format(nf_sub_rec)+'_maps_recon.fits')
    
    t1 = time.time()
    print('************************** All Done in {} minutes'.format((t1-t0)/60))
