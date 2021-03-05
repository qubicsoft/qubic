#!/bin/env python
from __future__ import division, print_function
import sys
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
from qubic import *
from pysimulators import FitsArray
import time
import os


# External library needed (not part of qubic yet)
import spectroimlib_tmp as si

### NB: Some of the pathes here are on my computer. You will have to change them
TD = False
path = QubicCalibration().path
if(TD):
	print('Technological Demonstrator')
	os.system('\cp '+path+'/TD_CalQubic_HornArray_v4.fits '+path+'/CalQubic_HornArray_v5.fits')
	os.system('\cp '+path+'/TD_CalQubic_DetArray_v3.fits '+path+'/CalQubic_DetArray_v4.fits')
else:
	print('First Instrument')
	os.system('rm -f '+path+'/CalQubic_HornArray_v5.fits')
	os.system('rm -f '+path+'/CalQubic_DetArray_v4.fits')


inst = QubicInstrument()
clf()
subplot(2,1,1)
inst.horn.plot()
subplot(2,1,2)
inst.detector.plot()

######## Default configuration
### Sky 
nside = 128
center = 0., -57.
center_gal = equ2gal(center[0], center[1])
dust_coeff = 1.39e-2
seed=None

### Detectors (for now using random pointing)
band = 150
TD = False
relative_bandwidth = 0.25
sz_ptg = 10.
nb_ptg = 1000
effective_duration = 30./365
ripples = False   
noiseless = False


### Mapmaking
tol = 1e-3

### Number of sub-bands to build the TOD
nf_sub_build = 10
nf_sub_rec = 3

parameters = {'nside':nside, 'center':center, 'dust_coeff': dust_coeff, 
				'band':band, 'relative_bandwidth':relative_bandwidth,
				'sz_ptg':sz_ptg, 'nb_ptg':nb_ptg, 'effective_duration':effective_duration, 
				'tol': tol, 'ripples':ripples,
				'nf_sub_build':nf_sub_build, 
				'nf_sub_rec': nf_sub_rec, 'noiseless':noiseless, 'seed':seed, 'TD':TD }


for k in parameters.keys(): print(k, parameters[k])



print('Creating input sky')
x0 = si.create_input_sky(parameters)

print('Creating pointing')
p = si.create_random_pointings(parameters['center'], parameters['nb_ptg'], parameters['sz_ptg'])

print('Creating TOD')
TOD = si.create_TOD(parameters, p, x0)

print('Doing Mapmaking on {} sub-map(s)'.format(nf_sub_rec))
maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, parameters, p, x0=x0)
if int(parameters['nf_sub_rec'])==1: maps_recon=np.reshape(maps_recon, np.shape(maps_convolved))
cov = np.sum(cov, axis=0)
maxcov = np.max(cov)
unseen = cov < maxcov*0.1
diffmap = maps_convolved - maps_recon
maps_convolved[:,unseen,:] = hp.UNSEEN
maps_recon[:,unseen,:] = hp.UNSEEN
diffmap[:,unseen,:] = hp.UNSEEN
therms = np.std(diffmap[:,~unseen,:], axis = 1)

stokes = ['I', 'Q', 'U']
for istokes in [0,1,2]:
	figure(istokes)
	if istokes==0: 
		xr=200
	else:
		xr=5
	for i in range(parameters['nf_sub_rec']):
		hp.gnomview(maps_convolved[i,:,istokes], rot=center_gal, reso=10, 
			sub=(parameters['nf_sub_rec'],3,3*i+1), min=-xr, max=xr, 
			title='Input '+stokes[istokes]+' SubFreq {}'.format(i))
		hp.gnomview(maps_recon[i,:,istokes], rot=center_gal, reso=10, 
			sub=(parameters['nf_sub_rec'],3,3*i+2), min=-xr, max=xr, 
			title='Output '+stokes[istokes]+' SubFreq {}'.format(i))
		hp.gnomview(diffmap[i,:,istokes], rot=center_gal, reso=10, 
			sub=(parameters['nf_sub_rec'],3,3*i+3), min=-xr, max=xr, 
			title='Residual '+stokes[istokes]+' SubFreq {}'.format(i))

