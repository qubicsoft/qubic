from __future__ import division
import sys
import os
import time
import pysm
import qubic 

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import SpectroImLib as si

from pysimulators import FitsArray
from pysm.nominal import models
from scipy.constants import c

### Instrument ###
d = qubic.qubicdict.qubicDict()
dp = qubic.qubicdict.qubicDict()
d.read_from_file("parameters.dict")
dp.read_from_file("parameters.dict")
dp['MultiBand'] = False
dp['nf_sub'] = 1

### Sky ###
sky_config = {
    'synchrotron': models('s1', d['nside']),
    'dust': models('d1', d['nside']),
    'freefree': models('f1', d['nside']), #not polarized
    'cmb': models('c1', d['nside']),
    'ame': models('a1', d['nside'])} #not polarized

planck_sky = si.Planck_sky(sky_config, d)
x0_planck = planck_sky.get_sky_map()

qubic_sky = si.Qubic_sky(sky_config, d)
x0_qubic = qubic_sky.get_sky_map()

### QUBIC TOD ###
p = qubic.get_pointing(d)
TODq = si.create_TOD(d, p, x0_qubic)

### Put Q and U to zero ###
x0_planck[..., 1:3] = 0

### Planck TOD ###
TODp = si.create_TOD(dp, p, x0_planck)

### Create difference TOD ###
TOD = TODq - TODp

### QUBIC TOD with I=0 ###
x01 = np.copy(x0_qubic) #shape is (num of sub-bands, npix, IQU)
TOD0 = si.create_TOD(d, p, x01)

##### Mapmaking #####

#Numbers of subbands for spectroimaging
noutmin = 2
noutmax = 4
path = 'bpmaps'

for nf_sub_rec in np.arange(noutmin, noutmax+1):
    maps_recon, cov, nus, nus_edge = si.reconstruct_maps(
        TOD, d, p, nf_sub_rec)
    cov = np.sum(cov, axis=0)
    maxcov = np.max(cov)
    unseen = cov < maxcov*0.1
    maps_recon[:,unseen,:] = hp.UNSEEN
    maps_recon += np.repeat(x0_planck, nf_sub_rec, axis=0)

for nf_sub_rec in np.arange(noutmin, noutmax+1):
    maps_recon0, cov, nus, nus_edge = si.reconstruct_maps(
        TOD0, d, p, nf_sub_rec)
    cov = np.sum(cov, axis=0)
    maxcov = np.max(cov)
    unseen = cov < maxcov*0.1
    maps_recon0[:,unseen,:] = hp.UNSEEN
