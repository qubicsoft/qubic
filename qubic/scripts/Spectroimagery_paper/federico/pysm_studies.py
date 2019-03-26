from __future__ import division
import sys
import os
import time

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import pysm

import qubic 
from pysimulators import FitsArray
from pysm.nominal import models
from scipy.constants import c

import SpectroImLib as si

def create_planck_bandwidth(planck_relative_bandwidth, planck_central_nu, length=100):
    halfband = planck_relative_bandwidth * planck_central_nu / 2
    bandwidths = np.zeros((len(planck_relative_bandwidth), length))
    v = []
    for i, hb in enumerate(halfband):
        bandwidths[i] = np.linspace(planck_central_nu[i] - hb, planck_central_nu[i] + hb, num=length)
        v.append((bandwidths[i], np.ones_like(bandwidths[i])))
    return v

def get_planck_resolution(nu, D=1):
    wl = c / nu / 1e9 # nust must be in Hz
    R = wl / D # D is the telescope diameter in meters
    return R * 60 * 180 / np.pi # result is expressed in arcmin

def read_map(nside, output_directory, output_prefix):
    map_list = [s for s in os.listdir(output_directory) if output_prefix in s]
    map_list = [m for m in map_list if 'total' in m]
    maps = np.zeros((len(map_list), hp.nside2npix(nside), 3))
    for i, title in enumerate(map_list):
        maps[i] = hp.read_map(title, field=(0, 1, 2)).T
    return maps

output_directory = "./" 
output_prefix_planck = "planck_sky"
output_prefix_qubic = "qubic_sky"

### Instrument ###
d = qubic.qubicdict.qubicDict()
dp = qubic.qubicdict.qubicDict()
d.read_from_file("parameters.dict")
dp.read_from_file("parameters.dict")
dp['MultiBand'] = False
dp['nf_sub'] = 1

# number of sub frequencies
_, nus_edge_in, central_nus, deltas, _, _ = qubic.compute_freq(
    d['filter_nu']/1e9,
    d['filter_relative_bandwidth'],
    d['nf_sub']) # Multiband instrument model

### Sky ###
sky_config = {
    'synchrotron': models('s1', d['nside']),
    'dust': models('d1', d['nside']),
    'freefree': models('f1', d['nside']), #not polarized
    'cmb': models('c1', d['nside']),
    'ame': models('a1', d['nside'])} #not polarized

sky = pysm.Sky(sky_config)

planck_central_nu = np.array([30, 44, 70, 100, 143, 217, 353, 545, 857])
planck_relative_bandwidth = np.array([0.2, 0.2, 0.2, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33])
planck_beams = np.array([33, 24, 14, 9.2, 7.1, 5.5, 5, 5, 5])
planck_Isensitivities_pixel = np.array([2, 2.7, 4.7, 2, 2.2, 4.8, 14.7, 147, 6700])
planck_Psensitivities_pixel = np.array([2.8, 3.9, 6.7, np.NaN, 4.2, 9.8, 29.8, np.NaN, np.NaN])
planck_channels = create_planck_bandwidth(planck_relative_bandwidth, planck_central_nu)
planck_channels_names = ['33_GHz', '44_GHz','70_GHz','100_GHz', '143_GHz', '217_GHz','353_GHz', '545_GHz','857_GHz']

planck_143_instrument = pysm.Instrument({
    'nside': d['nside'],
    'frequencies' : planck_central_nu[4:5], # GHz
    'use_smoothing' : True,
    'beams' : planck_beams[4:5], # arcmin 
    'add_noise' : True,  # If True `sens_I` and `sens_Q` are required
    'noise_seed' : 0,  # Not used if `add_noise` is False
    'sens_I': planck_Isensitivities_pixel[4:5] / planck_beams[4:5]**2, # Not used if `add_noise` is False
    'sens_P': planck_Psensitivities_pixel[4:5] / planck_beams[4:5]**2, # Not used if `add_noise` is False
    'use_bandpass' : True,  # If True pass banpasses  with the key `channels`
    'channel_names' : planck_channels_names[4:5],
    'channels' : planck_channels[4:5],
    'output_units' : 'uK_RJ',
    'output_directory' : output_directory,
    'output_prefix' : output_prefix_planck,
    'pixel_indices' : np.arange(hp.nside2npix(d['nside']))})

qubic_instrument = pysm.Instrument({
    'nside': d['nside'],
    'frequencies' : central_nus, # GHz
    'use_smoothing' : False,
    'beams': np.ones_like(central_nus), # arcmin 
    'add_noise': False,  # If True `sens_I` and `sens_Q` are required
    'noise_seed' : 0,  # Not used if `add_noise` is False
    'sens_I': np.ones_like(central_nus), # Not used if `add_noise` is False
    'sens_P': np.ones_like(central_nus), # Not used if `add_noise` is False
    'use_bandpass': False,  # If True pass banpasses  with the key `channels`
    'channel_names': np.ones_like(central_nus),
    'channels': np.ones_like(central_nus),
    'output_units': 'uK_RJ',
    'output_directory': output_directory,
    'output_prefix': output_prefix_qubic,
    'pixel_indices': np.arange(hp.nside2npix(d['nside']))})

planck_143_instrument.observe(sky)
qubic_instrument.observe(sky)

x0_planck = read_map(d['nside'], output_directory, output_prefix_planck)
x0_qubic = read_map(d['nside'], output_directory, output_prefix_qubic)

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
