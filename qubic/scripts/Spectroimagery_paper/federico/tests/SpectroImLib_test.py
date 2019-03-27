#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

'Test the functions in the SpectroImLib module.'

from __future__ import division

import unittest as ut

import healpy as hp
import numpy as np
import qubic
import pysm
from pysm.nominal import models

import SpectroImLib as si

d = qubic.qubicdict.qubicDict()
d.read_from_file("parameters.dict")
p = qubic.get_pointing(d)

sky_config = {
    'synchrotron': models('s1', d['nside']),
    'dust': models('d1', d['nside']),
    'freefree': models('f1', d['nside']), #not polarized
    'cmb': models('c1', d['nside']),
    'ame': models('a1', d['nside'])}  #not polarized

sky_q = si.Qubic_sky(sky_config, d)
sky_p = si.Planck_sky(sky_config, d)
sky_p_all = si.Planck_sky(sky_config, d, band=None)
        
class TestSpectroImLib(ut.TestCase):

    def test_get_simple_sky_map(self):

        sky_config0 = {'cmb': models('c1', d['nside'])}
        sky0 = si.sky(sky_config0, d, None, None, None).get_simple_sky_map()
        sky0_q = si.Qubic_sky(sky_config0, d).get_simple_sky_map()
        sky0_p = si.Planck_sky(sky_config0, d).get_simple_sky_map()
        self.assertTrue(np.all(sky0 == sky0_q))
        self.assertTrue(np.all(sky0_p == sky0_q))
        self.assertTrue(sky0_p.shape == sky0_q.shape)
        self.assertTrue(sky0_q.shape[0] == int(d['nf_sub']))
        self.assertTrue(sky0_q.shape[1] == 12*d['nside']**2)
        self.assertTrue(sky0_q.shape[2] == 3)
        
        sky = si.sky(sky_config, d, None, None, None).get_simple_sky_map()
        skyq = si.Qubic_sky(sky_config, d).get_simple_sky_map()
        skyp = si.Planck_sky(sky_config, d).get_simple_sky_map()
        for j in range(3):
            for i in range(len(sky0)):
                self.assertTrue(np.all(sky0[i, :, j] != sky[i, :, j]))
                self.assertTrue(np.all(sky0_q[i, :, j] == sky0_p[i, :, j]))
                self.assertTrue(np.all(skyq[i, :, j] == skyp[i, :, j]))

    def test_get_sky_map(self):
        
        maps_q = sky_q.get_sky_map()
        maps_p = sky_p.get_sky_map()        
        maps_p_all = sky_p_all.get_sky_map()
        
        self.assertTrue(maps_q.shape[0] == int(d['nf_sub']))
        self.assertTrue(maps_p.shape[0] == 1)
        self.assertTrue(maps_p_all.shape[0] == 9)
        self.assertTrue(maps_q.shape[1] == 12*d['nside']**2)
        self.assertTrue(maps_p.shape[1] == 12*d['nside']**2)
        self.assertTrue(maps_p_all.shape[1] == 12*d['nside']**2)
        self.assertTrue(maps_q.shape[2] == 3)
        self.assertTrue(maps_p.shape[2] == 3)
        self.assertTrue(maps_p_all.shape[2] == 3)

        instrument_mono = pysm.Instrument({
            'nside': d['nside'],
            'frequencies' : sky_p.planck_central_nus[4:5], # GHz
            'use_smoothing' : True,
            'beams' : sky_p.planck_beams[4:5], # arcmin 
            'add_noise' : True,  # If True `sens_I` and `sens_Q` are required
            'noise_seed' : 0,  # Not used if `add_noise` is False
            'sens_I': sky_p.get_planck_sensitivity("I")[4:5], # Not used if `add_noise` is False
            'sens_P': sky_p.get_planck_sensitivity("P")[4:5], # Not used if `add_noise` is False
            'use_bandpass' : False,  # If True pass banpasses  with the key `channels`
            'channel_names' : sky_p.planck_channels_names[4:5],
            'channels' : sky_p.planck_channels[4:5],
            'output_units' : 'uK_RJ',
            'output_directory' : "./",
            'output_prefix' : "planck_sky_mono",
            'pixel_indices' : np.arange(hp.nside2npix(d['nside']))})
        sky_p_mono = si.sky(sky_config, d, instrument_mono, "./", "planck_sky_mono")
        maps_p_mono = sky_p_mono.get_sky_map()   

        instrument_mono_noiseless = pysm.Instrument({
            'nside': d['nside'],
            'frequencies' : sky_p.planck_central_nus[4:5], # GHz
            'use_smoothing' : True,
            'beams' : sky_p.planck_beams[4:5], # arcmin 
            'add_noise' : False,  # If True `sens_I` and `sens_Q` are required
            'noise_seed' : 0,  # Not used if `add_noise` is False
            'sens_I': sky_p.get_planck_sensitivity("I")[4:5], # Not used if `add_noise` is False
            'sens_P': sky_p.get_planck_sensitivity("P")[4:5], # Not used if `add_noise` is False
            'use_bandpass' : False,  # If True pass banpasses  with the key `channels`
            'channel_names' : sky_p.planck_channels_names[4:5],
            'channels' : sky_p.planck_channels[4:5],
            'output_units' : 'uK_RJ',
            'output_directory' : "./",
            'output_prefix' : "planck_sky_mono_noiseless",
            'pixel_indices' : np.arange(hp.nside2npix(d['nside']))})
        sky_p_mono_noiseless = si.sky(sky_config, d, instrument_mono_noiseless,
                                   "./", "planck_sky_mono_noiseless")
        maps_p_mono_noiseless = sky_p_mono_noiseless.get_sky_map()   

        instrument_mono_noiseless_pointless = pysm.Instrument({
            'nside': d['nside'],
            'frequencies' : sky_p.planck_central_nus[4:5], # GHz
            'use_smoothing' : False,
            'beams' : sky_p.planck_beams[4:5], # arcmin 
            'add_noise' : False,  # If True `sens_I` and `sens_Q` are required
            'noise_seed' : 0,  # Not used if `add_noise` is False
            'sens_I': sky_p.get_planck_sensitivity("I")[4:5], # Not used if `add_noise` is False
            'sens_P': sky_p.get_planck_sensitivity("P")[4:5], # Not used if `add_noise` is False
            'use_bandpass' : False,  # If True pass banpasses  with the key `channels`
            'channel_names' : sky_p.planck_channels_names[4:5],
            'channels' : sky_p.planck_channels[4:5],
            'output_units' : 'uK_RJ',
            'output_directory' : "./",
            'output_prefix' : "planck_sky_mono_noiseless_pointless",
            'pixel_indices' : np.arange(hp.nside2npix(d['nside']))})
        sky_p_mono_noiseless_pointless = si.sky(
            sky_config, d, instrument_mono_noiseless_pointless, "./",
            "planck_sky_mono_noiseless_pointless")
        maps_p_mono_noiseless_pointless = sky_p_mono_noiseless_pointless.get_sky_map()   
        self.assertTrue(np.allclose(maps_p_mono_noiseless_pointless[0],
                                    pysm.Sky(sky_config).signal()(nu=143).T))
