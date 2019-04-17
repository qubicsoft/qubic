#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

'Test the functions in the SpectroImLib module.'

from __future__ import division

from pysm.nominal import models

import unittest as ut
import healpy as hp
import numpy as np
import SpectroImLib as si

import qubic
import pysm
import os


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
        sky0 = si.sky(sky_config0, d, None).get_simple_sky_map()
        sky0_q = si.Qubic_sky(sky_config0, d).get_simple_sky_map()
        sky0_p = si.Planck_sky(sky_config0, d).get_simple_sky_map()
        self.assertTrue(np.all(sky0 == sky0_q))
        self.assertTrue(np.all(sky0_p == sky0_q))
        self.assertTrue(sky0_p.shape == sky0_q.shape)
        self.assertTrue(sky0_q.shape[0] == int(d['nf_sub']))
        self.assertTrue(sky0_q.shape[1] == 12*d['nside']**2)
        self.assertTrue(sky0_q.shape[2] == 3)
        
        sky = si.sky(sky_config, d, None).get_simple_sky_map()
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

        
    def test_qubic_sky(self):

        maps_q = sky_q.get_sky_map()
        skyq = sky_q.get_simple_sky_map()
        self.assertTrue(np.allclose(maps_q, skyq))

        
    def test_planck_sky(self):

        maps_p = sky_p.get_sky_map()        
        maps_p_all = sky_p_all.get_sky_map()
        for i in [0, 1, 2, 3, 5, 6, 7, 8]:
            self.assertFalse(np.allclose(np.std(maps_p_all[i], axis=0),
                                         np.std(maps_p[0], axis=0), atol=1e-2))
            self.assertFalse(np.allclose(np.mean(maps_p_all[i], axis=0),
                                         np.mean(maps_p[0], axis=0), atol=1e-2))
        self.assertTrue(np.allclose(np.std(maps_p_all[4], axis=0),
                                    np.std(maps_p[0], axis=0), atol=1e-2))
        self.assertTrue(np.allclose(np.mean(maps_p_all[4], axis=0),
                                    np.mean(maps_p[0], axis=0), atol=1e-2))

        instrument_nless_pless = pysm.Instrument({
            'nside': d['nside'],
            'frequencies' : sky_p.planck_central_nus[4:5], # GHz
            'use_smoothing' : False,
            'beams' : None, # arcmin 
            'add_noise' : False,
            'noise_seed' : None,
            'sens_I': None, 
            'sens_P': None, 
            'use_bandpass' : True, 
            'channel_names' : sky_p.planck_channels_names[4:5],
            'channels' : sky_p.planck_channels[4:5],
            'output_units' : 'Jysr',
            'output_directory' : "./",
            'output_prefix' : "planck_sky_pless_nless",
            'pixel_indices' : None})
        sky_p_nless_pless = si.sky(sky_config, d, instrument_nless_pless)
        maps_p_nless_pless = sky_p_nless_pless.get_sky_map()
        disintegrated_maps = np.rollaxis(
            sky_p_nless_pless.sky.signal()(nu=sky_p.planck_channels[4:5][0][0]) *
            pysm.pysm.convert_units("uK_RJ", "Jysr",
            sky_p.planck_channels[4:5][0][0])[..., None, None], 2, 1) 
        integrated_maps = np.mean(disintegrated_maps, axis=0)
        self.assertTrue(np.allclose(maps_p_nless_pless, integrated_maps))

        instrument_noisy = pysm.Instrument({
            'nside': d['nside'],
            'frequencies' : sky_p.planck_central_nus[4:5], # GHz
            'use_smoothing' : False,
            'beams' : None, # arcmin 
            'add_noise' : True,
            'noise_seed' : 0,
            'sens_I': sky_p.get_planck_sensitivity("I")[4:5], 
            'sens_P': sky_p.get_planck_sensitivity("P")[4:5], 
            'use_bandpass' : False, 
            'channel_names' : sky_p.planck_channels_names[4:5],
            'channels' : None,
            'output_units' : 'uK_RJ',
            'output_directory' : "./",
            'output_prefix' : "planck_sky_noisy",
            'pixel_indices' : None})
        sky_p_noisy = si.sky(sky_config, d, instrument_noisy)
        maps_p_noisy, noise_map_produced = sky_p_noisy.get_sky_map(
            noise_map=True)
        noise_map = maps_p_noisy[0] - sky_p.sky.signal()(
            nu=sky_p.planck_central_nus[4:5]).T
        self.assertTrue(np.allclose(noise_map, noise_map_produced))
        c = np.sqrt(hp.nside2pixarea(d['nside'], degrees=True)) * 60 # arcmin
        C = pysm.pysm.convert_units("uK_RJ", "uK_CMB",
                                    sky_p.planck_central_nus[4:5][0])
        self.assertTrue(np.allclose(np.std(noise_map, axis=0)[0] * c * C,
                                    sky_p.get_planck_sensitivity("I")[4:5],
                                    atol=1e-1))
        self.assertTrue(np.allclose(np.std(noise_map, axis=0)[1] * c * C,
                                    sky_p.get_planck_sensitivity("P")[4:5],
                                    atol=1e-1))
        self.assertTrue(np.allclose(np.std(noise_map, axis=0)[2] * c * C,
                                    sky_p.get_planck_sensitivity("P")[4:5],
                                    atol=1e-1))

        sm_pns = sky_p.planck_beams[4:5][0] * np.pi / (60 * 180)
        jysky = sky_p.instrument.apply_bandpass(sky_p.sky.signal(), sky_p.sky)
        maps_a = sky_p.instrument.smoother(jysky)[0].T
        maps_b = hp.smoothing(jysky[0], fwhm=sm_pns).T
        self.assertTrue(np.allclose(maps_a, maps_b))
        
        # instrument_mono = pysm.Instrument({
        #     'nside': d['nside'],
        #     'frequencies' : sky_p.planck_central_nus[4:5], # GHz
        #     'use_smoothing' : True,
        #     'beams' : sky_p.planck_beams[4:5], # arcmin 
        #     'add_noise' : True,  
        #     'noise_seed' : 0, 
        #     'sens_I': sky_p.get_planck_sensitivity("I")[4:5],
        #     'sens_P': sky_p.get_planck_sensitivity("P")[4:5],
        #     'use_bandpass' : False,  
        #     'channel_names' : sky_p.planck_channels_names[4:5],
        #     'channels' : None,
        #     'output_units' : 'uK_RJ',
        #     'output_directory' : "./",
        #     'output_prefix' : "planck_sky_mono",
        #     'pixel_indices' : None})
        # sky_p_mono = si.sky(sky_config, d, instrument_mono)
        # maps_p_mono = sky_p_mono.get_sky_map()
        # self.assertTrue(maps_p_mono.shape == (1, 196608, 3))

        # instrument_mono_nless = pysm.Instrument({
        #     'nside': d['nside'],
        #     'frequencies' : sky_p.planck_central_nus[4:5], # GHz
        #     'use_smoothing' : True,
        #     'beams' : sky_p.planck_beams[4:5], # arcmin 
        #     'add_noise' : False,  
        #     'noise_seed' : None,  
        #     'sens_I': None, 
        #     'sens_P': None, 
        #     'use_bandpass' : False,  
        #     'channel_names' : sky_p.planck_channels_names[4:5],
        #     'channels' : None,
        #     'output_units' : 'uK_RJ',
        #     'output_directory' : "./",
        #     'output_prefix' : "planck_sky_nless_mono",
        #     'pixel_indices' : None})
        # sky_p_mono_nless = si.sky(sky_config, d, instrument_mono_nless)
        # maps_p_mono_nless = sky_p_mono_nless.get_sky_map()
        # self.assertTrue(maps_p_mono_nless.shape == (1, 196608, 3))

        instrument_mono_nless_pless = pysm.Instrument({
            'nside': d['nside'],
            'frequencies' : sky_p.planck_central_nus[4:5], # GHz
            'use_smoothing' : False,
            'beams' : None, # arcmin 
            'add_noise' : False,
            'noise_seed' : None,
            'sens_I': None, 
            'sens_P': None, 
            'use_bandpass' : False, 
            'channel_names' : sky_p.planck_channels_names[4:5],
            'channels' : None,
            'output_units' : 'uK_RJ',
            'output_directory' : "./",
            'output_prefix' : "planck_sky_pless_nless_mono",
            'pixel_indices' : None})
        sky_p_mono_nless_pless = si.sky(
            sky_config, d, instrument_mono_nless_pless)
        maps_p_mono_nless_pless = sky_p_mono_nless_pless.get_sky_map()
        self.assertTrue(np.allclose(maps_p_mono_nless_pless[0],
                                    pysm.Sky(sky_config).signal()(nu=143).T))
